# coach.py
# Chess coaching interface: Phase 4B classifier → hysteresis filter → RAG retrieval.
#
# Hysteresis (Schmitt trigger) prevents the coach from flip-flopping every move.
# A concept needs a HIGH probability to activate and only drops when it falls LOW.
# Call reset() at the start of each new game to clear the hysteresis state.
#
# Usage
# -----
#   from src.chess_coach.rag.coach import ChessCoach
#   coach = ChessCoach()
#   coach.reset()                                          # new game
#   result = coach.analyze(fen, history_uci=["e2e4", "c7c5", "g1f3"])
#   print(result["opening"])
#   for ann in result["annotations"]:
#       print(ann["annotation"])

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chess
import torch

_CKPT_PATH = Path("data/classifier_best.pt")

# Hysteresis thresholds — a concept must cross ACTIVATE to turn on,
# but only needs to stay above HOLD to remain the coaching theme.
ACTIVATE_THRESHOLD = 0.65
HOLD_THRESHOLD     = 0.40


def _replay_fens(start_fen: str, history_uci: list[str]) -> list[str]:
    """Return a list of FENs after each move in history_uci, starting position first."""
    board = chess.Board(start_fen)
    fens  = [board.fen()]
    for uci in history_uci:
        try:
            board.push_uci(uci)
            fens.append(board.fen())
        except Exception:
            break
    return fens


class ChessCoach:
    """Phase 4B concept classifier with hysteresis gating and RAG annotation retrieval.

    Parameters
    ----------
    ckpt_path       : path to trained checkpoint (default: data/classifier_best.pt)
    n_results       : number of annotations to retrieve per query
    use_hysteresis  : True → Schmitt-trigger concept gating (recommended for live use)
                      False → raw calibrated thresholds (for evaluation / debugging)
    """

    def __init__(
        self,
        ckpt_path:      str | Path = _CKPT_PATH,
        n_results:      int        = 5,
        use_hysteresis: bool       = True,
    ) -> None:
        self._n              = n_results
        self._use_hysteresis = use_hysteresis
        self._model          = None
        self._retriever      = None
        self._ckpt_path      = Path(ckpt_path)
        self._device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._active_concepts: set[str] = set()   # hysteresis state — persists across moves

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear hysteresis state. Call at the start of every new game."""
        self._active_concepts = set()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from src.chess_coach.ml.classifier import ChessConceptClassifier

        ckpt = torch.load(str(self._ckpt_path), map_location=self._device, weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)

        # Phase detection: nnue_proj key is unique to Phase 5D; spatial_proj to Phase 4B.
        is_phase5 = any(k.startswith("nnue_proj")     for k in sd)
        is_phase4 = any(k.startswith("spatial_proj")  for k in sd) and not is_phase5

        self._model = ChessConceptClassifier(phase4=is_phase4, phase5=is_phase5).to(self._device)
        self._model.load_state_dict(sd)
        self._model.eval()

        from src.chess_coach.rag.retriever import RAGRetriever
        self._retriever = RAGRetriever()

    # ── Hysteresis ────────────────────────────────────────────────────────────

    def _apply_hysteresis(
        self, raw_probs: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """Schmitt-trigger filter over concept probabilities.

        A concept activates when its probability crosses ACTIVATE_THRESHOLD (0.65).
        Once active it stays active as long as probability stays above HOLD_THRESHOLD (0.40).
        This prevents the coach from changing its mind on every quiet move.
        """
        prob_map  = dict(raw_probs)
        threshold = {c: HOLD_THRESHOLD if c in self._active_concepts else ACTIVATE_THRESHOLD
                     for c in prob_map}
        self._active_concepts = {c for c, p in raw_probs if p >= threshold[c]}
        return sorted(
            [(c, prob_map[c]) for c in new_active],
            key=lambda t: -t[1],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        fen:          str,
        history_uci:  list[str] | None = None,
        start_fen:    str               = chess.STARTING_FEN,
        eco_override: str | None        = None,
        threshold:    float | None      = None,
    ) -> dict:
        """Analyze a position and return concepts + coaching annotations.

        Parameters
        ----------
        fen          : FEN of the position to analyze
        history_uci  : UCI moves leading to this position (oldest first)
        start_fen    : FEN of the starting position (default: standard start)
        eco_override : force an ECO code instead of auto-detecting from history
        threshold    : ignored when use_hysteresis=True; used as global threshold otherwise

        Returns
        -------
        {
            "fen":         str,
            "opening":     {eco, opening, variation} | None,
            "concepts":    [(name, prob), ...],   # sorted by probability, hysteresis-gated
            "annotations": [{annotation, game, eco, move_san, ...}, ...],
        }
        """
        self._ensure_loaded()

        history_fens: list[str] = []
        if history_uci:
            history_fens = _replay_fens(start_fen, history_uci)

        if self._use_hysteresis:
            # Get all concept probabilities above the hold floor so the Schmitt
            # trigger can see concepts rising toward ACTIVATE and falling below HOLD.
            raw = self._model.predict_concepts(
                fen, history_rich=None, threshold=HOLD_THRESHOLD * 0.5
            )
            concepts = self._apply_hysteresis(raw)
        else:
            concepts = self._model.predict_concepts(
                fen, history_rich=None, threshold=threshold
            )

        concept_names = [name for name, _ in concepts]

        opening = (
            self._retriever._eco_db.get(eco_override) if eco_override
            else self._retriever.identify_opening(history_fens) if history_fens
            else None
        )

        annotations = self._retriever.retrieve(
            fen,
            history_fens = history_fens or None,
            concepts     = concept_names,
            eco_override = eco_override,
            n            = self._n,
        )

        return {
            "fen":         fen,
            "opening":     opening,
            "concepts":    concepts,
            "annotations": annotations,
        }

    def opening_name(self, history_uci: list[str], start_fen: str = chess.STARTING_FEN) -> str | None:
        """Return a human-readable opening name for a move sequence, or None."""
        self._ensure_loaded()
        history_fens = _replay_fens(start_fen, history_uci)
        info = self._retriever.identify_opening(history_fens)
        if info is None:
            return None
        name = info["opening"]
        if info.get("variation"):
            name += f", {info['variation']}"
        return name
