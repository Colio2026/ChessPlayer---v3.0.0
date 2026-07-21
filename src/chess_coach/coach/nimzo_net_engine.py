"""
coach/nimzo_net_engine.py
==========================
Drop-in replacement for StrategyEngine that uses ChessConceptClassifier
(Nimzo-Net) instead of rule-based extractors.

Public API mirrors StrategyEngine so coach_panel.py needs no structural
changes — only the InitWorker import changes.

    engine = NimzoNetEngine.from_config(config)
    output = engine.analyse(board, player_side='white')
    output = engine.analyse_from_pv(board, pv_uci, player_side, score_cp=None)
    engine.close()   # no-op — no Stockfish process to clean up
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import torch

# tools/ is in the project root, not in src/.  When the app runs as
# `python src/main.py`, only src/ lands on sys.path — tools becomes
# unreachable.  Ensure the project root is present so label_positions
# can be imported by predict_concepts().
_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from chess_coach.core.data_types        import CoachOutput
from chess_coach.core.board_utils       import get_phase
from chess_coach.core.conflict_resolver import ResolverResult
from chess_coach.coach.narrator         import assemble
from chess_coach.database.phrase_db     import PhraseDB
from chess_coach.ml.concept_signal_adapter import adapt, infer_strategy
from chess_coach.ml.paths               import CLASSIFIER_BEST as _CHECKPOINT
from chess_coach.ml.evaluate            import load_thresholds


class NimzoNetEngine:

    def __init__(self, model, phrase_db: PhraseDB) -> None:
        self._model     = model
        self._phrase_db = phrase_db

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "NimzoNetEngine":
        from chess_coach.ml.classifier import ChessConceptClassifier

        ckpt_path = Path(
            config.get('coach', {}).get('nimzo_checkpoint', str(_CHECKPOINT))
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"NimzoNet checkpoint not found: {ckpt_path}\n"
                "Run the training pipeline first: .\\retrain_and_reparse.ps1"
            )

        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        model = ChessConceptClassifier(phase4=True)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        db_path = config.get('coach', {}).get('phrase_db', '')
        phrase_db = PhraseDB(db_path)

        return cls(model, phrase_db)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(
        self,
        board:       chess.Board,
        player_side: str = 'white',
    ) -> CoachOutput:
        """Analyse a board position and return coaching output."""
        fen   = board.fen()
        phase = get_phase(board)

        concepts = self._model.predict_concepts(fen)

        signals = adapt(concepts, board, phase, player_side)
        primary, secondary, confidence, tie_band = infer_strategy(concepts)

        result = ResolverResult(
            primary    = primary,
            secondary  = secondary,
            confidence = confidence,
            tie_band   = tie_band,
        )

        weakness_squares = list({
            sq for sig in signals for sq in sig.key_squares
        })

        return assemble(
            result           = result,
            phase            = phase,
            signals          = signals,
            player_side      = player_side,
            phrase_db        = self._phrase_db,
            gm_precedents    = [],
            move_flags       = [],
            weakness_squares = weakness_squares,
        )

    def analyse_from_pv(
        self,
        board:       chess.Board,
        pv_uci:      list[str],
        player_side: str,
        score_cp:    int | None = None,
    ) -> CoachOutput:
        """
        Analyse the position reached after the first PV move.
        Each PV line gives a different coaching output based on where it leads.
        """
        if pv_uci:
            try:
                b2   = board.copy()
                move = chess.Move.from_uci(pv_uci[0])
                if move in b2.legal_moves:
                    b2.push(move)
                    return self.analyse(b2, player_side)
            except Exception:
                pass
        return self.analyse(board, player_side)

    def close(self) -> None:
        """No-op — NimzoNet has no subprocess to clean up."""
        pass
