"""
core/strategy_engine.py
========================
Public API entry point. Orchestrates all layers and returns CoachOutput.

Usage:
    from chess_coach import StrategyEngine
    engine = StrategyEngine(stockfish_path, db_path, pgn_index_path)
    output: CoachOutput = engine.analyse(board, move_history, player_side)

Layer execution order:
  1. phase_filter    — classify phase, re-weight signals
  2. extractors      — run all 6 extractors (with Stockfish eval if available)
  3. phase_filter    — apply weights to extractor output
  4. strategy scorers — blitz, flank, fortress, feint
  5. conflict_resolver — cascade rules → primary/secondary
  6. CoachOutput     — assemble (narrator/plan_recommender stubs for now)

Stockfish unavailable:
    If the engine cannot start, analyse() returns a CoachOutput with
    confidence=0.0 and strategy_primary='general' with a headline
    indicating the engine is unavailable. All signal lists are empty.
    The GUI can display this gracefully.
"""
from __future__ import annotations

import chess
from pathlib import Path
from dataclasses import dataclass, field

from core.data_types  import MetricSignal, CoachOutput, GMPrecedent
from core.board_utils import get_phase
from core.phase_filter import apply_phase_filter
from core.conflict_resolver import resolve
from extractors.king_safety    import extract_king_safety
from extractors.space_control  import extract_space_control
from extractors.piece_mobility import extract_piece_mobility
from extractors.pawn_structure import extract_pawn_structure
from extractors.material_balance import extract_material_balance
from extractors.tactic_scanner import extract_tactics
from strategies.blitz_detector   import score_blitz
from strategies.flank_detector   import score_flank
from strategies.fortress_detector import score_fortress
from strategies.feint_detector   import score_feint
from database.pattern_matcher import PatternMatcher
from database.phrase_db       import PhraseDB


class StrategyEngine:
    """
    Main entry point for the chess coach backend.

    Parameters
    ----------
    stockfish_path : str
        Absolute path to the Stockfish executable.
    db_path : str
        Path to chess_coach.db (phrase database). May be empty string
        if not yet built — narrator will use fallback phrases.
    pgn_index_path : str
        Path to game_index.db (GM precedent DB). May be empty string.
    movetime_ms : int
        Stockfish analysis time per position in milliseconds.
    """

    def __init__(
        self,
        stockfish_path: str = '',
        db_path: str = '',
        pgn_index_path: str = '',
        pgn_source_path: str = '',
        movetime_ms: int = 2000,
        auto_index: bool = True,
    ) -> None:
        self.stockfish_path  = stockfish_path
        self.db_path         = db_path
        self.pgn_index_path  = pgn_index_path
        self.pgn_source_path = pgn_source_path
        self.movetime_ms     = movetime_ms
        self._bridge         = None
        self._engine_ok      = False
        self._matcher        = PatternMatcher(pgn_index_path)
        self._phrase_db      = PhraseDB(db_path)

        # Auto-build coach_positions on first run if index.sqlite exists but
        # coach_positions is empty — uses existing game offsets, no re-parsing.
        if pgn_index_path:
            try:
                ensure_indexed(pgn_index_path, stockfish_path, movetime_ms)
            except Exception:
                pass

        if stockfish_path:
            self._try_start_engine()

        # ── Auto-index on first run ────────────────────────────────────────
        # If a game_index path is specified but the file doesn't exist yet,
        # and a source PGN is available, build the index automatically.
        if (auto_index and pgn_index_path and pgn_source_path
                and not Path(pgn_index_path).exists()
                and Path(pgn_source_path).exists()):
            self._auto_build_index()

        # ── Wire up DB layer ───────────────────────────────────────────────
        self._matcher = PatternMatcher(pgn_index_path) if pgn_index_path else PatternMatcher()
        self._phrase_db = PhraseDB(db_path) if db_path else PhraseDB()

    def _try_start_engine(self) -> None:
        try:
            from core.stockfish_bridge import StockfishBridge
            self._bridge = StockfishBridge(self.stockfish_path, self.movetime_ms)
            self._bridge.start()
            self._engine_ok = self._bridge.is_running
        except Exception:
            self._engine_ok = False
            self._bridge = None

    # ── Public API ────────────────────────────────────────────────────────

    def analyse(
        self,
        board: chess.Board,
        move_history: list[str] | None = None,
        player_side: str = 'white',
        history_boards: list[chess.Board] | None = None,
        history_signals: list[list[MetricSignal]] | None = None,
    ) -> CoachOutput:
        """
        Analyse a position and return a CoachOutput.

        Parameters
        ----------
        board : chess.Board
            Current position.
        move_history : list[str] | None
            UCI move strings, oldest first.
        player_side : str
            'white' | 'black' — the side being coached.
        history_boards : list[chess.Board] | None
            Board states from last N moves for trend detection.
        history_signals : list[list[MetricSignal]] | None
            Pre-computed signals from previous positions (optional optimisation).
        """
        if not self._engine_ok and self.stockfish_path:
            self._try_start_engine()

        # ── Graceful degradation if Stockfish unavailable ─────────────────
        if not self._engine_ok and self.stockfish_path:
            return self._unavailable_output(player_side, get_phase(board))

        # ── Step 1: Run extractors ────────────────────────────────────────
        eval_result = None
        if self._engine_ok and self._bridge:
            try:
                eval_result = self._bridge.get_eval(board.fen())
            except Exception:
                pass

        raw_signals = self._run_extractors(
            board, history_boards, eval_result
        )

        # ── Step 2: Phase filter — classify and re-weight ─────────────────
        phase, signals = apply_phase_filter(raw_signals, board)

        # ── Step 3: Strategy scoring ──────────────────────────────────────
        scores = {
            'blitz':   score_blitz(signals,   player_side, history_signals),
            'flank':   score_flank(signals,   player_side, history_signals),
            'fortress': score_fortress(signals, player_side, history_signals),
            'feint':   score_feint(
                signals, player_side, history_signals,
                db_confirmation=self._matcher.db_confirms_feint(board, phase)
                    if hasattr(self, '_matcher') else False,
            ),
        }

        # ── Step 4: Context lookup for conflict resolver ───────────────────
        context = _build_context(signals, player_side)

        # ── Step 5: Conflict resolution ───────────────────────────────────
        result = resolve(scores, context, phase, player_side)

        # ── Step 6: Assemble CoachOutput ──────────────────────────────────
        return CoachOutput(
            strategy_primary   = result.primary,
            strategy_secondary = result.secondary,
            confidence         = result.confidence,
            phase              = phase,
            headline           = _headline(result, scores, phase),
            plan_sentences     = _assemble_plan(
                self._phrase_db, result.primary, phase, signals
            ),
            tactic_hints       = _tactic_hints(signals, player_side),
            move_flags         = [],
            weakness_squares   = _weakness_squares(signals, player_side),
            gm_precedents      = (
                self._matcher.query(board, result.primary, phase)
                if hasattr(self, '_matcher') else []
            ),
            signal_dump        = signals,
        )

    def _auto_build_index(self) -> None:
        """Build game_index.db from the source PGN on first run."""
        try:
            from database.pgn_indexer import build_index
            print(f"chess_coach: building game index from {self.pgn_source_path} ...")
            build_index(
                pgn_path       = self.pgn_source_path,
                db_path        = self.pgn_index_path,
                stockfish_path = self.stockfish_path,
                movetime_ms    = min(self.movetime_ms, 200),  # fast for indexing
                verbose        = True,
            )
            print(f"chess_coach: game index ready at {self.pgn_index_path}")
        except Exception as e:
            print(f"chess_coach: auto-indexing failed ({e}) — pattern matching unavailable")

    def close(self) -> None:
        """Shut down the Stockfish engine and DB connections cleanly."""
        if self._bridge and self._engine_ok:
            try:
                self._bridge.stop()
            except Exception:
                pass
        if hasattr(self, '_matcher'):
            self._matcher.close()
        if hasattr(self, '_phrase_db'):
            self._phrase_db.close()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _run_extractors(
        self,
        board: chess.Board,
        history_boards: list[chess.Board] | None,
        eval_result,
    ) -> list[MetricSignal]:
        phase = get_phase(board)
        signals: list[MetricSignal] = []
        signals.extend(extract_king_safety(board, phase))
        signals.extend(extract_space_control(board, history_boards, phase))
        signals.extend(extract_piece_mobility(board, history_boards, phase))
        signals.extend(extract_pawn_structure(board, phase))
        signals.extend(extract_material_balance(board, eval_result, phase))
        signals.extend(extract_tactics(board, phase))
        return signals

    @staticmethod
    def _unavailable_output(player_side: str, phase: str) -> CoachOutput:
        return CoachOutput(
            strategy_primary   = 'general',
            strategy_secondary = None,
            confidence         = 0.0,
            phase              = phase,
            headline           = 'Stockfish engine unavailable — positional analysis only.',
            plan_sentences     = ['Check your Stockfish path in settings.',
                                  'Positional metrics will still function without engine eval.'],
            tactic_hints       = [],
            move_flags         = [],
            weakness_squares   = [],
            gm_precedents      = self._matcher.query(
                board, result.primary, phase
            ),
            signal_dump        = [],
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _assemble_plan(
    phrase_db: 'PhraseDB',
    strategy: str,
    phase: str,
    signals: list[MetricSignal],
) -> list[str]:
    """Fill the 4-slot plan using the phrase DB, fall back to action_hints."""
    if phrase_db.is_available:
        fragments = phrase_db.get_fragments(strategy, phase, signals)
        sentences = []
        for slot in ('diagnosis', 'evidence', 'plan', 'urgency'):
            sentences.extend(fragments.get(slot, []))
        if len(sentences) >= 2:
            return sentences[:4]

    # Fallback: use action_hints from the top signals
    hints = [s.action_hint for s in sorted(signals, key=lambda x: x.score, reverse=True)
             if s.action_hint][:4]
    if len(hints) >= 2:
        return hints
    return ['Position analysed.', 'Phrase database not yet available.']


def _build_context(signals: list[MetricSignal], player_side: str) -> dict:
    opp = 'black' if player_side == 'white' else 'white'
    def get(metric, side):
        return max((s.score for s in signals if s.metric_name == metric and s.side == side), default=0.0)
    return {
        'eval_deficit':  get('eval_deficit', player_side),
        'king_exposure': get('king_exposure', opp),
    }


def _headline(result, scores: dict, phase: str) -> str:
    strategy_names = {
        'blitz':   'Kingside Attack',
        'flank':   'Flank Squeeze',
        'fortress': 'Fortress Defence',
        'feint':   'Positional Feint',
        'general': 'Positional Play',
    }
    name = strategy_names.get(result.primary, result.primary.title())
    conf = result.confidence
    if result.tie_band and result.secondary:
        sec  = strategy_names.get(result.secondary, result.secondary.title())
        return f'{name} or {sec} — two plans in tension (confidence {conf:.0%})'
    return f'{name} recommended — {phase} ({conf:.0%} confidence)'


def _tactic_hints(signals: list[MetricSignal], player_side: str) -> list[str]:
    tactic_sigs = [s for s in signals
                   if s.metric_name.startswith('tactic_') and s.side == player_side]
    tactic_sigs.sort(key=lambda s: s.score, reverse=True)
    return [s.action_hint for s in tactic_sigs[:3] if s.action_hint]


def _weakness_squares(signals: list[MetricSignal], player_side: str) -> list[str]:
    opp = 'black' if player_side == 'white' else 'white'
    squares: list[str] = []
    for s in signals:
        if s.side == opp and s.score > 0.40:
            squares.extend(s.key_squares)
    return list(dict.fromkeys(squares))[:8]
