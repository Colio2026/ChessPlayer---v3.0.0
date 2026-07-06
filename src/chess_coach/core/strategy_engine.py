"""
core/strategy_engine.py
========================
Public API entry point. Orchestrates all layers and returns CoachOutput.

Usage
-----
    from chess_coach import StrategyEngine
    engine = StrategyEngine.from_config(cfg)          # preferred
    engine = StrategyEngine(stockfish_path, ...)      # manual construction

    output: CoachOutput = engine.analyse(board, player_side='white')

Config-driven construction (from_config)
-----------------------------------------
Reads coach section from the application config dict:
    cfg['coach']['pgn_source']   — PGN for GM precedents
    cfg['coach']['phrase_db']    — phrase database path
    cfg['coach']['min_rating']   — minimum rating filter
    cfg['coach']['movetime_ms']  — Stockfish analysis time
    cfg['coach']['auto_index']   — auto-build coach_positions on first run
    cfg['paths']['data_dir']     — base data directory
    cfg['engine']['path']        — Stockfish executable

Swapping from Carlsen.pgn to the 6M game database
---------------------------------------------------
Change coach.pgn_source in default.yaml to the new PGN path.
On the next StrategyEngine instantiation, ensure_indexed() detects
the source change and automatically rebuilds coach_positions from
the games already indexed by the browser indexer — no manual step.

Layer execution order
---------------------
  1. Extractors      — 6 extractors produce MetricSignal list
  2. Phase filter    — classify phase, re-weight signals
  3. Strategy scores — blitz, flank, fortress, feint (0.0–1.0)
  4. Conflict resolver — cascade rules → primary/secondary/confidence
  5. Plan recommender — move_flags + weakness_squares
  6. Narrator        — phrase DB slot-filling → CoachOutput
"""
from __future__ import annotations

import chess
from pathlib import Path

from chess_coach.core.data_types        import MetricSignal, CoachOutput
from chess_coach.core.board_utils       import get_phase
from chess_coach.core.phase_filter      import apply_phase_filter
from chess_coach.core.conflict_resolver import resolve
from chess_coach.extractors.king_safety      import extract_king_safety
from chess_coach.extractors.space_control    import extract_space_control
from chess_coach.extractors.piece_mobility   import extract_piece_mobility
from chess_coach.extractors.pawn_structure   import extract_pawn_structure
from chess_coach.extractors.material_balance import extract_material_balance
from chess_coach.extractors.tactic_scanner   import extract_tactics
from chess_coach.strategies.blitz_detector    import score_blitz
from chess_coach.strategies.flank_detector    import score_flank
from chess_coach.strategies.fortress_detector import score_fortress
from chess_coach.strategies.feint_detector    import score_feint
from chess_coach.database.pattern_matcher     import PatternMatcher
from chess_coach.database.phrase_db           import PhraseDB
from chess_coach.database.pgn_indexer         import ensure_indexed
from chess_coach.coach.narrator               import assemble as narrator_assemble
from chess_coach.coach.plan_recommender       import recommend as plan_recommend


class StrategyEngine:
    """
    Main entry point for the chess coach backend.

    Prefer StrategyEngine.from_config(cfg) for application use.
    Direct construction is available for testing.

    Parameters
    ----------
    stockfish_path : str
        Absolute path to Stockfish executable. Empty string = no engine.
    db_path : str
        Path to chess_coach.db (phrase DB). Empty = fallback phrases only.
    pgn_index_path : str
        Path to index.sqlite (shared with browser). Empty = no precedents.
    pgn_source_path : str
        Path to the PGN configured in coach.pgn_source.
        Used by ensure_indexed to detect source changes.
    movetime_ms : int
        Stockfish analysis time per position (ms).
    min_rating : int
        Minimum rating filter for GM precedent queries.
    auto_index : bool
        If True, call ensure_indexed() on construction.
    """

    def __init__(
        self,
        stockfish_path:  str  = '',
        db_path:         str  = '',
        pgn_index_path:  str  = '',
        pgn_source_path: str  = '',
        movetime_ms:     int  = 2000,
        min_rating:      int  = 0,
        auto_index:      bool = True,
        progress_cb            = None,
    ) -> None:
        self.stockfish_path  = stockfish_path
        self.db_path         = db_path
        self.pgn_index_path  = pgn_index_path
        self.pgn_source_path = pgn_source_path
        self.movetime_ms     = movetime_ms
        self._bridge         = None
        self._engine_ok      = False

        # ── Database layer ────────────────────────────────────────────────
        self._matcher   = PatternMatcher(pgn_index_path, min_rating=min_rating)
        self._phrase_db = PhraseDB(db_path)

        # ── Auto-index: build/rebuild coach_positions if needed ───────────
        # Detects source change automatically — swap coach.pgn_source in
        # default.yaml and restart; coach_positions rebuilds from the new
        # games already catalogued by the browser indexer.
        if auto_index and pgn_index_path:
            try:
                ensure_indexed(
                    db_path        = pgn_index_path,
                    stockfish_path = stockfish_path,
                    movetime_ms    = 5,
                    min_rating     = min_rating,
                    verbose        = False,
                    pgn_source     = pgn_source_path,
                    progress_cb    = progress_cb,
                    force          = False,
                )
            except Exception:
                pass

        # ── Stockfish ─────────────────────────────────────────────────────
        if stockfish_path:
            self._try_start_engine()

    @classmethod
    def from_config(cls, cfg: dict, progress_cb=None) -> 'StrategyEngine':
        """
        Construct from the application config dict.

        Reads coach section — see default.yaml for all keys.
        This is the preferred constructor for application code.
        """
        data_dir   = Path(cfg.get('paths', {}).get('data_dir', 'data'))
        coach_cfg  = cfg.get('coach', {})
        engine_cfg = cfg.get('engine', {})

        pgn_source = coach_cfg.get('pgn_source', '')
        if pgn_source and not Path(pgn_source).is_absolute():
            pgn_source = str(data_dir.parent / pgn_source)

        phrase_db = coach_cfg.get('phrase_db', 'data/chess_coach.db')
        if phrase_db and not Path(phrase_db).is_absolute():
            phrase_db = str(data_dir.parent / phrase_db)

        # index.sqlite is always in data_dir (created by browser indexer)
        index_path = str(data_dir / 'index.sqlite')

        return cls(
            stockfish_path  = engine_cfg.get('path', ''),
            db_path         = phrase_db,
            pgn_index_path  = index_path,
            pgn_source_path = pgn_source,
            movetime_ms     = coach_cfg.get('movetime_ms', 2000),
            min_rating      = coach_cfg.get('min_rating', 0),
            auto_index      = coach_cfg.get('auto_index', True),
            progress_cb     = progress_cb,
        )

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
        Analyse a position and return a fully assembled CoachOutput.

        Parameters
        ----------
        board : chess.Board
            Current position to analyse.
        move_history : list[str] | None
            UCI move strings, oldest first (optional context).
        player_side : str
            'white' | 'black' — the side being coached.
        history_boards : list[chess.Board] | None
            Recent board states for trend detection (flank/feint detectors).
        history_signals : list[list[MetricSignal]] | None
            Pre-computed signals from previous positions (optimisation).
        """
        if not self._engine_ok and self.stockfish_path:
            self._try_start_engine()

        phase = get_phase(board)

        # ── Step 1: Extractors ────────────────────────────────────────────
        eval_result = None
        if self._engine_ok and self._bridge:
            try:
                eval_result = self._bridge.get_eval(board.fen())
            except Exception:
                pass

        raw_signals = self._run_extractors(board, history_boards, eval_result, phase)

        # ── Step 2: Phase filter ──────────────────────────────────────────
        phase, signals = apply_phase_filter(raw_signals, board)

        # ── Step 3: Strategy scoring ──────────────────────────────────────
        db_feint = self._matcher.db_confirms_feint(board, phase)
        scores = {
            'blitz':    score_blitz(signals,    player_side, history_signals),
            'flank':    score_flank(signals,    player_side, history_signals),
            'fortress': score_fortress(signals, player_side, history_signals),
            'feint':    score_feint(signals,    player_side, history_signals,
                                   db_confirmation=db_feint),
        }

        # ── Step 4: Conflict resolution ───────────────────────────────────
        context = _build_context(signals, player_side)
        result  = resolve(scores, context, phase, player_side)

        # ── Step 5: Plan recommender ──────────────────────────────────────
        bridge_for_recommender = self._bridge if self._engine_ok else None
        move_flags, weakness_squares = plan_recommend(
            board, signals, player_side, result.primary,
            stockfish_bridge=bridge_for_recommender,
        )

        # ── Step 6: GM precedents ─────────────────────────────────────────
        gm_precedents = self._matcher.query(board, result.primary, phase)

        # ── Step 7: Narrator assembles CoachOutput ────────────────────────
        return narrator_assemble(
            result           = result,
            phase            = phase,
            signals          = signals,
            player_side      = player_side,
            phrase_db        = self._phrase_db,
            gm_precedents    = gm_precedents,
            move_flags       = move_flags,
            weakness_squares = weakness_squares,
        )

    def close(self) -> None:
        """Shut down engine and DB connections cleanly."""
        if self._bridge and self._engine_ok:
            try:
                self._bridge.stop()
            except Exception:
                pass
        self._matcher.close()
        self._phrase_db.close()

    # ── Internal ──────────────────────────────────────────────────────────

    def _try_start_engine(self) -> None:
        try:
            from core.stockfish_bridge import StockfishBridge
            self._bridge = StockfishBridge(self.stockfish_path, self.movetime_ms)
            self._bridge.start()
            self._engine_ok = self._bridge.is_running
        except Exception:
            self._engine_ok = False
            self._bridge    = None

    def _run_extractors(
        self,
        board: chess.Board,
        history_boards: list[chess.Board] | None,
        eval_result,
        phase: str,
    ) -> list[MetricSignal]:
        signals: list[MetricSignal] = []
        signals.extend(extract_king_safety(board, phase))
        signals.extend(extract_space_control(board, history_boards, phase))
        signals.extend(extract_piece_mobility(board, history_boards, phase))
        signals.extend(extract_pawn_structure(board, phase))
        signals.extend(extract_material_balance(board, eval_result, phase))
        signals.extend(extract_tactics(board, phase))
        return signals


# ── Module-level helpers ──────────────────────────────────────────────────────

def _build_context(signals: list[MetricSignal], player_side: str) -> dict:
    """Extract the two values the conflict resolver needs."""
    opp = 'black' if player_side == 'white' else 'white'
    def get(metric: str, side: str) -> float:
        return max(
            (s.score for s in signals if s.metric_name == metric and s.side == side),
            default=0.0
        )
    return {
        'eval_deficit':  get('eval_deficit',  player_side),
        'king_exposure': get('king_exposure', opp),
    }
