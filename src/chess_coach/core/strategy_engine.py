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
    cfg['coach']['phrase_db']    — phrase database path
    cfg['coach']['min_rating']   — minimum rating filter
    cfg['coach']['movetime_ms']  — Stockfish analysis time
    cfg['paths']['data_dir']     — base data directory
    cfg['engine']['path']        — Stockfish executable

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
    movetime_ms : int
        Stockfish analysis time per position (ms).
    min_rating : int
        Minimum rating filter for GM precedent queries.
    """

    def __init__(
        self,
        stockfish_path: str = '',
        db_path:        str = '',
        pgn_index_path: str = '',
        movetime_ms:    int = 2000,
        min_rating:     int = 0,
    ) -> None:
        self.stockfish_path = stockfish_path
        self.db_path        = db_path
        self.pgn_index_path = pgn_index_path
        self.movetime_ms    = movetime_ms
        self._bridge        = None
        self._engine_ok     = False

        # ── Database layer ────────────────────────────────────────────────
        self._matcher   = PatternMatcher(pgn_index_path, min_rating=min_rating)
        self._phrase_db = PhraseDB(db_path)

        # ── Stockfish ─────────────────────────────────────────────────────
        if stockfish_path:
            self._try_start_engine()

    @classmethod
    def from_config(cls, cfg: dict) -> 'StrategyEngine':
        """
        Construct from the application config dict.

        Reads coach section — see default.yaml for all keys.
        This is the preferred constructor for application code.
        """
        data_dir   = Path(cfg.get('paths', {}).get('data_dir', 'data'))
        coach_cfg  = cfg.get('coach', {})
        engine_cfg = cfg.get('engine', {})

        phrase_db = coach_cfg.get('phrase_db', 'data/chess_coach.db')
        if phrase_db and not Path(phrase_db).is_absolute():
            phrase_db = str(data_dir.parent / phrase_db)

        # index.sqlite is always in data_dir (created by browser indexer)
        index_path = str(data_dir / 'index.sqlite')

        return cls(
            stockfish_path = engine_cfg.get('path', ''),
            db_path        = phrase_db,
            pgn_index_path = index_path,
            movetime_ms    = coach_cfg.get('movetime_ms', 2000),
            min_rating     = coach_cfg.get('min_rating', 0),
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

    def analyse_from_pv(
        self,
        board:       chess.Board,
        pv_uci:      list[str],
        player_side: str = 'white',
        score_cp:    int | None = None,
    ) -> CoachOutput:
        """
        Explain Stockfish's recommended PV line using its own eval term breakdown
        plus extractor-based signal deltas (works even when SF16 NNUE gives no
        classical breakdown table).

        Falls back to analyse() if the bridge is unavailable or explain_pv fails.
        """
        if self._engine_ok and self._bridge and pv_uci:
            try:
                pv_exp = self._bridge.explain_pv(board, pv_uci, player_side)
                if pv_exp is not None:
                    game_theme    = self._get_game_theme(board, player_side)
                    phase         = get_phase(board)
                    gm_precedents = self._matcher.query(board, pv_exp.strategy, phase)
                    pv_signal_rows = self._compute_pv_signal_deltas(
                        board, pv_uci, player_side, phase
                    )
                    pv_line_text   = _format_pv_line(board, pv_uci, pv_exp.pv_san)
                    from chess_coach.coach.explainer import assemble as explainer_assemble
                    return explainer_assemble(
                        pv_exp, phase, gm_precedents, player_side,
                        game_theme=game_theme,
                        score_cp=score_cp,
                        pv_signal_rows=pv_signal_rows,
                        pv_uci=list(pv_uci),
                        pv_line_text=pv_line_text,
                    )
            except Exception as _exc:
                import traceback, sys
                print(f"[coach] analyse_from_pv error: {_exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
        return self.analyse(board, player_side=player_side)

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

    def _get_game_theme(
        self,
        board: chess.Board,
        player_side: str = 'white',
    ) -> dict:
        """
        Run all 6 extractors over the last 40 positions of the current game to
        derive the overall strategic theme.  Pure Python — no Stockfish call.

        Returns
        -------
        dict with keys:
            primary    : str   — dominant strategy label
            secondary  : str | None
            confidence : float
            signals    : list[MetricSignal]  — from current position (with history)
            n_positions: int   — how many positions were scanned
        """
        moves = list(board.move_stack)
        history_boards: list[chess.Board] = []
        if moves:
            b = chess.Board()
            window = moves[-40:]  # last 40 half-moves = 20 full moves
            for m in moves[:len(moves) - len(window)]:
                b.push(m)
            for m in window:
                b.push(m)
                history_boards.append(b.copy())

        phase = get_phase(board)
        raw_signals = self._run_extractors(board, history_boards or None, None, phase)
        _, signals = apply_phase_filter(raw_signals, board)

        db_feint = self._matcher.db_confirms_feint(board, phase)
        scores = {
            'blitz':    score_blitz(signals,    player_side, None),
            'flank':    score_flank(signals,    player_side, None),
            'fortress': score_fortress(signals, player_side, None),
            'feint':    score_feint(signals,    player_side, None,
                                   db_confirmation=db_feint),
        }
        context = _build_context(signals, player_side)
        result  = resolve(scores, context, phase, player_side)

        return {
            'primary':     result.primary,
            'secondary':   result.secondary,
            'confidence':  result.confidence,
            'signals':     signals,
            'n_positions': len(history_boards),
        }

    def _compute_pv_signal_deltas(
        self,
        board:       chess.Board,
        pv_uci:      list[str],
        player_side: str,
        phase:       str,
    ) -> list[tuple]:
        """
        Run all 6 extractors on the position before and after the PV line,
        then compute per-term deltas from the player's perspective.

        Returns list of (label, before, after, delta) tuples, where positional
        terms are in normalised [0-1] signal units and Material is in pawn units.
        This bypasses the SF16 NNUE limitation where get_eval_terms_fen() returns
        no classical breakdown table.
        """
        opp = 'black' if player_side == 'white' else 'white'

        board_after = board.copy()
        for uci in pv_uci:
            try:
                board_after.push(chess.Move.from_uci(uci))
            except Exception:
                break

        sigs_b = self._run_extractors(board,       None, None, phase)
        sigs_a = self._run_extractors(board_after, None, None, phase)

        def get(sigs, names, side):
            return sum(s.score for s in sigs if s.metric_name in names and s.side == side)

        # King safety: opponent exposed (good) minus self exposed (bad)
        _ks = {'king_exposure'}
        def ks(s):
            return get(s, _ks, opp) - get(s, _ks, player_side)

        # Mobility: player ratio minus opponent ratio
        _mob = {'piece_mobility_ratio'}
        def mob(s):
            return get(s, _mob, player_side) - get(s, _mob, opp)

        # Space: player space minus opponent space
        _spc = {'space_delta_queenside', 'space_delta_kingside'}
        def spc(s):
            return get(s, _spc, player_side) - get(s, _spc, opp)

        # Threats: player tactic signals minus opponent
        _thr = {'tactic_pin', 'tactic_fork', 'tactic_skewer', 'tactic_discovery'}
        def thr(s):
            return get(s, _thr, player_side) - get(s, _thr, opp)

        # Passed pawns: player minus opponent
        _pp = {'passed_pawn'}
        def ppwn(s):
            return get(s, _pp, player_side) - get(s, _pp, opp)

        # Pawn structure: outpost advantage minus weakness penalty
        _out = {'outpost_occupation'}
        _wk  = {'weak_pawns'}
        def pstr(s):
            my_  = get(s, _out, player_side) - get(s, _wk, player_side)
            opp_ = get(s, _out, opp)         - get(s, _wk, opp)
            return my_ - opp_

        # Material in actual pawn units (not normalised signal)
        PVAL = {chess.QUEEN: 9.0, chess.ROOK: 5.0,
                chess.BISHOP: 3.25, chess.KNIGHT: 3.0, chess.PAWN: 1.0}
        pc = chess.WHITE if player_side == 'white' else chess.BLACK
        oc = not pc

        def mat(b):
            my_  = sum(v * len(b.pieces(pt, pc)) for pt, v in PVAL.items())
            opp_ = sum(v * len(b.pieces(pt, oc)) for pt, v in PVAL.items())
            return my_ - opp_

        rows: list[tuple] = []
        for label, fn in [
            ('King safety',  ks),
            ('Mobility',     mob),
            ('Space',        spc),
            ('Threats',      thr),
            ('Passed pawns', ppwn),
        ]:
            b = round(fn(sigs_b), 2)
            a = round(fn(sigs_a), 2)
            rows.append((label, b, a, round(a - b, 2)))

        mb = round(mat(board),       2)
        ma = round(mat(board_after), 2)
        rows.append(('Material', mb, ma, round(ma - mb, 2)))

        pb = round(pstr(sigs_b), 2)
        pa = round(pstr(sigs_a), 2)
        rows.append(('Pawns', pb, pa, round(pa - pb, 2)))

        return rows

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

def _format_pv_line(
    board: chess.Board,
    pv_uci: list[str],
    pv_san: list[str],
) -> str:
    """
    Format PV moves with move numbers: '11. Nf3 Nc6  12. Bb5 a6'.
    Pairs white+black moves with two spaces between full-move groups.
    """
    if not pv_san:
        return ''
    tmp         = board.copy()
    pairs:  list[list[str]] = []
    current:    list[str]   = []
    for i, san in enumerate(pv_san):
        if tmp.turn == chess.WHITE:
            if current:
                pairs.append(current)
                current = []
            current.append(f"{tmp.fullmove_number}. {san}")
        else:
            if i == 0:
                current.append(f"{tmp.fullmove_number}... {san}")
            else:
                current.append(san)
        if i < len(pv_uci):
            try:
                tmp.push(chess.Move.from_uci(pv_uci[i]))
            except Exception:
                break
    if current:
        pairs.append(current)
    return '  '.join(' '.join(p) for p in pairs)


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
