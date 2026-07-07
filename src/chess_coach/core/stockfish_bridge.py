"""
core/stockfish_bridge.py
========================
Typed interface between the coach backend and Stockfish.

Wraps the existing UciEngine (engine/uci_engine.py) so that:
  - The coach never receives raw strings or untyped dicts.
  - Stockfish interaction is testable via mock substitution.
  - A single start/stop lifecycle is managed per StrategyEngine session.

All methods return typed results. Callers never touch UCI strings.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalResult:
    """
    Evaluation of a position returned by Stockfish.

    centipawns : int | None
        Score in centipawns from White's perspective.
        Positive = White advantage. Negative = Black advantage.
        None when Stockfish returns a forced mate score instead.

    mate_in : int | None
        Forced mate. Positive = White mates in N. Negative = Black mates in N.
        None when no forced mate exists.

    depth : int
        Search depth reached.

    is_mate : bool
        Convenience flag. True when mate_in is not None.
    """
    centipawns: Optional[int]
    mate_in:    Optional[int]
    depth:      int

    @property
    def is_mate(self) -> bool:
        return self.mate_in is not None

    def score_from_side(self, side: str) -> float:
        """
        Returns the raw centipawn score from the specified side's perspective.
        side: 'white' | 'black'
        Returns a large positive number for mate, large negative for getting mated.
        """
        cp = self.centipawns if self.centipawns is not None else (
            30000 if (self.mate_in or 0) > 0 else -30000
        )
        return cp if side == 'white' else -cp


@dataclass(frozen=True)
class MoveCandidate:
    """
    A single candidate move returned by get_top_moves().

    uci : str
        Move in UCI format e.g. 'e2e4', 'g1f3'.

    san : str
        Move in Standard Algebraic Notation e.g. 'e4', 'Nf3'.
        Computed from the board position supplied to get_top_moves().

    eval_result : EvalResult
        Stockfish's evaluation after this move is played.

    rank : int
        1-based ranking. rank=1 is Stockfish's best move.
    """
    uci:         str
    san:         str
    eval_result: EvalResult
    rank:        int


# ── Eval breakdown types ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalBreakdown:
    """
    Stockfish's own eval term scores for a position.

    All values are from White's perspective in pawn units.
    Positive = White advantage. Computed via Stockfish's 'eval' command.
    """
    material:     float = 0.0
    imbalance:    float = 0.0
    pawns:        float = 0.0
    knights:      float = 0.0
    bishops:      float = 0.0
    rooks:        float = 0.0
    queens:       float = 0.0
    mobility:     float = 0.0
    king_safety:  float = 0.0
    threats:      float = 0.0
    passed_pawns: float = 0.0
    space:        float = 0.0
    classical:    float = 0.0
    nnue:         float = 0.0
    final:        float = 0.0


@dataclass
class PVExplanation:
    """
    Explanation of a Stockfish PV line derived from eval term deltas.

    Produced by StockfishBridge.explain_pv(). Consumed by coach/explainer.py.
    The coach no longer generates its own strategic opinion — it explains WHY
    Stockfish's recommended line makes positional sense.
    """
    pv_uci:          list[str]
    eval_before:     EvalBreakdown
    eval_after:      EvalBreakdown
    deltas:          dict[str, float]
    dominant_term:   str
    dominant_delta:  float
    is_tactical:     bool
    strategy:        str
    confidence:      float
    pv_san:          list[str]
    tactic_move_idx: int


# ── Module-level helpers ──────────────────────────────────────────────────────

def _derive_strategy(
    dominant_term:  str,
    dominant_delta: float,
    before:         EvalBreakdown,
    after:          EvalBreakdown,
    player_side:    str,
) -> str:
    """Map the dominant eval term delta to one of the four strategy labels."""
    flip        = 1.0 if player_side == 'white' else -1.0
    total_delta = (after.final - before.final) * flip

    if dominant_term == 'king_safety':
        return 'blitz' if total_delta > 0 else 'fortress'
    if dominant_term in ('space', 'mobility'):
        return 'flank'
    if dominant_term == 'threats':
        return 'blitz' if total_delta > 0.5 else 'feint'
    if dominant_term in ('material', 'pawns', 'knights', 'bishops', 'rooks', 'queens'):
        return 'blitz' if dominant_delta > 0 else 'fortress'
    if dominant_term == 'passed_pawns':
        return 'flank'
    return 'flank'


# ── Bridge ────────────────────────────────────────────────────────────────────

class StockfishBridge:
    """
    Synchronous Stockfish wrapper for the coach backend.

    Lifecycle
    ---------
    bridge = StockfishBridge(stockfish_path)
    bridge.start()
    eval  = bridge.get_eval(fen)
    moves = bridge.get_top_moves(board, n=5)
    best  = bridge.get_best_move(board)
    bridge.stop()

    Thread safety: not thread-safe. The StrategyEngine runs analysis
    on a single thread and owns this bridge for its lifetime.

    Parameters
    ----------
    stockfish_path : Path | str
        Absolute or relative path to the Stockfish executable.
    movetime_ms : int
        Milliseconds given to Stockfish per position analysis.
        Default 2000ms (configurable via config['coach']['movetime_ms']).
    """

    def __init__(self, stockfish_path: Path | str, movetime_ms: int = 2000) -> None:
        self._path = Path(stockfish_path)
        self._movetime_ms = movetime_ms
        self._engine: object | None = None  # UciEngine instance

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Stockfish subprocess. Must be called before any analysis."""
        # Import here to avoid circular dependency — chess_coach is a sibling
        # package to chessplayer; we reach up via sys.path or relative import.
        try:
            from engine.uci_engine import UciEngine
        except ImportError:
            # Running tests or standalone: try an absolute path
            _root = Path(__file__).resolve().parents[3] / 'src' / 'chessplayer'
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from engine.uci_engine import UciEngine  # type: ignore

        self._engine = UciEngine(engine_exe=self._path)
        self._engine.start()  # type: ignore[union-attr]

    def stop(self) -> None:
        """Stop the Stockfish subprocess."""
        if self._engine is not None:
            self._engine.stop()  # type: ignore[union-attr]
            self._engine = None

    @property
    def is_running(self) -> bool:
        return self._engine is not None

    # ── Analysis ──────────────────────────────────────────────────────────────

    def get_eval(self, fen: str) -> EvalResult:
        """
        Evaluate a position given as a FEN string.

        Returns EvalResult with centipawns from White's perspective.
        Starting position should return approximately 0 ± 20 centipawns.

        Parameters
        ----------
        fen : str
            Full FEN string of the position to evaluate.
        """
        self._assert_running()
        board = chess.Board(fen)
        prefix_uci = self._board_to_uci_prefix(board)
        result = self._engine.analyze_movetime(prefix_uci, self._movetime_ms)  # type: ignore

        return EvalResult(
            centipawns = result.score_cp,
            mate_in    = result.score_mate,
            depth      = result.depth or 0,
        )

    def get_top_moves(self, board: chess.Board, n: int = 5) -> list[MoveCandidate]:
        """
        Return the top N candidate moves for the current position.

        Uses movetime analysis. For n > 1 this calls Stockfish multiple times
        using the exclude-move trick (simple, reliable, no multipv config needed).
        For production use with large n, consider multipv — but for the coach
        backend n is always ≤ 5.

        Parameters
        ----------
        board : chess.Board
            The current position. Not mutated.
        n : int
            Number of candidate moves to return. Clamped to number of legal moves.
        """
        self._assert_running()
        n = min(n, board.legal_moves.count())
        if n == 0:
            return []

        candidates: list[MoveCandidate] = []
        excluded: list[str] = []
        working_board = board.copy()

        for rank in range(1, n + 1):
            prefix_uci = self._board_to_uci_prefix(working_board)
            result = self._engine.analyze_movetime(prefix_uci, self._movetime_ms)  # type: ignore

            if not result.uci:
                break

            uci = result.uci
            # Skip already-returned moves
            if uci in excluded:
                break

            try:
                move = chess.Move.from_uci(uci)
                san = working_board.san(move)
            except (ValueError, chess.InvalidMoveError):
                san = uci

            candidates.append(MoveCandidate(
                uci         = uci,
                san         = san,
                eval_result = EvalResult(
                    centipawns = result.score_cp,
                    mate_in    = result.score_mate,
                    depth      = result.depth or 0,
                ),
                rank = rank,
            ))
            excluded.append(uci)

        return candidates

    def get_best_move(self, board: chess.Board) -> Optional[MoveCandidate]:
        """
        Return the single best move for the current position.

        Convenience wrapper around get_top_moves(n=1).
        Returns None if the position has no legal moves (checkmate/stalemate).
        """
        moves = self.get_top_moves(board, n=1)
        return moves[0] if moves else None

    def get_eval_breakdown(self, board: chess.Board) -> EvalBreakdown:
        """
        Return Stockfish's own eval term breakdown for a position.

        Uses the 'eval' command (near-instant, no search required).
        Values are in pawn units, White's perspective.
        Returns a zeroed EvalBreakdown on engine error.
        """
        self._assert_running()
        try:
            raw = self._engine.get_eval_terms_fen(board.fen())  # type: ignore
        except Exception:
            return EvalBreakdown()
        return EvalBreakdown(
            material     = raw.get('material',     0.0),
            imbalance    = raw.get('imbalance',    0.0),
            pawns        = raw.get('pawns',        0.0),
            knights      = raw.get('knights',      0.0),
            bishops      = raw.get('bishops',      0.0),
            rooks        = raw.get('rooks',        0.0),
            queens       = raw.get('queens',       0.0),
            mobility     = raw.get('mobility',     0.0),
            king_safety  = raw.get('king_safety',  0.0),
            threats      = raw.get('threats',      0.0),
            passed_pawns = raw.get('passed_pawns', raw.get('passed', 0.0)),
            space        = raw.get('space',        0.0),
            classical    = raw.get('classical',    0.0),
            nnue         = raw.get('nnue',         0.0),
            final        = raw.get('final',        0.0),
        )

    def explain_pv(
        self,
        board:       chess.Board,
        pv_uci:      list[str],
        player_side: str = 'white',
        max_moves:   int = 4,
    ) -> Optional[PVExplanation]:
        """
        Explain WHY Stockfish recommends the given PV line.

        Evaluates the current position and the position after max_moves of the PV
        using Stockfish's own eval term breakdown, then computes the dominant
        positional change to derive strategy and confidence.

        Returns None if the bridge is not running or PV is empty.
        """
        self._assert_running()
        if not pv_uci:
            return None

        # Build board states for each position in the PV
        boards: list[chess.Board] = [board.copy()]
        pv_san: list[str] = []
        for uci in pv_uci[:max_moves]:
            b = boards[-1].copy()
            try:
                move = chess.Move.from_uci(uci)
                pv_san.append(b.san(move))
                b.push(move)
            except Exception:
                break
            boards.append(b)

        if len(boards) < 2:
            return None

        # Eval breakdown for each position (current + each PV step)
        breakdowns: list[EvalBreakdown] = []
        for b in boards:
            try:
                bd = self.get_eval_breakdown(b)
            except Exception:
                bd = EvalBreakdown()
            breakdowns.append(bd)

        before = breakdowns[0]
        after  = breakdowns[-1]

        # Per-term deltas from the player's perspective
        flip = 1.0 if player_side == 'white' else -1.0
        positional_terms = (
            'king_safety', 'mobility', 'space', 'threats',
            'passed_pawns', 'material', 'pawns',
        )
        deltas: dict[str, float] = {
            term: (getattr(after, term, 0.0) - getattr(before, term, 0.0)) * flip
            for term in positional_terms
        }

        dominant_term  = max(deltas, key=lambda k: abs(deltas[k]))
        dominant_delta = deltas[dominant_term]

        # Find the single move with the biggest eval jump (tactic detection)
        max_jump        = 0.0
        tactic_move_idx = -1
        for i in range(1, len(breakdowns)):
            jump = abs(breakdowns[i].final - breakdowns[i - 1].final)
            if jump > max_jump:
                max_jump        = jump
                tactic_move_idx = i

        nnue_classical_gap = abs(after.nnue - after.classical)
        is_tactical = max_jump > 1.5 or nnue_classical_gap > 1.0

        strategy   = _derive_strategy(dominant_term, dominant_delta, before, after, player_side)
        raw_conf   = min(1.0, abs(dominant_delta) / 2.0)
        confidence = 0.50 + raw_conf * 0.50

        return PVExplanation(
            pv_uci          = list(pv_uci[:max_moves]),
            eval_before     = before,
            eval_after      = after,
            deltas          = deltas,
            dominant_term   = dominant_term,
            dominant_delta  = dominant_delta,
            is_tactical     = is_tactical,
            strategy        = strategy,
            confidence      = confidence,
            pv_san          = pv_san,
            tactic_move_idx = tactic_move_idx if max_jump > 1.5 else -1,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_running(self) -> None:
        if self._engine is None:
            raise RuntimeError(
                "StockfishBridge.start() must be called before analysis. "
                "Did you forget to call bridge.start()?"
            )

    @staticmethod
    def _board_to_uci_prefix(board: chess.Board) -> list[str]:
        """
        Convert a chess.Board's move stack to a UCI move list.
        Used by UciEngine.analyze_movetime() which expects a prefix list
        from the start position, not a FEN.
        """
        return [m.uci() for m in board.move_stack]
