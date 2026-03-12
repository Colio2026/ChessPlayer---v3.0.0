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
from dataclasses import dataclass
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
