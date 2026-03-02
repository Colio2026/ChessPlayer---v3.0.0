from __future__ import annotations

from dataclasses import dataclass
import chess
from core.log import log


@dataclass(frozen=True)
class MoveResult:
    ok: bool
    uci: str | None = None
    san: str | None = None
    reason: str | None = None
    promotion_required: bool = False
    promotion_uci_prefix: str | None = None  # e.g. "e7e8"


class GameSession:
    """python-chess Board + undo/redo."""

    def __init__(self) -> None:
        self.board = chess.Board()
        self._undo: list[chess.Move] = []
        self._redo: list[chess.Move] = []

    def reset(self) -> None:
        self.board.reset()
        self._undo.clear()
        self._redo.clear()

    def fen(self) -> str:
        return self.board.fen()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        mv = self._undo.pop()
        self.board.pop()
        self._redo.append(mv)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        mv = self._redo.pop()
        self.board.push(mv)
        self._undo.append(mv)
        return True

    def _needs_promotion(self, from_sq: chess.Square, to_sq: chess.Square) -> bool:
        piece = self.board.piece_at(from_sq)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        to_rank = chess.square_rank(to_sq)
        if piece.color == chess.WHITE and to_rank == 7:
            return True
        if piece.color == chess.BLACK and to_rank == 0:
            return True
        return False

    def try_move(self, from_sq: chess.Square, to_sq: chess.Square) -> MoveResult:
        if self._needs_promotion(from_sq, to_sq):
            prefix = chess.square_name(from_sq) + chess.square_name(to_sq)
            return MoveResult(ok=False, promotion_required=True, promotion_uci_prefix=prefix)

        mv = chess.Move(from_sq, to_sq)
        if mv not in self.board.legal_moves:
            log.info("Illegal move attempt: %s -> %s | fen=%s",
                     chess.square_name(from_sq), chess.square_name(to_sq), self.board.fen())
            return MoveResult(ok=False, reason="illegal")

        san = self.board.san(mv)
        self.board.push(mv)
        self._undo.append(mv)
        self._redo.clear()
        return MoveResult(ok=True, uci=mv.uci(), san=san)

    def try_promotion(self, uci_prefix: str, promo: str) -> MoveResult:
        promo = promo.lower().strip()
        promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
        if promo not in promo_map:
            return MoveResult(ok=False, reason="bad_promotion")

        mv = chess.Move.from_uci(uci_prefix + promo)
        if mv not in self.board.legal_moves:
            log.info("Illegal promotion attempt: %s%s | fen=%s", uci_prefix, promo, self.board.fen())
            return MoveResult(ok=False, reason="illegal")

        san = self.board.san(mv)
        self.board.push(mv)
        self._undo.append(mv)
        self._redo.clear()
        return MoveResult(ok=True, uci=mv.uci(), san=san)
