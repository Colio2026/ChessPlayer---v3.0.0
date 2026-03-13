from __future__ import annotations

from pathlib import Path

import chess
from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, QObject, Signal, Slot

from chessplayer.core.pgn_edit import PgnEditor


def _piece_code(piece: chess.Piece) -> str:
    color = "W" if piece.color == chess.WHITE else "B"
    letter_map = {
        chess.PAWN: "P",
        chess.KNIGHT: "N",
        chess.BISHOP: "B",
        chess.ROOK: "R",
        chess.QUEEN: "Q",
        chess.KING: "K",
    }
    return color + letter_map[piece.piece_type]


class BoardListModel(QAbstractListModel):
    FileRole = Qt.UserRole + 1
    RankRole = Qt.UserRole + 2
    CodeRole = Qt.UserRole + 3
    ImageRole = Qt.UserRole + 4

    def __init__(self, editor: PgnEditor, pieces_dir: Path) -> None:
        super().__init__()
        self._editor = editor
        self._pieces_dir = pieces_dir
        self._flip = False
        self._items: list[dict] = []
        self.rebuild()

    def roleNames(self):
        return {
            self.FileRole: b"file",
            self.RankRole: b"rank",
            self.CodeRole: b"code",
            self.ImageRole: b"image",
        }

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        item = self._items[index.row()]
        if role == self.FileRole:
            return item["file"]
        if role == self.RankRole:
            return item["rank"]
        if role == self.CodeRole:
            return item["code"]
        if role == self.ImageRole:
            return item["image"]
        return None

    def set_flipped(self, flipped: bool) -> None:
        self._flip = flipped
        self.rebuild()

    def flipped(self) -> bool:
        return self._flip

    def rebuild(self) -> None:
        b = self._editor.session.board
        items = []
        for sq in chess.SQUARES:
            piece = b.piece_at(sq)
            if not piece:
                continue
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            if self._flip:
                file = 7 - file
                rank = 7 - rank
            code = _piece_code(piece)
            img = (self._pieces_dir / f"{code}.png").resolve()
            items.append({"file": int(file), "rank": int(rank), "code": code, "image": img.as_uri()})

        self.beginResetModel()
        self._items = items
        self.endResetModel()


class BoardBridge(QObject):
    statusChanged = Signal(str)
    fenChanged = Signal(str)
    moveMade = Signal(str)               # SAN
    promotionRequested = Signal(str)     # uci prefix like e7e8

    def __init__(self, editor: PgnEditor, model: BoardListModel):
        super().__init__()
        self._editor = editor
        self._model = model

    @Slot(int, int, int, int)
    def attemptMove(self, fromFile: int, fromRank: int, toFile: int, toRank: int) -> None:
        if self._model.flipped():
            fromFile, fromRank = 7 - fromFile, 7 - fromRank
            toFile, toRank = 7 - toFile, 7 - toRank

        from_sq = chess.square(fromFile, fromRank)
        to_sq = chess.square(toFile, toRank)

        res = self._editor.try_user_move(from_sq, to_sq)
        if res.promotion_required and res.promotion_uci_prefix:
            self.statusChanged.emit("Promotion required")
            self.promotionRequested.emit(res.promotion_uci_prefix)
            self._model.rebuild()
            self.fenChanged.emit(self._editor.session.fen())
            return

        if not res.ok:
            self.statusChanged.emit("Illegal move")
            self._model.rebuild()
            self.fenChanged.emit(self._editor.session.fen())
            return

        self.statusChanged.emit(res.san or "Move")
        self.moveMade.emit(res.san or "")
        self._model.rebuild()
        self.fenChanged.emit(self._editor.session.fen())

    @Slot(str)
    def choosePromotion(self, promo: str) -> None:
        res = self._editor.resolve_promotion(promo)
        if not res.ok:
            self.statusChanged.emit("Illegal promotion")
        else:
            self.statusChanged.emit(res.san or "Promotion")
            self.moveMade.emit(res.san or "")
        self._model.rebuild()
        self.fenChanged.emit(self._editor.session.fen())

    @Slot(bool)
    def setFlipped(self, flipped: bool) -> None:
        self._model.set_flipped(flipped)
        self._model.rebuild()
        self.fenChanged.emit(self._editor.session.fen())

    @Slot()
    def stepBack(self) -> None:
        if self._editor.step_back():
            self._model.rebuild()
            self.statusChanged.emit("Back")
            self.fenChanged.emit(self._editor.session.fen())

    @Slot()
    def stepForward(self) -> None:
        if self._editor.step_forward_mainline():
            self._model.rebuild()
            self.statusChanged.emit("Forward")
            self.fenChanged.emit(self._editor.session.fen())
