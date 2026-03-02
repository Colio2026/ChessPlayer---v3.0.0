from __future__ import annotations

import chess.pgn
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt


class _Wrap:
    def __init__(self, node: chess.pgn.GameNode, parent: "_Wrap | None" = None) -> None:
        self.node = node
        self.parent = parent
        self.children: list[_Wrap] = [_Wrap(v, self) for v in node.variations]


class VariationTreeModel(QAbstractItemModel):
    def __init__(self) -> None:
        super().__init__()
        self._root: _Wrap | None = None

    def set_root(self, node: chess.pgn.GameNode | None) -> None:
        self.beginResetModel()
        self._root = _Wrap(node) if node else None
        self.endResetModel()

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self._root or column != 0:
            return QModelIndex()
        if not parent.isValid():
            if row != 0:
                return QModelIndex()
            return self.createIndex(0, 0, self._root)
        p = parent.internalPointer()
        if not isinstance(p, _Wrap) or row < 0 or row >= len(p.children):
            return QModelIndex()
        return self.createIndex(row, 0, p.children[row])

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        w = index.internalPointer()
        if not isinstance(w, _Wrap) or w.parent is None:
            return QModelIndex()
        if w.parent.parent is None:
            return self.createIndex(0, 0, w.parent)
        gp = w.parent.parent
        row = gp.children.index(w.parent)
        return self.createIndex(row, 0, w.parent)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if not self._root:
            return 0
        if not parent.isValid():
            return 1
        p = parent.internalPointer()
        if not isinstance(p, _Wrap):
            return 0
        return len(p.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 1

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole or not index.isValid():
            return None
        w = index.internalPointer()
        if not isinstance(w, _Wrap):
            return None
        if w.node.move is None:
            return "Game"
        try:
            return w.node.san()
        except Exception:
            return w.node.move.uci()
