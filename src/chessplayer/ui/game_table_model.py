from __future__ import annotations
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from pgn.models import GameMeta

COLUMNS = [("Date","date"),("White","white"),("Black","black"),("Result","result"),("ECO","eco"),("Opening","opening"),("Event","event"),("Site","site")]

class GameTableModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__(); self._rows: list[GameMeta] = []

    def set_rows(self, rows: list[GameMeta]) -> None:
        self.beginResetModel(); self._rows = rows; self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int: return len(self._rows)
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int: return len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole: return None
        return COLUMNS[section][0] if orientation == Qt.Horizontal else str(section+1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole: return None
        row = self._rows[index.row()]; key = COLUMNS[index.column()][1]
        val = getattr(row, key); return val if val is not None else ""

    def game_id_at(self, row: int) -> int | None:
        return int(self._rows[row].game_id) if 0 <= row < len(self._rows) else None
