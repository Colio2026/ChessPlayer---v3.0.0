from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from pgn.continuations import ContinuationStat


COLUMNS = [
    "Move",
    "Played",
    "White %",
    "W",
    "D",
    "L",
]


class ContinuationStatsModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[ContinuationStat] = []

    def set_rows(self, rows: list[ContinuationStat]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return COLUMNS[section]
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None

        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return row.san
            if col == 1:
                return str(row.count)
            if col == 2:
                return f"{row.white_score_pct:.1f}%"
            if col == 3:
                return str(row.white_wins)
            if col == 4:
                return str(row.draws)
            if col == 5:
                return str(row.black_wins)

        if role == Qt.TextAlignmentRole:
            if col == 0:
                return int(Qt.AlignVCenter | Qt.AlignLeft)
            return int(Qt.AlignCenter)

        return None
