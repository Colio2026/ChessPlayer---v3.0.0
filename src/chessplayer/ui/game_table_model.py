from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from chessplayer.pgn.models import GameMeta
from chessplayer.pgn.query import Query, default_multisort

if TYPE_CHECKING:
    from chessplayer.pgn.store import PgnStore

# Columns shown in the game browser table
COLUMNS: list[tuple[str, str]] = [
    ("Date",    "date"),
    ("White",   "white"),
    ("Black",   "black"),
    ("Result",  "result"),
    ("ECO",     "eco"),
    ("Opening", "opening"),
    ("Event",   "event"),
    ("Site",    "site"),
]

# How many rows are fetched from SQLite per chunk.
# Small enough to be instant, large enough that the user rarely
# triggers a second fetch while scrolling at normal speed.
_CHUNK_SIZE = 200


class GameTableModel(QAbstractTableModel):
    """
    Virtual, lazy-loading table model for the game browser.

    Qt asks for data as rows scroll into view. Rows are fetched from
    SQLite in chunks of _CHUNK_SIZE and cached indefinitely for the
    lifetime of the current query. Memory cost: ~400 bytes per cached
    GameMeta, so 6M games fully cached ≈ 2.4 GB — we never pre-fetch
    everything. In practice only the rows the user actually scrolls
    past are ever loaded.

    Call reset_query() whenever the source, filter, or sort changes.
    The model reports the true total row count immediately (via a fast
    COUNT(*) query) so the scroll bar always reflects the real size of
    the database.
    """

    def __init__(self) -> None:
        super().__init__()
        self._store:     "PgnStore | None" = None
        self._query:     Query             = Query()
        self._sort:      list[tuple[str, str]] = default_multisort()
        self._source_id: int | None        = None

        self._total:     int               = 0
        # Sparse cache: row_index → GameMeta
        self._cache:     dict[int, GameMeta] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def reset_query(
        self,
        store:     "PgnStore | None",
        query:     Query,
        sort:      list[tuple[str, str]],
        source_id: int | None,
    ) -> None:
        """
        Replace the active query. Clears all cached rows, re-counts
        the total, and notifies Qt that the model has been reset.
        """
        self.beginResetModel()
        self._store     = store
        self._query     = query
        self._sort      = sort
        self._source_id = source_id
        self._cache.clear()

        if store is not None:
            self._total = store.count_games(query, source_id)
        else:
            self._total = 0

        self.endResetModel()

    def game_id_at(self, row: int) -> int | None:
        meta = self._get_row(row)
        return int(meta.game_id) if meta is not None else None

    # ── QAbstractTableModel interface ─────────────────────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._total

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return COLUMNS[section][0]
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            meta = self._get_row(index.row())
            if meta is None:
                return ""
            val = getattr(meta, COLUMNS[index.column()][1])
            return val if val is not None else ""

        if role == Qt.TextAlignmentRole:
            # Centre the Result and ECO columns
            if index.column() in (3, 4):
                return Qt.AlignCenter

        return None

    # ── internal ─────────────────────────────────────────────────────────────

    def _get_row(self, row: int) -> GameMeta | None:
        """
        Return the GameMeta for the given row, fetching a chunk from
        SQLite if it is not already cached.
        """
        if row < 0 or row >= self._total or self._store is None:
            return None

        if row in self._cache:
            return self._cache[row]

        # Calculate which chunk this row belongs to and fetch it
        chunk_start = (row // _CHUNK_SIZE) * _CHUNK_SIZE
        rows = self._store.list_games(
            query=self._query,
            multisort=self._sort,
            page=0,                         # we use raw offset, not page number
            page_size=_CHUNK_SIZE,
            source_id=self._source_id,
        )

        # list_games uses page*page_size as offset — pass chunk_start as page=0
        # with an explicit offset by re-calling with the right page number
        page_num = chunk_start // _CHUNK_SIZE
        rows = self._store.list_games(
            query=self._query,
            multisort=self._sort,
            page=page_num,
            page_size=_CHUNK_SIZE,
            source_id=self._source_id,
        )

        for i, meta in enumerate(rows):
            self._cache[chunk_start + i] = meta

        return self._cache.get(row)
