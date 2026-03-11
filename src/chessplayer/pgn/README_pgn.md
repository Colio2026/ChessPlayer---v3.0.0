# pgn/  —  PGN Storage, Indexing, and Position Analysis

Handles all database interaction. Zero Qt dependencies.
Everything here is pure Python + sqlite3 + python-chess.

---

## Files

| File | Contents | Purpose |
|------|----------|---------|
| `store.py` | `PgnStore`, `IndexHandle`, `SourceRecord` | SQLite query layer. Read-only access to the game index. Fetches game metadata and seeks into original PGN files by byte offset to return full game text. |
| `indexer.py` | `build_or_rebuild_index_for_source()` | Scans PGN files, writes header metadata and byte offsets into the SQLite index. Supports `archive_file` (single .pgn) and `directory` (folder of .pgn files) sources. |
| `move_tree.py` | `MoveTree`, `build_tree()` | One-time position tree builder. Replays every game's mainline to record every `position → next move` transition. Saves to `data/trees/<sha1>.pkl.gz`. Query time is O(1). |
| `continuations.py` | `ContinuationStat`, `query_continuations()`, `common_continuations_from_store()` | Two query paths: fast MoveTree lookup (full library) and direct store scan (small filtered subset). |
| `models.py` | `GameMeta` | Frozen dataclass for one row of game metadata from the database. |
| `query.py` | `Query`, `Clause`, `compile_where()` | Filter specification for the game browser. Compiles to a parameterised SQL WHERE clause. |

---

## Database Schema

```sql
CREATE TABLE sources (
    source_id INTEGER PRIMARY KEY,
    type TEXT,      -- 'archive_file' or 'directory'
    path TEXT
);

CREATE TABLE games (
    game_id      INTEGER PRIMARY KEY,
    source_id    INTEGER,
    pgn_path     TEXT,          -- absolute path to the .pgn file
    offset_bytes INTEGER,       -- byte offset of this game in pgn_path
    white TEXT, black TEXT, result TEXT,
    event TEXT, site TEXT, date TEXT,
    eco TEXT, opening TEXT
);
```

No game text is stored in the database. `open_game_pgn_text(game_id)` seeks
to `offset_bytes` in `pgn_path` and parses one game from disk.

---

## MoveTree Position Key

```python
# FEN fields 0-3 only: piece placement + turn + castling + en-passant
# Halfmove clock and fullmove number are excluded
# → Transpositions from different move orders hash to the same node
pos_key = " ".join(board.fen().split()[:4])
```

MoveTree files live at:
```
data/trees/<sha1_of_source_path[:16]>.pkl.gz
```

---

## Adding a New Filter Field

1. Add a column to `games` in `indexer.py` `SCHEMA`.
2. Add a field to `Query` in `query.py`.
3. Add a clause to `compile_where()` in `query.py`.
4. Add a filter input to `QueryBuilder` in `ui/query_builder.py`.
