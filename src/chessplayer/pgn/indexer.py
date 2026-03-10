from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Optional

import chess.pgn

from utils.paths import resolve_path

ProgressCb = Callable[[int, str], None]
CancelCb = Callable[[], bool]

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    UNIQUE(type, path)
);

CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    pgn_path TEXT NOT NULL,
    offset_bytes INTEGER NOT NULL,

    white TEXT,
    black TEXT,
    result TEXT,
    event TEXT,
    site TEXT,
    date TEXT,
    eco TEXT,
    opening TEXT,

    FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE INDEX IF NOT EXISTS idx_games_white   ON games(white);
CREATE INDEX IF NOT EXISTS idx_games_black   ON games(black);
CREATE INDEX IF NOT EXISTS idx_games_event   ON games(event);
CREATE INDEX IF NOT EXISTS idx_games_eco     ON games(eco);
CREATE INDEX IF NOT EXISTS idx_games_opening ON games(opening);
CREATE INDEX IF NOT EXISTS idx_games_result  ON games(result);
CREATE INDEX IF NOT EXISTS idx_games_date    ON games(date);
CREATE INDEX IF NOT EXISTS idx_games_source  ON games(source_id);
"""


def _db_path(cfg: dict) -> Path:
    data_dir = resolve_path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "index.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _resolve_source_path(source_path: str | Path) -> Path:
    raw = Path(source_path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return resolve_path(str(raw)).resolve()


def _upsert_source(conn: sqlite3.Connection, source_type: str, source_path: str) -> int:
    conn.execute("INSERT OR IGNORE INTO sources(type, path) VALUES(?, ?)", (source_type, source_path))
    row = conn.execute(
        "SELECT source_id FROM sources WHERE type=? AND path=?",
        (source_type, source_path),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to upsert source")
    return int(row[0])


def _delete_games_for_source(conn: sqlite3.Connection, source_id: int) -> None:
    conn.execute("DELETE FROM games WHERE source_id=?", (source_id,))
    conn.commit()


def _iter_directory_pgns(src_path: Path) -> Iterable[Path]:
    yield from sorted(path for path in src_path.rglob("*.pgn") if path.is_file())


def _index_single_pgn_file(
    conn: sqlite3.Connection,
    source_id: int,
    pgn_file: Path,
    progress_cb: Optional[ProgressCb],
    cancel_cb: Optional[CancelCb],
    base_games_indexed: int = 0,
) -> int:
    games = 0
    if progress_cb:
        progress_cb(base_games_indexed, f"Opening {pgn_file.name} ...")

    with open(pgn_file, "rb") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        offset = text.tell()
        while True:
            if cancel_cb and cancel_cb():
                break
            game = chess.pgn.read_game(text)
            if game is None:
                break
            headers = game.headers
            conn.execute(
                """
                INSERT INTO games(
                    source_id, pgn_path, offset_bytes, white, black, result, event, site, date, eco, opening
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    str(pgn_file.resolve()),
                    int(offset),
                    headers.get("White"),
                    headers.get("Black"),
                    headers.get("Result"),
                    headers.get("Event"),
                    headers.get("Site"),
                    headers.get("Date"),
                    headers.get("ECO"),
                    headers.get("Opening"),
                ),
            )
            games += 1
            if games <= 100 or games % 100 == 0:
                conn.commit()
                if progress_cb:
                    progress_cb(base_games_indexed + games, f"Indexed {base_games_indexed + games} games from {pgn_file.name}")
            offset = text.tell()
    conn.commit()
    if progress_cb:
        progress_cb(base_games_indexed + games, f"Finished {pgn_file.name} ({games} games)")
    return games


def _rebuild_source(
    conn: sqlite3.Connection,
    source_id: int,
    src_type: str,
    src_path: Path,
    progress_cb: Optional[ProgressCb],
    cancel_cb: Optional[CancelCb],
) -> int:
    _delete_games_for_source(conn, source_id)
    total = 0

    if src_type == "archive_file":
        if progress_cb:
            progress_cb(0, f"Rebuilding PGN library: {src_path.name}")
        total = _index_single_pgn_file(conn, source_id, src_path, progress_cb, cancel_cb)
    elif src_type == "directory":
        files = list(_iter_directory_pgns(src_path))
        if progress_cb:
            progress_cb(0, f"Rebuilding PGN directory: {src_path}")
        for idx, file in enumerate(files, start=1):
            if cancel_cb and cancel_cb():
                break
            if progress_cb:
                progress_cb(total, f"Indexing file {idx}/{len(files)}: {file.name}")
            total += _index_single_pgn_file(
                conn,
                source_id,
                file,
                progress_cb,
                cancel_cb,
                base_games_indexed=total,
            )
    else:
        raise ValueError(f"Unknown source type: {src_type}")
    return total


def build_or_rebuild_index_for_source(
    cfg: dict,
    source_type: str,
    source_path: str | Path,
    progress_cb: Optional[ProgressCb] = None,
    cancel_cb: Optional[CancelCb] = None,
) -> Path:
    db_path = _db_path(cfg)
    conn = _connect(db_path)
    try:
        runtime_path = _resolve_source_path(source_path)
        if source_type == "archive_file" and not runtime_path.is_file():
            raise FileNotFoundError(f"PGN file not found: {runtime_path}")
        if source_type == "directory" and not runtime_path.is_dir():
            raise NotADirectoryError(f"PGN directory not found: {runtime_path}")
        source_id = _upsert_source(conn, source_type, str(runtime_path))
        _rebuild_source(conn, source_id, source_type, runtime_path, progress_cb, cancel_cb)
    finally:
        conn.close()
    return db_path

def build_or_rebuild_index_for_sources(
    cfg: dict,
    sources,
    progress_cb=None,
    cancel_cb=None,
):
    db_path = None
    for source_type, source_path in sources:
        if cancel_cb and cancel_cb():
            break
        db_path = build_or_rebuild_index_for_source(
            cfg=cfg,
            source_type=source_type,
            source_path=source_path,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )
    if db_path is None:
        db_path = _db_path(cfg)
    return db_path

def build_or_update_index(cfg: dict, progress_cb: Optional[ProgressCb], cancel_cb: Optional[CancelCb]) -> Path:
    active = cfg["pgn_sources"]["active_source"]
    return build_or_rebuild_index_for_source(
        cfg=cfg,
        source_type=active["type"],
        source_path=active["path"],
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )
