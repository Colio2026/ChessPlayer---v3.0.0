from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Callable, Optional

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

def _upsert_source(conn: sqlite3.Connection, source_type: str, source_path: str) -> int:
    conn.execute("INSERT OR IGNORE INTO sources(type, path) VALUES(?, ?)", (source_type, source_path))
    row = conn.execute("SELECT source_id FROM sources WHERE type=? AND path=?", (source_type, source_path)).fetchone()
    if row is None:
        raise RuntimeError("Failed to upsert source")
    return int(row[0])

def _index_single_pgn_file(
    conn: sqlite3.Connection,
    source_id: int,
    pgn_file: Path,
    progress_cb: Optional[ProgressCb],
    cancel_cb: Optional[CancelCb],
) -> int:
    games = 0
    with open(pgn_file, "rb") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        offset = text.tell()
        while True:
            if cancel_cb and cancel_cb():
                break
            game = chess.pgn.read_game(text)
            if game is None:
                break
            h = game.headers
            conn.execute(
                """
                INSERT INTO games(source_id, pgn_path, offset_bytes, white, black, result, event, site, date, eco, opening)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    str(pgn_file),
                    int(offset),
                    h.get("White"),
                    h.get("Black"),
                    h.get("Result"),
                    h.get("Event"),
                    h.get("Site"),
                    h.get("Date"),
                    h.get("ECO"),
                    h.get("Opening"),
                ),
            )
            games += 1
            if games % 1000 == 0:
                conn.commit()
                if progress_cb:
                    progress_cb(games, f"Indexed {games} games…")
            offset = text.tell()
    conn.commit()
    return games

def build_or_update_index(cfg: dict, progress_cb: Optional[ProgressCb], cancel_cb: Optional[CancelCb]) -> Path:
    db_path = _db_path(cfg)
    conn = _connect(db_path)

    active = cfg["pgn_sources"]["active_source"]
    src_type = active["type"]
    src_path = resolve_path(active["path"])

    source_id = _upsert_source(conn, src_type, str(src_path))

    existing = conn.execute("SELECT COUNT(1) FROM games WHERE source_id=?", (source_id,)).fetchone()[0]
    force = bool(cfg.get("indexing", {}).get("force_rebuild", False))

    if existing and not force:
        if progress_cb:
            progress_cb(int(existing), "Index exists; using existing index.")
        conn.close()
        return db_path

    if existing:
        conn.execute("DELETE FROM games WHERE source_id=?", (source_id,))
        conn.commit()

    if src_type == "archive_file":
        if progress_cb:
            progress_cb(0, f"Indexing archive: {src_path}")
        _index_single_pgn_file(conn, source_id, src_path, progress_cb, cancel_cb)
    elif src_type == "directory":
        total = 0
        for file in sorted(src_path.rglob("*.pgn")):
            if cancel_cb and cancel_cb():
                break
            total += _index_single_pgn_file(conn, source_id, file, progress_cb, cancel_cb)
            if progress_cb:
                progress_cb(total, f"Indexed {total} games (directory)…")
    else:
        raise ValueError(f"Unknown source type: {src_type}")

    conn.close()
    return db_path
