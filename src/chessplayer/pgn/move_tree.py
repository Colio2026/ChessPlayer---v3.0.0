from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chess
import chess.pgn

from pgn.continuations import ContinuationStat

ProgressCb = Callable[[int, str], None]
CancelCb   = Callable[[], bool]

_SEP       = "|"          # separator for move sequence keys
_MAX_DEPTH = 20           # maximum ply depth stored in the tree

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    move_seq    TEXT    PRIMARY KEY,   -- pipe-joined UCI moves e.g. "e2e4|e7e5"
    san         TEXT    NOT NULL,      -- SAN of the LAST move in the sequence
    count       INTEGER NOT NULL DEFAULT 0,
    white_wins  INTEGER NOT NULL DEFAULT 0,
    draws       INTEGER NOT NULL DEFAULT 0,
    black_wins  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_nodes_seq ON nodes(move_seq);
"""


# ─── helpers ──────────────────────────────────────────────────────────────────

def _seq_key(uci_moves: list[str]) -> str:
    return _SEP.join(uci_moves)


def _tree_db_path(cfg: dict, source_path: str) -> Path:
    """
    Derive the tree DB path from the source PGN path.
    e.g. data/Carlsen.pgn  →  data/trees/Carlsen_tree.sqlite
    """
    from utils.paths import resolve_path
    data_dir = resolve_path(cfg["paths"]["data_dir"])
    trees_dir = data_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_path).stem          # "Carlsen"
    return trees_dir / f"{stem}_tree.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ─── public API ───────────────────────────────────────────────────────────────

class MoveTree:
    """
    In-memory handle to a pre-built move tree stored in SQLite.
    Instantiate via MoveTree.load() after the tree has been built.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    # -- query -----------------------------------------------------------------

    def query(
        self,
        prefix_uci: list[str],
        max_out: int = 50,
    ) -> list[ContinuationStat]:
        """
        Return continuation stats one ply beyond prefix_uci.
        prefix_uci = [] returns first-move stats (root continuations).
        """
        prefix = _seq_key(prefix_uci)
        # A child key looks like: prefix|<one_more_move>
        # We match all nodes whose key starts with prefix + SEP (or is exactly
        # one move if prefix is empty) and contains no further SEP after that.
        if prefix:
            like_pat = f"{prefix}{_SEP}%"
        else:
            like_pat = f"%"          # root: any single move (no SEP in key)

        conn = _connect(self._db_path)
        try:
            if prefix:
                # children: key = prefix|<move>  — no further pipe after that
                rows = conn.execute(
                    """
                    SELECT san, count, white_wins, draws, black_wins
                    FROM nodes
                    WHERE move_seq LIKE ?
                      AND move_seq NOT LIKE ?
                    ORDER BY count DESC
                    LIMIT ?
                    """,
                    (like_pat, f"{prefix}{_SEP}%{_SEP}%", max_out),
                ).fetchall()
            else:
                # root: keys with no pipe at all
                rows = conn.execute(
                    """
                    SELECT san, count, white_wins, draws, black_wins
                    FROM nodes
                    WHERE move_seq NOT LIKE ?
                    ORDER BY count DESC
                    LIMIT ?
                    """,
                    (f"%{_SEP}%", max_out),
                ).fetchall()
        finally:
            conn.close()

        return [
            ContinuationStat(
                san=row[0],
                count=row[1],
                white_wins=row[2],
                draws=row[3],
                black_wins=row[4],
            )
            for row in rows
        ]

    def is_built(self) -> bool:
        """Return True if the tree DB exists and has been marked complete."""
        if not self._db_path.exists():
            return False
        try:
            conn = _connect(self._db_path)
            row = conn.execute(
                "SELECT value FROM meta WHERE key='status'"
            ).fetchone()
            conn.close()
            return row is not None and row[0] == "complete"
        except Exception:
            return False

    @staticmethod
    def load(cfg: dict, source_path: str) -> "MoveTree":
        """Return a MoveTree handle for the given source. Does not build."""
        db_path = _tree_db_path(cfg, source_path)
        return MoveTree(db_path)


# ─── builder ──────────────────────────────────────────────────────────────────

def build_tree(
    cfg: dict,
    source_path: str,
    pgn_store,                          # pgn.store.PgnStore
    source_id: int,
    progress_cb: Optional[ProgressCb] = None,
    cancel_cb:   Optional[CancelCb]   = None,
    max_depth:   int                   = _MAX_DEPTH,
) -> MoveTree:
    """
    Walk every game in source_id, build the position tree, persist to SQLite.
    Returns the completed MoveTree handle.
    """
    db_path = _tree_db_path(cfg, source_path)

    # Wipe any previous partial build
    if db_path.exists():
        db_path.unlink()

    conn = _connect(db_path)

    # accumulate into a dict before bulk-writing for speed
    # key → [san, count, white_wins, draws, black_wins]
    nodes: dict[str, list] = {}

    game_ids = pgn_store.list_game_ids_for_source(source_id)
    total    = len(game_ids)

    if progress_cb:
        progress_cb(0, f"Building move tree — 0 / {total} games processed")

    for idx, gid in enumerate(game_ids):
        if cancel_cb and cancel_cb():
            conn.close()
            raise InterruptedError("Tree build cancelled")

        try:
            pgn_text = pgn_store.open_game_pgn_text(gid)
            game     = chess.pgn.read_game(io.StringIO(pgn_text))
            if game is None:
                continue

            result = str(game.headers.get("Result", "")).strip()
            white_win = 1 if result == "1-0" else 0
            black_win = 1 if result == "0-1" else 0
            draw      = 1 if result not in ("1-0", "0-1") else 0

            board      = game.board()
            uci_moves: list[str] = []

            for move in game.mainline_moves():
                if len(uci_moves) >= max_depth:
                    break

                san = board.san(move)
                board.push(move)
                uci_moves.append(move.uci())

                key = _seq_key(uci_moves)
                if key in nodes:
                    nodes[key][1] += 1
                    nodes[key][2] += white_win
                    nodes[key][3] += draw
                    nodes[key][4] += black_win
                else:
                    nodes[key] = [san, 1, white_win, draw, black_win]

        except Exception:
            continue

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            # Flush to SQLite every 500 games
            conn.executemany(
                """
                INSERT INTO nodes(move_seq, san, count, white_wins, draws, black_wins)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(move_seq) DO UPDATE SET
                    count      = count      + excluded.count,
                    white_wins = white_wins + excluded.white_wins,
                    draws      = draws      + excluded.draws,
                    black_wins = black_wins + excluded.black_wins
                """,
                [(k, v[0], v[1], v[2], v[3], v[4]) for k, v in nodes.items()],
            )
            conn.commit()
            nodes.clear()

            if progress_cb:
                progress_cb(idx + 1, f"Building move tree — {idx + 1} / {total} games processed")

    # Mark complete
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('status','complete')")
    conn.execute(f"INSERT OR REPLACE INTO meta(key, value) VALUES('source','{source_path}')")
    conn.execute(f"INSERT OR REPLACE INTO meta(key, value) VALUES('total_games','{total}')")
    conn.commit()
    conn.close()

    if progress_cb:
        progress_cb(total, f"Move tree complete — {total} games indexed")

    return MoveTree(db_path)
