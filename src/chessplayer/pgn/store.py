from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import chess.pgn

from pgn.models import GameMeta
from pgn.query import Query, compile_where

@dataclass
class IndexHandle:
    db_path: Path

class PgnStore:
    def __init__(self, index: IndexHandle):
        self._db_path = index.db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def list_games(self, query: Query, multisort: list[tuple[str, str]], page: int, page_size: int) -> list[GameMeta]:
        where_sql, params = compile_where(query)
        order_sql = ", ".join([f"{col} {dir}" for col, dir in multisort]) if multisort else "game_id ASC"
        offset = max(page, 0) * page_size

        sql = f"""
        SELECT game_id, white, black, result, event, site, date, eco, opening, offset_bytes
        FROM games
        WHERE {where_sql}
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
        """

        conn = self._connect()
        rows = conn.execute(sql, [*params, page_size, offset]).fetchall()
        conn.close()

        out: list[GameMeta] = []
        for r in rows:
            out.append(GameMeta(
                game_id=int(r[0]), white=r[1], black=r[2], result=r[3],
                event=r[4], site=r[5], date=r[6], eco=r[7], opening=r[8],
                offset_bytes=int(r[9]),
            ))
        return out

    def open_game_pgn_text(self, game_id: int) -> str:
        conn = self._connect()
        row = conn.execute("SELECT pgn_path, offset_bytes FROM games WHERE game_id=?", (game_id,)).fetchone()
        conn.close()
        if not row:
            raise KeyError(f"game_id not found: {game_id}")

        pgn_path = Path(row[0])
        offset = int(row[1])

        with open(pgn_path, "rb") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
            text.seek(offset)
            game = chess.pgn.read_game(text)
            if game is None:
                raise RuntimeError("Failed to parse game at offset")
            return str(game)
