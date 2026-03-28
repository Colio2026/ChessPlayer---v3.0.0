from __future__ import annotations

import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import chess.pgn

from chessplayer.pgn.models import GameMeta
from chessplayer.pgn.query import Query, compile_where


@dataclass(frozen=True)
class IndexHandle:
    db_path: Path


@dataclass(frozen=True)
class SourceRecord:
    source_id: int
    source_type: str
    path: str


class PgnStore:
    def __init__(self, index: IndexHandle):
        self._db_path = index.db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _order_sql(self, multisort: list[tuple[str, str]]) -> str:
        if not multisort:
            return "game_id ASC"
        return ", ".join([f"{col} {direction}" for col, direction in multisort])

    def list_sources(self) -> list[SourceRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT source_id, type, path
                FROM sources
                ORDER BY source_id ASC
                """
            ).fetchall()
        finally:
            conn.close()

        return [
            SourceRecord(source_id=int(row[0]), source_type=str(row[1]), path=str(row[2]))
            for row in rows
        ]

    def get_source_id_by_path(
        self, source_path: str | Path, source_type: str | None = None
    ) -> int | None:
        raw = str(Path(source_path).expanduser())
        normalized = (
            str(Path(raw).expanduser().resolve())
            if Path(raw).expanduser().exists()
            else raw
        )

        conn = self._connect()
        try:
            if source_type is not None:
                row = conn.execute(
                    "SELECT source_id FROM sources WHERE type=? AND (path=? OR path=?)",
                    (source_type, raw, normalized),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT source_id FROM sources WHERE path=? OR path=?",
                    (raw, normalized),
                ).fetchone()
        finally:
            conn.close()

        return int(row[0]) if row else None

    def count_games(
        self,
        query: Query,
        source_id: int | None = None,
    ) -> int:
        """
        Return the total number of games matching query + source_id.
        Used by the lazy-loading table model to report rowCount to Qt
        without fetching any actual rows.
        """
        where_sql, params = compile_where(query)

        where_parts = [where_sql]
        sql_params: list[object] = list(params)
        if source_id is not None:
            where_parts.append("source_id = ?")
            sql_params.append(int(source_id))

        final_where = " AND ".join(part for part in where_parts if part)
        sql = f"SELECT COUNT(*) FROM games WHERE {final_where}"

        conn = self._connect()
        try:
            row = conn.execute(sql, sql_params).fetchone()
        finally:
            conn.close()

        return int(row[0]) if row else 0

    def list_games(
        self,
        query: Query,
        multisort: list[tuple[str, str]],
        page: int,
        page_size: int,
        source_id: int | None = None,
    ) -> list[GameMeta]:
        where_sql, params = compile_where(query)
        order_sql = self._order_sql(multisort)
        offset = max(page, 0) * page_size

        where_parts = [where_sql]
        sql_params: list[object] = list(params)
        if source_id is not None:
            where_parts.append("source_id = ?")
            sql_params.append(int(source_id))

        final_where = " AND ".join(part for part in where_parts if part)
        sql = f"""
        SELECT game_id, white, black, result, event, site, date, eco, opening, offset_bytes
        FROM games
        WHERE {final_where}
        ORDER BY {order_sql}
        LIMIT ? OFFSET ?
        """

        conn = self._connect()
        try:
            rows = conn.execute(sql, [*sql_params, page_size, offset]).fetchall()
        finally:
            conn.close()

        out: list[GameMeta] = []
        for row in rows:
            out.append(
                GameMeta(
                    game_id=int(row[0]),
                    white=row[1],
                    black=row[2],
                    result=row[3],
                    event=row[4],
                    site=row[5],
                    date=row[6],
                    eco=row[7],
                    opening=row[8],
                    offset_bytes=int(row[9]),
                )
            )
        return out

    def list_games_for_tree(
        self, source_id: int, after_game_id: int = 0
    ) -> list[tuple[int, str, int, str]]:
        """
        Return [(game_id, pgn_path, offset_bytes, result), ...] for tree building.

        Results are sorted by (pgn_path, offset_bytes) so callers can scan each
        file sequentially with at most one seek per chunk.

        If after_game_id > 0, only games with game_id > after_game_id are
        returned (incremental update).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT game_id, pgn_path, offset_bytes, result
                   FROM games
                   WHERE source_id = ? AND game_id > ?
                   ORDER BY pgn_path ASC, offset_bytes ASC""",
                (int(source_id), int(after_game_id)),
            ).fetchall()
        finally:
            conn.close()
        return [
            (int(r[0]), str(r[1]), int(r[2]), str(r[3]) if r[3] else "")
            for r in rows
        ]

    def list_game_offsets_for_path(self, source_id: int, pgn_path: str) -> list[int]:
        """
        Return sorted offset_bytes for every game in one PGN file within a source.
        Used by build_tree() to split work across parallel worker processes.
        Returns only integers (not full row tuples) to keep memory small.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT offset_bytes FROM games "
                "WHERE source_id=? AND pgn_path=? "
                "ORDER BY offset_bytes ASC",
                (int(source_id), pgn_path),
            ).fetchall()
        finally:
            conn.close()
        return [int(r[0]) for r in rows]

    def list_pgn_paths_for_source(self, source_id: int) -> list[str]:
        """Return distinct PGN file paths for a source, ordered alphabetically."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT pgn_path FROM games WHERE source_id=? ORDER BY pgn_path ASC",
                (int(source_id),),
            ).fetchall()
        finally:
            conn.close()
        return [str(r[0]) for r in rows]

    def count_games_for_source(self, source_id: int) -> int:
        """Return total number of games for a source."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM games WHERE source_id=?",
                (int(source_id),),
            ).fetchone()
        finally:
            conn.close()
        return int(row[0]) if row else 0

    def get_last_game_id_for_source(self, source_id: int) -> int:
        """Return the highest game_id for a source, or 0 if no games exist."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT MAX(game_id) FROM games WHERE source_id=?",
                (int(source_id),),
            ).fetchone()
        finally:
            conn.close()
        return int(row[0]) if (row and row[0] is not None) else 0

    def list_game_ids_for_source(self, source_id: int) -> list[int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT game_id FROM games WHERE source_id=? ORDER BY game_id ASC",
                (int(source_id),),
            ).fetchall()
        finally:
            conn.close()
        return [int(row[0]) for row in rows]

    def open_game_pgn_text(self, game_id: int) -> str:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT pgn_path, offset_bytes FROM games WHERE game_id=?",
                (game_id,),
            ).fetchone()
        finally:
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
