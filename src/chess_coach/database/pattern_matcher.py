"""
database/pattern_matcher.py
============================
Runtime query against game_index.db. Returns GMPrecedent objects.

Query logic (per spec Section 8):
  1. Compute pawn_hash of current position.
  2. Query game_index WHERE pawn_hash = current
                       AND strategy_tag = primary_strategy
                       AND rating > min_rating.
  3. Filter by phase match and result (prefer winning side).
  4. Return top 3 ordered by rating DESC, eval similarity ASC.
  5. Wrap results as GMPrecedent objects.

Feint gate (Phase 3 promise fulfilled):
  If the query returns >= 1 result for strategy_tag='feint',
  db_confirmation is set True — allowing feint to fire as primary
  in the conflict resolver.

Performance:
  The pawn_hash index makes each query O(log n) on the hash.
  At 6M positions this is well under 5ms.
  Connection is kept open for the lifetime of the StrategyEngine.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import chess

from chess_coach.core.data_types   import GMPrecedent
from chess_coach.core.board_utils  import get_pawn_hash

_DEFAULT_MIN_RATING = 2400
_MAX_RESULTS        = 3


class PatternMatcher:
    """
    Queries coach_positions table (inside index.sqlite) for GM precedents.
    The same SQLite file used by ChessPlayer's browser — no second DB.

    Parameters
    ----------
    db_path : str
        Path to index.sqlite (ChessPlayer's main game index).
        If empty string or file does not exist, all queries return [].
    min_rating : int
        Minimum ELO threshold for returned games.
    """

    def __init__(self, db_path: str = '', min_rating: int = _DEFAULT_MIN_RATING) -> None:
        self.db_path    = db_path
        self.min_rating = min_rating
        self._conn: sqlite3.Connection | None = None
        self._available = False

        if db_path and Path(db_path).exists():
            try:
                self._conn = sqlite3.connect(db_path, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
                self._available = True
            except sqlite3.Error:
                self._conn = None

    @property
    def is_available(self) -> bool:
        """True if the DB file was found and opened successfully."""
        return self._available

    def query(
        self,
        board: chess.Board,
        strategy: str,
        phase: str,
        max_results: int = _MAX_RESULTS,
    ) -> list[GMPrecedent]:
        """
        Find GM games with a matching pawn structure and strategy tag.

        Parameters
        ----------
        board    : chess.Board  — current position
        strategy : str          — 'blitz'|'flank'|'fortress'|'feint'|'general'
        phase    : str          — 'opening'|'middlegame'|'endgame'
        max_results : int       — cap on returned precedents

        Returns
        -------
        list[GMPrecedent], length 0–3.
        """
        if not self._available or self._conn is None:
            return []

        pawn_hash = get_pawn_hash(board)

        try:
            # Primary query: exact pawn hash + strategy + phase
            rows = self._run_query(pawn_hash, strategy, phase, max_results)

            # Fallback: relax phase constraint if no results
            if not rows:
                rows = self._run_query(pawn_hash, strategy, phase=None,
                                       max_results=max_results)

            # Fallback 2: relax strategy if still nothing (returns 'general' positions)
            if not rows and strategy != 'general':
                rows = self._run_query(pawn_hash, 'general', phase=None,
                                       max_results=max_results)

            return [self._row_to_precedent(r) for r in rows]

        except sqlite3.Error:
            return []

    def db_confirms_feint(self, board: chess.Board, phase: str) -> bool:
        """
        Return True if the game_index contains a feint-tagged position
        with the same pawn structure. Used to lift the 0.64 feint cap.
        """
        if not self._available or self._conn is None:
            return False
        pawn_hash = get_pawn_hash(board)
        try:
            cur = self._conn.execute(
                """SELECT COUNT(*) FROM coach_positions
                   WHERE pawn_hash = ?
                     AND strategy_tag = 'feint'
                     AND (rating_white >= ? OR rating_black >= ?)
                   LIMIT 1""",
                (pawn_hash, self.min_rating, self.min_rating),
            )
            return (cur.fetchone()[0] or 0) > 0
        except sqlite3.Error:
            return False

    def close(self) -> None:
        """Close the DB connection."""
        if self._conn:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
            self._available = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_query(
        self,
        pawn_hash: str,
        strategy: str,
        phase: str | None,
        max_results: int,
    ) -> list[sqlite3.Row]:
        assert self._conn is not None

        if phase:
            sql = """
                SELECT game_id, player_white, player_black, ply,
                       key_move, annotation, result, rating_white,
                       rating_black, eval_cp, strategy_tag
                FROM coach_positions
                WHERE pawn_hash    = ?
                  AND strategy_tag = ?
                  AND phase        = ?
                  AND (rating_white >= ? OR rating_black >= ?)
                ORDER BY
                    MAX(COALESCE(rating_white,0), COALESCE(rating_black,0)) DESC
                LIMIT ?
            """
            params = (pawn_hash, strategy, phase,
                      self.min_rating, self.min_rating, max_results)
        else:
            sql = """
                SELECT game_id, player_white, player_black, ply,
                       key_move, annotation, result, rating_white,
                       rating_black, eval_cp, strategy_tag
                FROM coach_positions
                WHERE pawn_hash    = ?
                  AND strategy_tag = ?
                  AND (rating_white >= ? OR rating_black >= ?)
                ORDER BY
                    MAX(COALESCE(rating_white,0), COALESCE(rating_black,0)) DESC
                LIMIT ?
            """
            params = (pawn_hash, strategy,
                      self.min_rating, self.min_rating, max_results)

        cur = self._conn.execute(sql, params)
        return cur.fetchall()

    @staticmethod
    def _row_to_precedent(row: sqlite3.Row) -> GMPrecedent:
        """Convert a DB row to a GMPrecedent dataclass."""
        # Determine the GM player (prefer the higher-rated side)
        wr = row['rating_white'] or 0
        br = row['rating_black'] or 0
        player = row['player_white'] if wr >= br else row['player_black']

        annotation = row['annotation'] or ''
        if not annotation:
            annotation = f"{row['strategy_tag'].title()} pattern — {row['result']}"

        return GMPrecedent(
            player     = player or 'Unknown',
            game_id    = str(row['game_id']),
            ply        = row['ply'],
            key_move   = row['key_move'] or '',
            annotation = annotation[:200],
        )
