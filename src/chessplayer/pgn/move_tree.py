from __future__ import annotations

"""
move_tree.py
────────────
One-time scan of a PGN library → compact position tree saved to disk.
Subsequent loads are instant (unpickle).  Queries are O(1) hash lookups.

Tree structure (in memory)
--------------------------
_data: dict[pos_key, dict[san, _NodeStat]]

pos_key   — FEN fields 0-3 joined: "rnbqkbnr/pppppppp/... w KQkq -"
            (piece placement + turn + castling + en-passant only —
             halfmove clock and fullmove number excluded so transpositions
             from different move orders all land on the same node)
san       — SAN of the move played FROM this position
_NodeStat — (count, white_wins, draws, black_wins) packed as a list
             for compact pickling

Disk format
-----------
Single gzip-pickle file in  <data_dir>/trees/<sha1_of_source_path>.pkl.gz
Typically 10-40 MB for 1 M games (after compression).
Build time: ~2-5 min for 1 M games on a laptop.
Load time:  < 1 second.
Query time: < 1 ms.
"""

import gzip
import hashlib
import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import chess
import chess.pgn

from pgn.continuations import ContinuationStat
from pgn.store import PgnStore
from utils.paths import resolve_path

ProgressCb = Callable[[int, str], None]
CancelCb   = Callable[[], bool]

# Indices into the per-node list  [count, white_wins, draws, black_wins]
_C  = 0
_WW = 1
_DR = 2
_BW = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pos_key(board: chess.Board) -> str:
    """Position key ignoring clocks — transposition-safe."""
    parts = board.fen().split()
    return " ".join(parts[:4])


def _tree_path(cfg: dict, source_path: str) -> Path:
    """Deterministic path for the tree file based on source path."""
    data_dir   = resolve_path(cfg["paths"]["data_dir"])
    trees_dir  = data_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha1(source_path.encode()).hexdigest()[:16]
    return trees_dir / f"{slug}.pkl.gz"


# ── MoveTree ──────────────────────────────────────────────────────────────────

class MoveTree:
    """
    In-memory position→continuations lookup table.

    Built once from a PGN library, persisted to disk, loaded on startup.
    All queries are pure dict lookups — O(1) regardless of database size.
    """

    def __init__(self) -> None:
        # pos_key -> { san -> [count, white_wins, draws, black_wins] }
        self._data: dict[str, dict[str, list[int]]] = {}
        self._total_games: int = 0
        self._source_path: str = ""

    # ── public API ────────────────────────────────────────────────────────────

    def is_built(self) -> bool:
        return bool(self._data)

    def total_games(self) -> int:
        return self._total_games

    def query(
        self,
        prefix_uci: list[str],
        max_out: int = 30,
    ) -> list[ContinuationStat]:
        """
        Return continuations from the position reached by prefix_uci.
        Empty prefix = starting position.
        Returns [] if the position was never seen.
        """
        board = chess.Board()
        for uci in prefix_uci:
            try:
                board.push(chess.Move.from_uci(uci))
            except Exception:
                return []

        node = self._data.get(_pos_key(board))
        if not node:
            return []

        out = [
            ContinuationStat(
                san        = san,
                count      = s[_C],
                white_wins = s[_WW],
                draws      = s[_DR],
                black_wins = s[_BW],
            )
            for san, s in node.items()
        ]
        out.sort(key=lambda x: (-x.count, -x.white_score_pct))
        return out[:max_out]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, cfg: dict) -> Path:
        path = _tree_path(cfg, self._source_path)
        payload = {
            "source_path":  self._source_path,
            "total_games":  self._total_games,
            "data":         self._data,
        }
        with gzip.open(path, "wb", compresslevel=1) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(cfg: dict, source_path: str) -> "MoveTree":
        """
        Load tree from disk.  Returns an empty (is_built=False) tree if the
        file does not exist or is corrupt.
        """
        tree = MoveTree()
        tree._source_path = source_path
        path = _tree_path(cfg, source_path)
        if not path.exists():
            return tree
        try:
            with gzip.open(path, "rb") as f:
                payload = pickle.load(f)
            tree._data        = payload["data"]
            tree._total_games = payload.get("total_games", 0)
        except Exception:
            # Corrupt file — return empty tree so the user can rebuild
            tree._data = {}
        return tree


# ── Builder ───────────────────────────────────────────────────────────────────

def build_tree(
    cfg:          dict,
    source_path:  str,
    pgn_store:    PgnStore,
    source_id:    int,
    progress_cb:  Optional[ProgressCb] = None,
    cancel_cb:    Optional[CancelCb]   = None,
) -> MoveTree:
    """
    Read every game in source_id from pgn_store, replay the mainline,
    and record every position → next-move transition.

    Runs in a background thread (called by _TreeWorker in variations_panel).
    Saves the tree to disk and returns it.

    Performance notes
    -----------------
    • python-chess board replay is the bottleneck: ~10-20k games/sec on
      a laptop.  1 M games takes 1-2 minutes.
    • We commit the tree to disk only once at the end, not incrementally,
      so disk I/O is not the bottleneck.
    • Memory: roughly 200-400 MB for 1 M games (before gzip compression).
    """
    game_ids = pgn_store.list_game_ids_for_source(source_id)
    total    = len(game_ids)

    data: dict[str, dict[str, list[int]]] = {}
    games_ok = 0

    for i, gid in enumerate(game_ids):
        if cancel_cb and cancel_cb():
            break

        if progress_cb and (i == 0 or (i + 1) % 500 == 0 or i + 1 == total):
            progress_cb(
                i + 1,
                f"Building tree: {i + 1:,} / {total:,} games …"
            )

        try:
            pgn_text = pgn_store.open_game_pgn_text(gid)
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if game is None:
                continue

            result = str(game.headers.get("Result", "")).strip()
            if result == "1-0":
                ww, dr, bw = 1, 0, 0
            elif result == "0-1":
                ww, dr, bw = 0, 0, 1
            else:
                ww, dr, bw = 0, 1, 0

            board = chess.Board()
            node  = game

            while node.variations:
                main = node.variations[0]
                key  = _pos_key(board)

                # Ensure the position node exists
                if key not in data:
                    data[key] = {}
                pos_node = data[key]

                # SAN of the move played from this position
                try:
                    san = board.san(main.move)
                except Exception:
                    break

                if san not in pos_node:
                    pos_node[san] = [0, 0, 0, 0]
                s = pos_node[san]
                s[_C]  += 1
                s[_WW] += ww
                s[_DR] += dr
                s[_BW] += bw

                board.push(main.move)
                node = main

            games_ok += 1

        except Exception:
            continue

    tree = MoveTree()
    tree._data        = data
    tree._total_games = games_ok
    tree._source_path = source_path

    if progress_cb:
        progress_cb(total, f"Saving tree ({games_ok:,} games) …")

    tree.save(cfg)
    return tree
