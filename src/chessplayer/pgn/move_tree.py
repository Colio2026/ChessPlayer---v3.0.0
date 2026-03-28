from __future__ import annotations

"""
move_tree.py
────────────
One-time scan of a PGN library → compact position tree saved to disk.
Subsequent loads are instant (unpickle).  Queries are O(1) hash lookups.

Tree structure (in memory)
--------------------------
_data: dict[pos_key, dict[uci, _NodeStat]]

pos_key   — tuple of 11 bitboard integers from chess.Board
            (pawns, knights, bishops, rooks, queens, kings,
             white_occupied, black_occupied, turn, castling_rights, ep_square)
            Transposition-safe; O(1) to compute vs O(64) for board.fen().

uci       — UCI string of the move ("e2e4").  Stored as UCI so no
            legal_moves call is needed at build time.  Converted to SAN
            only in query() — at most max_out calls per lookup.

_NodeStat — [count, white_wins, draws, black_wins] packed as a list
             for compact pickling.

Disk format
-----------
Single gzip-pickle file in  <data_dir>/trees/<sha1_of_source_path>.pkl.gz
Versioned payload (_TREE_VERSION=2). Trees built with version < 2 have
incompatible FEN string keys and trigger an automatic full rebuild.

Performance
-----------
Full rebuild:
  Fetches only offset_bytes integers per file from the DB (no bulk row load),
  splits into _MAX_CHUNK_GAMES-sized chunks, and fans out to a
  ProcessPoolExecutor.  Each worker runs _GameVisitor which:
    - receives board in pre-move state for FREE from python-chess — no second
      board replay, no extra legal_moves call per move
    - skips annotated sidelines via SkipType.SKIP
    - uses O(1) bitboard tuple as key instead of board.fen()
  Progress is reported after every chunk → smooth counter updates.

Incremental update:
  Loads existing tree, scans only new games (game_id > last_game_id).
  Single-threaded; near-instant for small additions.
"""

import gc
import gzip
import hashlib
import io
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import Callable, Optional

import chess
import chess.pgn
import chess.polyglot

from chessplayer.pgn.continuations import ContinuationStat
from chessplayer.pgn.store import PgnStore
from chessplayer.utils.paths import resolve_path

ProgressCb = Callable[[int, str], None]
CancelCb   = Callable[[], bool]

# Indices into the per-node list  [count, white_wins, draws, black_wins]
_C  = 0
_WW = 1
_DR = 2
_BW = 3

_TREE_VERSION    = 3    # bump when key/value format changes → triggers full rebuild
_MAX_CHUNK_GAMES = 500  # games per worker chunk — bounds IPC payload AND drives progress frequency
_MAX_FULL_MOVES  = 30   # only index the first N full moves — positions beyond this are
                         # almost always unique to one game and useless for continuations


# ── Position key ──────────────────────────────────────────────────────────────

def _pos_key(board: chess.Board) -> int:
    """
    Polyglot Zobrist hash — transposition-safe position key as a single int.

    chess.polyglot.zobrist_hash() XORs pre-computed 64-bit values for every
    piece placement, castling right, ep file, and side to move.  The result
    is a single Python int (~28 bytes) vs a tuple of 11 large ints (~452 bytes).
    For a 3 M-position tree that is ~1.3 GB of key memory saved.

    Collision probability is negligible (< 1 in 2^64 per pair of positions).
    """
    return chess.polyglot.zobrist_hash(board)


def _tree_path(cfg: dict, source_path: str) -> Path:
    """Deterministic path for the tree file based on source path."""
    data_dir  = resolve_path(cfg["paths"]["data_dir"])
    trees_dir = data_dir / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    slug = hashlib.sha1(source_path.encode()).hexdigest()[:16]
    return trees_dir / f"{slug}.pkl.gz"


# ── Visitor ───────────────────────────────────────────────────────────────────

class _GameVisitor(chess.pgn.BaseVisitor):
    """
    Accumulates position→move statistics via python-chess's visitor protocol.

    visit_move(board, move) is called by python-chess for each main-line move
    with the board already maintained in the pre-move state — provided for
    free, eliminating the need for a second board replay or board.san() call.

    begin_variation() returning SkipType.SKIP tells python-chess to discard
    the entire variation subtree without calling parse_san() for any of its
    moves — a significant saving for heavily annotated game collections.

    The visitor instance is REUSED across games (factory = lambda: visitor).
    begin_game() resets per-game state before each new game.
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        self._ww: int = 0
        self._dr: int = 1
        self._bw: int = 0

    def begin_game(self) -> None:
        self._ww, self._dr, self._bw = 0, 1, 0   # default: draw

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        if tagname == "Result":
            if tagvalue == "1-0":    self._ww, self._dr, self._bw = 1, 0, 0
            elif tagvalue == "0-1":  self._ww, self._dr, self._bw = 0, 0, 1

    def begin_variation(self) -> chess.pgn.SkipType:
        return chess.pgn.SkipType.SKIP

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        if board.fullmove_number > _MAX_FULL_MOVES:
            return   # late-game positions are unique per game; skip to save memory
        key  = _pos_key(board)   # O(1) Zobrist int — board is pre-move state
        uci  = move.uci()        # O(1) — no legal_moves call
        data = self._data
        if key not in data:
            data[key] = {}
        node = data[key]
        if uci not in node:
            node[uci] = [0, 0, 0, 0]
        s       = node[uci]
        s[_C]  += 1
        s[_WW] += self._ww
        s[_DR] += self._dr
        s[_BW] += self._bw

    def result(self) -> object:
        return self   # non-None → game found; None returned by read_game() signals EOF


# ── Scan helpers ──────────────────────────────────────────────────────────────

def _scan_chunk(args: tuple) -> tuple[dict, int]:
    """
    Process a contiguous slice of games from a PGN file starting from a
    known byte offset.  Safe to run in a worker process.

    Parameters (packed as a tuple for ProcessPoolExecutor compatibility)
    ────────────────────────────────────────────────────────────────────
    pgn_path_str : str — absolute path to the PGN file
    first_offset : int — byte offset to seek before reading; 0 = file start
    n_games      : int — number of games to process from that position

    Returns (partial_data, games_ok).  Caller merges partial_data.
    """
    pgn_path_str, first_offset, n_games = args
    data:     dict = {}
    games_ok: int  = 0

    visitor = _GameVisitor(data)
    factory = lambda: visitor   # reuse; begin_game() resets per-game state

    try:
        with open(pgn_path_str, "rb") as raw:
            if first_offset > 0:
                raw.seek(first_offset)
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")

            for _ in range(n_games):
                found = chess.pgn.read_game(text, Visitor=factory)
                if found is None:
                    break
                games_ok += 1
    except Exception:
        pass

    return data, games_ok


def _merge_partial(base: dict, patch: dict) -> None:
    """Merge patch into base in-place."""
    for key, moves in patch.items():
        if key not in base:
            base[key] = {uci: list(s) for uci, s in moves.items()}
        else:
            b = base[key]
            for uci, s in moves.items():
                if uci not in b:
                    b[uci] = list(s)
                else:
                    bs      = b[uci]
                    bs[_C]  += s[_C]
                    bs[_WW] += s[_WW]
                    bs[_DR] += s[_DR]
                    bs[_BW] += s[_BW]


def _make_chunks(pgn_path: str, offsets: list[int], n_workers: int) -> list[tuple]:
    """
    Split a sorted list of game-start offsets into worker-sized chunks.

    Each chunk = (pgn_path, first_offset, n_games) accepted by _scan_chunk.
    chunk_size = min(per_worker_ceiling, _MAX_CHUNK_GAMES) so that:
      - libraries smaller than n_workers * MAX get exactly n_workers chunks
      - larger libraries are split finely for frequent progress callbacks
        and bounded IPC payload (~500 games × ~300 bytes/pos × 30 pos ≈ 4 MB)
    """
    if not offsets:
        return []
    n          = len(offsets)
    per_worker = max(1, (n + n_workers - 1) // n_workers)
    chunk_size = min(per_worker, _MAX_CHUNK_GAMES)
    return [
        (pgn_path, offsets[i], min(chunk_size, n - i))
        for i in range(0, n, chunk_size)
    ]


# ── MoveTree ──────────────────────────────────────────────────────────────────

class MoveTree:
    """
    In-memory position→continuations lookup table.

    Built once from a PGN library, persisted to disk, loaded on startup.
    All queries are pure dict lookups — O(1) regardless of database size.
    """

    def __init__(self) -> None:
        self._data: dict[int, dict[str, list[int]]] = {}
        self._total_games:  int = 0
        self._source_path:  str = ""
        self._last_game_id: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def is_built(self) -> bool:
        return bool(self._data)

    def total_games(self) -> int:
        return self._total_games

    def last_game_id(self) -> int:
        return self._last_game_id

    def query(self, prefix_uci: list[str], max_out: int = 30) -> list[ContinuationStat]:
        """
        Return continuations from the position reached by prefix_uci.
        Empty prefix = starting position.
        Returns [] if the position was never seen.

        SAN conversion happens here (at most max_out times per call) rather
        than at build time — avoiding legal_moves generation for every game.
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

        out = []
        for uci, s in node.items():
            try:
                move = chess.Move.from_uci(uci)
                san  = board.san(move)
            except Exception:
                continue
            out.append(ContinuationStat(
                san        = san,
                count      = s[_C],
                white_wins = s[_WW],
                draws      = s[_DR],
                black_wins = s[_BW],
            ))
        out.sort(key=lambda x: (-x.count, -x.white_score_pct))
        return out[:max_out]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, cfg: dict) -> Path:
        path = _tree_path(cfg, self._source_path)
        payload = {
            "version":      _TREE_VERSION,
            "source_path":  self._source_path,
            "total_games":  self._total_games,
            "last_game_id": self._last_game_id,
            "data":         self._data,
        }
        with gzip.open(path, "wb", compresslevel=1) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load(cfg: dict, source_path: str) -> "MoveTree":
        """
        Load tree from disk.  Returns an empty (is_built=False) tree if the
        file does not exist, is corrupt, or was built with an older key format.
        An empty tree triggers a full rebuild in build_tree().
        """
        tree = MoveTree()
        tree._source_path = source_path
        path = _tree_path(cfg, source_path)
        if not path.exists():
            return tree
        try:
            with gzip.open(path, "rb") as f:
                payload = pickle.load(f)
            if payload.get("version", 1) < _TREE_VERSION:
                return tree   # old key format (FEN string or bitboard tuple); must rebuild
            tree._data         = payload["data"]
            tree._total_games  = payload.get("total_games", 0)
            tree._last_game_id = payload.get("last_game_id", 0)
        except Exception:
            tree._data = {}
        return tree


# ── Builder ───────────────────────────────────────────────────────────────────

def build_tree(
    cfg:         dict,
    source_path: str,
    pgn_store:   PgnStore,
    source_id:   int,
    progress_cb: Optional[ProgressCb] = None,
    cancel_cb:   Optional[CancelCb]   = None,
    total_cb:    Optional[Callable[[int], None]] = None,
    incremental: bool = False,
) -> MoveTree:
    """
    Build (or incrementally update) the move tree for a PGN source.

    Full rebuild (incremental=False)
    ─────────────────────────────────
    Fetches only offset_bytes integers per file (no bulk record load), splits
    into _MAX_CHUNK_GAMES-sized chunks, fans out to ProcessPoolExecutor.
    progress_cb fires after every chunk → smooth X / Y counter in the UI.

    Incremental update (incremental=True)
    ───────────────────────────────────────
    Loads the existing tree, queries only games with game_id > last_game_id,
    and merges them.  Single-threaded; near-instant for small additions.
    """

    # ── Load base tree ────────────────────────────────────────────────────────
    if incremental:
        tree         = MoveTree.load(cfg, source_path)
        after_gid    = tree._last_game_id
        data         = tree._data
        games_ok     = tree._total_games
    else:
        tree              = MoveTree()
        tree._source_path = source_path
        after_gid         = 0
        data              = {}
        games_ok          = 0

    # ── Incremental path ──────────────────────────────────────────────────────
    if incremental:
        records = pgn_store.list_games_for_tree(source_id, after_game_id=after_gid)
        total   = len(records)

        if total_cb:
            total_cb(total)   # switches bar from indeterminate to percentage mode

        if total == 0:
            if progress_cb:
                progress_cb(0, "Tree is already up to date — no new games.")
            if not tree._source_path:
                tree._source_path = source_path
            return tree

        if progress_cb:
            progress_cb(0, f"Updating tree: {total:,} new games …")

        # Extract per-file offsets from records and split into bounded chunks.
        # The old approach passed the entire file group as one chunk, which
        # built a single multi-GB partial dict before merging → memory crash.
        n_workers  = min(os.cpu_count() or 1, 8)
        all_chunks: list[tuple] = []
        for pgn_path, group_iter in groupby(records, key=lambda r: r[1]):
            group   = list(group_iter)
            offsets = [r[2] for r in group]   # offset_bytes, already sorted
            all_chunks.extend(_make_chunks(pgn_path, offsets, n_workers))

        processed = 0
        if len(all_chunks) <= n_workers:
            # Few chunks — sequential avoids process-spawn overhead
            for chunk in all_chunks:
                if cancel_cb and cancel_cb():
                    break
                partial_data, partial_ok = _scan_chunk(chunk)
                _merge_partial(data, partial_data)
                games_ok  += partial_ok
                processed += partial_ok
                if progress_cb:
                    progress_cb(processed, f"Updating tree: {processed:,} / {total:,} games …")
        else:
            # Large update — fan out to worker processes, same as full rebuild
            with ProcessPoolExecutor(max_workers=n_workers) as exe:
                futs = [exe.submit(_scan_chunk, chunk) for chunk in all_chunks]
                for fut in as_completed(futs):
                    if cancel_cb and cancel_cb():
                        exe.shutdown(wait=False, cancel_futures=True)
                        break
                    try:
                        partial_data, partial_ok = fut.result()
                    except Exception:
                        continue
                    _merge_partial(data, partial_data)
                    games_ok  += partial_ok
                    processed += partial_ok
                    if progress_cb:
                        progress_cb(processed, f"Updating tree: {processed:,} / {total:,} games …")
            del futs, all_chunks   # free cached partial_data in completed futures

        last_game_id = max((r[0] for r in records), default=after_gid)

    # ── Full-rebuild path ─────────────────────────────────────────────────────
    else:
        pgn_paths = pgn_store.list_pgn_paths_for_source(source_id)
        total     = pgn_store.count_games_for_source(source_id)

        if total == 0 or not pgn_paths:
            if progress_cb:
                progress_cb(0, "No games to process.")
            return tree

        if total_cb:
            total_cb(total)   # switches bar from indeterminate to percentage mode

        if progress_cb:
            progress_cb(0, f"Building tree: {total:,} games …")

        n_workers  = min(os.cpu_count() or 1, 8)
        all_chunks: list[tuple] = []
        for pgn_path in pgn_paths:
            offsets = pgn_store.list_game_offsets_for_path(source_id, pgn_path)
            all_chunks.extend(_make_chunks(pgn_path, offsets, n_workers))

        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futs = [exe.submit(_scan_chunk, chunk) for chunk in all_chunks]
            for fut in as_completed(futs):
                if cancel_cb and cancel_cb():
                    exe.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    partial_data, partial_ok = fut.result()
                except Exception:
                    continue
                _merge_partial(data, partial_data)
                games_ok += partial_ok
                if progress_cb:
                    progress_cb(
                        games_ok,
                        f"Building tree: {games_ok:,} / {total:,} games …"
                    )
        # Each Future caches its result (partial_data dict) until the Future
        # object is freed.  800 futures × ~2.4 MB each ≈ 1.9 GB still live
        # at save time unless we explicitly release the list here.
        del futs, all_chunks

        last_game_id = pgn_store.get_last_game_id_for_source(source_id)

    # ── Persist ───────────────────────────────────────────────────────────────
    tree._data         = data
    tree._total_games  = games_ok
    tree._source_path  = source_path
    tree._last_game_id = last_game_id

    if progress_cb:
        progress_cb(total, f"Saving tree ({games_ok:,} games) …")

    gc.collect()   # free stale refs from build process before serialising
    tree.save(cfg)
    return tree