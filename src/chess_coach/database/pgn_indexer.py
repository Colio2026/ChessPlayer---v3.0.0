"""
database/pgn_indexer.py
========================
Builds the coach_positions table inside the EXISTING index.sqlite that
ChessPlayer's indexer.py already maintains.

Design principle
----------------
The existing indexer already parsed every PGN game and stored:
  - pgn_path      : absolute path to the PGN file
  - offset_bytes  : byte offset of that game within the file

This module reads those records and seeks directly to each game —
zero double-parsing. The PGN is never walked twice.

The coach_positions table is written into the SAME index.sqlite file.
No second database is needed.

Auto-trigger
------------
StrategyEngine calls ensure_indexed() on first run. If coach_positions
is empty it builds the table from whatever games already exist in index.sqlite.

CLI usage (optional manual rebuild)
-------------------------------------
    python -m chess_coach.database.pgn_indexer \\
        --db  data/index.sqlite \\
        [--stockfish /path/to/stockfish] \\
        [--movetime 500] \\
        [--min-rating 0]

Schema added to index.sqlite
-----------------------------
    coach_positions (id, game_id, ply, fen, pawn_hash, strategy_tag,
                     phase, eval_cp, player_white, player_black,
                     rating_white, rating_black, result, key_move, annotation)
"""
from __future__ import annotations

import argparse
import io
import multiprocessing
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import Callable, Optional

import chess
import chess.pgn
import chess.polyglot

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chess_coach.core.board_utils    import get_phase, get_pawn_hash
from chess_coach.core.data_types     import MetricSignal
from chess_coach.extractors.king_safety      import extract_king_safety
from chess_coach.extractors.space_control    import extract_space_control
from chess_coach.extractors.piece_mobility   import extract_piece_mobility
from chess_coach.extractors.pawn_structure   import extract_pawn_structure
from chess_coach.extractors.material_balance import extract_material_balance
from chess_coach.extractors.tactic_scanner   import extract_tactics
from chess_coach.strategies.blitz_detector    import score_blitz
from chess_coach.strategies.flank_detector    import score_flank
from chess_coach.strategies.fortress_detector import score_fortress
from chess_coach.strategies.feint_detector    import score_feint

_MIN_PLY            = 10
_MAX_PLY            = 60
_STRATEGY_THRESHOLD = 0.40
_BATCH_SIZE         = 50   # commit every N games

# Tracks which PGN source was last indexed — enables automatic re-index
# when coach.pgn_source changes in config (e.g. swapping to 6M game DB)
_CREATE_META = """
CREATE TABLE IF NOT EXISTS coach_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS coach_positions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      INTEGER NOT NULL,
    ply          INTEGER NOT NULL,
    fen          TEXT    NOT NULL,
    pawn_hash    TEXT    NOT NULL,
    strategy_tag TEXT    NOT NULL,
    phase        TEXT    NOT NULL,
    eval_cp      INTEGER,
    player_white TEXT,
    player_black TEXT,
    rating_white INTEGER,
    rating_black INTEGER,
    result       TEXT,
    key_move     TEXT,
    annotation   TEXT,
    FOREIGN KEY(game_id) REFERENCES games(game_id)
);
CREATE INDEX IF NOT EXISTS idx_cp_pawn_hash ON coach_positions (pawn_hash);
CREATE INDEX IF NOT EXISTS idx_cp_strategy  ON coach_positions (strategy_tag, phase);
"""


# ── Public API ─────────────────────────────────────────────────────────────

def ensure_indexed(
    db_path: str,
    stockfish_path: str = '',
    movetime_ms: int = 5,
    min_rating: int = 0,
    verbose: bool = False,
    pgn_source: str = '',
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    force: bool = False,
) -> bool:
    """
    Build coach_positions if it doesn't exist, is empty, or the
    configured pgn_source has changed since the last index run.

    Called automatically by StrategyEngine on first use.
    Returns True if indexing was performed, False if already up to date.

    pgn_source : str
        Path to the PGN file configured in coach.pgn_source.
        When this changes (e.g. swapping Carlsen.pgn → 6M game DB),
        coach_positions is wiped and rebuilt automatically from the new
        games already catalogued in index.sqlite by the browser indexer.
    """
    if not Path(db_path).exists():
        return False

    conn = sqlite3.connect(db_path)
    conn.executescript(_CREATE_META)
    conn.executescript(_CREATE_TABLE)
    conn.commit()

    # Check if source has changed since last index run
    count = conn.execute("SELECT COUNT(*) FROM coach_positions").fetchone()[0]
    stored_source = _get_meta(conn, 'pgn_source')
    source_changed = pgn_source and stored_source and stored_source != pgn_source
    conn.close()

    if count > 0 and not source_changed and not force:
        return False   # already indexed with the same source

    if source_changed and verbose:
        print(f"coach_indexer: pgn_source changed "
              f"({stored_source!r} → {pgn_source!r}) — rebuilding coach_positions")

    build_from_existing_index(
        db_path        = db_path,
        stockfish_path = stockfish_path,
        movetime_ms    = movetime_ms,
        min_rating     = min_rating,
        verbose        = verbose,
        pgn_source     = pgn_source,
        progress_cb    = progress_cb,
    )
    return True


def _get_meta(conn: sqlite3.Connection, key: str) -> str:
    """Read a value from coach_meta table."""
    try:
        row = conn.execute("SELECT value FROM coach_meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else ''
    except sqlite3.Error:
        return ''


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Write a value to coach_meta table."""
    conn.execute(
        "INSERT OR REPLACE INTO coach_meta(key, value) VALUES (?, ?)",
        (key, value)
    )


# Computed once at import time; workers restore these into sys.path on spawn
_CHESS_COACH_DIR = str(Path(__file__).resolve().parent.parent)   # src/chess_coach/
_SRC_DIR         = str(Path(__file__).resolve().parent.parent.parent)  # src/


def _process_chunk(args: tuple) -> tuple:
    """
    Worker: process a slice of game rows and write coach_positions to a temp SQLite file.
    Returns (games_processed, positions_indexed, skipped, temp_sqlite_path).
    Must be a module-level function so ProcessPoolExecutor can pickle it on Windows.
    """
    # On Windows the worker starts a fresh interpreter — restore sys.path so
    # chess_coach imports (and core.stockfish_bridge) resolve correctly.
    for _p in (_SRC_DIR, _CHESS_COACH_DIR):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    chunk_id, rows_chunk, db_path, stockfish_path, movetime_ms, min_rating = args

    tmp_path = db_path + f".chunk_{chunk_id}.tmp"
    conn = sqlite3.connect(tmp_path)
    conn.executescript(_CREATE_TABLE)
    conn.commit()

    bridge = _start_engine(stockfish_path, movetime_ms, False)
    seen: dict = {}
    batch: list = []
    n_games = n_positions = n_skipped = 0

    for pgn_path, group_iter in groupby(
        sorted(rows_chunk, key=lambda r: (r[1], r[2])),
        key=lambda r: r[1],
    ):
        group = list(group_iter)
        try:
            with open(pgn_path, 'rb') as raw:
                text = io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline='')
                for row in group:
                    game_id, _, offset_bytes, white, black, result = row
                    try:
                        text.seek(offset_bytes)
                        game = chess.pgn.read_game(text)
                    except Exception:
                        n_skipped += 1
                        continue
                    if game is None:
                        n_skipped += 1
                        continue

                    rating_w = _parse_elo(game.headers.get('WhiteElo', ''))
                    rating_b = _parse_elo(game.headers.get('BlackElo', ''))
                    if min_rating > 0 and rating_w < min_rating and rating_b < min_rating:
                        n_skipped += 1
                        continue

                    positions = _walk_game(
                        game, game_id, white or '', black or '',
                        rating_w, rating_b, result or '',
                        bridge, seen,
                    )
                    batch.extend(positions)
                    n_games     += 1
                    n_positions += len(positions)

                    if len(batch) >= _BATCH_SIZE * 10:
                        _flush(conn, batch)
                        batch.clear()

                text.detach()
        except (FileNotFoundError, OSError):
            n_skipped += len(group)

    if batch:
        _flush(conn, batch)
    conn.close()

    if bridge:
        try: bridge.stop()
        except Exception: pass

    return n_games, n_positions, n_skipped, tmp_path


def build_from_existing_index(
    db_path: str,
    stockfish_path: str = '',
    movetime_ms: int    = 5,
    min_rating: int     = 0,
    verbose: bool       = True,
    pgn_source: str     = '',
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Populate coach_positions by reading games already catalogued in index.sqlite.
    Spawns one worker process per CPU core; each writes to a temp SQLite file
    which is merged into the main DB at the end.

    Returns summary: {games_processed, positions_indexed, skipped}.
    """
    # ── Setup main DB ─────────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    conn.executescript(_CREATE_TABLE)
    conn.executescript(_CREATE_META)
    conn.execute("DELETE FROM coach_positions")
    if pgn_source:
        _set_meta(conn, 'pgn_source', pgn_source)
    conn.commit()

    rows = conn.execute("""
        SELECT game_id, pgn_path, offset_bytes, white, black, result
        FROM games ORDER BY game_id ASC
    """).fetchall()
    conn.close()

    rows.sort(key=lambda r: (r[1], r[2]))
    total = len(rows)

    if progress_cb:
        progress_cb(0, total, f"Coach has analysed 0% of games in PGN"
                              f" — this may take a while (0 / {total:,})")

    # ── Parallel processing ───────────────────────────────────────────────────
    n_workers  = max(1, min(multiprocessing.cpu_count() - 1, 8))
    chunk_size = max(50, total // max(1, n_workers * 100))
    chunks     = [rows[i:i + chunk_size] for i in range(0, total, chunk_size)]

    if verbose:
        print(f"Spawning {n_workers} workers, {len(chunks)} chunks, {total:,} games")

    args_list = [
        (i, chunk, db_path, stockfish_path, movetime_ms, min_rating)
        for i, chunk in enumerate(chunks)
    ]

    overall    = {'games_processed': 0, 'positions_indexed': 0, 'skipped': 0}
    temp_paths: list[str] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_chunk, a): a[0] for a in args_list}
        for future in as_completed(futures):
            try:
                n_games, n_pos, n_skip, tmp_path = future.result()
            except Exception as exc:
                if verbose:
                    print(f"Chunk failed: {exc}")
                continue

            overall['games_processed']  += n_games
            overall['positions_indexed'] += n_pos
            overall['skipped']           += n_skip
            temp_paths.append(tmp_path)

            done = overall['games_processed']
            pct  = int(done / total * 100) if total else 0
            msg  = (f"Coach has analysed {pct}% of games in PGN"
                    f" — this may take a while ({done:,} / {total:,})")
            if progress_cb:
                progress_cb(done, total, msg)
            if verbose:
                print(f"  {msg}")

    # ── Merge temp files into main DB ─────────────────────────────────────────
    main_conn = sqlite3.connect(db_path)
    for tmp_path in temp_paths:
        try:
            safe = tmp_path.replace("'", "''")
            main_conn.execute(f"ATTACH DATABASE '{safe}' AS tmp_db")
            main_conn.execute("""
                INSERT INTO coach_positions
                    (game_id, ply, fen, pawn_hash, strategy_tag, phase,
                     eval_cp, player_white, player_black, rating_white,
                     rating_black, result, key_move, annotation)
                SELECT game_id, ply, fen, pawn_hash, strategy_tag, phase,
                       eval_cp, player_white, player_black, rating_white,
                       rating_black, result, key_move, annotation
                FROM tmp_db.coach_positions
            """)
            main_conn.execute("DETACH DATABASE tmp_db")
            main_conn.commit()
        except Exception as exc:
            if verbose:
                print(f"Merge failed for {tmp_path}: {exc}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    n        = overall['games_processed']
    done_msg = f"Coach ready  ·  {n:,} games indexed"
    if progress_cb:
        progress_cb(total, total, done_msg)
    if verbose:
        print(f"\nDone. {n} games processed, "
              f"{overall['positions_indexed']} positions indexed, "
              f"{overall['skipped']} skipped.")

    main_conn.close()
    return overall


# ── Internal helpers ────────────────────────────────────────────────────────

def _parse_elo(val: str) -> int:
    """Parse ELO from a PGN header value. Returns 0 for missing or non-numeric."""
    try:
        v = (val or '').strip().strip('?')
        return int(v) if v.isdigit() else 0
    except (ValueError, AttributeError):
        return 0


def _load_game_at_offset(pgn_path: str, offset_bytes: int) -> Optional[chess.pgn.Game]:
    """Seek to a byte offset in the PGN and parse exactly one game."""
    with open(pgn_path, 'rb') as raw:
        text = io.TextIOWrapper(raw, encoding='utf-8', errors='replace', newline='')
        text.seek(offset_bytes)
        return chess.pgn.read_game(text)


def _walk_game(
    game: chess.pgn.Game,
    game_id: int,
    white: str, black: str,
    rating_w: int, rating_b: int,
    result: str,
    bridge,
    seen: dict,
) -> list[tuple]:
    """Walk mainline positions ply 10–60 and return DB row tuples.

    seen: shared dict mapping Zobrist hash → (strategy_tag, phase, eval_cp).
    Positions already in seen skip extractors and Stockfish entirely — the
    first occurrence of any position pays the full cost; every repeat is free.
    """
    rows: list[tuple] = []
    board = game.board()
    node  = game
    ply   = 0

    for move in game.mainline_moves():
        annotation = ''
        next_node  = node.next()
        if next_node and next_node.comment:
            annotation = next_node.comment.strip()[:500]

        board.push(move)
        ply += 1

        if _MIN_PLY <= ply <= _MAX_PLY:
            z_hash    = chess.polyglot.zobrist_hash(board)
            pawn_hash = get_pawn_hash(board)
            fen       = board.fen()

            if z_hash in seen:
                # Reuse cached result — skip extractors and Stockfish
                strategy_tag, phase, eval_cp = seen[z_hash]
            else:
                phase   = get_phase(board)
                eval_cp = None
                if bridge:
                    try:
                        ev = bridge.get_eval(fen)
                        eval_cp = ev.centipawns
                    except Exception:
                        pass
                strategy_tag = _classify(board, phase)
                seen[z_hash] = (strategy_tag, phase, eval_cp)

            rows.append((
                game_id, ply, fen, pawn_hash, strategy_tag, phase,
                eval_cp, white, black, rating_w, rating_b,
                result, move.uci(), annotation,
            ))

        elif ply > _MAX_PLY:
            break

        node = next_node or node

    return rows


def _classify(board: chess.Board, phase: str) -> str:
    """Classify a position using the live detector stack.

    Scores from both sides and uses the dominant perspective so that
    positions where Black is the active player are not misclassified
    as 'fortress' simply because White's score is low.
    """
    try:
        sigs: list[MetricSignal] = []
        sigs.extend(extract_king_safety(board, phase))
        sigs.extend(extract_space_control(board, phase=phase))
        sigs.extend(extract_piece_mobility(board, phase=phase))
        sigs.extend(extract_pawn_structure(board, phase))
        sigs.extend(extract_material_balance(board, phase=phase))
        sigs.extend(extract_tactics(board, phase))

        w_scores = {
            'blitz':    score_blitz(sigs,    'white'),
            'flank':    score_flank(sigs,    'white'),
            'fortress': score_fortress(sigs, 'white'),
            'feint':    score_feint(sigs,    'white'),
        }
        b_scores = {
            'blitz':    score_blitz(sigs,    'black'),
            'flank':    score_flank(sigs,    'black'),
            'fortress': score_fortress(sigs, 'black'),
            'feint':    score_feint(sigs,    'black'),
        }
        # Use whichever side shows the stronger pattern
        scores = b_scores if max(b_scores.values()) > max(w_scores.values()) else w_scores
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] >= _STRATEGY_THRESHOLD else 'general'
    except Exception:
        return 'general'


def _flush(conn: sqlite3.Connection, batch: list[tuple]) -> None:
    conn.executemany(
        """INSERT INTO coach_positions
           (game_id, ply, fen, pawn_hash, strategy_tag, phase,
            eval_cp, player_white, player_black, rating_white,
            rating_black, result, key_move, annotation)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        batch,
    )
    conn.commit()


def _start_engine(stockfish_path: str, movetime_ms: int, verbose: bool):
    if not stockfish_path:
        return None
    try:
        from core.stockfish_bridge import StockfishBridge
        bridge = StockfishBridge(stockfish_path, movetime_ms)
        bridge.start()
        if not bridge.is_running:
            if verbose: print("WARNING: Stockfish failed to start — eval_cp will be NULL")
            return None
        return bridge
    except Exception as e:
        if verbose: print(f"WARNING: Stockfish unavailable ({e})")
        return None


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build coach_positions table inside an existing index.sqlite.'
    )
    parser.add_argument('--db',         required=True, help='Path to existing index.sqlite')
    parser.add_argument('--stockfish',  default='',    help='Path to Stockfish (optional)')
    parser.add_argument('--movetime',   type=int, default=5)
    parser.add_argument('--min-rating', type=int, default=0)
    parser.add_argument('--quiet',      action='store_true')
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: DB not found: {args.db}")
        sys.exit(1)

    build_from_existing_index(
        db_path        = args.db,
        stockfish_path = args.stockfish,
        movetime_ms    = args.movetime,
        min_rating     = args.min_rating,
        verbose        = not args.quiet,
    )



if __name__ == '__main__':
    main()
