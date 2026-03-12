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
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.pgn

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.board_utils    import get_phase, get_pawn_hash
from core.data_types     import MetricSignal
from extractors.king_safety      import extract_king_safety
from extractors.space_control    import extract_space_control
from extractors.piece_mobility   import extract_piece_mobility
from extractors.pawn_structure   import extract_pawn_structure
from extractors.material_balance import extract_material_balance
from extractors.tactic_scanner   import extract_tactics
from strategies.blitz_detector    import score_blitz
from strategies.flank_detector    import score_flank
from strategies.fortress_detector import score_fortress
from strategies.feint_detector    import score_feint

_MIN_PLY            = 10
_MAX_PLY            = 60
_STRATEGY_THRESHOLD = 0.40
_BATCH_SIZE         = 50   # commit every N games

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
    movetime_ms: int = 500,
    min_rating: int = 0,
    verbose: bool = False,
) -> bool:
    """
    Build coach_positions if it doesn't exist or is empty.
    Called automatically by StrategyEngine on first use.
    Returns True if indexing was performed, False if already populated.
    """
    if not Path(db_path).exists():
        return False

    conn = sqlite3.connect(db_path)
    conn.executescript(_CREATE_TABLE)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM coach_positions").fetchone()[0]
    conn.close()

    if count > 0:
        return False   # already indexed

    build_from_existing_index(
        db_path        = db_path,
        stockfish_path = stockfish_path,
        movetime_ms    = movetime_ms,
        min_rating     = min_rating,
        verbose        = verbose,
    )
    return True


def build_from_existing_index(
    db_path: str,
    stockfish_path: str = '',
    movetime_ms: int    = 500,
    min_rating: int     = 0,
    verbose: bool       = True,
) -> dict:
    """
    Populate coach_positions by reading games already catalogued in index.sqlite.
    Uses pgn_path + offset_bytes to seek directly — no re-parsing of the full file.

    Returns summary: {games_processed, positions_indexed, skipped}.
    """
    bridge = _start_engine(stockfish_path, movetime_ms, verbose)

    conn = sqlite3.connect(db_path)
    conn.executescript(_CREATE_TABLE)
    conn.commit()

    # Clear any partial previous run
    conn.execute("DELETE FROM coach_positions")
    conn.commit()

    # Fetch all games — existing schema has white/black/result but not ELO
    rows = conn.execute("""
        SELECT game_id, pgn_path, offset_bytes, white, black, result
        FROM games
        ORDER BY game_id ASC
    """).fetchall()

    stats = {'games_processed': 0, 'positions_indexed': 0, 'skipped': 0}
    batch: list[tuple] = []

    for row in rows:
        game_id, pgn_path, offset_bytes, white, black, result = row
        rating_w = rating_b = 0   # existing indexer doesn't store ELO

        if min_rating > 0 and rating_w < min_rating and rating_b < min_rating:
            stats['skipped'] += 1
            continue

        try:
            game = _load_game_at_offset(pgn_path, offset_bytes)
        except Exception:
            stats['skipped'] += 1
            continue

        if game is None:
            stats['skipped'] += 1
            continue

        # Override headers from DB values (more reliable than PGN headers)
        positions = _walk_game(
            game, game_id, white or '', black or '',
            rating_w, rating_b, result or '',
            bridge,
        )

        batch.extend(positions)
        stats['games_processed'] += 1
        stats['positions_indexed'] += len(positions)

        if len(batch) >= _BATCH_SIZE * 10:
            _flush(conn, batch)
            batch.clear()

        if verbose and stats['games_processed'] % 100 == 0:
            print(f"  {stats['games_processed']} games, "
                  f"{stats['positions_indexed']} positions indexed...")

    if batch:
        _flush(conn, batch)

    conn.close()
    if bridge:
        try: bridge.stop()
        except Exception: pass

    if verbose:
        print(f"\nDone. {stats['games_processed']} games processed, "
              f"{stats['positions_indexed']} positions indexed, "
              f"{stats['skipped']} skipped.")
    return stats


# ── Internal helpers ────────────────────────────────────────────────────────

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
) -> list[tuple]:
    """Walk mainline positions ply 10–60 and return DB row tuples."""
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
            fen       = board.fen()
            pawn_hash = get_pawn_hash(board)
            phase     = get_phase(board)

            eval_cp: Optional[int] = None
            if bridge:
                try:
                    ev = bridge.get_eval(fen)
                    eval_cp = ev.centipawns
                except Exception:
                    pass

            strategy_tag = _classify(board, phase)

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
    """Classify a position using the live detector stack."""
    try:
        sigs: list[MetricSignal] = []
        sigs.extend(extract_king_safety(board, phase))
        sigs.extend(extract_space_control(board, phase=phase))
        sigs.extend(extract_piece_mobility(board, phase=phase))
        sigs.extend(extract_pawn_structure(board, phase))
        sigs.extend(extract_material_balance(board, phase=phase))
        sigs.extend(extract_tactics(board, phase))

        scores = {
            'blitz':    score_blitz(sigs,    'white'),
            'flank':    score_flank(sigs,    'white'),
            'fortress': score_fortress(sigs, 'white'),
            'feint':    score_feint(sigs,    'white'),
        }
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
    parser.add_argument('--movetime',   type=int, default=500)
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
