#!/usr/bin/env python3
"""
ingest_puzzle_pgns.py  —  Convert theme-organised puzzle PGNs into training data
----------------------------------------------------------------------------------
Reads PGN files from a folder tree where each subfolder name IS the concept label.
No annotation text required — the folder name provides the label.

Expected layout
---------------
    puzzles/
        back_rank/       ← label = "back_rank"
            file1.pgn
            file2.pgn
        outpost/         ← label = "outpost"
            ...
        pin/
            ...

Each game is converted to one JSONL entry per position visited in the game.
The final position of a puzzle solution is the most useful, so by default only
the last position is emitted.  Use --all-positions to emit every ply.

Usage
-----
    # Append puzzle data to existing training file
    python tools/ingest_puzzle_pgns.py --input data/puzzles --output data/training_raw.jsonl --append

    # Dry run — print stats only
    python tools/ingest_puzzle_pgns.py --input data/puzzles --count-only

    # Emit every position in each game (more data, more noise)
    python tools/ingest_puzzle_pgns.py --input data/puzzles --output data/training_raw.jsonl --all-positions --append
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import chess
    import chess.pgn
except ImportError:
    sys.exit("chess package missing — run: python -m pip install chess")

# Ensure project root is on sys.path so src.* imports work when running
# this script directly (python tools/ingest_puzzle_pgns.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chess_coach.ml.concept_vocab import CONCEPT_TO_IDX


def get_phase(board: chess.Board) -> str:
    move_no = board.fullmove_number
    queens  = len(board.pieces(chess.QUEEN, chess.WHITE)) + \
              len(board.pieces(chess.QUEEN, chess.BLACK))
    pieces  = sum(len(board.pieces(pt, c))
                  for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                  for c in (chess.WHITE, chess.BLACK))
    if move_no <= 10:
        return "opening"
    if queens == 0 or pieces <= 12:
        return "endgame"
    return "middlegame"


def board_to_example(board: chess.Board, themes: list[str], move_uci: str = "") -> dict:
    return {
        "fen":      board.fen(),
        "move_uci": move_uci,
        "themes":   themes,
        "phase":    get_phase(board),
        "comment":  "",
    }


def ingest(args: argparse.Namespace) -> None:
    root = Path(args.input)
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    # Discover all subfolders — each becomes a label
    pgn_files: list[tuple[Path, str]] = []  # (path, label)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        label = folder.name.lower().replace("-", "_").replace(" ", "_")
        if label not in CONCEPT_TO_IDX:
            print(f"  WARNING: folder '{folder.name}' → label '{label}' "
                  f"is not in concept_vocab — skipping")
            continue
        for pgn_path in sorted(folder.rglob("*.pgn")):
            pgn_files.append((pgn_path, label))

    if not pgn_files:
        sys.exit("No matching PGN files found under any known-concept subfolder.")

    print(f"Found {len(pgn_files)} PGN files across "
          f"{len({lbl for _, lbl in pgn_files})} concept folders.")

    if args.count_only:
        label_counts: Counter = Counter()
        for pgn_path, label in pgn_files:
            with open(pgn_path, errors="replace") as fh:
                while chess.pgn.read_game(fh):
                    label_counts[label] += 1
        print("\nGame counts per concept:")
        for label, n in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {label:<22} {n:>6}")
        print(f"\nTotal games: {sum(label_counts.values()):,}")
        return

    out_path = Path(args.output)
    mode     = "a" if args.append else "w"
    if not args.append and out_path.exists():
        ans = input(f"{out_path} already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit("Aborted.")

    t0             = time.time()
    total_games    = 0
    total_examples = 0
    label_counts   = Counter()
    skipped_games  = 0

    with open(out_path, mode, encoding="utf-8") as out_fh:
        for pgn_path, label in pgn_files:
            with open(pgn_path, errors="replace") as fh:
                while True:
                    try:
                        game = chess.pgn.read_game(fh)
                    except Exception:
                        skipped_games += 1
                        continue
                    if game is None:
                        break

                    total_games += 1
                    themes = [label]   # folder name is the only label

                    moves = list(game.mainline_moves())
                    if args.all_positions:
                        # Emit every position: FEN before each move + that move as context
                        board = game.board()
                        for move in moves:
                            ex = board_to_example(board, themes, move_uci=move.uci())
                            out_fh.write(json.dumps(ex) + "\n")
                            total_examples += 1
                            board.push(move)
                    else:
                        # Emit the initial puzzle position + first solution move as context.
                        # This is where the tactical motif lives — the position you're
                        # asked to evaluate, with the key move that executes the concept.
                        board    = game.board()
                        move_uci = moves[0].uci() if moves else ""
                        ex = board_to_example(board, themes, move_uci=move_uci)
                        out_fh.write(json.dumps(ex) + "\n")
                        total_examples += 1

                    label_counts[label] += 1

                    if total_games % 200 == 0:
                        elapsed = time.time() - t0
                        rate    = total_games / elapsed
                        print(f"\r  {total_games:,} games  {total_examples:,} examples  "
                              f"{rate:.0f} games/s", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\r  {total_games:,} games  {total_examples:,} examples  "
          f"({elapsed:.1f}s total)                ")

    print("\nExamples written per concept:")
    for label, n in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<22} {n:>6}")

    if skipped_games:
        print(f"\n  {skipped_games} games skipped due to parse errors.")

    print(f"\nDone — {total_examples:,} examples written to {out_path}")
    print("Next: python -m src.chess_coach.ml.train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert theme-folder puzzle PGNs into JSONL training data"
    )
    parser.add_argument("--input",         required=True,
                        help="Root puzzle folder (contains concept subfolders)")
    parser.add_argument("--output",        default="data/training_raw.jsonl")
    parser.add_argument("--append",        action="store_true",
                        help="Append to existing output file instead of overwriting")
    parser.add_argument("--count-only",    action="store_true",
                        help="Count games per concept without writing output")
    parser.add_argument("--all-positions", action="store_true",
                        help="Emit every ply (default: final position only)")
    args = parser.parse_args()
    ingest(args)


if __name__ == "__main__":
    main()
