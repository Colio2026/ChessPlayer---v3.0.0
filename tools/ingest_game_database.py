#!/usr/bin/env python3
"""
ingest_game_database.py  —  Extract training positions from master-game PGN databases
--------------------------------------------------------------------------------------
Streams through large PGN files (Caissabase, Carlsen, Lichess Elite, etc.),
samples positions from each game, runs the algorithmic concept detectors from
label_positions.py, and writes JSONL training examples.

This is the primary source for structural / strategic concepts that the Lichess
puzzle CSV doesn't tag: battery, blockade, pawn_storm, king_activity,
space_advantage, minority_attack, color_complex, development_lead,
piece_activity, square_control, and all pawn-structure concepts.

The next move actually played in the game is recorded as move_uci — this gives
the model a move feature from the same position a master was thinking about.

Usage
-----
    # Scan Caissabase (recommended: cap at 50k per concept)
    python tools/ingest_game_database.py --input data/Caissabase.pgn --append

    # Multiple files
    python tools/ingest_game_database.py --input data/Carlsen.pgn data/lichess_elite_2020-10.pgn --append

    # Dry run — show label distribution without writing
    python tools/ingest_game_database.py --input data/Caissabase.pgn --count-only --limit 5000

    # Only target concepts that are still under-represented
    python tools/ingest_game_database.py --input data/Caissabase.pgn --target battery,blockade,minority_attack --append
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
    sys.exit("chess package missing — run: pip install chess")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chess_coach.ml.concept_vocab import CONCEPT_TO_IDX
from tools.label_positions import label_position, DETECTABLE_CONCEPTS

_VALID = set(CONCEPT_TO_IDX.keys()) & DETECTABLE_CONCEPTS


def _get_phase(board: chess.Board) -> str:
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


def _is_exchange_sacrifice(board: chess.Board, move: chess.Move) -> bool:
    """The move gives a rook for a minor piece."""
    moving   = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    if not moving or not captured:
        return False
    return (moving.piece_type == chess.ROOK
            and captured.piece_type in {chess.BISHOP, chess.KNIGHT}
            and captured.color != moving.color)


def _process_game(
    game:        chess.pgn.Game,
    sample_rate: int,
    target:      set[str] | None,
) -> list[dict]:
    """Extract sampled positions from one game. Returns list of example dicts."""
    examples: list[dict] = []
    board = game.board()
    moves = list(game.mainline_moves())

    for i, move in enumerate(moves):
        # Only sample every Nth position — skip the very early opening theory
        if i < 4 or i % sample_rate != 0:
            board.push(move)
            continue

        labels = label_position(board) & _VALID

        # Exchange sacrifice: detect from the move about to be played
        if _is_exchange_sacrifice(board, move) and "exchange_sacrifice" in CONCEPT_TO_IDX:
            labels = labels | {"exchange_sacrifice"}

        if target:
            labels = labels & target

        if labels:
            examples.append({
                "fen":      board.fen(),
                "move_uci": move.uci(),
                "themes":   sorted(labels),
                "comment":  "",
                "phase":    _get_phase(board),
            })

        board.push(move)

    return examples


def ingest(args: argparse.Namespace) -> None:
    inputs = [Path(p) for p in args.input]
    for p in inputs:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    target: set[str] | None = None
    if args.target:
        target = set(t.strip() for t in args.target.split(",")) & _VALID
        if not target:
            sys.exit("--target contained no valid detectable concepts")
        print(f"Targeting: {sorted(target)}")

    out_path = Path(args.output)
    mode     = "a" if args.append else "w"
    if not args.append and not args.count_only and out_path.exists():
        ans = input(f"{out_path} exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit("Aborted.")

    cap           = args.max_per_concept if args.max_per_concept > 0 else None
    concept_counts: Counter = Counter()
    label_counts:   Counter = Counter()
    total_games    = 0
    total_examples = 0
    games_skipped  = 0
    t0             = time.time()

    open_fh = open(out_path, mode, encoding="utf-8") if not args.count_only else None

    try:
        for pgn_path in inputs:
            print(f"\nStreaming {pgn_path} ({pgn_path.stat().st_size / 1e9:.2f} GB) …")
            with open(pgn_path, encoding="utf-8", errors="replace") as fh:
                while True:
                    if args.limit and total_games >= args.limit:
                        break

                    try:
                        game = chess.pgn.read_game(fh)
                    except Exception:
                        games_skipped += 1
                        continue
                    if game is None:
                        break

                    total_games += 1

                    try:
                        examples = _process_game(game, args.sample_rate, target)
                    except Exception:
                        games_skipped += 1
                        continue

                    for ex in examples:
                        themes = ex["themes"]

                        if cap:
                            allowed = [t for t in themes if concept_counts[t] < cap]
                            if not allowed:
                                continue
                            ex["themes"] = allowed
                            themes = allowed

                        label_counts.update(themes)
                        concept_counts.update(themes)
                        total_examples += 1

                        if open_fh:
                            open_fh.write(json.dumps(ex) + "\n")

                    if total_games % 1000 == 0:
                        elapsed = time.time() - t0
                        rate    = total_games / elapsed
                        print(f"\r  {total_games:>7,} games  {total_examples:>8,} examples  "
                              f"{rate:>6,.0f} games/s", end="", flush=True)

                    # Stop early once all targeted concepts are capped
                    if cap and target and all(concept_counts[c] >= cap for c in target):
                        print("\n  All target concepts capped — stopping early.")
                        break
    finally:
        if open_fh:
            open_fh.close()

    elapsed = time.time() - t0
    print(f"\r  {total_games:>7,} games  {total_examples:>8,} examples  ({elapsed:.1f}s)  "
          "                ")

    print(f"\nLabel distribution:")
    for label, n in label_counts.most_common():
        bar = "#" * min(40, n // max(1, max(label_counts.values()) // 40))
        print(f"  {label:<25} {n:>8,}  {bar}")

    if games_skipped:
        print(f"\n  {games_skipped} games skipped (parse errors).")
    if not args.count_only:
        print(f"\nDone — {total_examples:,} examples written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract algorithmically-labeled training positions from PGN game databases"
    )
    parser.add_argument("--input",           required=True, nargs="+",
                        help="One or more PGN files (Caissabase, Carlsen, Lichess Elite, …)")
    parser.add_argument("--output",          default="data/training_raw.jsonl")
    parser.add_argument("--append",          action="store_true")
    parser.add_argument("--count-only",      action="store_true",
                        help="Print label distribution without writing output")
    parser.add_argument("--limit",           type=int, default=0,
                        help="Stop after N games per file (0 = no limit)")
    parser.add_argument("--sample-rate",     type=int, default=5,
                        help="Sample every Nth move from each game (default 5)")
    parser.add_argument("--max-per-concept", type=int, default=50_000,
                        help="Cap examples per concept label (0 = no cap)")
    parser.add_argument("--target",          default="",
                        help="Comma-separated concept labels to collect. "
                             "Only positions with at least one target label are kept. "
                             "Default: all detectable concepts.")
    args = parser.parse_args()
    ingest(args)


if __name__ == "__main__":
    main()
