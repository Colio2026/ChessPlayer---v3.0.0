#!/usr/bin/env python3
"""
ingest_lichess_csv.py  —  Convert the Lichess puzzle CSV into training data
---------------------------------------------------------------------------
Reads lichess_db_puzzle.csv (or its .zst compressed form), maps Lichess
theme tags to our concept vocab, and ALSO runs the algorithmic position
detectors from label_positions.py.  This gives every detectable structural
concept (passed pawn, pin, fork, bad bishop, …) for free on every puzzle.

Expected CSV columns (Lichess format):
  PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays,
  Themes, GameUrl, OpeningTags

The FEN is the position BEFORE the opponent's trigger move.  We apply
Moves[0] to reach the actual puzzle position, then record Moves[1] as
the move_uci context feature.

Usage
-----
    # Full run — append to training_raw.jsonl (recommended)
    python tools/ingest_lichess_csv.py --input data/lichess_db_puzzle.csv --append

    # Dry run — show label distribution for first 50 000 puzzles
    python tools/ingest_lichess_csv.py --input data/lichess_db_puzzle.csv --count-only --limit 50000

    # Rating filter — only well-tested puzzles
    python tools/ingest_lichess_csv.py --input data/lichess_db_puzzle.csv --min-rating 1200 --max-rating 2400 --append
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

try:
    import chess
    import chess.pgn
except ImportError:
    sys.exit("chess package missing — run: pip install chess")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chess_coach.ml.concept_vocab import CONCEPT_TO_IDX
from tools.label_positions import label_position, DETECTABLE_CONCEPTS

# ── Lichess theme tag → our concept label ─────────────────────────────────────
LICHESS_TAG_MAP: dict[str, str] = {
    # Core tactics
    "pin":                "pin",
    "fork":               "fork",
    "skewer":             "skewer",
    "discoveredAttack":   "discovered_attack",
    "doubleCheck":        "discovered_attack",
    "xRayAttack":         "skewer",
    "deflection":         "deflection",
    "attraction":         "decoy",
    "decoy":              "decoy",
    "interference":       "interference",
    "clearance":          "clearance",
    "overloading":        "overloading",
    "capturingDefender":  "overloading",
    "sacrifice":          "sacrifice",
    "trappedPiece":       "trapped_piece",
    "intermezzo":         "zwischenzug",
    "zwischenzug":        "zwischenzug",
    "zugzwang":           "zugzwang",
    # Mating patterns
    "backRankMate":       "back_rank",
    "mate":               "mating_attack",
    "mateIn1":            "mating_attack",
    "mateIn2":            "mating_attack",
    "mateIn3":            "mating_attack",
    "mateIn4":            "mating_attack",
    "mateIn5":            "mating_attack",
    "smotheredMate":      "mating_attack",
    "anastasiaMate":      "mating_attack",
    "arabianMate":        "mating_attack",
    "hookMate":           "mating_attack",
    "bodenMate":          "mating_attack",
    "doubleBishopMate":   "mating_attack",
    "vukovicMate":        "mating_attack",
    "epauletteMate":      "mating_attack",
    # Strategic / dynamic
    "equality":           "counterplay",   # defensive resource achieving balance = counterplay
    "defensiveMove":      "prophylaxis",
    "exposedKing":        "king_safety",
    "kingsideAttack":     "attacking_chances",
    "queensideAttack":    "attacking_chances",
    # Endgame
    "rookEndgame":        "endgame_technique",
    "pawnEndgame":        "endgame_technique",
    "knightEndgame":      "endgame_technique",
    "bishopEndgame":      "endgame_technique",
    "queenEndgame":       "endgame_technique",
    "queenRookEndgame":   "endgame_technique",
    # Pawn
    "advancedPawn":       "passed_pawn",
    "promotionDefense":   "passed_pawn",
}

# Only keep labels that exist in our concept vocab
_VALID = set(CONCEPT_TO_IDX.keys())


def _map_lichess_themes(theme_str: str) -> set[str]:
    labels: set[str] = set()
    for tag in theme_str.split():
        mapped = LICHESS_TAG_MAP.get(tag)
        if mapped and mapped in _VALID:
            labels.add(mapped)
    return labels


def _is_exchange_sacrifice(board: chess.Board, move_uci: str) -> bool:
    """The move gives a rook for a minor piece."""
    try:
        move     = chess.Move.from_uci(move_uci)
        moving   = board.piece_at(move.from_square)
        captured = board.piece_at(move.to_square)
        if not moving or not captured:
            return False
        return (moving.piece_type == chess.ROOK
                and captured.piece_type in {chess.BISHOP, chess.KNIGHT}
                and captured.color != moving.color)
    except Exception:
        return False


def _process_row(row: dict) -> dict | None:
    """Parse one CSV row into a JSONL-ready dict.  Returns None to skip."""
    try:
        fen   = row["FEN"]
        moves = row["Moves"].split()
        if len(moves) < 2:
            return None

        board = chess.Board(fen)
        trigger = chess.Move.from_uci(moves[0])
        if trigger not in board.legal_moves:
            return None
        board.push(trigger)

        puzzle_fen = board.fen()
        move_uci   = moves[1]           # first solver move = key tactic move

        # Labels from Lichess tags
        themes = _map_lichess_themes(row["Themes"])

        # Labels from algorithmic detectors — fills in structural concepts
        # that Lichess doesn't tag (passed pawn, bad bishop, pawn structure, …)
        algo_labels = label_position(board)
        themes.update(algo_labels & _VALID)

        # Exchange sacrifice: rook captures bishop/knight
        if (_is_exchange_sacrifice(board, move_uci)
                and "exchange_sacrifice" in _VALID):
            themes.add("exchange_sacrifice")

        if not themes:
            return None

        return {
            "fen":      puzzle_fen,
            "move_uci": move_uci,
            "themes":   sorted(themes),
            "comment":  "",
            "phase":    "puzzle",
        }
    except Exception:
        return None


def ingest(args: argparse.Namespace) -> None:
    csv_path = Path(args.input)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    out_path = Path(args.output)
    mode     = "a" if args.append else "w"
    if not args.append and out_path.exists():
        ans = input(f"{out_path} already exists. Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit("Aborted.")

    # Read CSV — handle very large file with streaming
    cap           = args.max_per_concept if args.max_per_concept > 0 else None
    concept_counts: Counter = Counter()   # tracks written examples per concept for cap
    print(f"Reading {csv_path} ..."
          + (f"  (cap: {cap:,} per concept)" if cap else ""))
    t0            = time.time()
    total_read    = 0
    total_written = 0
    skipped       = 0
    label_counts: Counter = Counter()

    open_fn = open
    if csv_path.suffix == ".zst":
        try:
            import zstandard
            open_fn = lambda p, **kw: zstandard.open(p, "rt", **kw)  # noqa
        except ImportError:
            sys.exit("zstandard package needed for .zst files — pip install zstandard")

    with open_fn(csv_path, encoding="utf-8", errors="replace") as fh, \
            open(out_path, mode, encoding="utf-8") as out_fh:

        reader = csv.DictReader(fh)
        rows_buffer: list[dict] = []

        for row in reader:
            # Rating filter
            try:
                rating = int(row.get("Rating", "0"))
                if rating < args.min_rating or rating > args.max_rating:
                    continue
            except ValueError:
                pass

            # Popularity filter (NbPlays) — skip brand-new puzzles with no data
            try:
                if int(row.get("NbPlays", "0")) < args.min_plays:
                    continue
            except ValueError:
                pass

            total_read += 1
            if args.limit and total_read > args.limit:
                break

            rows_buffer.append(dict(row))

            if len(rows_buffer) >= args.batch_size:
                results = _flush_batch(rows_buffer, args, out_fh,
                                       label_counts, concept_counts, cap)
                total_written += results[0]
                skipped       += results[1]
                rows_buffer    = []

                elapsed = time.time() - t0
                rate    = total_read / elapsed
                print(f"\r  {total_read:>8,} read  {total_written:>8,} written  "
                      f"{rate:>6,.0f}/s", end="", flush=True)

        # flush remainder
        if rows_buffer:
            results = _flush_batch(rows_buffer, args, out_fh,
                                   label_counts, concept_counts, cap)
            total_written += results[0]
            skipped       += results[1]

    elapsed = time.time() - t0
    print(f"\r  {total_read:>8,} read  {total_written:>8,} written  "
          f"({elapsed:.1f}s)                        ")

    print(f"\nLabel distribution (all concepts):")
    for label, n in label_counts.most_common():
        print(f"  {label:<25} {n:>8,}")
    print(f"\nSkipped (parse errors / no labels): {skipped:,}")
    print(f"Done — {total_written:,} examples written to {out_path}")


def _flush_batch(
    rows: list[dict],
    args: argparse.Namespace,
    out_fh,
    label_counts: Counter,
    concept_counts: Counter,
    cap: int | None,
) -> tuple[int, int]:
    written = skipped = 0

    if args.count_only:
        # In count-only mode ignore cap — show raw distribution
        for row in rows:
            ex = _process_row(row)
            if ex:
                label_counts.update(ex["themes"])
                written += 1
            else:
                skipped += 1
        return written, skipped

    # Use multiprocessing for CPU-heavy algorithmic detection
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_process_row, rows)
    else:
        results = [_process_row(r) for r in rows]

    for ex in results:
        if not ex:
            skipped += 1
            continue

        if cap:
            # Keep only labels that are still under the cap
            allowed = [t for t in ex["themes"] if concept_counts[t] < cap]
            if not allowed:
                skipped += 1
                continue
            ex["themes"] = allowed

        label_counts.update(ex["themes"])
        concept_counts.update(ex["themes"])
        out_fh.write(json.dumps(ex) + "\n")
        written += 1

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Lichess puzzle CSV into JSONL training data"
    )
    parser.add_argument("--input",       required=True,
                        help="Path to lichess_db_puzzle.csv (or .csv.zst)")
    parser.add_argument("--output",      default="data/training_raw.jsonl")
    parser.add_argument("--append",      action="store_true",
                        help="Append to existing output file (do not overwrite)")
    parser.add_argument("--count-only",  action="store_true",
                        help="Print label distribution without writing output")
    parser.add_argument("--limit",       type=int, default=0,
                        help="Stop after this many rows (0 = no limit)")
    parser.add_argument("--min-rating",  type=int, default=800)
    parser.add_argument("--max-rating",  type=int, default=2800)
    parser.add_argument("--min-plays",   type=int, default=100,
                        help="Skip puzzles with fewer than this many plays")
    parser.add_argument("--batch-size",  type=int, default=2000,
                        help="Rows per processing batch")
    parser.add_argument("--max-per-concept", type=int, default=50_000,
                        help="Cap examples per concept label (0 = no cap). Prevents "
                             "common structural concepts from dominating the dataset.")
    parser.add_argument("--workers",     type=int,
                        default=max(1, cpu_count() - 1),
                        help="Worker processes for algorithmic detection (default: all CPUs - 1)")
    args = parser.parse_args()
    ingest(args)


if __name__ == "__main__":
    main()
