#!/usr/bin/env python3
"""
scrape_lichess_puzzles.py  —  Extract themed puzzles from the Lichess puzzle CSV
----------------------------------------------------------------------------------
Streams through data/lichess_db_puzzle.csv (download from database.lichess.org/#puzzles)
and writes PGN files organised by concept folder.

Output
------
    data/puzzles/
        fork/
            batch_0001.pgn   (20 puzzles per file)
            batch_0002.pgn
        back_rank/
            ...
        _index.jsonl         (counts per theme — allows resume)

Usage
-----
    # Extract all concept themes (default, skips meta-tags like short/long)
    python assets/webscraping/scrape_lichess_puzzles.py

    # Specific themes only (Lichess camelCase slugs)
    python assets/webscraping/scrape_lichess_puzzles.py --themes fork,pin,backRankMate

    # Cap per theme
    python assets/webscraping/scrape_lichess_puzzles.py --max-per-theme 2000

    # Check how many puzzles per theme exist before extracting
    python assets/webscraping/scrape_lichess_puzzles.py --count-only

Dependencies
------------
    python -m pip install chess
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import chess
    import chess.pgn
except ImportError:
    sys.exit("Run:  python -m pip install chess")

CSV_PATH       = Path("data/lichess_db_puzzle.csv")
OUTPUT_ROOT    = Path("data/puzzles")
INDEX_PATH     = OUTPUT_ROOT / "_index.jsonl"
PUZZLES_PER_FILE = 20

# Lichess camelCase slug → (output folder name, concept_vocab label or None)
# folder name = the label used by ingest_puzzle_pgns.py
# None = meta / difficulty tag, no chess concept → skipped by default
THEMES: dict[str, tuple[str, str | None]] = {

    # ── tactical motifs ───────────────────────────────────────────────────────
    "advancedPawn":       ("passed_pawn",        "passed_pawn"),
    "attackingF2F7":      ("mating_attack",       "mating_attack"),
    "attraction":         ("decoy",               "decoy"),
    "backRankMate":       ("back_rank",           "back_rank"),
    "capturingDefender":  ("deflection",          "deflection"),
    "clearance":          ("clearance",           "clearance"),
    "crushing":           ("combination",         "combination"),
    "defensiveMove":      ("prophylaxis",         "prophylaxis"),
    "deflection":         ("deflection",          "deflection"),
    "discoveredAttack":   ("discovered_attack",   "discovered_attack"),
    "doubleCheck":        ("discovered_attack",   "discovered_attack"),
    "fork":               ("fork",                "fork"),
    "hangingPiece":       ("trapped_piece",       "trapped_piece"),
    "interference":       ("interference",        "interference"),
    "pin":                ("pin",                 "pin"),
    "promotion":          ("passed_pawn",         "passed_pawn"),
    "sacrifice":          ("sacrifice",           "sacrifice"),
    "skewer":             ("skewer",              "skewer"),
    "trappedPiece":       ("trapped_piece",       "trapped_piece"),
    "underPromotion":     ("passed_pawn",         "passed_pawn"),
    "xRayAttack":         ("discovered_attack",   "discovered_attack"),
    "zugzwang":           ("zugzwang",            "zugzwang"),

    # ── mating patterns ───────────────────────────────────────────────────────
    "arabianMate":        ("mating_attack",       "mating_attack"),
    "anastasiaMate":      ("mating_attack",       "mating_attack"),
    "balestraMate":       ("mating_attack",       "mating_attack"),
    "blindSwineMate":     ("mating_attack",       "mating_attack"),
    "bodenMate":          ("mating_attack",       "mating_attack"),
    "cornerMate":         ("mating_attack",       "mating_attack"),
    "doubleBishopMate":   ("mating_attack",       "mating_attack"),
    "dovetailMate":       ("mating_attack",       "mating_attack"),
    "epauletteMate":      ("mating_attack",       "mating_attack"),
    "hookMate":           ("mating_attack",       "mating_attack"),
    "killBoxMate":        ("mating_attack",       "mating_attack"),
    "mate":               ("mating_attack",       "mating_attack"),
    "mateIn1":            ("mating_attack",       "mating_attack"),
    "mateIn2":            ("mating_attack",       "mating_attack"),
    "mateIn3":            ("mating_attack",       "mating_attack"),
    "mateIn4":            ("mating_attack",       "mating_attack"),
    "mateIn5":            ("mating_attack",       "mating_attack"),
    "morphysMate":        ("mating_attack",       "mating_attack"),
    "operaMate":          ("mating_attack",       "mating_attack"),
    "pillsburysMate":     ("mating_attack",       "mating_attack"),
    "smotheredMate":      ("mating_attack",       "mating_attack"),
    "swallowstailMate":   ("mating_attack",       "mating_attack"),
    "triangleMate":       ("mating_attack",       "mating_attack"),
    "vukovicMate":        ("mating_attack",       "mating_attack"),

    # ── endgame types — split across concepts rather than all → endgame_technique ─
    # endgame alone has 3M+ puzzles so endgame_technique still hits 5000 cap easily
    "endgame":            ("endgame_technique",   "endgame_technique"),
    "knightEndgame":      ("endgame_technique",   "endgame_technique"),
    "queenEndgame":       ("endgame_technique",   "endgame_technique"),
    "queenRookEndgame":   ("endgame_technique",   "endgame_technique"),
    "pawnEndgame":        ("opposition",          "opposition"),          # pawn endings = king opposition
    "rookEndgame":        ("rook_seventh",        "rook_seventh"),        # rook endings = 7th rank / open file
    "bishopEndgame":      ("bad_bishop",          "bad_bishop"),          # bishop endings = good vs bad bishop

    # ── positional / strategic ────────────────────────────────────────────────
    "castling":           ("development_lead",    "development_lead"),
    "exposedKing":        ("king_safety",         "king_safety"),
    "opening":            ("development_lead",    "development_lead"),
    "kingsideAttack":     ("pawn_storm",          "pawn_storm"),          # kingside attack often = pawn storm
    "queensideAttack":    ("minority_attack",     "minority_attack"),     # queenside attack includes minority attack
    "equality":           ("fortification",       "fortification"),       # achieving equality = fortress defense
    "advantage":          ("piece_activity",      "piece_activity"),      # advantage comes from active pieces
    "veryLong":           ("king_activity",       "king_activity"),       # long endgames feature king marches

    # ── themes added in pass 2 ───────────────────────────────────────────────
    "collinearMove":      ("battery",             "battery"),
    "discoveredCheck":    ("discovered_attack",   "discovered_attack"),
    "enPassant":          ("pawn_break",          "pawn_break"),
    "intermezzo":         ("zwischenzug",         "zwischenzug"),
    "quietMove":          ("tempo",               "tempo"),
    # note: "overloading" does not exist as a theme in the Lichess CSV — removed

    # ── meta / difficulty — no concept label, skipped unless explicitly named ─
    "long":               ("long",                None),
    "master":             ("master",              None),
    "masterVsMaster":     ("master_vs_master",    None),
    "middlegame":         ("middlegame",           None),
    "oneMove":            ("one_move",             None),
    "short":              ("short",               None),
    "superGM":            ("super_gm",            None),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def row_to_pgn(row: dict, primary_slug: str, label: str | None) -> str | None:
    """Convert one CSV row to a PGN string. Returns None on parse failure."""
    try:
        board = chess.Board(row["FEN"])
    except Exception:
        return None

    game = chess.pgn.Game()
    game.headers["Event"]   = f"Lichess Puzzle {row['PuzzleId']}"
    game.headers["Site"]    = row.get("GameUrl", "")
    game.headers["White"]   = "Puzzle"
    game.headers["Black"]   = "Puzzle"
    game.headers["Result"]  = "*"
    game.headers["Rating"]  = row.get("Rating", "?")
    game.headers["Themes"]  = row.get("Themes", primary_slug)
    if label:
        game.headers["ConceptLabel"] = label
    game.setup(board)

    node = game
    for uci in row.get("Moves", "").split():
        try:
            move = chess.Move.from_uci(uci)
            if move not in node.board().legal_moves:
                break
            node = node.add_variation(move)
        except Exception:
            break

    return str(game)


def write_batch(pgn_list: list[str], folder: Path, batch_num: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"batch_{batch_num:04d}.pgn").write_text(
        "\n\n".join(pgn_list), encoding="utf-8"
    )


def load_index() -> dict[str, int]:
    """Returns {folder_name: puzzles_already_written}."""
    counts: dict[str, int] = defaultdict(int)
    if INDEX_PATH.exists():
        for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                counts[rec["folder"]] = rec.get("total", 0)
            except Exception:
                pass
    return dict(counts)


def save_index(counts: dict[str, int]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for folder, total in counts.items():
            f.write(json.dumps({"folder": folder, "total": total}) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if not CSV_PATH.exists():
        sys.exit(
            f"CSV not found at {CSV_PATH}\n"
            "Download from: https://database.lichess.org/#puzzles\n"
            "Then place the unzipped .csv at data/lichess_db_puzzle.csv"
        )

    # Build the set of slugs to extract
    if args.themes:
        requested = [s.strip() for s in args.themes.split(",")]
        unknown   = [s for s in requested if s not in THEMES]
        if unknown:
            print(f"WARNING — unknown slugs (will skip): {unknown}")
        active = {s: THEMES[s] for s in requested if s in THEMES}
    else:
        # Default: only slugs that map to a real concept label
        active = {s: v for s, v in THEMES.items() if v[1] is not None}

    if not active:
        sys.exit("No valid themes selected.")

    print(f"Themes to extract: {len(active)}")
    for slug, (folder, label) in sorted(active.items()):
        print(f"  {slug:<22} -> {folder}/  (label={label})")

    if args.count_only:
        print(f"\nScanning {CSV_PATH} for theme counts …")
        counts: dict[str, int] = defaultdict(int)
        t0 = time.time()
        with open(CSV_PATH, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                puzzle_themes = set(row["Themes"].split())
                for slug in active:
                    if slug in puzzle_themes:
                        counts[slug] += 1
                if i % 200_000 == 0:
                    print(f"  {i:,} rows scanned ...", flush=True)
        print(f"\nDone ({time.time()-t0:.0f}s).\n")
        print(f"{'Slug':<24} {'Count':>8}")
        print("-" * 34)
        for slug, (folder, _) in sorted(active.items()):
            print(f"{slug:<24} {counts.get(slug, 0):>8,}")
        return

    # Resume: track how many already written per folder
    done_per_folder: dict[str, int] = load_index()

    # Per-folder buffers and batch counters
    buffers:    dict[str, list[str]] = defaultdict(list)
    batch_nums: dict[str, int]       = {}
    written:    dict[str, int]       = dict(done_per_folder)

    for slug, (folder, _) in active.items():
        if folder not in batch_nums:
            batch_nums[folder] = (done_per_folder.get(folder, 0) // PUZZLES_PER_FILE) + 1

    max_pt = args.max_per_theme

    print(f"\nStreaming {CSV_PATH} ({CSV_PATH.stat().st_size / 1e9:.2f} GB) …\n")
    t0         = time.time()
    rows_read  = 0
    total_kept = 0

    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows_read += 1
            puzzle_themes = set(row["Themes"].split())

            for slug, (folder, label) in active.items():
                if slug not in puzzle_themes:
                    continue
                if written.get(folder, 0) >= max_pt:
                    continue

                pgn = row_to_pgn(row, slug, label)
                if pgn is None:
                    continue

                buffers[folder].append(pgn)
                written[folder] = written.get(folder, 0) + 1
                total_kept += 1

                if len(buffers[folder]) >= PUZZLES_PER_FILE:
                    write_batch(buffers[folder], OUTPUT_ROOT / folder, batch_nums[folder])
                    batch_nums[folder] += 1
                    buffers[folder] = []

            # Progress every 100k rows
            if rows_read % 100_000 == 0:
                elapsed = time.time() - t0
                rate    = rows_read / elapsed
                pct     = 100 * rows_read / 4_300_000  # approx total rows
                print(f"  {rows_read:>7,} rows  {total_kept:>7,} kept  "
                      f"{rate:>7,.0f} rows/s  ~{pct:.0f}%", flush=True)

            # Stop early if all themes are at max
            if all(written.get(THEMES[s][0], 0) >= max_pt for s in active):
                print("  All themes reached max - stopping early.")
                break

    # Flush remaining buffers
    for folder, pgn_list in buffers.items():
        if pgn_list:
            write_batch(pgn_list, OUTPUT_ROOT / folder, batch_nums[folder])

    # Save updated index
    save_index(written)

    elapsed = time.time() - t0
    print(f"\n-- Done ({elapsed:.0f}s, {rows_read:,} rows scanned) "
          f"-----------------------------")
    print(f"\n{'Folder':<24} {'Written':>8}")
    print("-" * 34)
    for slug, (folder, _) in sorted(active.items()):
        n = written.get(folder, 0)
        if n:
            print(f"{folder:<24} {n:>8,}")

    print(f"\nTotal: {total_kept:,} puzzles written to {OUTPUT_ROOT}/")
    print(f"\nNext steps:")
    print(f"  python tools/ingest_puzzle_pgns.py "
          f"--input data/puzzles --output data/training_raw.jsonl --append")
    print(f"  python -m src.chess_coach.ml.train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Lichess puzzles by theme from the downloaded CSV"
    )
    parser.add_argument(
        "--themes", default="",
        help="Comma-separated Lichess theme slugs (default: all concept themes). "
             "E.g. fork,pin,backRankMate,zugzwang"
    )
    parser.add_argument(
        "--max-per-theme", type=int, default=5000,
        help="Max puzzles to write per theme folder (default 5000)."
    )
    parser.add_argument(
        "--count-only", action="store_true",
        help="Scan CSV and print how many puzzles exist per theme — no files written."
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
