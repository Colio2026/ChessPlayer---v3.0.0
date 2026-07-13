#!/usr/bin/env python3
"""
remap_concepts.py  —  Apply v2 → v3 concept vocab changes to training_raw.jsonl
---------------------------------------------------------------------------------
Reads the existing JSONL, renames/removes concept labels in-place, and writes
the result back.  Run this INSTEAD of re-ingesting all data sources.

v3 changes applied:
  Rename : exchange_sacrifice → sacrifice
           minority_attack    → pawn_storm
           tempo              → initiative
           square_control     → piece_activity
           discovered_attack  → discovery      (v1 residual)
           decoy              → deflection     (v1 residual)
           counterplay        → attacking_chances (v1 residual)
  Remove : bishop_quality  (split back to bad_bishop/good_bishop; re-detect via algo)
           pawn_weakness    (removed concept)
           color_complex    (removed concept)
           endgame_technique (replaced by specific endgame types; re-detect via algo)
           combination, pawn_break, simplification, fortification, coordination (v1)

Note: bad_bishop, good_bishop, clearance, x_ray, double_check, promotion,
      shouldering, and all specific endgame types must be repopulated by
      re-running parse_annotated_pgn.py and ingest_lichess_csv.py.

Usage
-----
    python tools/remap_concepts.py
    python tools/remap_concepts.py --input data/training_raw.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

REMAP: dict[str, str] = {
    # v1 residuals
    "discovered_attack":  "discovery",
    "decoy":              "deflection",
    "counterplay":        "attacking_chances",
    # v2 → v3
    "exchange_sacrifice": "sacrifice",
    "minority_attack":    "pawn_storm",
    "tempo":              "initiative",
    "square_control":     "piece_activity",
}

REMOVE: frozenset[str] = frozenset({
    # v1 residuals
    "combination",
    "pawn_break",
    "simplification",
    "fortification",
    "coordination",
    # v2 → v3 removals
    "bishop_quality",     # can't split back; re-detect with algo detectors
    "pawn_weakness",
    "color_complex",
    "endgame_technique",  # can't determine type; re-detect with algo detectors
})


def remap_themes(themes: list[str]) -> list[str]:
    return sorted({REMAP.get(t, t) for t in themes if t not in REMOVE})


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap v2 → v3 concept labels in JSONL")
    parser.add_argument("--input",   default="data/training_raw.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing changes")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    tmp = src.with_suffix(".tmp")

    before_counts: Counter = Counter()
    after_counts:  Counter = Counter()
    total = kept = dropped = 0

    with open(src, encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            old_themes = ex.get("themes", [])
            before_counts.update(old_themes)

            new_themes = remap_themes(old_themes)
            after_counts.update(new_themes)

            if not new_themes:
                dropped += 1
                continue

            ex["themes"] = new_themes
            kept += 1
            if not args.dry_run:
                fout.write(json.dumps(ex) + "\n")

    print(f"\nProcessed : {total:,} lines")
    print(f"Kept      : {kept:,}  (have at least one label after remap)")
    print(f"Dropped   : {dropped:,}  (all labels were removed)")

    print("\nBefore → After label counts (changed concepts only):")
    changed = set(REMAP) | REMOVE
    for old in sorted(changed):
        b = before_counts[old]
        if b == 0:
            continue
        new_name = REMAP.get(old, "[removed]")
        print(f"  {old:<25} {b:>8,}  →  {new_name}")

    print("\nAfter counts for merged/new labels:")
    for new in sorted(set(REMAP.values())):
        print(f"  {new:<25} {after_counts[new]:>8,}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        tmp.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp), str(src))
        print(f"\nWrote remapped data to {src}")


if __name__ == "__main__":
    main()
