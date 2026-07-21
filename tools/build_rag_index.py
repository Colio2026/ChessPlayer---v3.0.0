#!/usr/bin/env python3
"""Build the two data files needed for RAG-based chess coaching.

Outputs
-------
data/eco_db.json       — normalized-FEN → {eco, opening, variation, depth}
                         derived from the authoritative eco.pgn opening database
data/rag_index.jsonl   — filtered annotated positions from training_raw.jsonl
                         only records with genuine human commentary (>= 80 chars)

Usage
-----
    python tools/build_rag_index.py
    python tools/build_rag_index.py --force      # rebuild even if outputs exist
    python tools/build_rag_index.py --min-len 60 # lower comment threshold
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import chess
import chess.pgn

_ECO_PGN   = Path("data/annotated_pgns/ECO_code_openings/eco.pgn")
_JSONL     = Path("data/training_raw.jsonl")
_ECO_OUT   = Path("data/eco_db.json")
_RAG_OUT   = Path("data/rag_index.jsonl")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_fen(fen: str) -> str:
    """Return first 4 FEN fields (strip halfmove / fullmove counters)."""
    return " ".join(fen.split()[:4])


def _is_engine_annotation(text: str) -> bool:
    """Heuristic: engine-generated lines start with standard prefixes."""
    strip = text.strip()
    prefixes = ("Inaccuracy.", "Mistake.", "Blunder.", "Best was", "Better was",
                "!!", "!?", "?!", "??", "!", "?")
    return strip.startswith(prefixes) or len(strip) < 30


# ── Phase 1: ECO database ─────────────────────────────────────────────────────

def build_eco_db(eco_pgn_path: Path, out_path: Path) -> None:
    """Parse eco.pgn and build FEN→ECO lookup dict.

    Every intermediate position in each ECO line is recorded, tagged with its
    depth (ply count).  When a position is reached by two different ECO lines
    the deepest (most specific) entry wins.
    """
    print(f"Building ECO database from {eco_pgn_path} ...")
    t0 = time.time()

    eco_db: dict[str, dict] = {}
    skipped = 0
    total   = 0

    with open(eco_pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            total += 1
            eco       = game.headers.get("ECO", "")
            opening   = game.headers.get("Opening", "")
            variation = game.headers.get("Variation", "")

            if not eco:
                skipped += 1
                continue

            board = game.board()
            depth = 0
            # Record each position in the mainline
            for node in game.mainline():
                board.push(node.move)
                depth += 1
                norm = _norm_fen(board.fen())
                existing = eco_db.get(norm)
                if existing is None or depth > existing["depth"]:
                    eco_db[norm] = {
                        "eco":       eco,
                        "opening":   opening,
                        "variation": variation,
                        "depth":     depth,
                    }

    print(f"  {total} ECO entries → {len(eco_db):,} unique positions  "
          f"({skipped} skipped, {time.time()-t0:.1f}s)")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eco_db, f, separators=(",", ":"))

    print(f"ECO database → {out_path}  ({out_path.stat().st_size / 1e3:.0f} KB)")


# ── Phase 2: RAG annotation index ─────────────────────────────────────────────

_KEEP_KEYS = {"fen", "eco", "opening", "annotation", "themes",
              "phase", "game", "move_san", "fullmove", "side", "source"}


def build_rag_index(jsonl_path: Path, out_path: Path, min_len: int) -> None:
    """Filter training_raw.jsonl for genuine human commentary records.

    Keeps: annotation length >= min_len AND not flagged as engine-generated.
    Drops: algo_features, history_rich, move_uci, _ac to keep the file compact.
    """
    print(f"\nBuilding RAG index from {jsonl_path}  (min_len={min_len}) ...")
    t0 = time.time()
    total = kept = 0

    with open(jsonl_path, encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                ex = json.loads(raw)
                ann = ex.get("annotation", "")
                if not ann or len(ann) < min_len:
                    continue
                if _is_engine_annotation(ann):
                    continue
                record = {k: ex[k] for k in _KEEP_KEYS if k in ex}
                fout.write(json.dumps(record, separators=(",", ":")) + "\n")
                kept += 1
            except Exception:
                pass
            if total % 200_000 == 0:
                print(f"  {total:,} scanned  {kept:,} kept ...", end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"  {total:,} scanned  {kept:,} kept  ({elapsed:.1f}s)")
    print(f"RAG index → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build eco_db.json and rag_index.jsonl for chess coaching RAG"
    )
    ap.add_argument("--force",   action="store_true", help="rebuild even if outputs exist")
    ap.add_argument("--min-len", type=int, default=80,
                    help="minimum annotation length to include in RAG index (default: 80)")
    ap.add_argument("--eco-pgn", default=str(_ECO_PGN))
    ap.add_argument("--jsonl",   default=str(_JSONL))
    args = ap.parse_args()

    eco_pgn_path = Path(args.eco_pgn)
    jsonl_path   = Path(args.jsonl)

    for p in (eco_pgn_path, jsonl_path):
        if not p.exists():
            sys.exit(f"Not found: {p}")

    if not _ECO_OUT.exists() or args.force:
        build_eco_db(eco_pgn_path, _ECO_OUT)
    else:
        print(f"eco_db.json exists — skipping  (--force to rebuild)")

    if not _RAG_OUT.exists() or args.force:
        build_rag_index(jsonl_path, _RAG_OUT, args.min_len)
    else:
        print(f"rag_index.jsonl exists — skipping  (--force to rebuild)")

    print("\nDone.")
    print("  Next: from src.chess_coach.rag.retriever import RAGRetriever")


if __name__ == "__main__":
    main()
