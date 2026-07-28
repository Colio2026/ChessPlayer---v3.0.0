#!/usr/bin/env python3
"""Build Syzygy tablebase feature cache from training_raw.jsonl.

Probes data/syzygy/ for every training position with ≤7 pieces and writes
a 3-column float32 cache used by the Phase 6B gating network.

Feature layout (N, 3)
---------------------
    tb[0]  wdl_norm    : WDL normalised to [-1, -0.5, 0, 0.5, 1]
                         (2=win→1.0, 1=cursed win→0.5, 0=draw→0.0,
                          -1=blessed loss→-0.5, -2=loss→-1.0)
    tb[1]  dtz_norm    : DTZ / 100.0 clamped to [-1.0, 1.0]
                         (sign: positive = fewer moves to zeroing for side to move)
    tb[2]  is_tb       : 1.0 if tablebase covers this position, 0.0 otherwise

Positions with >7 pieces are left as all-zeros (is_tb=0.0).
The gating network uses is_tb as a hard signal to boost the endgame expert.

Usage
-----
    python tools/build_tablebase_cache.py
    python tools/build_tablebase_cache.py --force       # rebuild even if exists
    python tools/build_tablebase_cache.py --max-pieces 5  # only probe ≤5-piece positions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TB_DIM = 3   # [wdl_norm, dtz_norm, is_tb]

_WDL_NORM = {2: 1.0, 1: 0.5, 0: 0.0, -1: -0.5, -2: -1.0}


def _norm_dtz(dtz: int) -> float:
    return float(np.clip(dtz / 100.0, -1.0, 1.0))


def build(
    jsonl_path:  Path,
    cache_path:  Path,
    syzygy_path: Path,
    max_pieces:  int  = 7,
) -> None:
    import chess
    import chess.syzygy

    # ── Determine N from algo_cache or JSONL ──────────────────────────────────
    algo_path = Path("data/algo_cache.npy")
    if algo_path.exists():
        probe = np.load(str(algo_path), mmap_mode="r")
        n     = probe.shape[0]
        del probe
    else:
        print("algo_cache.npy not found — counting JSONL rows ...")
        n = sum(1 for line in open(jsonl_path, "rb") if line.strip())
        print(f"  {n:,} rows")

    size_mb = n * TB_DIM * 4 / 1e6
    print(f"Allocating tb_cache  {n:,} × {TB_DIM}  ({size_mb:.1f} MB) ...")
    arr = np.lib.format.open_memmap(
        str(cache_path), mode="w+", dtype="float32", shape=(n, TB_DIM)
    )

    # ── Open tablebase ────────────────────────────────────────────────────────
    print(f"Opening Syzygy tablebase at {syzygy_path} ...")
    try:
        tb = chess.syzygy.open_tablebase(str(syzygy_path))
    except Exception as e:
        print(f"ERROR: could not open tablebase — {e}")
        print("  Download with: python tools/download_syzygy.py")
        del arr
        return

    # ── Pre-flight probe ──────────────────────────────────────────────────────
    test_fen = "8/8/8/3k4/8/8/3PK3/8 w - - 0 1"   # K+P vs K, white wins
    try:
        test_board = chess.Board(test_fen)
        wdl_test   = tb.probe_wdl(test_board)
        assert wdl_test == 2, f"expected WDL=2 (win), got {wdl_test}"
        print(f"  Probe OK  (K+P vs K WDL={wdl_test})")
    except Exception as e:
        print(f"WARNING: pre-flight probe failed — {e}")
        print("  Continuing — verify data/syzygy/ contains 3-4-5 piece files.")

    # ── Pass: scan JSONL, probe each position ─────────────────────────────────
    print(f"Probing positions (≤{max_pieces} pieces) from {jsonl_path} ...")
    t0           = time.time()
    n_probed     = 0
    n_covered    = 0
    n_skipped    = 0
    n_errors     = 0
    n_processed  = 0

    with open(jsonl_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ex  = json.loads(raw)
                ac  = ex.get("_ac")
                fen = ex.get("fen", "")
                if ac is None or not (0 <= ac < n) or not fen:
                    continue

                n_processed += 1
                board = chess.Board(fen)
                piece_count = board.occupied.bit_count()

                if piece_count > max_pieces:
                    n_skipped += 1
                    continue

                n_probed += 1
                wdl = dtz = None
                try:
                    wdl = tb.probe_wdl(board)
                except Exception:
                    pass
                try:
                    dtz = tb.probe_dtz(board)
                except Exception:
                    pass

                if wdl is not None:
                    arr[ac, 0] = _WDL_NORM.get(wdl, 0.0)
                    arr[ac, 1] = _norm_dtz(dtz) if dtz is not None else 0.0
                    arr[ac, 2] = 1.0
                    n_covered += 1
                else:
                    n_errors += 1

            except Exception:
                pass

            if n_processed % 50_000 == 0 and n_processed > 0:
                elapsed = time.time() - t0
                rate    = n_processed / elapsed if elapsed > 0 else 0
                print(
                    f"  {n_processed:>8,} processed  "
                    f"{n_probed:>7,} probed  "
                    f"{n_covered:>7,} covered  "
                    f"{rate:>6,.0f} pos/s",
                    end="\r", flush=True,
                )

    tb.close()
    elapsed = time.time() - t0
    del arr   # flush mmap

    print(f"\n  {n_processed:,} processed  {n_probed:,} probed  "
          f"{n_covered:,} covered  {n_errors} probe errors  ({elapsed:.0f}s)")

    # ── Validate ──────────────────────────────────────────────────────────────
    result  = np.load(str(cache_path), mmap_mode="r")
    tb_rows = int((result[:, 2] == 1.0).sum())
    del result

    print(f"\nTablebase cache → {cache_path}  ({cache_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Covered positions : {tb_rows:,} / {n:,}  "
          f"({tb_rows / n * 100:.1f}% of training set)")
    if tb_rows == 0:
        print("  WARNING: zero covered positions — check data/syzygy/ files.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build Syzygy tablebase feature cache (tb_cache.npy)"
    )
    p.add_argument("--jsonl",       default="data/training_raw.jsonl")
    p.add_argument("--output",      default="data/tb_cache.npy")
    p.add_argument("--syzygy-path", default="data/syzygy",
                   help="Directory containing .rtbw/.rtbz files")
    p.add_argument("--force",       action="store_true",
                   help="Rebuild even if cache already exists")
    p.add_argument("--max-pieces",  type=int, default=7,
                   help="Only probe positions with ≤N pieces (default: 7)")
    args = p.parse_args()

    jsonl_path  = Path(args.jsonl)
    cache_path  = Path(args.output)
    syzygy_path = Path(args.syzygy_path)

    if not jsonl_path.exists():
        sys.exit(f"Not found: {jsonl_path}")
    if not syzygy_path.exists():
        sys.exit(f"Syzygy directory not found: {syzygy_path}\n"
                 "Run: python tools/download_syzygy.py")

    if cache_path.exists() and not args.force:
        print(f"tb_cache already exists: {cache_path}  (pass --force to rebuild)")
        return

    build(jsonl_path, cache_path, syzygy_path, max_pieces=args.max_pieces)
    print("\nDone.  Next: python -m src.chess_coach.ml.train --phase6")


if __name__ == "__main__":
    main()
