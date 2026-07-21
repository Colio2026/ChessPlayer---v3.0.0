# build_board_cache.py
# Pre-computes fen_to_tensor() for every training position and writes the
# result to data/board_cache.npy (shape: N × 1001, float32).
#
# Eliminates per-example FEN parsing during training — every __getitem__ becomes
# a single mmap row lookup instead of a full board traversal.
#
# Run AFTER build_algo_cache.py (requires _ac index on each JSONL record).
#
# Usage:
#   python tools/build_board_cache.py
#   python tools/build_board_cache.py --force          # rebuild even if cache exists
#
# Output: data/board_cache.npy  float32 (N, 1001)
# Positions stored at index _ac (same indexing as all other caches).
# Missing positions (no _ac) keep the default zero row.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

_ALGO_CACHE_PATH  = Path("data/algo_cache.npy")
_BOARD_CACHE_PATH = Path("data/board_cache.npy")
_JSONL_PATH       = Path("data/training_raw.jsonl")

BOARD_DIMS = 1001   # fen_to_tensor output size (matches INPUT_SIZE in board_encoder.py)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    args = ap.parse_args()

    if _BOARD_CACHE_PATH.exists() and not args.force:
        size_gb = _BOARD_CACHE_PATH.stat().st_size / 1e9
        print(f"board_cache.npy already exists ({size_gb:.2f} GB). Use --force to rebuild.")
        sys.exit(0)

    if not _JSONL_PATH.exists():
        sys.exit(f"ERROR: {_JSONL_PATH} not found. Run parse_annotated_pgn.py first.")

    # Determine N from algo_cache (authoritative row count).
    # Fallback: scan JSONL for max(_ac) + 1.  Do NOT count lines — caches are
    # indexed by _ac values, not by line position, so a line count can be wrong
    # if the JSONL has been reordered or appended since the last full build.
    if _ALGO_CACHE_PATH.exists():
        N = int(np.load(_ALGO_CACHE_PATH, mmap_mode="r").shape[0])
        print(f"Row count from algo_cache.npy: {N:,}")
    else:
        print("algo_cache.npy not found — scanning JSONL for max(_ac) ...")
        import json as _json
        max_ac = -1
        with open(_JSONL_PATH, "rb") as _f:
            for _raw in _f:
                _s = _raw.strip()
                if not _s:
                    continue
                try:
                    _ac = _json.loads(_s).get("_ac")
                    if _ac is not None and _ac > max_ac:
                        max_ac = _ac
                except Exception:
                    pass
        if max_ac < 0:
            sys.exit("ERROR: No _ac indices found in JSONL. Run build_algo_cache.py first.")
        N = max_ac + 1
        print(f"  max(_ac) + 1 = {N:,} rows")

    size_gb = N * BOARD_DIMS * 4 / 1e9
    print(f"Allocating board_cache.npy: {N:,} rows × {BOARD_DIMS} cols = {size_gb:.2f} GB")

    cache = np.lib.format.open_memmap(
        str(_BOARD_CACHE_PATH), mode="w+", dtype=np.float32, shape=(N, BOARD_DIMS)
    )

    from src.chess_coach.ml.board_encoder import fen_to_tensor

    print(f"\nProcessing positions from {_JSONL_PATH} ...")
    t0     = time.time()
    done   = 0
    skip   = 0
    errors = 0
    LOG_EVERY = 50_000

    with open(_JSONL_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                ex  = json.loads(line)
                idx = ex.get("_ac")
                if idx is None:
                    skip += 1
                    continue
                fen = ex.get("fen", "")
                if not fen:
                    skip += 1
                    continue
                cache[idx] = fen_to_tensor(fen).numpy()
                done += 1
                if done % LOG_EVERY == 0:
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    remain  = (N - done) / rate if rate > 0 else 0
                    print(f"  {done:>8,} done  {skip:,} skipped  "
                          f"{rate:,.0f}/s  ETA {remain/60:.1f}m")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  WARNING: {e}")

    elapsed = time.time() - t0
    cache.flush()
    print(f"\nDone: {done:,} positions in {elapsed/60:.1f}m  "
          f"({done/(elapsed+1e-9):,.0f}/s)  "
          f"{errors} errors  {skip} skipped")
    print(f"Cache: {_BOARD_CACHE_PATH}  ({_BOARD_CACHE_PATH.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
