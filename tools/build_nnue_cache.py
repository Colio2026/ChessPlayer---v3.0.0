# build_nnue_cache.py
# Batch-computes NNUE Feature Transformer activations for every training position
# and writes them to data/nnue_cache.npy (shape: N × 2048, float32).
#
# Run AFTER build_algo_cache.py (requires _ac index on each JSONL record).
#
# Usage:
#   python tools/build_nnue_cache.py
#   python tools/build_nnue_cache.py --force          # rebuild even if cache exists
#   python tools/build_nnue_cache.py --nnue data/nn.nnue
#
# Output: data/nnue_cache.npy  float32 (N, 2048)  where N = row count of algo_cache.npy.
# Positions are stored at index _ac (same indexing as algo_cache and sf_cache).
# Missing positions (no _ac) get a zero row.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so 'tools.nnue_reader' is importable
# when this script is run as  python tools/build_nnue_cache.py  from the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

_DEFAULT_NNUE_PATH = Path("data/nn.nnue")
_ALGO_CACHE_PATH   = Path("data/algo_cache.npy")
_NNUE_CACHE_PATH   = Path("data/nnue_cache.npy")
_JSONL_PATH        = Path("data/training_raw.jsonl")


def _find_nnue(override: str | None = None) -> Path | None:
    if override:
        p = Path(override)
        return p if p.exists() else None
    if _DEFAULT_NNUE_PATH.exists():
        return _DEFAULT_NNUE_PATH
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",  action="store_true", help="rebuild even if cache exists")
    ap.add_argument("--nnue",   default=None,        help="path to .nnue file (default: data/nn.nnue)")
    args = ap.parse_args()

    if _NNUE_CACHE_PATH.exists() and not args.force:
        print(f"nnue_cache.npy already exists ({_NNUE_CACHE_PATH.stat().st_size/1e9:.2f} GB). "
              f"Use --force to rebuild.")
        sys.exit(0)

    # --- Locate .nnue file ---
    nnue_path = _find_nnue(args.nnue)
    if nnue_path is None:
        print("WARNING: .nnue file not found. Writing zero cache so pipeline can continue.")
        print("  Expected location: data/nn.nnue")
        print("  Export from SF binary with: \"uci\\nisready\\nexport_net data/nn.nnue\\nquit\" | stockfish")
        N = int(np.load(_ALGO_CACHE_PATH, mmap_mode='r').shape[0]) if _ALGO_CACHE_PATH.exists() else 0
        if N > 0:
            np.save(_NNUE_CACHE_PATH, np.zeros((N, 2048), dtype=np.float32))
            print(f"  Zero cache written: {_NNUE_CACHE_PATH} ({N:,} rows)")
        sys.exit(0)

    # --- Determine N ---
    if _ALGO_CACHE_PATH.exists():
        N = np.load(_ALGO_CACHE_PATH, mmap_mode='r').shape[0]
    else:
        print("algo_cache.npy not found — counting rows from JSONL (Phase 5 mode) ...")
        N = sum(1 for line in open(_JSONL_PATH, "rb") if line.strip())
        print(f"  {N:,} rows")
    print(f"Target cache size: {N:,} rows × 2048 cols = {N * 2048 * 4 / 1e9:.2f} GB")

    # --- Load FT weights ---
    print(f"\nLoading NNUE Feature Transformer from {nnue_path} ...")
    t0 = time.time()
    from tools.nnue_reader import load_feature_transformer, compute_activations, NNUE_SIZE
    biases, weights = load_feature_transformer(str(nnue_path))
    print(f"  Loaded in {time.time()-t0:.1f}s  "
          f"| biases {biases.shape}  weights {weights.shape}  "
          f"| bias range [{biases.min()}, {biases.max()}]")

    # --- Allocate output cache (memory-mapped for large files) ---
    cache = np.lib.format.open_memmap(
        str(_NNUE_CACHE_PATH), mode='w+', dtype=np.float32, shape=(N, NNUE_SIZE)
    )

    # --- Stream through JSONL and fill cache ---
    if not _JSONL_PATH.exists():
        sys.exit(f"ERROR: {_JSONL_PATH} not found.")

    print(f"\nProcessing positions from {_JSONL_PATH} ...")
    t1     = time.time()
    done   = 0
    skip   = 0
    errors = 0
    LOG_EVERY = 25_000

    with open(_JSONL_PATH, encoding='utf-8', errors='replace') as f:
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
                cache[idx] = compute_activations(fen, biases, weights)
                done += 1
                if done % LOG_EVERY == 0:
                    elapsed = time.time() - t1
                    rate    = done / elapsed
                    remain  = (N - done) / rate if rate > 0 else 0
                    print(f"  {done:>8,} done  {skip:,} skipped  "
                          f"{rate:,.0f}/s  ETA {remain/60:.1f}m")
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  WARNING: {e}")

    elapsed = time.time() - t1
    print(f"\nDone: {done:,} positions in {elapsed/60:.1f}m  "
          f"({done/(elapsed+1e-9):.0f}/s)  "
          f"{errors} errors  {skip} skipped")
    print(f"Cache: {_NNUE_CACHE_PATH}  "
          f"({_NNUE_CACHE_PATH.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
