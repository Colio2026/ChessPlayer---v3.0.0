#!/usr/bin/env python3
"""Strip algo_features from training_raw.jsonl and write a memory-mapped numpy cache.

Run once after parse + ingest, before training:
    python tools/build_algo_cache.py

Without this, loading a Phase 4 JSONL with 1663 floats embedded per example
creates ~13 GB of Python objects in RAM during dataset init — more than most
machines have. This script moves the dense floats to a binary file that the
dataset memory-maps, bringing dataset RAM from ~13 GB down to ~1.5 GB.

Outputs
-------
data/algo_cache.npy          float32 (N, algo_dim) array on disk (memory-mapped by dataset)
data/training_raw.jsonl      rewritten in place without algo_features; each line gains "_ac" index
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    jsonl_path = Path("data/training_raw.jsonl")
    cache_path = Path("data/algo_cache.npy")
    tmp_path   = Path("data/training_raw_stripped.jsonl")

    if not jsonl_path.exists():
        sys.exit(f"Not found: {jsonl_path}")

    if not jsonl_path.exists():
        sys.exit(f"Not found: {jsonl_path}")

    # ── Pass 1: count lines and detect algo_dim ────────────────────────────────
    print(f"Scanning {jsonl_path}  ({jsonl_path.stat().st_size / 1e9:.2f} GB) ...")
    n = 0
    algo_dim: int | None = None
    with open(jsonl_path, "rb") as fbin:
        for raw_line in fbin:
            stripped = raw_line.strip()
            if not stripped:
                continue
            n += 1
            if algo_dim is None:
                try:
                    ex = json.loads(stripped)
                    af = ex.get("algo_features")
                    if af:
                        algo_dim = len(af)
                except Exception:
                    pass
    if algo_dim is None:
        algo_dim = 1663   # Phase 4 default
    print(f"  {n:,} lines,  algo_dim={algo_dim}")

    # ── Allocate disk-backed memmap (writes directly to disk, minimal RAM) ─────
    cache_gb = n * algo_dim * 4 / 1e9
    print(f"Allocating cache  {n:,} × {algo_dim}  ({cache_gb:.2f} GB on disk) ...")
    arr = np.lib.format.open_memmap(
        str(cache_path), mode="w+", dtype="float32", shape=(n, algo_dim)
    )
    zero_row = np.zeros(algo_dim, dtype="float32")

    # ── Pass 2: strip algo_features → cache, write stripped JSONL ─────────────
    print("Stripping algo_features ...")
    row = 0
    with open(jsonl_path, encoding="utf-8", errors="replace") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                af = ex.pop("algo_features", None)
                ex["_ac"] = row
                arr[row] = np.array(af, dtype="float32") if af else zero_row
                fout.write(json.dumps(ex, separators=(",", ":")) + "\n")
            except Exception:
                arr[row] = zero_row
                fout.write(line + "\n")   # keep malformed lines unchanged
            row += 1
            if row % 100_000 == 0:
                print(f"  {row:,} / {n:,}", end="\r", flush=True)

    del arr   # flush memmap to disk

    print(f"\nCache written → {cache_path}  ({cache_path.stat().st_size / 1e9:.2f} GB)")

    tmp_path.replace(jsonl_path)
    new_sz = jsonl_path.stat().st_size / 1e9
    print(f"JSONL stripped → {jsonl_path}  ({new_sz:.2f} GB)")
    print("Done.  Run training next: python -m src.chess_coach.ml.train --phase4")


if __name__ == "__main__":
    main()
