#!/usr/bin/env python3
"""Strip algo_features from training_raw.jsonl and write memory-mapped numpy caches.

Run once after parse + ingest, before training:
    python tools/build_algo_cache.py

Outputs
-------
data/algo_cache.npy     float32 (N, algo_dim) — v4 spatial features (1811-dim)
data/v3_cache.npy       float32 (N, 59)       — v3 summary features (algo_feature_vector)
data/training_raw.jsonl rewritten without algo_features; each line gains "_ac" index

The v3 cache eliminates the per-example algo_feature_vector(fen) call in dataset
__getitem__, which was the main training bottleneck (~30 ms/example on CPU).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V3_DIM = 59  # algo_feature_vector() output size (Phase 3 summary bits)


def _peek_has_field(jsonl_path: Path, field: str) -> bool:
    """Return True if the first parseable line of the JSONL contains field."""
    with open(jsonl_path, "rb") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                try:
                    return field in json.loads(stripped)
                except Exception:
                    pass
    return False


def _scan(jsonl_path: Path) -> tuple[int, int]:
    """Count non-empty lines and detect algo_dim from the first line that has it."""
    print(f"Scanning {jsonl_path}  ({jsonl_path.stat().st_size / 1e9:.2f} GB) ...")
    n = 0
    algo_dim: int | None = None
    with open(jsonl_path, "rb") as f:
        for raw in f:
            stripped = raw.strip()
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
        algo_dim = 1811   # Phase 4 default (B1-B7 including king_safety_vec)
    print(f"  {n:,} lines,  algo_dim={algo_dim}")
    return n, algo_dim


def _build_both(
    jsonl_path: Path,
    tmp_path:   Path,
    algo_arr:   np.ndarray,
    v3_arr:     np.ndarray,
    n:          int,
    algo_dim:   int,
) -> None:
    """
    Pass 2: strip algo_features → algo_arr, compute v3 → v3_arr, rewrite JSONL.
    Each surviving line gains an '_ac' field that is its row index in both caches.
    """
    from label_positions import algo_feature_vector   # tools/ is on sys.path

    zero_algo = np.zeros(algo_dim, dtype="float32")
    zero_v3   = np.zeros(V3_DIM,   dtype="float32")
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
                algo_arr[row] = np.array(af, dtype="float32") if af else zero_algo
                try:
                    v3_arr[row] = algo_feature_vector(ex["fen"]).astype("float32")
                except Exception:
                    v3_arr[row] = zero_v3
                fout.write(json.dumps(ex, separators=(",", ":")) + "\n")
            except Exception:
                algo_arr[row] = zero_algo
                v3_arr[row]   = zero_v3
                fout.write(line + "\n")   # keep malformed lines unchanged
            row += 1
            if row % 100_000 == 0:
                print(f"  {row:,} / {n:,}", end="\r", flush=True)
    print()


def _build_v3_only(jsonl_path: Path, v3_arr: np.ndarray, n: int) -> None:
    """
    V3-only pass: JSONL already stripped and has '_ac' indices.
    Computes algo_feature_vector(fen) for every example and stores at _ac row.
    """
    from label_positions import algo_feature_vector

    zero_v3 = np.zeros(V3_DIM, dtype="float32")
    row = 0

    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if row >= n:
                break   # JSONL has more lines than the cache; skip extras
            try:
                ex     = json.loads(line)
                ac_idx = ex.get("_ac", row)
                if 0 <= ac_idx < n:
                    v3_arr[ac_idx] = algo_feature_vector(ex["fen"]).astype("float32")
            except Exception:
                v3_arr[row] = zero_v3
            row += 1
            if row % 100_000 == 0:
                print(f"  {row:,} / {n:,}", end="\r", flush=True)
    print()


def main() -> None:
    jsonl_path      = Path("data/training_raw.jsonl")
    algo_cache_path = Path("data/algo_cache.npy")
    v3_cache_path   = Path("data/v3_cache.npy")
    tmp_path        = Path("data/training_raw_stripped.jsonl")

    if not jsonl_path.exists():
        sys.exit(f"Not found: {jsonl_path}")

    need_v3 = not v3_cache_path.exists()
    if not need_v3:
        print("Both caches already exist. Delete data/v3_cache.npy to rebuild.")
        return

    has_algo_features = _peek_has_field(jsonl_path, "algo_features")

    if has_algo_features:
        # ── Full build: algo_cache + v3_cache in a single pass ────────────────
        n, algo_dim = _scan(jsonl_path)

        cache_gb = n * algo_dim * 4 / 1e9
        print(f"Allocating algo cache  {n:,} × {algo_dim}  ({cache_gb:.2f} GB) ...")
        algo_arr = np.lib.format.open_memmap(
            str(algo_cache_path), mode="w+", dtype="float32", shape=(n, algo_dim)
        )

        v3_gb = n * V3_DIM * 4 / 1e9
        print(f"Allocating v3 cache    {n:,} × {V3_DIM}  ({v3_gb:.3f} GB) ...")
        v3_arr = np.lib.format.open_memmap(
            str(v3_cache_path), mode="w+", dtype="float32", shape=(n, V3_DIM)
        )

        print("Building caches and stripping algo_features from JSONL ...")
        _build_both(jsonl_path, tmp_path, algo_arr, v3_arr, n, algo_dim)

        del algo_arr, v3_arr   # flush memmaps to disk

        tmp_path.replace(jsonl_path)
        print(f"Algo cache  → {algo_cache_path}  ({algo_cache_path.stat().st_size / 1e9:.2f} GB)")
        print(f"V3 cache    → {v3_cache_path}  ({v3_cache_path.stat().st_size / 1e6:.1f} MB)")
        print(f"JSONL       → {jsonl_path}  ({jsonl_path.stat().st_size / 1e9:.2f} GB)")

    else:
        # ── V3-only build: JSONL already stripped, algo_cache already exists ──
        if not algo_cache_path.exists():
            sys.exit(
                "algo_cache.npy not found and JSONL has no algo_features.\n"
                "Re-parse from scratch: .\\retrain_and_reparse.ps1"
            )
        probe = np.load(str(algo_cache_path), mmap_mode="r")
        n     = probe.shape[0]
        del probe

        v3_gb = n * V3_DIM * 4 / 1e9
        print(f"Allocating v3 cache  {n:,} × {V3_DIM}  ({v3_gb:.3f} GB) ...")
        v3_arr = np.lib.format.open_memmap(
            str(v3_cache_path), mode="w+", dtype="float32", shape=(n, V3_DIM)
        )

        print(f"Computing v3 features from {jsonl_path} ({n:,} examples) ...")
        print("This may take 10-20 minutes depending on CPU speed.")
        _build_v3_only(jsonl_path, v3_arr, n)
        del v3_arr

        print(f"V3 cache → {v3_cache_path}  ({v3_cache_path.stat().st_size / 1e6:.1f} MB)")

    print("\nDone.  Run training next:")
    print("  python -m src.chess_coach.ml.train --phase4")


if __name__ == "__main__":
    main()
