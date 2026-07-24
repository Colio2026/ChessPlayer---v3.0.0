#!/usr/bin/env python3
"""Strip algo_features from training_raw.jsonl and write memory-mapped numpy caches.

Run once after parse + ingest, before training:
    python tools/build_algo_cache.py

Outputs
-------
data/algo_cache.npy     float32 (N, algo_dim) — v4 spatial features (3779-dim)
data/v3_cache.npy       float32 (N, 82)       — v3 summary features (algo_feature_vector)
data/training_raw.jsonl rewritten without algo_features; each line gains "_ac" index

The v3 cache eliminates the per-example algo_feature_vector(fen) call in dataset
__getitem__, which was the main training bottleneck (~30 ms/example on CPU).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V3_DIM = 82  # algo_feature_vector() output size — 36 per-color×2 + 10 global


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
        algo_dim = 3779   # Phase 4 default (B1-B9 full spatial block)
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


def _reindex(jsonl_path: Path) -> int:
    """Rewrite JSONL adding _ac=row_number to every line that lacks it.

    Returns the number of rows written.  Fast sequential pass — no cache
    rebuild needed.  Triggered when caches exist but the JSONL was replaced
    (e.g. re-parsed) without going through build_algo_cache.
    """
    tmp = jsonl_path.with_suffix(".reindex.tmp")
    row = 0
    with open(jsonl_path, encoding="utf-8", errors="replace") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if "_ac" not in ex:
                    ex["_ac"] = row
                fout.write(json.dumps(ex, separators=(",", ":")) + "\n")
            except Exception:
                fout.write(line + "\n")
            row += 1
            if row % 200_000 == 0:
                print(f"  {row:,} lines reindexed ...", end="\r", flush=True)
    print()
    tmp.replace(jsonl_path)
    return row


def _verify(jsonl_path: Path, algo_cache_path: Path, n_samples: int) -> None:
    """Spot-check N random positions: recompute algo_feature_vector_v4 and compare to cache."""
    import random
    from label_positions import algo_feature_vector_v4

    cache = np.load(str(algo_cache_path), mmap_mode="r")
    print(f"\nVerifying {n_samples} random positions from {jsonl_path.name} against {algo_cache_path.name} ...")

    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    rng = random.Random(42)
    samples = rng.sample(all_lines, min(n_samples, len(all_lines)))

    max_err = 0.0
    mean_err = 0.0
    checked = 0
    mismatches = 0

    for raw in samples:
        try:
            ex  = json.loads(raw)
            ac  = ex.get("_ac")
            fen = ex.get("fen", "")
            if ac is None or not fen:
                continue
            fresh = algo_feature_vector_v4(fen).astype("float32")
            cached = np.array(cache[ac], dtype="float32")
            diff   = float(np.abs(fresh - cached).max())
            mean_err += diff
            max_err   = max(max_err, diff)
            checked  += 1
            if diff > 1e-4:
                mismatches += 1
        except Exception as exc:
            print(f"  Warning: {exc}")

    if checked == 0:
        print("  No valid samples found — cannot verify.")
        return

    mean_err /= checked
    print(f"  Checked  : {checked}")
    print(f"  Max Δ    : {max_err:.6f}")
    print(f"  Mean Δ   : {mean_err:.6f}")
    print(f"  Mismatches (Δ > 1e-4): {mismatches}")
    if mismatches == 0:
        print("  ✓ Cache matches freshly computed values.")
    else:
        print(f"  ✗ {mismatches} positions differ — cache may be stale. Re-run with --force.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Rebuild caches even if they already exist (use after a fresh parse)")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="Spot-check N random positions: recompute vs cached values. Exits after check.")
    args = ap.parse_args()

    jsonl_path      = Path("data/training_raw.jsonl")
    algo_cache_path = Path("data/algo_cache.npy")
    v3_cache_path   = Path("data/v3_cache.npy")
    tmp_path        = Path("data/training_raw_stripped.jsonl")

    if not jsonl_path.exists():
        sys.exit(f"Not found: {jsonl_path}")

    if args.verify:
        if not algo_cache_path.exists():
            sys.exit(f"Cannot verify — algo_cache not found: {algo_cache_path}")
        _verify(jsonl_path, algo_cache_path, args.verify)
        return

    if args.force:
        # Delete stale caches so the full build runs below.
        for p in (algo_cache_path, v3_cache_path):
            if p.exists():
                p.unlink()
                print(f"  Removed stale cache: {p}")

    need_algo = not algo_cache_path.exists()
    need_v3   = not v3_cache_path.exists()
    has_ac    = _peek_has_field(jsonl_path, "_ac")

    if not need_algo and not need_v3 and has_ac:
        print("Both caches exist and JSONL has _ac indices — nothing to do.")
        return

    if not need_algo and not need_v3 and not has_ac:
        # Caches are fine but JSONL was replaced without _ac indices (e.g. re-parse).
        # Fast reindex pass only — no cache rebuild needed.
        print("Caches exist but JSONL is missing _ac indices — reindexing ...")
        n = _reindex(jsonl_path)
        print(f"Reindexed {n:,} lines → {jsonl_path}")
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
        # ── Stripped JSONL: algo_features already extracted in a prior run ─────
        # Recompute both caches from FEN. algo_feature_vector_v4 recomputes the
        # 3779-dim features; algo_feature_vector recomputes the 82-dim v3 bits.
        # _ac indices are already stamped — no JSONL rewrite needed unless missing.
        from label_positions import algo_feature_vector, algo_feature_vector_v4

        ALGO_DIM = 3779

        print(f"Scanning {jsonl_path} for line count ...")
        n = 0
        with open(jsonl_path, "rb") as f:
            for raw in f:
                if raw.strip():
                    n += 1
        print(f"  {n:,} lines")

        if need_algo:
            algo_gb = n * ALGO_DIM * 4 / 1e9
            print(f"Allocating algo cache  {n:,} × {ALGO_DIM}  ({algo_gb:.2f} GB) ...")
            algo_arr = np.lib.format.open_memmap(
                str(algo_cache_path), mode="w+", dtype="float32", shape=(n, ALGO_DIM)
            )
        else:
            algo_arr = None

        if need_v3:
            v3_gb = n * V3_DIM * 4 / 1e9
            print(f"Allocating v3 cache    {n:,} × {V3_DIM}  ({v3_gb:.3f} GB) ...")
            v3_arr = np.lib.format.open_memmap(
                str(v3_cache_path), mode="w+", dtype="float32", shape=(n, V3_DIM)
            )
        else:
            v3_arr = None

        needs_rewrite = not has_ac
        zero_algo = np.zeros(ALGO_DIM, dtype="float32")
        zero_v3   = np.zeros(V3_DIM,   dtype="float32")
        row = 0

        import time as _time
        t0 = _time.time()

        ctx = open(tmp_path, "w", encoding="utf-8") if needs_rewrite else None
        with open(jsonl_path, encoding="utf-8", errors="replace") as fin:
            for raw_line in fin:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    ex     = json.loads(line)
                    ac_idx = ex.get("_ac", row)
                    fen    = ex.get("fen", "")
                    if algo_arr is not None:
                        try:
                            algo_arr[ac_idx] = algo_feature_vector_v4(fen).astype("float32")
                        except Exception:
                            algo_arr[ac_idx] = zero_algo
                    if v3_arr is not None:
                        try:
                            v3_arr[ac_idx] = algo_feature_vector(fen).astype("float32")
                        except Exception:
                            v3_arr[ac_idx] = zero_v3
                    if ctx is not None:
                        ex["_ac"] = row
                        ctx.write(json.dumps(ex, separators=(",", ":")) + "\n")
                except Exception:
                    if algo_arr is not None:
                        algo_arr[row] = zero_algo
                    if v3_arr is not None:
                        v3_arr[row] = zero_v3
                    if ctx is not None:
                        ctx.write(line + "\n")
                row += 1
                if row % 50_000 == 0:
                    elapsed = _time.time() - t0
                    rate    = row / elapsed if elapsed > 0 else 0
                    remain  = (n - row) / rate if rate > 0 else 0
                    print(f"  {row:>8,} / {n:,}  {rate:,.0f}/s  ETA {remain/60:.1f}m", end="\r", flush=True)
        print()

        if ctx is not None:
            ctx.close()
            tmp_path.replace(jsonl_path)

        if algo_arr is not None:
            del algo_arr
            print(f"Algo cache → {algo_cache_path}  ({algo_cache_path.stat().st_size / 1e9:.2f} GB)")
        if v3_arr is not None:
            del v3_arr
            print(f"V3 cache   → {v3_cache_path}  ({v3_cache_path.stat().st_size / 1e6:.1f} MB)")

    print("\nDone.  Run training next:")
    print("  python -m src.chess_coach.ml.train --phase4")


if __name__ == "__main__":
    main()
