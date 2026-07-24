#!/usr/bin/env python3
"""Build Stockfish classical eval feature cache from training_raw.jsonl.

Run after build_algo_cache.py (requires _ac indices to be in JSONL):
    python tools/build_sf_cache.py

Outputs
-------
data/sf_cache.npy  float32 (N, 14) — 7 SF classical eval terms × 2 sides (white/black)

Feature layout
--------------
    white side:  indices 0-6  → Mobility, King safety, Threats, Passed, Space, Pawns, Imbalance
    black side:  indices 7-13 → same order
    sf[3]  = white Passed  (main signal for passed_pawn quality)
    sf[10] = black Passed

Design note
-----------
Stockfish is driven in per-batch mode: N positions are piped to a fresh SF
process via subprocess.run(input=...).  Python's communicate() handles all
pipe I/O in background threads — the only reliable way to drive a Windows
console subprocess.  Persistent pipe handles raised OSError [Errno 22]
(ERROR_INVALID_PARAMETER) on every eval write regardless of flags or mode.

If Stockfish is not found the cache is written as all-zeros so the pipeline
still runs — the model trains and works, just without SF classical eval signal.

Override engine path with STOCKFISH_PATH environment variable or --sf-path flag.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SF_DIM   = 14
SF_TERMS = ["Mobility", "King safety", "Threats", "Passed", "Space", "Pawns", "Imbalance"]

_CANDIDATE_SF_PATHS = [
    Path("assets/engines/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"),
    Path("assets/engines/stockfish-windows-x86-64-avx2/stockfish/stockfish.exe"),
    Path("assets/engines/stockfish/stockfish.exe"),
    Path("stockfish.exe"),
    Path("stockfish"),
]

_DEFAULT_BATCH_SIZE    = 1000
_DEFAULT_BATCH_TIMEOUT = 120.0   # seconds per batch (generous; 1000 × ~1 ms/eval = ~1 s real work)


def _find_sf() -> str | None:
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).exists():
        return env
    for p in _CANDIDATE_SF_PATHS:
        resolved = p.resolve()
        if resolved.exists():
            return str(resolved)
    return None


# ── SF classical eval output parser ──────────────────────────────────────────

def _parse_classical_table(lines: list[str]) -> np.ndarray:
    """Extract per-side MG values from the SF classical eval ASCII table.

    Returns a 14-dim float32 vector: [white×7, black×7] in SF_TERMS order.
    SF16 cells contain two values ("MG  EG"); we take MG (the first token).
    Unknown or '---' entries default to 0.0.
    """
    out = np.zeros(SF_DIM, dtype=np.float32)
    term_map = {t.lower(): i for i, t in enumerate(SF_TERMS)}

    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) < 5:
            continue
        idx = term_map.get(parts[1].strip().lower())
        if idx is None:
            continue

        def _first_num(cell: str) -> float:
            # SF16 cells contain "MG  EG"; take MG. SF14 single-value cells still work.
            for tok in cell.split():
                try:
                    return float(tok)
                except ValueError:
                    pass
            return 0.0

        out[idx]     = _first_num(parts[2])   # white MG
        out[7 + idx] = _first_num(parts[3])   # black MG

    return out


# ── Batch SF driver ───────────────────────────────────────────────────────────

def _batch_stdin(fens: list[str]) -> bytes:
    """Build complete stdin bytes for a batch of N positions.

    Protocol sent to SF:
        uci                     → uciok  (+ option lines)
        isready                 → readyok
        for each fen:
            position fen <fen>
            eval
            isready             → readyok  (marks end of that eval's output)
        quit
    """
    cmds = ["uci", "isready"]
    for fen in fens:
        cmds.extend([f"position fen {fen}", "eval", "isready"])
    cmds.append("quit")
    return "\n".join(cmds).encode()


def _run_sf_batch(
    sf_path: str,
    batch: list[tuple[int, str]],
    timeout: float = _DEFAULT_BATCH_TIMEOUT,
) -> tuple[list[np.ndarray], bool]:
    """Run SF on a batch of (ac, fen) pairs; return (results, ok).

    subprocess.run(input=...) calls communicate() internally: one thread
    writes all stdin, another reads all stdout — no deadlock, no handle
    lifecycle issues.

    Output parsing splits on "readyok\\n":
        segs[0]     UCI init output (id name …, uciok)
        segs[1]     classical eval block for batch[0]
        segs[2]     classical eval block for batch[1]
        …
        segs[N]     classical eval block for batch[N-1]
    """
    fens = [fen for _, fen in batch]
    try:
        result = subprocess.run(
            [sf_path],
            input=_batch_stdin(fens),
            capture_output=True,
            timeout=timeout,
        )
        stdout = (
            result.stdout
            .decode("utf-8", errors="replace")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        segs = stdout.split("readyok\n")
        arrays = [
            _parse_classical_table(
                (segs[i + 1] if i + 1 < len(segs) else "").split("\n")
            )
            for i in range(len(fens))
        ]
        return arrays, True
    except subprocess.TimeoutExpired:
        return [np.zeros(SF_DIM, dtype=np.float32)] * len(batch), False
    except Exception:
        return [np.zeros(SF_DIM, dtype=np.float32)] * len(batch), False


# ── Cache builder ─────────────────────────────────────────────────────────────

def build(
    jsonl_path: Path,
    cache_path: Path,
    sf_path: str | None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    batch_timeout: float = _DEFAULT_BATCH_TIMEOUT,
) -> None:
    algo_path = Path("data/algo_cache.npy")
    if algo_path.exists():
        probe = np.load(str(algo_path), mmap_mode="r")
        n     = probe.shape[0]
        del probe
    else:
        print("algo_cache.npy not found — counting rows from JSONL ...")
        n = sum(1 for line in open(jsonl_path, "rb") if line.strip())
        print(f"  {n:,} rows")

    size_mb = n * SF_DIM * 4 / 1e6
    print(f"Allocating sf_cache  {n:,} × {SF_DIM}  ({size_mb:.1f} MB) ...")
    arr = np.lib.format.open_memmap(
        str(cache_path), mode="w+", dtype="float32", shape=(n, SF_DIM)
    )

    if sf_path is None:
        print("Warning: Stockfish not found.")
        print("  Set STOCKFISH_PATH env var or pass --sf-path to populate the cache.")
        print("  Writing zero cache — pipeline still runs; SF features will be inactive.")
        del arr
        return

    print(f"Stockfish: {sf_path}")

    # ── Pre-flight probe ──────────────────────────────────────────────────────
    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    probe_results, probe_ok = _run_sf_batch(sf_path, [(0, startpos)], timeout=30.0)
    if not probe_ok:
        print("Warning: SF probe timed out after 30 s — writing zero cache.")
        del arr
        return
    vec = probe_results[0]
    if not np.any(vec):
        print("Warning: SF probe returned an all-zero vector.")
        print("  Check that the SF binary runs standalone and outputs a classical eval table.")
        print("  Continuing — verify non-zero rows in the final summary.")
    else:
        print(f"  Probe OK  (Mobility white={vec[0]:.2f}  black={vec[7]:.2f})")

    # ── Pass 1: collect (ac_idx, fen) pairs ──────────────────────────────────
    print("Collecting positions from JSONL ...")
    pairs: list[tuple[int, str]] = []
    with open(jsonl_path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ex = json.loads(raw)
                ac = ex.get("_ac")
                if ac is None or not (0 <= ac < n):
                    continue
                pairs.append((ac, ex["fen"]))
            except Exception:
                pass
    print(f"  {len(pairs):,} positions to evaluate")

    # ── Pass 2: batch evaluate ────────────────────────────────────────────────
    t0             = time.time()
    n_batches      = (len(pairs) + batch_size - 1) // batch_size
    failed_batches = 0

    for b in range(n_batches):
        batch = pairs[b * batch_size : (b + 1) * batch_size]
        results, ok = _run_sf_batch(sf_path, batch, timeout=batch_timeout)
        if not ok:
            failed_batches += 1
        for (ac, _), vec in zip(batch, results):
            arr[ac] = vec

        done    = min((b + 1) * batch_size, len(pairs))
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 0
        print(
            f"  batch {b + 1:>5}/{n_batches}  "
            f"{done:>8,}/{len(pairs):,}  "
            f"{rate:>7,.0f} pos/s  "
            f"failed_batches={failed_batches}",
            end="\r", flush=True,
        )

    elapsed = time.time() - t0
    print(f"\n  {len(pairs):,} positions  {failed_batches} failed batches  ({elapsed:.0f}s)")

    # ── Validate ──────────────────────────────────────────────────────────────
    nonzero_rows = int(np.any(arr != 0, axis=1).sum())
    zero_frac    = 1.0 - nonzero_rows / n
    del arr   # flush mmap to disk

    print(f"SF cache → {cache_path}  ({cache_path.stat().st_size / 1e6:.1f} MB)")
    if zero_frac > 0.95:
        print(f"\nWARNING: {zero_frac * 100:.1f}% of rows are all-zero — SF features are inactive.")
        print("  Check that Stockfish binary is correct and outputs a classical eval table.")
    else:
        print(f"  Non-zero rows: {nonzero_rows:,} / {n:,}  ({(1 - zero_frac) * 100:.1f}% populated)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build Stockfish classical eval feature cache (sf_cache.npy)"
    )
    p.add_argument("--jsonl",          default="data/training_raw.jsonl",
                   help="JSONL file with _ac indices (output of build_algo_cache.py)")
    p.add_argument("--output",         default="data/sf_cache.npy")
    p.add_argument("--sf-path",        default=None,
                   help="Path to Stockfish binary (default: auto-detect)")
    p.add_argument("--force",          action="store_true",
                   help="Rebuild even if cache already exists")
    p.add_argument("--batch-size",     type=int,   default=_DEFAULT_BATCH_SIZE,
                   help=f"Positions per SF invocation (default: {_DEFAULT_BATCH_SIZE})")
    p.add_argument("--batch-timeout",  type=float, default=_DEFAULT_BATCH_TIMEOUT,
                   help=f"Seconds before a batch is abandoned as zero (default: {_DEFAULT_BATCH_TIMEOUT})")
    args = p.parse_args()

    jsonl_path = Path(args.jsonl)
    cache_path = Path(args.output)

    if not jsonl_path.exists():
        sys.exit(f"Not found: {jsonl_path}")

    if cache_path.exists() and not args.force:
        print(f"SF cache already exists: {cache_path}  (pass --force to rebuild)")
        return

    sf_path = args.sf_path or _find_sf()
    build(jsonl_path, cache_path, sf_path,
          batch_size=args.batch_size, batch_timeout=args.batch_timeout)
    print("\nDone.  Run training next:")
    print("  python -m src.chess_coach.ml.train --phase4")


if __name__ == "__main__":
    main()
