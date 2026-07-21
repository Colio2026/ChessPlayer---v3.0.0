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
    """Extract per-side values from the SF classical eval ASCII table.

    Returns a 14-dim float32 vector: [white×7, black×7] in SF_TERMS order.
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

        def _f(raw: str) -> float:
            raw = raw.strip()
            return 0.0 if (not raw or raw == "---") else float(raw) if _is_num(raw) else 0.0

        def _is_num(s: str) -> bool:
            try:
                float(s)
                return True
            except ValueError:
                return False

        out[idx]     = _f(parts[2])   # white
        out[7 + idx] = _f(parts[3])   # black

    return out


# ── Persistent SF process ─────────────────────────────────────────────────────

class _SFProc:
    """Long-lived Stockfish subprocess using UCI stdin/stdout protocol."""

    def __init__(self, sf_path: str) -> None:
        self._p = subprocess.Popen(
            [sf_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._write("uci")
        self._drain_until("uciok")
        # Disable NNUE so the classical eval table shows per-side values.
        # Silently ignored by engines that don't support this option.
        self._write("setoption name Use NNUE value false")
        self._write("isready")
        self._drain_until("readyok")

    def _write(self, cmd: str) -> None:
        self._p.stdin.write(cmd + "\n")
        self._p.stdin.flush()

    def _drain_until(self, keyword: str, limit: int = 80) -> list[str]:
        lines: list[str] = []
        for _ in range(limit):
            line = self._p.stdout.readline()
            if not line:
                break
            lines.append(line)
            if keyword in line:
                break
        return lines

    def eval_fen(self, fen: str) -> np.ndarray:
        self._write(f"position fen {fen}")
        self._write("eval")
        # SF16+ ends the eval block with "Classical evaluation"
        # older SF may use "Final evaluation" — read whichever comes first
        lines = self._drain_until("Classical evaluation", limit=80)
        if not any("Classical evaluation" in l for l in lines):
            lines += self._drain_until("Final evaluation", limit=20)
        return _parse_classical_table(lines)

    def close(self) -> None:
        try:
            self._write("quit")
            self._p.wait(timeout=3)
        except Exception:
            self._p.kill()


# ── Cache builder ─────────────────────────────────────────────────────────────

def build(jsonl_path: Path, cache_path: Path, sf_path: str | None) -> None:
    algo_path = Path("data/algo_cache.npy")
    if algo_path.exists():
        probe = np.load(str(algo_path), mmap_mode="r")
        n     = probe.shape[0]
        del probe
    else:
        print("algo_cache.npy not found — counting rows from JSONL (Phase 5 mode) ...")
        n = sum(1 for line in open(jsonl_path, "rb") if line.strip())
        print(f"  {n:,} rows")

    size_mb = n * SF_DIM * 4 / 1e6
    print(f"Allocating sf_cache  {n:,} × {SF_DIM}  ({size_mb:.1f} MB) ...")
    arr = np.lib.format.open_memmap(
        str(cache_path), mode="w+", dtype="float32", shape=(n, SF_DIM)
    )

    sf: _SFProc | None = None
    if sf_path:
        try:
            sf = _SFProc(sf_path)
            print(f"Stockfish: {sf_path}")
        except Exception as exc:
            print(f"Warning: Stockfish failed to start ({exc}) — writing zero cache")
    else:
        print("Warning: Stockfish not found.")
        print("  Set STOCKFISH_PATH env var or pass --sf-path to populate the cache.")
        print("  Writing zero cache — pipeline still runs; SF features will be inactive.")

    t0 = time.time()
    done = errors = 0

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
                if sf is not None:
                    try:
                        arr[ac] = sf.eval_fen(ex["fen"])
                    except Exception:
                        errors += 1
                done += 1
                if done % 50_000 == 0:
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    print(f"  {done:>8,} / {n:,}  {rate:>7,.0f}/s  errors={errors}",
                          end="\r", flush=True)
            except Exception:
                pass

    print(f"\n  {done:,} processed  {errors} errors  ({time.time() - t0:.0f}s)")
    del arr   # flush mmap to disk

    if sf is not None:
        sf.close()

    print(f"SF cache → {cache_path}  ({cache_path.stat().st_size / 1e6:.1f} MB)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build Stockfish classical eval feature cache (sf_cache.npy)"
    )
    p.add_argument("--jsonl",    default="data/training_raw.jsonl",
                   help="JSONL file with _ac indices (output of build_algo_cache.py)")
    p.add_argument("--output",   default="data/sf_cache.npy")
    p.add_argument("--sf-path",  default=None,
                   help="Path to Stockfish binary (default: auto-detect)")
    p.add_argument("--force",    action="store_true",
                   help="Rebuild even if cache already exists")
    args = p.parse_args()

    jsonl_path = Path(args.jsonl)
    cache_path = Path(args.output)

    if not jsonl_path.exists():
        sys.exit(f"Not found: {jsonl_path}")

    if cache_path.exists() and not args.force:
        print(f"SF cache already exists: {cache_path}  (pass --force to rebuild)")
        return

    sf_path = args.sf_path or _find_sf()
    build(jsonl_path, cache_path, sf_path)
    print("\nDone.  Run training next:")
    print("  python -m src.chess_coach.ml.train --phase4")


if __name__ == "__main__":
    main()
