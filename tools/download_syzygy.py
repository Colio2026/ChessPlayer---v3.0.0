#!/usr/bin/env python3
"""Download all Syzygy 3-4-5 piece tablebase files from tablebase.sesse.net.

Usage:
    python tools/download_syzygy.py

Output: data/syzygy/  (~938 MB, ~150 files)
Safe to re-run — already-downloaded files are skipped.
"""

from __future__ import annotations

import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://tablebase.sesse.net/syzygy/3-4-5/"
OUT_DIR  = Path("data/syzygy")

# Server cert has a hostname mismatch (their issue) — bypass verification.
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
        return r.read()


def _download_file(url: str, dest: Path) -> float:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120, context=_SSL) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    return dest.stat().st_size / 1e6


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file listing from {BASE_URL} ...")
    html = _fetch(BASE_URL).decode("utf-8", errors="replace")

    files = sorted(set(re.findall(r'href="([^"]+\.(?:rtbw|rtbz))"', html)))
    files = [f for f in files if "/" not in f]   # strip any absolute paths

    if not files:
        sys.exit("No .rtbw/.rtbz files found — the page layout may have changed.")

    print(f"{len(files)} files to download.\n")

    downloaded = skipped = failed = 0

    for i, fname in enumerate(files, 1):
        dest = OUT_DIR / fname
        url  = BASE_URL + fname

        if dest.exists():
            skipped += 1
            print(f"[{i:>3}/{len(files)}] SKIP  {fname}")
            continue

        print(f"[{i:>3}/{len(files)}] {fname} ...", end=" ", flush=True)
        try:
            mb = _download_file(url, dest)
            print(f"OK  ({mb:.1f} MB)")
            downloaded += 1
        except Exception as exc:
            print(f"FAILED — {exc}")
            if dest.exists():
                dest.unlink()
            failed += 1
        time.sleep(0.05)   # polite pause between requests

    total_mb = sum(f.stat().st_size for f in OUT_DIR.iterdir()) / 1e6
    print(f"\nDone.  Downloaded={downloaded}  Skipped={skipped}  Failed={failed}")
    print(f"Total in data/syzygy/: {total_mb:.0f} MB")
    print("\nVerify:")
    print('  python -c "import chess, chess.syzygy; tb = chess.syzygy.open_tablebase(\'data/syzygy\'); b = chess.Board(\'8/8/8/3k4/8/8/3PK3/8 w - - 0 1\'); print(tb.probe_wdl(b))"')


if __name__ == "__main__":
    main()
