#!/usr/bin/env python3
"""
scrape_lichess_studies.py  —  Download annotated studies from lichess.org
--------------------------------------------------------------------------
Targets studies sorted by number of likes (best quality proxy available).
Also searches several chess topic terms to pull in thematic annotated content.

Private / deleted studies return 404 — they are counted, logged, and skipped.
Only studies with substantial prose annotation are kept (engine arrows alone
don't count as annotation for our training purposes).

Output: data/annotated_pgns/lichess_batch_0001.pgn  (20 studies per file)
        data/annotated_pgns/lichess_index.jsonl      (metadata + resume state)

Usage
-----
    # Default — grab up to 2000 quality studies headless
    python assets/scrape_lichess_studies.py

    # Test first 50
    python assets/scrape_lichess_studies.py --max-studies 50

    # With a Lichess API token (higher rate limit, avoids 429s on long runs)
    python assets/scrape_lichess_studies.py --token lip_xxxxxxxx

    # Resume an interrupted run automatically
    python assets/scrape_lichess_studies.py

Dependencies
------------
    python -m pip install requests beautifulsoup4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Run:  python -m pip install requests beautifulsoup4")

# ── constants ─────────────────────────────────────────────────────────────────

BASE          = "https://lichess.org"
STUDY_PGN_API = "https://lichess.org/api/study/{sid}.pgn"
DEFAULT_OUT   = Path(__file__).parent.parent / "data" / "annotated_pgns"
BATCH_SIZE    = 20

# Discovery sources — ordered by expected annotation quality
# Each entry is (url_template, label) where {page} is substituted
DISCOVERY_SOURCES = [
    # Most-liked studies overall — best single source
    ("https://lichess.org/study/all/likes?page={page}", "popular"),

    # Topic searches — pull thematic annotated content specifically
    ("https://lichess.org/study/search?q=annotated+games&page={page}",    "search:annotated-games"),
    ("https://lichess.org/study/search?q=opening+theory&page={page}",     "search:opening-theory"),
    ("https://lichess.org/study/search?q=masterclass&page={page}",        "search:masterclass"),
    ("https://lichess.org/study/search?q=endgame+technique&page={page}",  "search:endgame"),
    ("https://lichess.org/study/search?q=chess+strategy&page={page}",     "search:strategy"),
    ("https://lichess.org/study/search?q=grandmaster+games&page={page}",  "search:gm-games"),
    ("https://lichess.org/study/search?q=tactics+explained&page={page}",  "search:tactics"),
]

# Minimum prose characters to consider a study worth keeping.
# Filters out studies that are just arrows and engine symbols with no text.
MIN_PROSE_CHARS = 400

# Delay between API calls (seconds).  Lichess asks for < 10 req/s without token.
GUEST_DELAY = 0.8
TOKEN_DELAY = 0.3


# ── annotation quality ────────────────────────────────────────────────────────

_BOARD_MARKER_RE = re.compile(r'\[%[^\]]+\]')   # [%csl ...] [%cal ...] engine markers

def prose_char_count(pgn_text: str) -> int:
    """
    Count characters of meaningful prose inside { } comment blocks.
    Strips board markers and pure symbol comments (!, ?, +, #).
    Returns 0 if the study has no real text annotation.
    """
    total = 0
    for block in re.findall(r'\{([^}]*)\}', pgn_text):
        clean = _BOARD_MARKER_RE.sub('', block)
        clean = re.sub(r'[\s!?+#\-=<>]+', ' ', clean).strip()
        # Must contain actual alphabetic words to count
        if re.search(r'[a-zA-Z]{3,}', clean):
            total += len(clean)
    return total


# ── discovery ─────────────────────────────────────────────────────────────────

_STUDY_HREF_RE = re.compile(r'^/study/([A-Za-z0-9]{8})$')

def _extract_study_ids_from_html(html: str) -> list[str]:
    """Parse study IDs from a Lichess study list/search page."""
    soup = BeautifulSoup(html, "html.parser")
    ids: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        m = _STUDY_HREF_RE.match(a["href"])
        if m:
            sid = m.group(1)
            if sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


def discover_study_ids(
    session:     requests.Session,
    max_per_src: int,
    delay:       float,
) -> list[str]:
    """
    Crawl all DISCOVERY_SOURCES and return a deduplicated list of study IDs,
    most-popular first.
    """
    seen: set[str]   = set()
    all_ids: list[str] = []

    for url_tpl, label in DISCOVERY_SOURCES:
        print(f"\nSource: {label}")
        collected = 0
        page = 1

        while collected < max_per_src:
            url = url_tpl.format(page=page)
            try:
                resp = session.get(url, timeout=20)
            except Exception as e:
                print(f"  Request error: {e}")
                break

            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} — stopping this source")
                break

            ids = _extract_study_ids_from_html(resp.text)
            new = [sid for sid in ids if sid not in seen]

            for sid in new:
                seen.add(sid)
                all_ids.append(sid)
            collected += len(new)

            print(f"  Page {page}: +{len(new)} new  (source total: {collected})")

            if not ids:
                break   # no more pages for this source

            page += 1
            time.sleep(delay)

    print(f"\n{len(all_ids)} unique study IDs discovered across all sources.")
    return all_ids


# ── download ──────────────────────────────────────────────────────────────────

def download_study(session: requests.Session, sid: str) -> tuple[str | None, str]:
    """
    Download a study's PGN via the Lichess API.

    Returns (pgn_text, status) where status is one of:
        'ok'       — downloaded and passes quality filter
        'private'  — 404 / 401 (private or deleted)
        'no_annot' — downloaded but annotation too sparse
        'error'    — network / unexpected error
    """
    url = STUDY_PGN_API.format(sid=sid)
    try:
        resp = session.get(url, timeout=30, headers={"Accept": "application/x-chess-pgn"})
    except Exception as e:
        return None, f"error:{e}"

    if resp.status_code in (401, 403, 404):
        return None, "private"

    if resp.status_code != 200:
        return None, f"error:HTTP {resp.status_code}"

    text = resp.text
    if not ("[Event" in text or "[White" in text):
        return None, "error:not-pgn"

    if prose_char_count(text) < MIN_PROSE_CHARS:
        return None, "no_annot"

    return text, "ok"


# ── batch writing ─────────────────────────────────────────────────────────────

def write_batch(
    out_dir:   Path,
    batch_num: int,
    pgn_list:  list[str],
    meta_list: list[dict],
    idx_file,
) -> None:
    filename = f"lichess_batch_{batch_num:04d}.pgn"
    (out_dir / filename).write_text("\n\n".join(pgn_list), encoding="utf-8")
    rec = {
        "batch":      batch_num,
        "filename":   filename,
        "count":      len(pgn_list),
        "study_ids":  [m["sid"]   for m in meta_list],
        "prose_chars":[m["prose"] for m in meta_list],
    }
    idx_file.write(json.dumps(rec) + "\n")
    idx_file.flush()
    print(f"  → wrote {filename}  ({len(pgn_list)} studies)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download annotated Lichess studies sorted by popularity",
    )
    parser.add_argument("--max-studies",  type=int,   default=2000,
                        help="Max studies to keep (default 2000).")
    parser.add_argument("--max-per-src",  type=int,   default=500,
                        help="Max study IDs to collect per source (default 500).")
    parser.add_argument("--min-prose",    type=int,   default=MIN_PROSE_CHARS,
                        help=f"Min prose chars to keep a study (default {MIN_PROSE_CHARS}).")
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE,
                        help=f"Studies per output file (default {BATCH_SIZE}).")
    parser.add_argument("--token",        default="",
                        help="Lichess API token (optional, increases rate limit).")
    parser.add_argument("--output",       default=str(DEFAULT_OUT))
    parser.add_argument("--delay",        type=float, default=0.0,
                        help="Override auto delay between requests.")
    args = parser.parse_args()

    global MIN_PROSE_CHARS
    MIN_PROSE_CHARS = args.min_prose

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "lichess_index.jsonl"

    # ── session setup ─────────────────────────────────────────────────────────
    session = requests.Session()
    session.headers.update({
        "User-Agent": "ChessPlayer-coach-trainer/1.0 (training data collector)",
    })
    if args.token:
        session.headers["Authorization"] = f"Bearer {args.token}"
        delay = args.delay or TOKEN_DELAY
        print("Using API token — increased rate limit.")
    else:
        delay = args.delay or GUEST_DELAY
        print("No token — using guest rate limit (0.8s between calls).")
        print("Get a free token at https://lichess.org/account/oauth/token\n")

    # ── load already-downloaded study IDs ────────────────────────────────────
    done_ids: set[str] = set()
    existing_batches   = 0
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.update(rec.get("study_ids", []))
                    existing_batches = max(existing_batches, rec.get("batch", 0))
                except Exception:
                    pass
    if done_ids:
        print(f"Resuming — {len(done_ids)} studies already saved.")

    # ── discover study IDs ────────────────────────────────────────────────────
    print("\n── Discovering study IDs ─────────────────────────────────────────────")
    all_ids = discover_study_ids(session, args.max_per_src, delay)

    # ── download studies ──────────────────────────────────────────────────────
    print("\n── Downloading & filtering studies ───────────────────────────────────")
    ok = skip = private = no_annot = error = 0
    pgn_buf:  list[str]  = []
    meta_buf: list[dict] = []
    batch_num = existing_batches + 1

    with open(index_path, "a", encoding="utf-8") as idx:
        for i, sid in enumerate(all_ids, 1):
            if ok >= args.max_studies:
                print(f"\nReached --max-studies limit ({args.max_studies}).")
                break

            if sid in done_ids:
                skip += 1
                continue

            print(f"[{i}/{len(all_ids)}] {sid}", end="  ")
            sys.stdout.flush()

            pgn, status = download_study(session, sid)

            if status == "ok":
                prose = prose_char_count(pgn)
                pgn_buf.append(pgn.strip())
                meta_buf.append({"sid": sid, "prose": prose})
                ok += 1
                print(f"✓  ({prose} prose chars)")

                if len(pgn_buf) >= args.batch_size:
                    write_batch(out_dir, batch_num, pgn_buf, meta_buf, idx)
                    pgn_buf  = []
                    meta_buf = []
                    batch_num += 1

            elif status == "private":
                private += 1
                print("— private/deleted")

            elif status == "no_annot":
                no_annot += 1
                print("✗  sparse annotation, skipped")

            else:
                error += 1
                print(f"✗  {status}")

            time.sleep(delay)

        # Flush remaining
        if pgn_buf:
            write_batch(out_dir, batch_num, pgn_buf, meta_buf, idx)

    total_attempted = ok + private + no_annot + error
    batch_count     = (existing_batches + (ok // args.batch_size)
                       + (1 if ok % args.batch_size else 0))

    print(f"""
── Done ──────────────────────────────────────────────────────────────────
Kept       : {ok}  studies  →  batch files: lichess_batch_0001 … {batch_num:04d}
Skipped    : {skip}  (already had from previous run)
Private    : {private}  (404 / deleted — expected, ignored)
No annot   : {no_annot}  (moves only, filtered out)
Errors     : {error}

Output     : {out_dir}
Index      : {index_path}

Next step:
    python tools/parse_annotated_pgn.py --input "{out_dir}" --output data/training_raw.jsonl
""")


if __name__ == "__main__":
    main()
