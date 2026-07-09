#!/usr/bin/env python3
"""
scrape_chessgames.py  —  Download annotated PGNs from chessgames.com
---------------------------------------------------------------------
1. Searches  https://www.chessgames.com/perl/ezsearch.pl?search=annotated
2. Collects game IDs from search results (paginates automatically)
3. Clicks "download-w/annot." on each game page to get the annotated PGN
4. Saves in batches of 20 games per file  →  data/annotated_pgns/batch_001.pgn …

Usage
-----
    python assets/scrape_chessgames.py --max-games 10 --headed
    python assets/scrape_chessgames.py --max-games 2000          # bulk headless
    python assets/scrape_chessgames.py                           # resume

Dependencies
------------
    pip install playwright beautifulsoup4
    python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, Page, Download, TimeoutError as PWTimeout
except ImportError:
    sys.exit("Run:  python -m pip install playwright && python -m playwright install chromium")

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Run:  python -m pip install beautifulsoup4")

# ── constants ─────────────────────────────────────────────────────────────────

SEARCH_URL  = "https://www.chessgames.com/perl/ezsearch.pl?search=annotated"
BASE_URL    = "https://www.chessgames.com"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "annotated_pgns"
BATCH_SIZE  = 20   # games per output file

GID_RE = re.compile(r'[?&]gid=(\d+)')

# ── helpers ───────────────────────────────────────────────────────────────────

def load_downloaded(index_path: Path) -> set[str]:
    """Return set of gids already written (so we can resume)."""
    ids: set[str] = set()
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    # Index entries are per-batch; each has a 'gids' list
                    ids.update(rec.get("gids", []))
                except Exception:
                    pass
    return ids


def write_batch(
    out_dir:   Path,
    batch_num: int,
    pgn_list:  list[str],
    meta_list: list[dict],
    idx_file,
) -> None:
    """Write one batch file and append its index entry."""
    filename = f"batch_{batch_num:04d}.pgn"
    content  = "\n\n".join(pgn_list)
    (out_dir / filename).write_text(content, encoding="utf-8")
    record = {
        "batch":    batch_num,
        "filename": filename,
        "count":    len(pgn_list),
        "gids":     [m["gid"]   for m in meta_list],
        "titles":   [m["title"] for m in meta_list],
    }
    idx_file.write(json.dumps(record) + "\n")
    idx_file.flush()
    print(f"\n  → wrote {filename}  ({len(pgn_list)} games)")


# ── discovery ─────────────────────────────────────────────────────────────────

def discover_game_ids(page: Page, max_pages: int, delay: float) -> list[dict]:
    """Crawl search results and return unique game dicts {gid, title, url}."""
    seen:  set[str]   = set()
    games: list[dict] = []

    for pg in range(1, max_pages + 1):
        url = SEARCH_URL if pg == 1 else f"{SEARCH_URL}&pn={pg}"
        print(f"Search page {pg}: {url}")

        page.goto(url, wait_until="load", timeout=30_000)
        page.wait_for_timeout(1000)
        soup = BeautifulSoup(page.content(), "html.parser")

        found = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "chessgame" not in href:
                continue
            m = GID_RE.search(href)
            if not m:
                continue
            gid = m.group(1)
            if gid in seen:
                continue
            seen.add(gid)
            title = a.get_text(strip=True) or f"game_{gid}"
            full  = BASE_URL + href if href.startswith("/") else href
            games.append({"gid": gid, "title": title, "url": full})
            found += 1

        print(f"  +{found} new games  (total: {len(games)})")
        if found == 0:
            break
        time.sleep(delay)

    return games


# ── PGN download ──────────────────────────────────────────────────────────────

def _fetch_pgn_url(page: Page, url: str) -> str | None:
    """Fetch a URL via the browser's request API and return text if it's a PGN."""
    try:
        resp = page.request.get(url, timeout=15_000)
        text = resp.text()
        if "[Event" in text or "[White" in text:
            return text
    except Exception:
        pass
    return None


def download_annotated_pgn(page: Page, game: dict) -> str | None:
    """
    Navigate to a game page and get the annotated PGN.

    Priority:
      1. Find a link whose text/href contains 'w/annot' → click it, intercept download
      2. Construct the annotated URL from the nph-chesspgn href directly
      3. Try known chessgames annotated PGN URL patterns
    """
    page.goto(game["url"], wait_until="load", timeout=30_000)
    page.wait_for_timeout(1000)

    # Collect all <a href> elements and their text
    links: list[tuple[str, str]] = []   # (text, href)
    for a in page.query_selector_all("a[href]"):
        try:
            t = (a.inner_text() or "").strip()
            h = (a.get_attribute("href") or "").strip()
            links.append((t, h))
        except Exception:
            pass

    # ── Pass 1: find the "download-w/annot" link ──────────────────────────────
    annot_href: str | None = None
    for text, href in links:
        text_l = text.lower()
        href_l = href.lower()
        # Match "download-w/annot", "download w/annot", or any nph-chesspgn with annot
        if ("w/annot" in text_l or "w/annot" in href_l
                or ("download" in text_l and "annot" in text_l)):
            annot_href = href
            break

    if annot_href:
        full_url = BASE_URL + annot_href if annot_href.startswith("/") else annot_href
        print(f"    w/annot link: {full_url}")

        # Try as a direct-fetch URL first (faster, no download dialog)
        pgn = _fetch_pgn_url(page, full_url)
        if pgn:
            return pgn

        # Fall back to click + intercept download
        try:
            loc = page.locator(f'a[href="{annot_href}"]').first
            with page.expect_download(timeout=15_000) as dl_info:
                loc.click()
            dl: Download = dl_info.value
            text = Path(dl.path()).read_text(encoding="utf-8", errors="replace")
            if "[Event" in text or "[White" in text:
                return text
        except Exception as e:
            print(f"    click failed: {e}")

    # ── Pass 2: any nph-chesspgn link, prefer one with annot in its URL ───────
    pgn_hrefs = [h for _, h in links if "nph-chesspgn" in h or "chesspgn" in h]
    # Sort so annotated variants (containing 'ann' or 'annot') come first
    pgn_hrefs.sort(key=lambda h: ("ann" not in h.lower()))
    for href in pgn_hrefs:
        full_url = BASE_URL + href if href.startswith("/") else href
        pgn = _fetch_pgn_url(page, full_url)
        if pgn:
            print(f"    fallback PGN URL: {full_url}")
            return pgn

    # ── Pass 3: try known chessgames annotated URL patterns for this gid ──────
    gid = game["gid"]
    candidates = [
        f"{BASE_URL}/perl/nph-chesspgn?text=1&gid={gid}",
        f"{BASE_URL}/perl/nph-chesspgn?text=1&gid={gid}&ann=1",
        f"{BASE_URL}/perl/nph-chesspgn?text=2&gid={gid}",
    ]
    for url in candidates:
        pgn = _fetch_pgn_url(page, url)
        if pgn:
            print(f"    direct URL worked: {url}")
            return pgn

    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download annotated PGNs from chessgames.com (batched output)",
    )
    parser.add_argument("--headed",     action="store_true",
                        help="Show the browser window.")
    parser.add_argument("--max-games",  type=int,   default=10,
                        help="Max games to download (default 10).")
    parser.add_argument("--max-pages",  type=int,   default=50,
                        help="Max search result pages to crawl (default 50).")
    parser.add_argument("--output",     default=str(DEFAULT_OUT))
    parser.add_argument("--delay",      type=float, default=1.5,
                        help="Seconds between game requests (default 1.5).")
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE,
                        help=f"Games per output file (default {BATCH_SIZE}).")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    downloaded = load_downloaded(index_path)
    if downloaded:
        print(f"Resuming — {len(downloaded)} games already saved.")

    # Figure out where to resume batch numbering
    existing_batches = sorted(out_dir.glob("batch_*.pgn"))
    batch_num = len(existing_batches) + 1

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not args.headed,
            slow_mo=300 if args.headed else 0,
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            accept_downloads=True,
        )
        page = ctx.new_page()

        # ── discover ───────────────────────────────────────────────────────────
        print("\n── Discovering games ─────────────────────────────────────────────")
        all_games = discover_game_ids(page, args.max_pages, args.delay)
        print(f"\n{len(all_games)} unique games found in search results.")

        if not all_games:
            print("No games found. Try --headed to inspect the page.")
            browser.close()
            return

        # ── download & batch ───────────────────────────────────────────────────
        print(f"\n── Downloading PGNs (max {args.max_games}, "
              f"batched {args.batch_size} per file) ────────────────")
        ok = skip = fail = 0
        pgn_buf:  list[str]  = []   # PGN text for the current batch
        meta_buf: list[dict] = []   # metadata for the current batch

        with open(index_path, "a", encoding="utf-8") as idx:
            for i, game in enumerate(all_games, 1):
                if ok + len(downloaded) >= args.max_games:
                    print(f"\nReached --max-games limit ({args.max_games}).")
                    break

                if game["gid"] in downloaded:
                    skip += 1
                    continue

                label = game["title"][:52].ljust(52)
                print(f"[{i}/{len(all_games)}] {label}  gid={game['gid']}", end="  ")
                sys.stdout.flush()

                try:
                    pgn = download_annotated_pgn(page, game)
                except Exception as exc:
                    pgn = None
                    print(f"\n    Exception: {exc}")

                if pgn:
                    pgn_buf.append(pgn.strip())
                    meta_buf.append(game)
                    ok += 1
                    print("✓")

                    # Flush batch when full
                    if len(pgn_buf) >= args.batch_size:
                        write_batch(out_dir, batch_num, pgn_buf, meta_buf, idx)
                        pgn_buf  = []
                        meta_buf = []
                        batch_num += 1
                else:
                    fail += 1
                    if args.headed:
                        shot = out_dir / f"debug_gid{game['gid']}.png"
                        try:
                            page.screenshot(path=str(shot))
                            print(f"✗  (screenshot: {shot.name})")
                        except Exception:
                            print("✗")
                    else:
                        print("✗")

                time.sleep(args.delay)

            # Flush any remaining games that didn't fill a full batch
            if pgn_buf:
                write_batch(out_dir, batch_num, pgn_buf, meta_buf, idx)

        browser.close()

    print(f"""
── Done ──────────────────────────────────────────────────────────────────
Downloaded : {ok}  games  →  {(ok // args.batch_size) + (1 if ok % args.batch_size else 0)} batch file(s)
Skipped    : {skip}  (already had)
Failed     : {fail}

Output     : {out_dir}
Index      : {index_path}
""")
    if fail > 0:
        print(
            "If failures persist with --headed you'll see exactly which button\n"
            "is on the page — report the label text and I'll update the selector.\n"
        )


if __name__ == "__main__":
    main()
