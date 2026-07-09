#!/usr/bin/env python3
"""
scrape_gameknot.py  —  Playwright-based scraper for gameknot.com annotated games
----------------------------------------------------------------------------------
Automates the exact click sequence the site requires:
    1. Visit game page  (https://gameknot.com/annotation.pl/<slug>?gm=<id>)
    2. Click "Interactive"
    3. Click "Save/Export"
    4. Click "Get PGN"
    5. Extract the PGN text that appears and save it

Game discovery crawls:  https://gameknot.com/best-annotated-games.pl

Usage
-----
    # First run — grab everything (up to 2000 games), headless
    python assets/scrape_gameknot.py

    # Watch the browser (useful for debugging selector issues)
    python assets/scrape_gameknot.py --headed

    # Resume interrupted run (already-saved game IDs are skipped)
    python assets/scrape_gameknot.py

    # Limit to 100 games, slower delay
    python assets/scrape_gameknot.py --max-games 100 --delay 2.5

Dependencies
------------
    pip install playwright
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
    from playwright.sync_api import (
        sync_playwright,
        Page,
        TimeoutError as PWTimeout,
        Response,
    )
except ImportError:
    sys.exit(
        "playwright not installed.\n"
        "Run:  pip install playwright\n"
        "      python -m playwright install chromium"
    )

# ── constants ─────────────────────────────────────────────────────────────────

LISTING_URL = "https://gameknot.com/best-annotated-games.pl"
BASE_URL    = "https://gameknot.com"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "annotated_pgns"

# Matches annotation links on the listing page
GAME_LINK_RE = re.compile(r'/annotation\.pl/([^?#"\']+)\?gm=(\d+)', re.IGNORECASE)

# How long (ms) to wait for UI elements before giving up on a game
UI_TIMEOUT = 12_000


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_filename(slug: str, game_id: str) -> str:
    clean = re.sub(r"[^\w\-]", "_", slug)[:80].strip("_")
    return f"{clean}_gm{game_id}.pgn"


def looks_like_pgn(text: str) -> bool:
    s = text.strip()
    return bool(s) and ("[Event" in s or "[White" in s)


def load_downloaded_ids(index_path: Path) -> set[str]:
    ids: set[str] = set()
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["game_id"])
                except Exception:
                    pass
    return ids


# ── game discovery ────────────────────────────────────────────────────────────

def discover_games(page: Page, max_pages: int, delay: float) -> list[dict]:
    """
    Crawl the best-annotated-games listing and return unique game metadata.
    Handles pagination automatically.
    """
    seen:  set[str]   = set()
    games: list[dict] = []
    pg = 1

    while pg <= max_pages:
        url = LISTING_URL if pg == 1 else f"{LISTING_URL}?pg={pg}"
        print(f"\nListing page {pg}: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)

        found = 0
        for a in page.query_selector_all("a[href]"):
            href = a.get_attribute("href") or ""
            m    = GAME_LINK_RE.search(href)
            if not m:
                continue
            slug, gid = m.group(1), m.group(2)
            if gid in seen:
                continue
            seen.add(gid)
            full = BASE_URL + href if href.startswith("/") else href
            # Ensure no stray .pgn suffix on the page URL
            full = re.sub(r"\.pgn$", "", full)
            title = (a.inner_text() or slug.replace("-", " ").title()).strip()
            games.append({"url": full, "slug": slug, "game_id": gid, "title": title})
            found += 1

        print(f"  +{found} new games  (total so far: {len(games)})")
        if found == 0:
            break   # no more pages

        pg += 1
        time.sleep(delay)

    return games


# ── PGN extraction ────────────────────────────────────────────────────────────

def _try_network_intercept(page: Page, game_url: str) -> str | None:
    """
    Navigate to the game page and listen for any network response that looks
    like a PGN.  Some sites serve PGN via a background XHR/fetch.
    Returns the PGN text if found, None otherwise.
    """
    captured: list[str] = []

    def on_response(resp: Response) -> None:
        ctype = resp.headers.get("content-type", "")
        if "pgn" not in ctype and "text" not in ctype:
            return
        try:
            body = resp.text()
            if looks_like_pgn(body):
                captured.append(body.strip())
        except Exception:
            pass

    page.on("response", on_response)
    try:
        page.goto(game_url, wait_until="domcontentloaded", timeout=30_000)
        # Give any deferred network requests a moment to fire
        page.wait_for_timeout(1500)
    finally:
        page.remove_listener("response", on_response)

    return captured[0] if captured else None


def _click_text(page: Page, texts: list[str], timeout: int = UI_TIMEOUT) -> bool:
    """
    Try to click the first visible element whose inner text matches any of
    the given strings (case-insensitive, partial match).
    Returns True on success.
    """
    for text in texts:
        try:
            # getByText is the most robust selector for text-matched buttons
            loc = page.get_by_text(re.compile(re.escape(text), re.IGNORECASE)).first
            loc.wait_for(state="visible", timeout=timeout)
            loc.click(timeout=timeout)
            return True
        except PWTimeout:
            continue
        except Exception:
            continue
    return False


def _extract_pgn_from_dom(page: Page) -> str | None:
    """
    After the Get PGN button is clicked, look for the PGN text in the page.
    Tries: <textarea>, <pre>, any element whose text starts with '[Event'.
    """
    # 1. Textarea (most common pattern for "copy this text" UIs)
    for sel in ["textarea", "pre", ".pgn-text", "#pgn-text", "[class*='pgn']"]:
        try:
            el = page.locator(sel).first
            el.wait_for(state="visible", timeout=4_000)
            text = el.input_value() if sel == "textarea" else el.inner_text()
            if looks_like_pgn(text):
                return text.strip()
        except Exception:
            pass

    # 2. Scan every visible text block for PGN content
    try:
        for el in page.query_selector_all("*"):
            try:
                txt = el.inner_text()
                if looks_like_pgn(txt) and len(txt) > 100:
                    return txt.strip()
            except Exception:
                continue
    except Exception:
        pass

    return None


def get_pgn(page: Page, game: dict) -> str | None:
    """
    Full extraction pipeline for one game:
        1. Network intercept (catches background XHR PGN responses)
        2. UI automation: Interactive → Save/Export → Get PGN → extract text
    """
    # ── Pass 1: network intercept ─────────────────────────────────────────────
    pgn = _try_network_intercept(page, game["url"])
    if pgn:
        return pgn

    # ── Pass 2: UI automation ─────────────────────────────────────────────────
    # We're already on the game page from pass 1.  Now click through the UI.

    # Step A: click "Interactive" (various label spellings seen on chess sites)
    clicked = _click_text(page, ["Interactive", "interactive", "Play through"])
    if not clicked:
        print("    ✗  Could not find 'Interactive' button")
        return None
    page.wait_for_timeout(800)

    # Step B: click "Save/Export" or "Export" or "Save"
    clicked = _click_text(page, ["Save/Export", "Save / Export", "Export", "Save"])
    if not clicked:
        print("    ✗  Could not find 'Save/Export' button")
        return None
    page.wait_for_timeout(600)

    # Step C: click "Get PGN" (sometimes also labelled "PGN", "Download PGN")
    clicked = _click_text(page, ["Get PGN", "Get pgn", "PGN", "Download PGN"])
    if not clicked:
        print("    ✗  Could not find 'Get PGN' button")
        return None
    page.wait_for_timeout(800)

    # Step D: extract the displayed PGN
    pgn = _extract_pgn_from_dom(page)
    if pgn:
        return pgn

    # Step E: check if a new network response fired during the click sequence
    #         (some sites serve PGN via fetch after a button click)
    print("    ✗  PGN not found in DOM after UI sequence")
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape annotated PGNs from gameknot.com using browser automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--headed",    action="store_true",
                        help="Show the browser window (useful for debugging).")
    parser.add_argument("--max-games", type=int,   default=2000,
                        help="Max games to download (default 2000).")
    parser.add_argument("--max-pages", type=int,   default=100,
                        help="Max listing pages to crawl (default 100).")
    parser.add_argument("--output",    default=str(DEFAULT_OUT),
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--delay",     type=float, default=1.5,
                        help="Seconds between games (default 1.5).")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    downloaded = load_downloaded_ids(index_path)
    if downloaded:
        print(f"Resuming — {len(downloaded)} games already saved, will skip them.")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not args.headed,
            # Slow down actions by 300ms in headed mode so you can follow along
            slow_mo=300 if args.headed else 0,
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        # ── discover games ────────────────────────────────────────────────────
        print("\n── Discovering games ─────────────────────────────────────────────")
        all_games = discover_games(page, args.max_pages, args.delay)
        print(f"\n{len(all_games)} unique games found in the listing.")

        if not all_games:
            print(
                "\nNo games found.  The listing page structure may have changed.\n"
                "Try running with --headed to watch what the browser sees."
            )
            browser.close()
            return

        # ── download PGNs ─────────────────────────────────────────────────────
        print("\n── Downloading PGNs ──────────────────────────────────────────────")
        ok = skip = fail = 0

        with open(index_path, "a", encoding="utf-8") as idx:
            for i, game in enumerate(all_games, 1):
                if len(downloaded) + ok >= args.max_games:
                    print(f"\nReached --max-games limit ({args.max_games}).")
                    break

                if game["game_id"] in downloaded:
                    skip += 1
                    continue

                label = game["title"][:55].ljust(55)
                print(f"[{i}/{len(all_games)}] {label}  gm={game['game_id']}", end="  ")
                sys.stdout.flush()

                try:
                    pgn = get_pgn(page, game)
                except Exception as exc:
                    pgn = None
                    print(f"\n    Exception: {exc}")

                if pgn:
                    filename = safe_filename(game["slug"], game["game_id"])
                    (out_dir / filename).write_text(pgn, encoding="utf-8")
                    idx.write(json.dumps({
                        "game_id":  game["game_id"],
                        "title":    game["title"],
                        "slug":     game["slug"],
                        "url":      game["url"],
                        "filename": filename,
                    }) + "\n")
                    idx.flush()
                    ok += 1
                    print("✓")
                else:
                    fail += 1
                    # Save a screenshot so you can see what went wrong
                    if args.headed:
                        shot = out_dir / f"debug_gm{game['game_id']}.png"
                        try:
                            page.screenshot(path=str(shot))
                            print(f"✗  (screenshot: {shot.name})")
                        except Exception:
                            print("✗")
                    else:
                        print("✗")

                time.sleep(args.delay)

        browser.close()

    print(f"""
── Done ──────────────────────────────────────────────────────────────────
Downloaded : {ok}
Skipped    : {skip}  (already had)
Failed     : {fail}

If you got many failures, run with --headed to watch the UI flow and see
where the click sequence breaks.  The selectors may need tuning for the
exact button labels gameknot uses.

Output     : {out_dir}
Index      : {index_path}

Next step:
    python tools/parse_annotated_pgn.py --input "{out_dir}" --output data/training_raw.jsonl
""")


if __name__ == "__main__":
    main()
