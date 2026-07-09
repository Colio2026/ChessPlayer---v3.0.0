#!/usr/bin/env python3
"""
scrape_lichess_studies.py  --  Download Lichess studies for positional concept training data
---------------------------------------------------------------------------------------------
Searches Lichess study search (JSON API) for each concept using broad keyword terms,
downloads the full PGN for each study found, and saves organised by concept folder.

The search endpoint returns 16 results per page. We paginate until we hit max_per_concept
or run out of results. Each study is filtered by minimum prose character count so
purely diagrammatic or empty studies are skipped.

Output
------
    data/annotated_pgns/lichess_studies/
        outpost/
            batch_0001.pgn
        isolated_pawn/
            ...
        _index.jsonl

Usage
-----
    # All 22 concepts that lack puzzle data (default)
    python assets/scraping/scrape_lichess_studies.py --token lip_xxxx

    # Specific concepts
    python assets/scraping/scrape_lichess_studies.py --concepts outpost,blockade --token lip_xxxx

    # More studies per concept (default 200)
    python assets/scraping/scrape_lichess_studies.py --max-per-concept 400 --token lip_xxxx

Auth token
----------
    lichess.org/account/oauth/token  (tick "study:read" for private access)

Dependencies
------------
    python -m pip install requests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Run:  python -m pip install requests")

# ── config ────────────────────────────────────────────────────────────────────

OUTPUT_ROOT  = Path("data/annotated_pgns/lichess_studies")
INDEX_PATH   = OUTPUT_ROOT / "_index.jsonl"
BATCH_SIZE    = 20
MIN_PROSE     = 2000   # minimum prose chars — only keep well-annotated studies
MIN_GAMES     = 10     # OR keep if study has this many games (puzzle collections)
BASE_DELAY    = 1.5    # seconds between requests normally
BACKOFF_DELAY = 120    # seconds to sleep after a 429

SEARCH_URL  = "https://lichess.org/study/search"
STUDY_URL   = "https://lichess.org/api/study/{sid}.pgn"

# Headers that make Lichess return JSON instead of the React shell
JSON_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Accept":           "application/json",
}

# ── concept → search terms ────────────────────────────────────────────────────
# Multiple terms per concept. Searches are run in order; results are deduplicated.
# Broader terms come last so the most specific ones fill the quota first.

CONCEPT_SEARCHES: dict[str, list[str]] = {

    # pawn structure cluster
    "isolated_pawn": [
        "isolated pawn", "IQP", "isolated queen pawn", "isolani",
        "isolated d pawn", "pawn isolation",
    ],
    "backward_pawn": [
        "backward pawn", "backward d pawn", "backward c pawn",
        "weak pawn backward",
    ],
    "doubled_pawn": [
        "doubled pawns", "doubled pawn weakness", "pawn structure doubled",
    ],
    "pawn_majority": [
        "pawn majority", "queenside majority", "kingside majority",
        "mobile majority",
    ],
    "pawn_chain": [
        "pawn chain", "nimzowitsch pawn", "pawn wedge", "pawn formation",
    ],
    "pawn_weakness": [
        "pawn weakness", "weak pawns", "pawn defect", "pawn structure weakness",
    ],
    "pawn_island": [
        "pawn islands", "scattered pawns", "disconnected pawns",
    ],

    # piece concepts cluster
    "outpost": [
        "outpost", "knight outpost", "ideal square", "knight domination",
        "strong square knight", "support point",
    ],
    "blockade": [
        "blockade", "nimzowitsch blockade", "pawn blockade",
        "blockader", "knight blockade",
    ],
    "good_bishop": [
        "good bishop", "active bishop", "bishop vs knight open",
        "bishop outshines", "two bishops endgame",
    ],
    "bishop_pair": [
        "bishop pair", "two bishops", "pair of bishops",
        "bishop pair advantage", "retain bishops",
    ],
    "exchange_sacrifice": [
        "exchange sacrifice", "rook for bishop", "rook for knight",
        "petrosian exchange", "positional exchange sacrifice",
    ],
    "overloading": [
        "overloading", "overloaded piece", "overloaded defender",
        "piece overloaded", "defending too much",
    ],

    # positional / strategic cluster
    "open_file": [
        "open file", "rook open file", "file control", "rook file",
        "half open file", "seize the file",
    ],
    "weak_square": [
        "weak square", "color weakness", "hole in position",
        "weak dark squares", "weak light squares",
    ],
    "space_advantage": [
        "space advantage", "spatial advantage", "cramped position",
        "lack of space", "squeeze position",
    ],
    "color_complex": [
        "color complex", "colour complex", "light square strategy",
        "dark square strategy", "wrong color bishop",
    ],
    "coordination": [
        "piece coordination", "piece harmony", "coordinated pieces",
        "harmonious position", "pieces work together",
    ],
    "simplification": [
        "simplification", "liquidation", "trading advantage",
        "simplify to win", "convert advantage endgame",
    ],
    "square_control": [
        "square control", "key square control", "central control",
        "dominant square", "occupy key square",
    ],

    # concepts that lost their puzzle source in the last scraper update
    "counterplay": [
        "counterplay", "dynamic compensation", "active counterplay",
        "compensation for pawn", "fighting back",
    ],
    "attacking_chances": [
        "kingside attack strategy", "attacking chances", "build attack",
        "attacking play positional",
    ],
}


# ── helpers ───────────────────────────────────────────────────────────────────

_BOARD_MARKER_RE = re.compile(r'\[%[^\]]+\]')


def prose_char_count(pgn_text: str) -> int:
    total = 0
    for block in re.findall(r'\{([^}]*)\}', pgn_text):
        clean = _BOARD_MARKER_RE.sub('', block)
        clean = re.sub(r'[\s!?+#\-=<>]+', ' ', clean).strip()
        if re.search(r'[a-zA-Z]{3,}', clean):
            total += len(clean)
    return total


def game_count(pgn_text: str) -> int:
    return pgn_text.count("[Event ")


def content_hash(pgn_text: str) -> str:
    """Hash just the prose comments so duplicate studies (re-uploaded by others) are caught."""
    prose = " ".join(re.findall(r'\{([^}]+)\}', pgn_text))
    return hashlib.md5(prose.encode("utf-8", errors="replace")).hexdigest()


def is_worth_keeping(pgn_text: str) -> tuple[bool, str]:
    """Return (keep, reason). Keep if long annotation OR many games (puzzle collection)."""
    prose = prose_char_count(pgn_text)
    games = game_count(pgn_text)
    if prose >= MIN_PROSE:
        return True, f"{prose:,} prose chars"
    if games >= MIN_GAMES:
        return True, f"{games} games (puzzle set)"
    return False, f"sparse({prose} chars, {games} games)"


def search_page(session: requests.Session, query: str, page: int) -> list[str]:
    """Return study IDs from one page of search results. Empty list = no more pages."""
    try:
        resp = session.get(
            SEARCH_URL,
            params={"q": query, "order": "hot", "page": page},
            headers=JSON_HEADERS,
            timeout=20,
        )
    except Exception as e:
        print(f"    search error: {e}")
        return []

    if resp.status_code == 429:
        print(f"    rate-limited on search — sleeping {BACKOFF_DELAY}s...")
        time.sleep(BACKOFF_DELAY)
        return search_page(session, query, page)
    if resp.status_code != 200:
        print(f"    search HTTP {resp.status_code}")
        return []

    try:
        data = resp.json()
        results = data.get("paginator", {}).get("currentPageResults", [])
        return [r["id"] for r in results if "id" in r]
    except Exception as e:
        print(f"    JSON parse error: {e}")
        return []


def download_study(session: requests.Session, sid: str) -> tuple[str | None, str]:
    """Download one study PGN. Returns (pgn_text | None, status_string)."""
    try:
        resp = session.get(
            STUDY_URL.format(sid=sid),
            timeout=30,
            headers={"Accept": "application/x-chess-pgn"},
        )
    except Exception as e:
        return None, f"error:{e}"

    if resp.status_code in (401, 403, 404):
        return None, "private"
    if resp.status_code == 429:
        print(f"    rate-limited on download — sleeping {BACKOFF_DELAY}s...")
        time.sleep(BACKOFF_DELAY)
        return download_study(session, sid)
    if resp.status_code != 200:
        return None, f"error:HTTP {resp.status_code}"

    text = resp.text
    if "[Event" not in text and "[White" not in text:
        return None, "error:not-pgn"

    keep, reason = is_worth_keeping(text)
    if not keep:
        return None, reason

    return text, "ok"


def load_index() -> dict[str, set[str]]:
    done: dict[str, set[str]] = defaultdict(set)
    if INDEX_PATH.exists():
        for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[rec["concept"]].update(rec.get("study_ids", []))
            except Exception:
                pass
    return dict(done)


def append_index(idx_file, concept: str, batch_num: int, study_ids: list[str]) -> None:
    idx_file.write(json.dumps({
        "concept":   concept,
        "batch":     batch_num,
        "study_ids": study_ids,
    }) + "\n")
    idx_file.flush()


def write_batch(folder: Path, batch_num: int, pgn_list: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"batch_{batch_num:04d}.pgn").write_text(
        "\n\n".join(pgn_list), encoding="utf-8"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if args.concepts:
        selected = [c.strip() for c in args.concepts.split(",")]
        unknown  = [c for c in selected if c not in CONCEPT_SEARCHES]
        if unknown:
            print(f"Unknown concepts (ignored): {unknown}")
        active = {c: CONCEPT_SEARCHES[c] for c in selected if c in CONCEPT_SEARCHES}
    else:
        active = dict(CONCEPT_SEARCHES)

    session = requests.Session()
    session.headers["User-Agent"] = "ChessCoachTrainer/3.0 (education)"
    if args.token:
        session.headers["Authorization"] = f"Bearer {args.token}"
        print(f"Token set. Base delay: {BASE_DELAY}s, backoff on 429: {BACKOFF_DELAY}s.")
    else:
        print(f"No token. Base delay: {BASE_DELAY}s. Get one at lichess.org/account/oauth/token")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    done = load_index()
    seen_hashes: set[str] = set()   # cross-concept duplicate detection
    total_saved = 0

    with open(INDEX_PATH, "a", encoding="utf-8") as idx_file:
        for concept, queries in active.items():
            already_saved = done.get(concept, set())
            print(f"\n{'='*55}")
            print(f"  {concept}  ({len(already_saved)} already saved)")
            print(f"{'='*55}")

            candidate_ids: list[str] = []
            seen_ids: set[str] = set(already_saved)

            for query in queries:
                if len(candidate_ids) + len(already_saved) >= args.max_per_concept:
                    break
                print(f"  Searching '{query}'...")
                page = 1
                while True:
                    ids = search_page(session, query, page)
                    time.sleep(BASE_DELAY)
                    if not ids:
                        break
                    new = [s for s in ids if s not in seen_ids]
                    for sid in new:
                        seen_ids.add(sid)
                        candidate_ids.append(sid)
                    print(f"    page {page}: {len(ids)} results, {len(new)} new")
                    if len(ids) < 16:
                        break
                    page += 1
                    if len(candidate_ids) + len(already_saved) >= args.max_per_concept * 3:
                        break

            if not candidate_ids:
                print(f"  No new studies found.")
                continue

            print(f"  {len(candidate_ids)} candidates to download...")

            folder    = OUTPUT_ROOT / concept
            batch_num = (len(already_saved) // BATCH_SIZE) + 1
            pgn_buf:  list[str] = []
            id_buf:   list[str] = []
            saved = private = sparse = dupes = 0

            for sid in candidate_ids:
                if saved + len(already_saved) >= args.max_per_concept:
                    break

                pgn, status = download_study(session, sid)
                time.sleep(BASE_DELAY)

                if status == "ok":
                    h = content_hash(pgn)
                    if h in seen_hashes:
                        dupes += 1
                        print(f"  = {sid}  (duplicate content, skipped)")
                        continue
                    seen_hashes.add(h)

                    _, reason = is_worth_keeping(pgn)
                    pgn_buf.append(pgn.strip())
                    id_buf.append(sid)
                    saved += 1
                    print(f"  + {sid}  ({reason})")

                    if len(pgn_buf) >= BATCH_SIZE:
                        write_batch(folder, batch_num, pgn_buf)
                        append_index(idx_file, concept, batch_num, id_buf)
                        print(f"    -> saved batch_{batch_num:04d}.pgn")
                        pgn_buf = []
                        id_buf  = []
                        batch_num += 1

                elif status == "private":
                    private += 1
                else:
                    print(f"  - {sid}  ({status})")
                    sparse += 1

            if pgn_buf:
                write_batch(folder, batch_num, pgn_buf)
                append_index(idx_file, concept, batch_num, id_buf)
                print(f"    -> saved batch_{batch_num:04d}.pgn")

            total_saved += saved
            print(f"  Saved: {saved}  |  dupes: {dupes}  |  private: {private}  |  sparse: {sparse}")

    print(f"""
{'='*55}
Done. Total studies saved: {total_saved}
Output: {OUTPUT_ROOT}

Next steps:
    python tools/parse_annotated_pgn.py \\
        --input data/annotated_pgns/lichess_studies \\
        --output data/training_raw.jsonl --append
    python -m src.chess_coach.ml.train
""")


def main() -> None:
    global MIN_PROSE
    parser = argparse.ArgumentParser(
        description="Download Lichess studies for chess concept training data"
    )
    parser.add_argument("--token",           default="",
                        help="Lichess OAuth token (study:read scope)")
    parser.add_argument("--concepts",        default="",
                        help="Comma-separated concepts to target (default: all 22)")
    parser.add_argument("--max-per-concept", type=int, default=200,
                        help="Max studies to save per concept (default 200)")
    parser.add_argument("--min-prose",       type=int, default=MIN_PROSE,
                        help=f"Min prose chars to keep a study (default {MIN_PROSE})")
    args = parser.parse_args()
    MIN_PROSE = args.min_prose
    run(args)


if __name__ == "__main__":
    main()
