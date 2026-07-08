#!/usr/bin/env python3
"""
parse_annotated_pgn.py  —  Convert annotated PGN files into training data
--------------------------------------------------------------------------
Walks every .pgn file in --input, extracts (FEN, annotation, themes) for
every annotated move, and writes one JSONL record per example.

WHAT COUNTS AS ONE EXAMPLE
    Any move in a PGN game that has a { comment } of at least 25 characters.
    The FEN is the board BEFORE the move was played (what the annotator saw).

DATA FORMAT  (one JSON object per line in the output file)
    {
      "fen":        "r1bq1rk1/...",        # board position FEN
      "move_san":   "Nd5",                 # the move played from this position
      "move_uci":   "f3d5",               # UCI form
      "annotation": "The knight reaches its ideal outpost...",
      "themes":     ["outpost", "blockade"],  # extracted chess concepts (may be [])
      "phase":      "middlegame",          # opening / middlegame / endgame
      "fullmove":   14,                    # move number
      "side":       "white",              # whose move it is
      "source":     "my_system.pgn",      # source filename (not full path)
      "game":       "Nimzowitsch vs Systemsson, 1927"  # Event + players
    }

LABELED vs UNLABELED EXAMPLES
    Labeled   — themes list is non-empty  → used to train the concept classifier
    Unlabeled — themes list is []         → still valuable for the retrieval layer
                                            and future LLM fine-tuning

Usage
-----
    # Parse a single file
    python tools/parse_annotated_pgn.py --input data/annotated_pgns/my_system.pgn

    # Parse an entire directory (recurses into subdirectories)
    python tools/parse_annotated_pgn.py --input data/annotated_pgns/

    # Append to an existing dataset (safe to run multiple times)
    python tools/parse_annotated_pgn.py --input data/annotated_pgns/ --append

Dependencies
------------
    python -m pip install chess
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import chess
    import chess.pgn
except ImportError:
    sys.exit("Run:  python -m pip install chess")


# ── concept keyword map ───────────────────────────────────────────────────────
# Each key is the theme label stored in the dataset.
# Each value is a list of substrings to search for (case-insensitive) in comments.
# Add new themes freely — they flow into the classifier's output layer automatically.

CONCEPT_KEYWORDS: dict[str, list[str]] = {

    # ── Tactical themes ───────────────────────────────────────────────────────
    "pin": [
        "pin", "pinned", "absolute pin", "relative pin", "pinning",
    ],
    "fork": [
        "fork", "double attack", "forking",
    ],
    "skewer": [
        "skewer", "skewering",
    ],
    "discovered_attack": [
        "discovered attack", "discovery", "unmasking",
    ],
    "deflection": [
        "deflect", "deflection", "lure away", "draw away",
    ],
    "decoy": [
        "decoy", "lure", "entice",
    ],
    "overloading": [
        "overloaded", "overloading", "too many duties", "cannot defend both",
        "guard too many", "protecting too much"
    ],
    "zwischenzug": [
        "zwischenzug", "in-between move", "intermezzo", "intermediate move",
    ],
    "interference": [
        "interference", "interfer",
    ],
    "clearance": [
        "clearance", "clearing the",
    ],
    "back_rank": [
        "back rank", "back-rank", "first rank mate", "back rank weakness",
    ],
    "sacrifice": [
        "sacrifice", "sacrific",
        "sac ",                    # "the sac on h7" (space prevents matching "sack")
        "gives the exchange",
        "offers the exchange",
    ],
    "exchange_sacrifice": [
        "exchange sacrifice", "rook for bishop", "rook for knight",
        "sacrifices the exchange",
    ],
    "combination": [
        "combination", "combinational", "tactical sequence",
    ],
    "mating_attack": [
        "mate", "checkmate", "mating", "mating net", "mating attack",
    ],
    "trapped_piece": [
        "trapped", "has no escape", "caught", "no retreat",
    ],

    # ── Piece concepts ────────────────────────────────────────────────────────
    "outpost": [
        "outpost", "outpost square", "ideal square", "ideal post",
        "cannot be dislodged", "cannot be driven away", "cannot be chased",
        "no way to drive", "strong post", "permanent post",
        "strong position for the knight", "knight cannot be",
        "knight stands well", "knight is well placed",
    ],
    "blockade": [
        "blockade", "blockader", "blockading", "sitting in front",
        "blocked in", "block the pawn",
    ],
    "bad_bishop": [
        "bad bishop", "bad piece", "problem child", "shut in", "shut out",
        "imprisoned bishop", "wrong colored bishop", "blocked bishop",
        "passive bishop", "inactive bishop", "locked bishop",
        "bishop is inferior", "bishop inside the pawn chain",
        "bishop condemned", "bishop is passive",
    ],
    "good_bishop": [
        "good bishop", "active bishop", "activates the white bishop", "activates the black bishop"
    ],
    "bishop_pair": [
        "bishop pair", "two bishops", "pair of bishops",
    ],
    "piece_activity": [
        "active", "activity", "active piece", "activation", "piece play",
        "centralize", "centralized", "centralizing", "mobilize",
        "active bishop", "passive bishop", "active role",
        "aggressively posted", "well posted", "actively placed",
        "bishop proves", "bishop is strong", "bishop is superior",
        "piece becomes active", "pieces become active",
    ],
    "overprotection": [
        "overprotect", "overprotection", "over-protect",
    ],
    "battery": [
        "battery", "doubled rooks", "queen and rook", "rook and queen",
        "doubling on",
    ],
    "rook_seventh": [
        "seventh rank", "seventh row", "on the seventh", "rook on the 7",
        "absolute seventh",
    ],
    "rook_open_file": [
        "rook on the open", "open file for the rook", "rook seizes",
        "rook occupies", "rook enters",
    ],

    # ── Pawn structure ────────────────────────────────────────────────────────
    "passed_pawn": [
        "passed pawn", "passer", "queening", "promotion", "advancing pawn",
        "unstoppable pawn", "past the pawn",
    ],
    "isolated_pawn": [
        "isolated", "isolating", "IQP", "isolated queen pawn", "isolated pawn",
    ],
    "backward_pawn": [
        "backward pawn", "backward d-pawn", "cannot advance", "cannot be supported",
    ],
    "doubled_pawn": [
        "doubled pawn", "doubled pawns",
    ],
    "pawn_majority": [
        "pawn majority", "queenside majority", "kingside majority",
        "mobile majority",
    ],
    "pawn_chain": [
        "pawn chain", "chain", "head of the chain", "base of the chain",
        "pawn wedge",
    ],
    "pawn_break": [
        "pawn break", "pawn advance", "break through", "break open",
    ],
    "pawn_storm": [
        "pawn storm", "pawn advance", "storming", "pawn avalanche",
    ],
    "pawn_weakness": [
        "pawn weakness", "weak pawn", "pawn defect",
    ],
    "pawn_island": [
        "pawn island", "isolated group",
    ],

    # ── King and structure ────────────────────────────────────────────────────
    "king_safety": [
        "king safety", "king weakness", "weakened king", "exposed king",
        "attack on the king", "king in danger", "king under attack",
    ],
    "king_activity": [
        "king march", "king to the center", "active king", "king joins",
        "king walk", "king becomes active",
    ],
    "weak_square": [
        "weak square", "color weakness", "weakened squares", "hole",
        "weak on the", "light square weakness", "dark square weakness",
    ],
    "open_file": [
        "open file", "half-open", "semi-open file", "open d-file",
        "open e-file", "open c-file",
    ],

    # ── Strategic themes ──────────────────────────────────────────────────────
    "space_advantage": [
        "space advantage", "more space", "cramped", "lack of space",
        "restricts", "restricted", "constricted",
    ],
    "initiative": [
        "initiative", "keeps the initiative", "seizes the initiative",
        "maintains pressure", "attacking chances",
    ],
    "tempo": [
        "tempo", "tempi", "gains a tempo", "loss of time", "wasted tempo", "gains time", "gain of tempo", "loss of tempo"
    ],
    "zugzwang": [
        "zugzwang", "compulsion to move", "any move worsens", "forced to move"
    ],
    "prophylaxis": [
        "prophylaxis", "prophylactic", "prevents", "stops the threat",
        "anticipate", "forestall", "in order to prevent",
        "restraining", "restraint", "restrain the",
        "to prevent", "avoiding", "denying",
    ],
    "minority_attack": [
        "minority attack",
    ],
    "simplification": [
        "simplif",                 # simplify, simplification, simplified
        "swap off",                # "swap off the knights"
        "trade off all",           # "trade off all the pieces"
        "reduce to a",             # "reduce to a winning endgame"
        "into a won endgame",
        "liquidat",                # liquidate, liquidation
    ],
    "positional_sacrifice": [
        "positional sacrifice", "long-term sacrifice", "positional pawn sacrifice",
    ],
    "fortification": [
        "fortress", "fortify", "impregnable", "cannot be broken",
    ],
    "coordination": [
        "coordination", "cooperate", "harmonious", "pieces work together",
    ],
    "color_complex": [
        "color complex", "light squares", "dark squares", "color weakness",
        "wrong color",
    ],
    "endgame_technique": [
        "technique", "technical", "conversion", "realize the advantage", "convert the advantage",
    ],
    "opposition": [
        "opposition", "key square", "king opposition",
    ],

    # ── New themes ────────────────────────────────────────────────────────────
    "counterplay": [
        "counterplay", "counter-play", "counter play",
        "compensation", "compensat",
        "counter chances", "counter-attack", "counter attack",
        "dynamic chances", "sufficient compensation",
        "in return", "good counterchances",
    ],

    "development_lead": [
        "lead in development", "development advantage", "better development",
        "ahead in development", "lagging in development",
        "behind in development", "developmental lead",
        "undeveloped", "poorly developed", "rapid development",
        "quick development", "lost time", "waste time",
        "gain in development",
    ],

    "attacking_chances": [
        "kingside attack", "king-side attack", "king side attack",
        "queenside attack", "queen-side attack", "queen side attack",
        "attack on the king", "attacking chances", "assault",
        "attacking possibilities", "attacking play", "attacking position",
        "attack steadily", "press the attack", "dangerous attack",
        "the attack is", "start a", "launch",
    ],

    "square_control": [
        "control of", "controls the", "control the",
        "dominate", "dominates", "key square",
        "important square", "occupy", "occupies the",
        "seize", "seizes", "strong point",
        "command of", "commands the",
    ],
}


# ── helpers ───────────────────────────────────────────────────────────────────

_BOARD_MARKER_RE = re.compile(r'\[%[^\]]+\]')

def clean_comment(raw: str) -> str:
    """Strip board markers and normalize whitespace."""
    c = _BOARD_MARKER_RE.sub(' ', raw)
    return re.sub(r'\s+', ' ', c).strip()


def extract_themes(annotation: str) -> list[str]:
    """Return sorted list of matching concept labels."""
    text = annotation.lower()
    return sorted(
        theme for theme, keywords in CONCEPT_KEYWORDS.items()
        if any(kw in text for kw in keywords)
    )


def get_phase(board: chess.Board) -> str:
    queens = (len(board.pieces(chess.QUEEN, chess.WHITE))
              + len(board.pieces(chess.QUEEN, chess.BLACK)))
    pieces = len(board.piece_map())
    if board.fullmove_number <= 10:
        return "opening"
    if queens == 0 or pieces <= 12:
        return "endgame"
    return "middlegame"


def game_header(game: chess.pgn.Game) -> str:
    white  = game.headers.get("White",  "?")
    black  = game.headers.get("Black",  "?")
    date   = game.headers.get("Date",   "?")[:4]
    event  = game.headers.get("Event",  "")
    parts  = [f"{white} vs {black}"]
    if date and date != "?":
        parts.append(date)
    if event:
        parts.append(event)
    return ", ".join(parts)


# ── PGN parsing ───────────────────────────────────────────────────────────────

def parse_file(pgn_path: Path, min_comment_len: int = 25, progress_every: int = 500):
    """
    Generator — yields one training example dict at a time.
    Prints a progress line every `progress_every` games so you can see it's alive.
    """
    import time
    source     = pgn_path.name
    games_read = 0
    examples_yielded = 0
    t_start    = time.time()

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            games_read += 1
            if games_read % progress_every == 0:
                elapsed = time.time() - t_start
                rate    = games_read / elapsed if elapsed > 0 else 0
                print(f"\r    {games_read:,} games read  |  {examples_yielded:,} examples  |  "
                      f"{rate:.0f} games/min  |  {elapsed/60:.1f} min elapsed",
                      end="", flush=True)

            header_str = game_header(game)
            board      = game.board()
            node       = game

            while node.variations:
                node       = node.variations[0]
                fen_before = board.fen()
                move       = node.move
                board.push(move)

                comment = clean_comment(node.comment)
                if len(comment) < min_comment_len:
                    continue

                themes = extract_themes(comment)
                phase  = get_phase(chess.Board(fen_before))

                try:
                    san = chess.Board(fen_before).san(move)
                except Exception:
                    san = move.uci()

                examples_yielded += 1
                yield {
                    "fen":        fen_before,
                    "move_san":   san,
                    "move_uci":   move.uci(),
                    "annotation": comment,
                    "themes":     themes,
                    "phase":      phase,
                    "fullmove":   board.fullmove_number,
                    "side":       "white" if not board.turn else "black",
                    "source":     source,
                    "game":       header_str,
                }

    elapsed = time.time() - t_start
    print(f"\r    {games_read:,} games read  |  {examples_yielded:,} examples  |  "
          f"done in {elapsed/60:.1f} min                    ")


# ── statistics ────────────────────────────────────────────────────────────────

def print_stats(examples: list[dict], out_path: Path) -> None:
    labeled   = [e for e in examples if e["themes"]]
    unlabeled = [e for e in examples if not e["themes"]]

    theme_counts: Counter = Counter()
    for e in labeled:
        theme_counts.update(e["themes"])

    phase_counts: Counter = Counter(e["phase"] for e in examples)

    source_counts: Counter = Counter()
    for e in examples:
        source_counts[e["source"]] += 1

    multi_labeled = [e for e in labeled if len(e["themes"]) >= 2]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║               TRAINING DATASET STATISTICS                   ║
╚══════════════════════════════════════════════════════════════╝

Total examples    : {len(examples):,}
  Labeled         : {len(labeled):,}   ← usable for classifier training
  Unlabeled       : {len(unlabeled):,}   ← usable for retrieval / LLM fine-tuning
  Multi-labeled   : {len(multi_labeled):,}   ← have ≥2 themes (richest examples)

Viability check
  Minimum (5,000 labeled) : {"✓  PASS" if len(labeled) >= 5_000  else f"✗  need {5_000  - len(labeled):,} more"}
  Good     (50,000 labeled): {"✓  PASS" if len(labeled) >= 50_000 else f"✗  need {50_000 - len(labeled):,} more"}

Phase distribution
  Opening    : {phase_counts['opening']:,}
  Middlegame : {phase_counts['middlegame']:,}
  Endgame    : {phase_counts['endgame']:,}

Top 20 themes (of {len(theme_counts)} total)""")

    for theme, count in theme_counts.most_common(20):
        bar = "█" * min(40, count // max(1, len(labeled) // 400))
        print(f"  {theme:<25} {count:>5}  {bar}")

    print(f"""
Source files  ({len(source_counts)} files)""")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {src:<50} {count:>5} examples")

    print(f"""
Output written to: {out_path}
""")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse annotated PGN files into training data (JSONL format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to a .pgn file or a directory of .pgn files.")
    parser.add_argument("--output", "-o",
                        default="data/training_raw.jsonl",
                        help="Output JSONL file (default: data/training_raw.jsonl).")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output instead of overwriting.")
    parser.add_argument("--min-comment", type=int, default=25,
                        help="Minimum comment length in characters (default 25).")
    parser.add_argument("--count-only", action="store_true",
                        help="Just count and report stats, don't write output file.")
    args = parser.parse_args()

    # ── find PGN files ────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if input_path.is_dir():
        pgn_files = sorted(input_path.rglob("*.pgn"))
    elif input_path.is_file():
        pgn_files = [input_path]
    else:
        sys.exit(f"Not found: {args.input}")

    print(f"Found {len(pgn_files)} PGN file(s) in {args.input}")

    out_path = Path(args.output)
    if not args.count_only:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── stream parse → write  (no full list in RAM) ───────────────────────────
    # Track stats with counters instead of holding every example in memory.
    total = labeled_n = unlabeled_n = multi_n = 0
    theme_counts: Counter  = Counter()
    phase_counts: Counter  = Counter()
    source_counts: Counter = Counter()

    mode = "a" if args.append else "w"
    out_f = open(out_path, mode, encoding="utf-8") if not args.count_only else None

    try:
        for i, pgn_path in enumerate(pgn_files, 1):
            print(f"\n  [{i}/{len(pgn_files)}] {pgn_path.name}")
            try:
                for ex in parse_file(pgn_path, args.min_comment):
                    if out_f:
                        out_f.write(json.dumps(ex) + "\n")

                    total += 1
                    themes = ex["themes"]
                    if themes:
                        labeled_n += 1
                        theme_counts.update(themes)
                        if len(themes) >= 2:
                            multi_n += 1
                    else:
                        unlabeled_n += 1
                    phase_counts[ex["phase"]] += 1
                    source_counts[ex["source"]] += 1

            except Exception as exc:
                print(f"\n    ERROR in {pgn_path.name}: {exc}")
    finally:
        if out_f:
            out_f.close()

    if total == 0:
        sys.exit("\nNo examples extracted. Check that your PGN files have { comments }.")

    # ── print statistics ──────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║               TRAINING DATASET STATISTICS                   ║
╚══════════════════════════════════════════════════════════════╝

Total examples    : {total:,}
  Labeled         : {labeled_n:,}   ← usable for classifier training
  Unlabeled       : {unlabeled_n:,}   ← usable for retrieval / LLM fine-tuning
  Multi-labeled   : {multi_n:,}   ← have ≥2 themes (richest examples)

Viability check
  Minimum (5,000 labeled) : {"✓  PASS" if labeled_n >= 5_000  else f"✗  need {5_000  - labeled_n:,} more"}
  Good     (50,000 labeled): {"✓  PASS" if labeled_n >= 50_000 else f"✗  need {50_000 - labeled_n:,} more"}

Phase distribution
  Opening    : {phase_counts['opening']:,}
  Middlegame : {phase_counts['middlegame']:,}
  Endgame    : {phase_counts['endgame']:,}

Top 20 themes (of {len(theme_counts)} total)""")

    for theme, count in theme_counts.most_common(20):
        bar = "█" * min(40, count // max(1, labeled_n // 400))
        print(f"  {theme:<25} {count:>5}  {bar}")

    print(f"\nSource files  ({len(source_counts)} files)")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {src:<50} {count:>5} examples")

    if not args.count_only:
        print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()
