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
        "break the pin", "breaking the pin", "unpin", "unpinned",
        "bound hand and foot",           # "bound hand and foot" = completely pinned
        "nail",                          # "nailed down", "nailed to"
        "pin is decisive", "pin decides",
        "exploit the pin", "maintain the pin",
        "the pin on",                    # "the pin on the rook", "the pin on f3"
        "absolute pin",                  # repeated from above but common phrase
        "pin to the king", "pinned to the king",
    ],
    "fork": [
        "fork", "double attack", "forking",
        "knight fork", "royal fork", "rook fork",  # named fork types
        "attacks two pieces", "attacks two",        # describing a fork without the word
        "simultaneous attack on",
        "threatens two", "win two pieces",
        "fork the king", "fork threat",
        "forks the",                               # "forks the queen and rook"
    ],
    "skewer": [
        "skewer", "skewering",
        "reverse pin",                  # technical alternate name for skewer
        "skewers the",                  # "skewers the rook", "skewers the queen"
        "attacks through",              # "the bishop attacks through the rook"
        "forced to move to avoid",      # "king forced to move to avoid losing"
        "in front of the king",         # piece is in front → skewer potential
    ],
    "discovered_attack": [
        "discovered attack", "discovery", "unmasking",
        "x-ray", "x ray attack",                        # Lichess: xRayAttack
        "double check",                                  # Lichess: doubleCheck
        "discovered check",                             # Lichess: discoveredCheck
        "reveal", "reveals a check", "uncovers",
    ],
    "deflection": [
        "deflect", "deflection", "lure away", "draw away",
        "capturing defender", "capturing the defender",  # Lichess: capturingDefender
    ],
    "decoy": [
        "decoy", "lure", "entice",
        "attraction",                                    # Lichess: attraction
        "draw the king",
        "force the king",
        "lure the king",
        "tempt",
        "bait",
        "draw the queen",
        "pulled to",
        "drag the king",
    ],
    "overloading": [
        "overloaded", "overloading", "too many duties", "cannot defend both",
        "guard too many", "protecting too much",
        "two duties", "two tasks", "serving two",
        "overtaxed", "over-taxed", "over taxed",
        "juggling", "cannot be in two places",
        "defend two", "protect two",
    ],
    "zwischenzug": [
        "zwischenzug", "in-between move", "intermezzo", "intermediate move",
        "in between move", "interpolat",
        "before recapturing", "before taking",
        "before capturing",
        "check first", "with check first",
        "plays in between", "prior to recapturing",
        "intermediate check", "in-between check",
        "before the capture", "before taking back",
        "first wins", "intermediate winning",
    ],
    "interference": [
        "interference", "interfer",
        "blocks the line",             # "blocks the line of communication"
        "cuts off",                    # "cuts off the rook from defending"
        "interposing",
        "intercept",
        "blocking the connection",
        "cutting off communication",
        "interrupt",
        "blocks the diagonal",
        "blocks the file",
        "block the connection",
    ],
    "clearance": [
        "clearance", "clearing the",
        "vacate", "vacating",
        "clear the square", "clears the square",
        "clear the diagonal", "clears the diagonal",
        "clear the file", "clears the file",
        "make room for",
        "vacates",
        "clear the way", "clears the way",
        "remove from the diagonal",
        "empty the square",
    ],
    "back_rank": [
        "back rank", "back-rank", "backrank", "first rank mate", "back rank weakness",
        "back rank mate",                             # Lichess: backRankMate
    ],
    "sacrifice": [
        "sacrifice", "sacrific",
        "sac ",                    # "the sac on h7" (space prevents matching "sack")
        "gives the exchange",
        "offers the exchange",
        # merged from positional_sacrifice
        "positional sacrifice", "long-term sacrifice", "positional pawn sacrifice",
        "strategic sacrifice", "long term compensation",
        "material is only temporary", "gives up material for",
        "long-term compensation", "long term investment",
        "sacrifices a pawn for", "pawn is only temporary",
    ],
    "exchange_sacrifice": [
        "exchange sacrifice", "rook for bishop", "rook for knight",
        "sacrifices the exchange",
        "gives up the exchange",
        "positional exchange sacrifice",
        "rook versus bishop", "rook versus knight",
        "give the exchange",
        "giving the exchange",
        "exchange for positional",
        "exchange for compensation",
        "exchange for a pawn",
        "exchange is justified",
    ],
    "combination": [
        "combination", "combinational",
        "brilliant combination", "beautiful combination",
        "winning combination",
        "tactical motif",
        "forcing sequence", "forced sequence",
        "series of forcing moves",
        "tactical blow",
    ],
    "mating_attack": [
        "checkmate", "mating", "mating net", "mating attack",
        "smothered mate", "smothered",
        "arabian mate", "anastasia", "hook mate", "boden", "epaulette mate",
        "pillsbury", "opera mate", "morphy's mate", "morphy mate",
        "vukovic", "corner mate", "swallow's tail", "swallowstail",
        "blind swine", "triangle mate", "balestra",
    ],
    "trapped_piece": [
        "trapped", "has no escape", "no retreat",
        "no safe square", "no good square", "nowhere to go",
        "cannot escape", "piece is lost", "piece cannot move",
    ],

    # ── Piece concepts ────────────────────────────────────────────────────────
    "outpost": [
        "outpost", "outpost square", "ideal square", "ideal post",
        "cannot be dislodged", "cannot be driven away", "cannot be chased",
        "no way to drive", "strong post", "permanent post",
        "strong position for the knight", "knight cannot be",
        "knight stands well", "knight is well placed",
        "support point",               # Nimzowitsch / Steinitz term for outpost-like square
        "stronghold",                  # "stronghold on e5", "stronghold for the knight"
        "cannot be forced away",
        "knight is firmly placed",
        "no way to attack",            # "no way to attack the knight"
        "denying the opposing pieces",
        "strong square for",
    ],
    "blockade": [
        "blockade", "blockader", "blockading", "sitting in front",
        "blocked in", "block the pawn",
        "the blockader", "elastic blockader", "blockading piece",
        "knight blockades", "bishop blockades",
    ],
    "bad_bishop": [
        "bad bishop", "bad piece", "problem child", "shut in", "shut out",
        "imprisoned bishop", "wrong colored bishop", "blocked bishop",
        "passive bishop", "inactive bishop", "locked bishop",
        "bishop is inferior", "bishop inside the pawn chain",
        "bishop condemned", "bishop is passive",
        "bishop blocked by its own pawns",
        "bishop of the wrong color",
        "bishop is blocked",
        "exchange off his bad bishop", "exchange the bad bishop",
        "inferior to the knight",      # bad bishop is inferior to knight in closed positions
        "bishop has no scope",
        "bishop is useless",
        "suffers from bad bishop",
        "same color as",               # "same color as the pawns" = bad bishop
        "color of a bishop",
    ],
    "good_bishop": [
        "good bishop", "active bishop", "activates the white bishop", "activates the black bishop",
        "superior bishop",
        "bishop proves superior",
        "bishop is strong",
        "dominant bishop",
        "bishop is active",
        "bishop controls",
        "bishop sweeps",
        "bishop outshines",
        "bishop reigns",
        "long diagonal",
        "bishop exploits the diagonal",
        "bishop dominates",
        "bishop is superior to the knight",
    ],
    "bishop_pair": [
        "bishop pair", "two bishops", "pair of bishops",
        "the two bishops",
        "bishop pair advantage",
        "give up the bishop pair", "surrendering the bishop pair",
        "relinquish the bishops",
        "bishop pair is strong", "bishop pair proves",
        "both bishops",
        "double bishops",
        "retain the bishops",
        "bishop pair compensates",
    ],
    "piece_activity": [
        "piece play", "activation", "piece activity",
        "centralize", "centralized", "centralizing", "mobilize",
        "aggressively posted", "well posted", "actively placed",
        "piece becomes active", "pieces become active",
        "active rook", "rook becomes active", "activate the rook",
        "activate the knight", "knight becomes active",
        "pieces are active", "pieces become dominant",
        "piece coordination improves",
    ],
    "battery": [
        "battery", "doubled rooks", "queen and rook", "rook and queen",
        "doubling on",
        "stacking", "stack the rook", "stack on", "piling up", "pile up on",
        "rooks on the same", "double on the", "collinear",
        "queen and bishop battery", "bishop battery",
        "rook battery", "queen behind the rook",
        "double the rooks", "rooks are doubled",
        "align along", "lined up",
    ],
    "rook_seventh": [
        "seventh rank", "seventh row", "on the seventh", "rook on the 7",
        "absolute seventh",
        "pig on the", "pigs on the", "pig on 7",
        "second rank",
        "master of the seventh", "dominate the seventh",
        "rook on the seventh", "rooks on the seventh",
        "invades the seventh", "penetrates to the seventh",
        "on the 7th rank", "rook reaches the seventh",
        "two rooks on the seventh", "seventh rank is decisive",
    ],

    # ── Pawn structure ────────────────────────────────────────────────────────
    "passed_pawn": [
        "passed pawn", "passer", "queening", "promotion", "advancing pawn",
        "unstoppable pawn", "past the pawn",
        "underpromot", "under-promot", "queening square",
    ],
    "isolated_pawn": [
        "isolated", "isolating", "IQP", "isolated queen pawn", "isolated pawn",
    ],
    "backward_pawn": [
        "backward pawn", "backward d-pawn",
        "saddled with",
        "laggard pawn",
        "chronic weakness",
        "fixed weakness",
        "immovable pawn",
        "doom of the backward pawn",
        "backward pawn as a weakness",
        "weak pawn on",
        "transfers the weakness",
        "cannot be advanced",
        "square in front of the pawn",
        "pawn cannot move",
        "the pawn is a target",
    ],
    "doubled_pawn": [
        "doubled pawn", "doubled pawns",
        "double the pawns", "doubling the pawns",
        "saddled with doubled",
        "doubled c-pawns", "doubled f-pawns", "doubled b-pawns",
        "structural damage",
        "pawn on the same file",
        "two pawns on the same",
        "weakened pawn structure",
        "broken pawn structure",
    ],
    "pawn_majority": [
        "pawn majority", "queenside majority", "kingside majority",
        "mobile majority",
        "extra pawn",
        "numerical pawn advantage",
        "outnumber the pawns",
        "more pawns on the",
        "passed pawn from the majority",
        "pawn roller",
        "advancing majority",
        "majority on the queenside", "majority on the kingside",
    ],
    "pawn_chain": [
        "pawn chain", "head of the chain", "base of the chain",
        "pawn wedge", "attack the base", "attack the chain",
        "break the chain", "undermine the chain",
    ],
    "pawn_break": [
        "pawn break", "pawn advance", "break through", "break open",
        "liberating move",
        "freeing advance",
        "pawn breaks open",
        "advance the pawn",
        "push the pawn",
        "break the center",
        "en passant",                  # Lichess: enPassant → pawn_break
        "pawn lever",
        "structural break",
        "open the position",
        "explode the center",
    ],
    "pawn_storm": [
        "pawn storm", "pawn advance", "storming", "pawn avalanche",
        "advance the pawns",
        "advance on the kingside",
        "advance on the queenside",
        "attack with pawns",
        "h4-h5", "g4-g5", "f4-f5",    # typical kingside storm moves
        "a4-a5", "b4-b5", "c4-c5",    # typical queenside storm moves
        "pawn roller",
        "rolling pawns",
        "advancing the pawns",
    ],
    "pawn_weakness": [
        "pawn weakness", "weak pawn", "pawn defect",
        "pawn structure weakness",
        "structural defect",
        "weak pawns on",
        "exploiting the weak pawn",
        "target the pawn",
        "attack the weakness",
        "pressure on the pawn",
        "exploit the weakness",
        "probe the weakness",
    ],
    "pawn_island": [
        "pawn island", "isolated group",
        "three pawn islands", "two pawn islands",
        "split pawns",
        "scattered pawns",
        "disconnected pawns",
        "pawn islands advantage",
        "fewer pawn islands",
        "more pawn islands",
    ],

    # ── King and structure ────────────────────────────────────────────────────
    "king_safety": [
        "king safety", "king weakness", "weakened king", "exposed king",
        "attack on the king", "king in danger", "king under attack",
    ],
    "king_activity": [
        "king march", "king to the center", "active king", "king joins",
        "king walk", "king becomes active",
        "shouldering",
        "outflanking", "outflank",
        "king advances", "king penetrates", "king invades",
        "king centralizes", "king supports",
        "king escorts the pawn", "king enters the game",
        "king takes part", "king is a strong piece",
        "king becomes a fighter", "king joins the attack",
        "king marches to", "active use of the king",
        "king activity", "king centralization",
        "king participates", "king in the center",
    ],
    "weak_square": [
        "weak square", "color weakness", "weakened squares", "hole",
        "weak on the", "light square weakness", "dark square weakness",
    ],
    "open_file": [
        "open file", "half-open", "semi-open file", "open d-file",
        "open e-file", "open c-file", "opens the file",
        "half open",
        "opening the file",
        "open the f-file", "open the g-file", "open the b-file",
        "files in the centre",
        "control of the file",
        "file is open",
        "file for his rook",
        "exploit the file",
        # merged from rook_open_file
        "rook on the open", "open file for the rook", "file for the rook",
        "rook seizes", "rook occupies", "rook enters",
        "rooks need open files",
        "occupy the open file", "occupy the file",
        "seize the file", "seizes the file",
        "file belongs to",
        "pressure down the file",
        "control the file", "controls the file", "dominate the file",
        "rook penetrates", "rook dominates the file",
        "rooks have open files", "rooks will penetrate",
        "rook lifts",
        "files for the rooks",
        "open the file",
        "half open file",
    ],

    # ── Strategic themes ──────────────────────────────────────────────────────
    "space_advantage": [
        "space advantage", "more space", "cramped", "lack of space",
        "constricted",
        "squeeze", "squeezing", "squeezed",
        "strangle", "strangled",
        "stifling", "stifle",
        "choke", "choking",
        "suffocating", "suffocation",
        "no room to maneuver", "no maneuvering room", "maneuvering room",
        "lack of elbow room", "elbow room",
        "no room to breathe", "room to breathe",
        "acute lack of space",
        "control of space",
        "greater mobility",
        "pieces lack mobility",
        "free his game", "free the game",
        "cannot maneuver",
        "obstructing each other",
        "pieces obstruct",
        "siege",
    ],
    "tempo": [
        "tempo", "tempi", "gains a tempo", "loss of time", "wasted tempo",
        "gains time", "gain of tempo", "loss of tempo",
        "loses a tempo", "tempo advantage", "with gain of time",
        "time advantage", "with tempo",
    ],
    "zugzwang": [
        "zugzwang", "compulsion to move", "any move worsens", "forced to move",
        "every move loses",
        "all moves are bad",
        "no good moves",
        "in zugzwang",
        "mutual zugzwang",
        "whoever moves loses",
        "must move",
        "move is a liability",
        "right to move",
        "pass would win",
    ],
    "prophylaxis": [
        "prophylaxis", "prophylactic",
        "anticipate", "forestall", "in order to prevent",
        "restraining", "restraint", "restrain the",
        "defensive move",
        "nip in the bud",
        "prevent the advance", "prevent the break",
        "stop the plan", "hinder the plan",
    ],
    "minority_attack": [
        "minority attack", "minority structure",
        "advances the b-pawn", "b4-b5", "b5 against",
        "creates a weakness on c6", "weakness on c6",
        "queenside minority",
    ],
    "simplification": [
        "simplif",                 # simplify, simplification, simplified
        "swap off",                # "swap off the knights"
        "trade off all",           # "trade off all the pieces"
        "reduce to a",             # "reduce to a winning endgame"
        "into a won endgame",
        "liquidat",                # liquidate, liquidation
        "trading down", "trades down",
        "exchange all", "exchanges all",
        "converts the advantage", "technique is simple",
        "into a winning", "transition to",
    ],
    "fortification": [
        "fortress", "fortify", "impregnable", "cannot be broken",
        "defensive fortress", "drawn fortress", "draws by fortress",
        "cannot be infiltrated", "cannot be breached", "cannot break through",
        "defensive wall", "draw by defense",
        "king fortress", "king's fortress",
        "stalemate", "stalemated", "stalemate trick", "stalemate trap",
        "draw by repetition", "draws by repetition", "threefold repetition",
        "three-fold repetition", "threefold repetition",
        "forces a draw", "forces the draw", "secures a draw",
        "holds the draw", "saves the draw", "drawing resource",
        "overprotect", "overprotection", "over-protect",
        "extra protection", "extra defender",
        "redundant defense",
        "prophylactic protection",
        "secure the key square",
        "defend in advance",
    ],
    "coordination": [
        "coordination", "cooperate", "harmonious", "pieces work together",
        "in concert", "working together", "pieces combine", "acting together",
        "pieces cooperate", "work in tandem", "work together",
        "discoordinat",          # discoordinate, discoordination
        "poorly coordinated", "lack of coordination",
        "pieces act together", "pieces collaborate",
    ],
    "color_complex": [
        "color complex", "colour complex",
        "light squares", "dark squares",
        "color weakness", "colour weakness",
        "light-squared", "dark-squared",
        "light square bishop", "dark square bishop",
        "wrong color bishop", "wrong colour bishop", "wrong-colored bishop",
        "bishop of the wrong color", "bishop of wrong color",
        "control the light", "control the dark",
    ],
    "endgame_technique": [
        "endgame technique", "technical endgame",
        "conversion", "realize the advantage", "convert the advantage",
        "rook endgame", "rook ending",
        "bishop endgame", "bishop ending",
        "pawn endgame", "pawn ending",
        "knight endgame", "knight ending",
        "queen endgame", "queen ending",
        "lucena",                           # Lucena position (rook + pawn vs rook)
        "philidor",                         # Philidor position (defensive drawing technique)
        "build a bridge", "bridge building",# Lucena winning method
        "tarrasch rule", "rook behind",     # rook behind passed pawn rule
        "shoulder", "cut off the king",     # king cutting technique
        "theoretically drawn", "theoretical win", "book draw",
        "drawn endgame", "drawn ending", "drawn position",
        "theoretical draw", "objectively drawn",
    ],
    "opposition": [
        "opposition", "king opposition",
        "triangulation", "triangulates",
        "corresponding square", "key square", "critical square",
        "takes the opposition", "seize the opposition",
    ],

    # ── New themes ────────────────────────────────────────────────────────────
    "counterplay": [
        "counterplay", "counter-play", "counter play",
        "counter chances", "counter-attack", "counter attack",
        "dynamic chances", "sufficient compensation",
        "good counterchances",
        "perpetual check", "draws by perpetual",
        "gives perpetual", "force perpetual", "forces perpetual",
        "draw by perpetual",
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
        "direct attack", "launch the attack", "launch the kingside",
        "open lines for the attack", "attacking ambitions",
    ],

    "square_control": [
        "key square control", "central control", "control of the center",
        "dominates the center", "control the center",
        "important square", "crucial square", "strategic square",
        "occupy the key square", "occupy the center",
        "outpost square", "strong outpost square",
        "control of key squares", "grip on the center",
        "dominate the center", "center control",
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

def _inject_folder_concept(themes: list[str], folder_concept: str | None) -> list[str]:
    if folder_concept and folder_concept not in themes:
        return sorted(themes + [folder_concept])
    return themes


def parse_file(pgn_path: Path, min_comment_len: int = 25, progress_every: int = 500,
               folder_concept: str | None = None):
    """
    Generator — yields one training example dict at a time.
    Prints a progress line every `progress_every` games so you can see it's alive.

    folder_concept: if set (e.g. "backward_pawn" for files under lichess_studies/),
                    that label is guaranteed on every yielded example regardless of
                    whether the keyword pass finds it.
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

            # ── game-level root comment (no move yet) ─────────────────────────
            # Captures definition/explanation games like "Example 1" that have a
            # rich comment before any moves (or instead of moves).
            root_comment = clean_comment(game.comment)
            if len(root_comment) >= min_comment_len:
                themes = _inject_folder_concept(extract_themes(root_comment), folder_concept)
                examples_yielded += 1
                yield {
                    "fen":        board.fen(),
                    "move_san":   "",
                    "move_uci":   "",
                    "annotation": root_comment,
                    "themes":     themes,
                    "phase":      get_phase(board),
                    "fullmove":   board.fullmove_number,
                    "side":       "white" if board.turn == chess.WHITE else "black",
                    "source":     source,
                    "game":       header_str,
                }

            # ── per-move comments ─────────────────────────────────────────────
            node = game
            while node.variations:
                node       = node.variations[0]
                fen_before = board.fen()
                move       = node.move
                board.push(move)

                comment = clean_comment(node.comment)
                if len(comment) < min_comment_len:
                    continue

                themes = _inject_folder_concept(extract_themes(comment), folder_concept)
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
            # Detect folder concept for files under lichess_studies/<concept>/
            folder_concept = None
            parts = pgn_path.parts
            if "lichess_studies" in parts:
                idx = parts.index("lichess_studies")
                if idx + 1 < len(parts):
                    candidate = parts[idx + 1]
                    if candidate in CONCEPT_KEYWORDS:
                        folder_concept = candidate

            print(f"\n  [{i}/{len(pgn_files)}] {pgn_path.name}"
                  + (f"  [guaranteed: {folder_concept}]" if folder_concept else ""))
            try:
                for ex in parse_file(pgn_path, args.min_comment, folder_concept=folder_concept):
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
