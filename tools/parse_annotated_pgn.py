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
      "fen":          "r1bq1rk1/...",       # board position FEN
      "move_san":     "Nd5",                # the move played from this position
      "move_uci":     "f3d5",              # UCI form
      "annotation":   "The knight reaches its ideal outpost...",
      "themes":       ["outpost", "blockade"],  # extracted chess concepts (may be [])
      "phase":        "middlegame",         # opening / middlegame / endgame
      "fullmove":     14,                   # move number
      "side":         "white",             # whose move it is
      "source":       "my_system.pgn",     # source filename (not full path)
      "game":         "Nimzowitsch vs Systemsson, 1927",  # Event + players
      "history_uci":  ["e2e4", "c7c5", ...],  # last MAX_HISTORY half-moves before FEN
      "eco":          "B54",               # ECO code from PGN header (null if absent)
      "opening":      "Sicilian Defense",  # opening name (null if absent)
      "algo_features": [0.0, 1.0, ...]    # 59 pre-computed structural bits (if available)
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

MAX_HISTORY = 60   # half-moves of game context stored per labeled example (last N)

try:
    from tools.label_positions import algo_feature_vector_v4 as _algo_fn
    _ALGO_AVAILABLE = True
except ImportError:
    _ALGO_AVAILABLE = False

# Silver-label concepts: deterministic structural detectors only.
# Fired on positions whose keyword pass returns an empty themes list.
_SILVER_CONCEPTS: frozenset[str] = frozenset({
    "pawn_island", "isolated_pawn", "passed_pawn",
    "bishop_pair", "rook_seventh", "pawn_majority",
})
try:
    from tools.label_positions import label_position as _label_fn
    _SILVER_AVAILABLE = True
except ImportError:
    _SILVER_AVAILABLE = False


# ── concept keyword map ───────────────────────────────────────────────────────
# Each key is the theme label stored in the dataset.
# Each value is a list of substrings to search for (case-insensitive) in comments.
# Add new themes freely — they flow into the classifier's output layer automatically.

CONCEPT_KEYWORDS: dict[str, list[str]] = {

    # ── Tactical themes ───────────────────────────────────────────────────────
    "pin": [
        "pin", "pinned", "absolute pin", "relative pin", "pinning",
        "break the pin", "breaking the pin", "unpin", "unpinned",
        "bound hand and foot",
        "pin is decisive", "pin decides",
        "exploit the pin", "maintain the pin",
        "the pin on",
        "pin to the king", "pinned to the king",
    ],
    "fork": [
        "fork", "double attack", "forking",
        "knight fork", "royal fork", "rook fork",
        "fork the king", "fork threat",
        "forks the",
    ],
    "skewer": [
        "skewer", "skewering",
        "reverse pin",
        "skewers the",
        "attacks through",
    ],
    "discovery": [
        "discovered attack", "discovery", "unmasking",
        "discovered check",
        "reveal", "reveals a check", "uncovers",
    ],
    # Phase 3 note: x_ray, double_check, clearance split out from discovery below
    "x_ray": [
        "x-ray", "x ray", "x-ray attack", "x ray attack",
        "x-ray defense",
        "defends through", "x-ray through",
        "guards through", "protects through",
        "sees through",
    ],
    "double_check": [
        "double check", "double-check",
        "two pieces give check", "check from two pieces",
        "simultaneous check",
    ],
    "clearance": [
        "clearance", "clearance sacrifice", "clearing the",
        "vacate", "vacating", "vacates",
        "clear the square", "clears the square",
        "clear the diagonal", "clears the diagonal",
        "clear the file", "clears the file",
        "make room for",
        "clear the way", "clears the way",
        "remove from the diagonal",
        "empty the square", "moves out of the way"
    ],
    "deflection": [
        "deflect", "deflection", "lure away", "draw away",
        "capturing defender", "capturing the defender",
        "decoy", "lure", "entice",
        "attraction",
        "lure the king",
        "tempt",
        "bait",
        "draw the queen",
    ],
    "overloading": [
        "overloaded", "overloading", "too many duties", "cannot defend both",
        "guard too many", "protecting too much",
        "overtaxed", "over-taxed", "over taxed",
        "overburdened", "over-burdened",
        "juggling", "cannot be in two places",
        "defend two", "protect two",
    ],
    "zwischenzug": [
        "zwischenzug", "in-between move", "intermezzo", "intermediate move",
        "in between move", "interpolat",
        "plays in between",
        "intermediate check", "in-between check",
    ],
    "interference": [
        "interference", "interfer",
        "blocks the line",
        "interposing",
        "disconnect", "disconnects",
        "cutting off communication",
        "blocks the diagonal",
        "blocks the file",
    ],
    "back_rank": [
        "back rank", "back-rank", "backrank", "first rank mate", "back rank weakness",
        "back rank mate",                             # Lichess: backRankMate
    ],
    "sacrifice": [
        "sacrifice", "sacrific",
        "sac ",
        "exchange sacrifice", "positional exchange sacrifice",
        "gives the exchange", "give the exchange", "giving the exchange",
        "gives up the exchange", "sacrifices the exchange",
        "rook for bishop", "rook for knight",
        "exchange for positional", "exchange for a pawn",
        "positional sacrifice", "long-term sacrifice", "positional pawn sacrifice",
        "strategic sacrifice",
        "material is only temporary", "gives up material for",
        "long term investment",
        "sacrifices a pawn for",
    ],
    "mating_attack": [
        "checkmate", "mating", "mating net", "mating attack",
        "threatens mate", "threatening checkmate", "threat of mate",
        "forced mate", "mate is inevitable", "delivering checkmate",
        "smothered mate", "smothered",
        "arabian mate", "anastasia", "hook mate", "boden", "epaulette mate",
        "opera mate", "morphy's mate", "morphy mate",
        # folded from attacking_chances — direct king-attack keywords
        "kingside attack", "king-side attack", "king side attack",
        "queenside attack", "queen-side attack", "queen side attack",
        "attack on the king", "assault", "dangerous attack",
        "direct attack", "launch the attack", "launch the kingside",
        "open lines for the attack",
    ],
    "trapped_piece": [
        "trapped", "has no escape", "no retreat",
        "no safe square", "no good square", "nowhere to go",
        "cannot escape", "piece cannot move",
    ],

    # ── Piece concepts ────────────────────────────────────────────────────────
    "outpost": [
        "outpost", "outpost square", "ideal post",
        "cannot be dislodged",
        "no way to drive", "strong post", "permanent post",
        "strong position for the knight", "knight cannot be",
        "knight stands well", "knight is well placed",
        "stronghold",
        "knight is firmly placed",
        "strong square for",
        "strong square", "protected square", "protected strong square",
        "hole in the position", "creates a hole",
        "bridgehead", "beachhead",
        "absolute outpost", "relative outpost",
        "secure base", "secure outpost",
        "cannot be chased", "cannot be driven away", "cannot be expelled",
    ],
    "blockade": [
        "blockade", "blockader", "blockading",
        "block the pawn",
        "the blockader", "elastic blockader", "blockading piece",
        "knight blockades", "bishop blockades",
    ],
    "bad_bishop": [
        "bad bishop",
        "imprisoned bishop", "wrong colored bishop", "blocked bishop",
        "passive bishop", "inactive bishop", "locked bishop",
        "bishop is inferior", "bishop inside the pawn chain",
        "bishop condemned", "bishop is passive",
        "bishop blocked by its own pawns",
        "bishop of the wrong color",
        "bishop is blocked",
        "exchange off his bad bishop", "exchange the bad bishop",
        "inferior to the knight",
        "bishop has no scope",
        "bishop is useless",
        "suffers from bad bishop",
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
        "bishop pair", "two bishops",
        "the two bishops",
        "bishop pair advantage",
        "give up the bishop pair", "surrendering the bishop pair",
        "relinquish the bishops",
        "bishop pair is strong", "bishop pair proves",
        "bishops are strong", "better bishops",
        "retain the bishops",
    ],
    "piece_activity": [
        "piece activity",
        "centralize", "centralized", "centralizing", "mobilize",
        "aggressively posted", "well posted", "actively placed",
        "piece becomes active", "pieces become active",
        "active rook", "rook becomes active", "activate the rook",
        "activate the knight", "knight becomes active",
        "pieces are active", "pieces become dominant",
        "more active", "better activity", "more aggressive",
        "greater activity", "superior activity",
        # square control (folded in)
        "key square control", "central control", "control of the center",
        "dominates the center", "control the center",
        "important square", "crucial square", "strategic square",
        "occupy the key square", "occupy the center",
        "control of key squares", "grip on the center",
        "dominate the center", "center control",
    ],
    "battery": [
        "battery",
        "doubled rooks", "doubling on",
        "piling up", "pile up on",
        "rooks on the same", "double on the",
        "queen and bishop battery", "bishop battery",
        "rook battery", "queen behind the rook",
        "double the rooks", "rooks are doubled",
        "align along",
    ],
    "rook_seventh": [
        "seventh rank", "seventh row", "on the seventh", "rook on the 7",
        "absolute seventh",
        "master of the seventh", "dominate the seventh",
        "rook on the seventh", "rooks on the seventh",
        "invades the seventh", "penetrates to the seventh",
        "on the 7th rank", "rook reaches the seventh",
        "two rooks on the seventh", "seventh rank is decisive",
    ],

    # ── Pawn structure ────────────────────────────────────────────────────────
    "passed_pawn": [
        "passed pawn", "passer",
        "unstoppable pawn", "advancing pawn", "past the pawn", "is now passed"
    ],
    "promotion": [
        "promotion", "queening", "queening square",
        "underpromot", "under-promot",
        "promotes to queen", "promotes to knight", "promotes to rook", "promotes to bishop",
        "knight promotion", "queen promotion",
        "pawn queens", "pawn promotes", "queen the pawn", "promote the pawn",
        "promotion race", "pawn race",
        "on the verge of queening", "about to queen",
        "prevent the promotion", "stop the pawn from queening",
    ],
    "isolated_pawn": [
        "isolated", "isolating", "IQP", "isolated queen pawn", "isolated pawn",
        "isolani", "isolani pawn", "isolated queen's pawn",
    ],
    "backward_pawn": [
        "backward pawn", "backward d-pawn", "backward c-pawn", "backward e-pawn",
        "backward pawn as a weakness", "is backwards", "now backwards", "becomes backwards", "the backwards",
        "weak pawn on", "backward pawn on", "backward queen pawn",
    ],
    "doubled_pawn": [
        "doubled pawn", "doubled pawns",
        "double the pawns", "doubling the pawns",
        "doubled c-pawns", "doubled f-pawns", "doubled b-pawns",
        "pawn on the same file",
        "two pawns on the same",
    ],
    "pawn_chain": [
        "pawn chain", "head of the chain", "base of the chain",
        "pawn wedge", "the chain",
        "break the chain", "undermine the chain",
        "undermine the base", "destroy the chain",
        "locked pawns", "locked pawn structure", "locked structure",
        "chain base", "base of the pawn chain",
    ],
    "pawn_majority": [
        "pawn majority", "queenside majority", "kingside majority",
        "mobile majority",
        "passed pawn from the majority",
        "majority on the queenside", "majority on the kingside",
        "numerical advantage on the queenside", "numerical advantage on the kingside",
        "extra pawn on the queenside", "extra pawn on the kingside",
    ],
    "pawn_storm": [
        "pawn storm", "pawn advance", "storming", "pawn avalanche",
        "advance the pawns",
        "advance on the kingside",
        "advance on the queenside",
        "attack with pawns",
        "pawn roller", "rolling pawns",
        "advancing the pawns",
        "minority attack", "minority structure", "queenside minority",
        "creates a weakness on c6", "weakness on c6",
        "create a weakness in the pawn chain",
        "weaken the majority",
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
        "vulnerable king", "open king position", "king lacks shelter",
        "kingside weaknesses", "stripped of pawns",
        "king is exposed", "king is unsafe",
        "dangerous for the king", "king comes under fire",
        "uncastled king", "king in the center",
        "king hunt",
    ],
    "king_activity": [
        "king march", "king to the center", "active king",
        "king walk", "king becomes active",
        "outflanking", "outflank",
        "king advances", "king penetrates", "king invades",
        "king centralizes", "king supports",
        "king escorts the pawn", "king enters the game",
        "king joins the attack",
        "king marches to", "active use of the king",
        "king activity", "king centralization",
    ],
    "shouldering": [
        "shouldering", "body-check", "body check", "shoulder charge",
        "cuts off the king", "king cannot pass",
        "v-maneuver", "v maneuver",
        "restricts the opposing king", "forces the king to go around",
        "prevents the king from reaching", "obstructs the king",
        "king is cut off", "king is blocked from",
    ],
    "weak_square": [
        "color weakness", "colour weakness", "hole",
        "weak on the", "light square weakness", "dark square weakness",
        "light-squared weakness", "dark-squared weakness",
        "control the light squares", "control the dark squares",
        "weak color complex", "weak colour complex", "color complex weakness",
        "d5 hole", "e4 hole", "d4 hole", "c5 hole", "f5 hole",
        "square complex", "weak square complex",
    ],
    "open_file": [
        "open file", "half-open", "semi-open file", "open d-file",
        "open e-file", "open c-file", "opens the file",
        "half open", "half open file",
        "opening the file", "open the file",
        "open the f-file", "open the g-file", "open the b-file",
        "file is open", "file is now open",
        "rook on the open", "open file for the rook",
        "rooks need open files", "rooks have open files",
        "occupy the open file",
        "seize the file", "seizes the file",
        "pressure down the file",
        "control the file", "controls the file", "dominate the file",
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
        "no room to maneuver", "no maneuvering room",
        "acute lack of space",
        "control of space",
        "greater mobility",
        "pieces lack mobility",
        "free his game", "free the game",
        "siege",
    ],
    "initiative": [
        "initiative", "seize the initiative", "holds the initiative",
        "keeps the initiative", "surrenders the initiative",
        "dictates the pace", "forces the opponent to react",
        "keeps the pressure", "maintains the pressure",
        "opponent is on the defensive", "opponent cannot react",
        "tempo", "tempi", "gains a tempo", "wasted tempo",
        "gain of tempo", "loss of tempo", "loses a tempo",
        "tempo advantage", "with tempo",
        # folded from attacking_chances — dynamic/counterplay keywords
        "attacking chances", "attacking possibilities", "attacking play",
        "attacking position", "attack steadily", "press the attack",
        "attacking ambitions",
        "counterplay", "counter-play", "counter play",
        "counter chances", "counter-attack", "counter attack",
        "dynamic chances", "sufficient compensation", "good counterchances",
    ],
    "zugzwang": [
        "zugzwang", "compulsion to move", "any move worsens",
        "every move loses",
        "all moves are bad",
        "no good move"
        "in zugzwang",
        "mutual zugzwang",
        "whoever moves loses",
        "in a bind", "cannot improve the position",
    ],
    "prophylaxis": [
        "prophylaxis", "prophylactic", "prophylactic move",
        "forestall", "in order to prevent",
        "restraining", "restraint", "restrain the",
        "nip in the bud",
        "prevent the advance", "prevent the break",
        "stop the plan", "hinder the plan",
        "overprotect", "overprotection", "over-protect",
        "prevent the knight from", "prevent the bishop from",
    ],
    "rook_endgame": [
        "rook endgame", "rook ending", "rook and pawn endgame",
        "lucena", "philidor",
        "build a bridge", "bridge building",
        "tarrasch rule", "rook behind",
        "rook vs pawn", "rook ending technique",
    ],
    "pawn_endgame": [
        "pawn endgame", "pawn ending", "king and pawn endgame",
        "pawn endgame technique", "pure pawn ending",
    ],
    "bishop_endgame": [
        "bishop endgame", "bishop ending",
        "opposite colored bishops", "opposite-colored bishops",
        "same-colored bishops", "same colored bishops",
        "bishop and pawn endgame",
    ],
    "knight_endgame": [
        "knight endgame", "knight ending",
        "knight versus pawn", "knight and pawn endgame",
    ],
    "queen_endgame": [
        "queen endgame", "queen ending",
        "queen versus pawn", "queen and pawn endgame",
        "queen endgame technique",
    ],
    "drawn_position": [
        "drawn position", "theoretical draw", "objectively drawn",
        "theoretically drawn", "book draw",
        "stalemate", "stalemate trick", "stalemate trap",
        "perpetual check", "draws by perpetual", "force perpetual",
        "gives perpetual", "draw by perpetual",
        "threefold repetition", "draw by repetition",
        "insufficient material", "dead position",
        "fortress", "impenetrable fortress",
        "drawing technique", "hold the draw",
    ],
    "opposition": [
        "opposition", "king opposition",
        "direct opposition", "diagonal opposition",
        "distant opposition", "long-range opposition",
        "virtual opposition", "rectangular opposition",
        "triangulation", "triangulates",
        "corresponding square",
        "takes the opposition", "seize the opposition",
        "holds the opposition", "has the opposition", "loses the opposition",
    ],

    "development_lead": [
        "lead in development", "development advantage", "better development",
        "lagging in development", "behind in development",
        "developmental lead",
        "undeveloped", "poorly developed",
        "rapid development", "quick development",
        "gain in development",
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
    source        = pgn_path.name
    games_read    = 0
    games_corrupt = 0
    examples_yielded = 0
    t_start       = time.time()

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            games_read += 1

            # Any illegal SAN means board.move_stack is unreliable — discard entire game
            if game.errors:
                games_corrupt += 1
                continue

            if games_read % progress_every == 0:
                elapsed = time.time() - t_start
                rate    = games_read / elapsed if elapsed > 0 else 0
                print(f"\r    {games_read:,} games read  |  {examples_yielded:,} examples  |  "
                      f"{rate:.0f} games/min  |  {elapsed/60:.1f} min elapsed",
                      end="", flush=True)

            header_str = game_header(game)
            eco        = game.headers.get("ECO") or None
            opening    = game.headers.get("Opening") or None
            board      = game.board()

            # ── game-level root comment (no move yet) ─────────────────────────
            # Captures definition/explanation games like "Example 1" that have a
            # rich comment before any moves (or instead of moves).
            root_comment = clean_comment(game.comment)
            if len(root_comment) >= min_comment_len:
                root_fen = board.fen()
                themes   = _inject_folder_concept(extract_themes(root_comment), folder_concept)
                if not themes and _SILVER_AVAILABLE:
                    silver = _label_fn(board) & _SILVER_CONCEPTS
                    themes = sorted(silver)
                extra    = {}
                if _ALGO_AVAILABLE:
                    try:
                        extra["algo_features"] = _algo_fn(root_fen).tolist()
                    except Exception:
                        pass
                examples_yielded += 1
                yield {
                    "fen":          root_fen,
                    "move_san":     "",
                    "move_uci":     "",
                    "annotation":   root_comment,
                    "themes":       themes,
                    "phase":        get_phase(board),
                    "fullmove":     board.fullmove_number,
                    "side":         "white" if board.turn == chess.WHITE else "black",
                    "source":       source,
                    "game":         header_str,
                    "history_rich": [],
                    "eco":          eco,
                    "opening":      opening,
                    **extra,
                }

            # ── per-move comments ─────────────────────────────────────────────
            node         = game
            history_rich: list[dict] = []   # built incrementally as moves are pushed

            while node.variations:
                node = node.variations[0]

                fen_before = board.fen()
                move       = node.move

                # Capture piece / capture info BEFORE push, check flag AFTER
                piece_obj    = board.piece_at(move.from_square)
                captured_obj = board.piece_at(move.to_square)
                color_before = board.turn
                board.push(move)
                is_check = board.is_check()

                history_rich.append({
                    "uci":      move.uci(),
                    "piece":    piece_obj.piece_type if piece_obj else None,
                    "captured": captured_obj.piece_type if captured_obj else None,
                    "is_check": is_check,
                    "color":    1 if color_before == chess.WHITE else 0,
                })

                comment = clean_comment(node.comment)
                if len(comment) < min_comment_len:
                    continue

                themes = _inject_folder_concept(extract_themes(comment), folder_concept)
                if not themes and _SILVER_AVAILABLE:
                    silver = _label_fn(chess.Board(fen_before)) & _SILVER_CONCEPTS
                    themes = sorted(silver)

                phase  = get_phase(chess.Board(fen_before))

                try:
                    san = chess.Board(fen_before).san(move)
                except Exception:
                    san = move.uci()

                extra = {}
                if _ALGO_AVAILABLE:
                    try:
                        extra["algo_features"] = _algo_fn(fen_before).tolist()
                    except Exception:
                        pass

                examples_yielded += 1
                yield {
                    "fen":          fen_before,
                    "move_san":     san,
                    "move_uci":     move.uci(),
                    "annotation":   comment,
                    "themes":       themes,
                    "phase":        phase,
                    "fullmove":     board.fullmove_number,
                    "side":         "white" if not board.turn else "black",
                    "source":       source,
                    "game":         header_str,
                    "history_rich": list(history_rich[-MAX_HISTORY:]),
                    "eco":          eco,
                    "opening":      opening,
                    **extra,
                }

    elapsed = time.time() - t_start
    corrupt_note = f"  |  {games_corrupt:,} corrupt skipped" if games_corrupt else ""
    print(f"\r    {games_read:,} games read  |  {examples_yielded:,} examples"
          f"{corrupt_note}  |  done in {elapsed/60:.1f} min                    ")


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
