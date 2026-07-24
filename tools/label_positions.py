"""
label_positions.py  —  Algorithmic chess concept detector
----------------------------------------------------------
Takes a python-chess Board and returns the set of concept labels
that are provably present in the position.  Labels are position-level:
a concept is added if it exists for EITHER side so the coach learns to
identify the motif regardless of whose favour it serves.

Usage
-----
    from tools.label_positions import label_position
    labels = label_position(board)          # frozenset[str]

Only concepts that can be computed exactly from a single FEN are
detected here.  Dynamic concepts that need move history (tempo,
sacrifice, counterplay, combination) are left to keyword matching
and the future CNN+history architecture.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import chess
import numpy as np

# ── Optional Syzygy endgame tablebases ────────────────────────────────────────
# Set the SYZYGY_PATH environment variable or place .rtbw/.rtbz files under
# data/syzygy/ to enable tablebase-assisted draw detection.  Probed only for
# positions with ≤7 pieces; silently disabled if the path does not exist.
#
# Download (3-6 piece, ~1 GB):  https://tablebase.lichess.ovh/tables/standard/
# 7-piece Syzygy (~18 GB) can be added later; 5-piece covers most endgames.
_SYZYGY_PATH   = os.environ.get("SYZYGY_PATH", str(Path("data/syzygy").resolve()))
_syzygy_tb     = None   # chess.syzygy.Tablebase or None
_syzygy_loaded = False  # True once we have attempted to open


def _get_syzygy():
    """Return an open Syzygy Tablebase, or None if unavailable."""
    global _syzygy_tb, _syzygy_loaded
    if _syzygy_loaded:
        return _syzygy_tb
    _syzygy_loaded = True
    try:
        if Path(_SYZYGY_PATH).exists():
            import chess.syzygy  # type: ignore
            _syzygy_tb = chess.syzygy.open_tablebase(_SYZYGY_PATH)
    except Exception:
        _syzygy_tb = None
    return _syzygy_tb


def _is_syzygy_draw(board: chess.Board) -> bool:
    """True if Syzygy tablebases prove the position is a draw (WDL == 0).

    Only probed for positions with ≤7 pieces; returns False if tablebases are
    unavailable or the position has too many pieces for the available tables.
    """
    if bin(int(board.occupied)).count("1") > 7:
        return False
    tb = _get_syzygy()
    if tb is None:
        return False
    try:
        return tb.probe_wdl(board) == 0
    except Exception:
        return False

# Concepts this module can detect.  Caller can intersect against this
# to know which labels came from detectors vs. other sources.
DETECTABLE_CONCEPTS: frozenset[str] = frozenset({
    # Pawn structure
    "passed_pawn",
    "isolated_pawn",
    "doubled_pawn",
    "pawn_island",
    "pawn_majority",
    "backward_pawn",
    "pawn_chain",
    "pawn_storm",
    "promotion",
    # Piece quality / placement
    "bad_bishop",
    "good_bishop",
    "bishop_pair",
    "battery",
    "blockade",
    "outpost",
    "rook_seventh",
    "piece_activity",
    "trapped_piece",
    # King
    "king_activity",
    "king_safety",
    "back_rank",
    "opposition",
    # Squares / files
    "open_file",
    "weak_square",
    "space_advantage",
    # Tactics (detectable from single position)
    "pin",
    "fork",
    "skewer",
    "overloading",
    "x_ray",
    "discovery",
    "double_check",
    "mating_attack",
    "interference",
    "sacrifice",
    "clearance",
    "deflection",
    "zwischenzug",
    # Endgame geometry
    "zugzwang",
    "shouldering",
    # Strategic
    "development_lead",
    "initiative",
    "prophylaxis",
    # Endgame types (by material composition)
    "rook_endgame",
    "pawn_endgame",
    "bishop_endgame",
    "knight_endgame",
    "queen_endgame",
    "drawn_position",
})

# ── Bitboard shift helpers ────────────────────────────────────────────────────
# Raw 64-bit integer operations. Much faster than python-chess SquareSet loops
# for bulk pawn structure computation.

_BB_ALL = chess.BB_ALL  # 0xFFFF_FFFF_FFFF_FFFF

def _bb_north(bb: int) -> int:
    return (bb << 8) & _BB_ALL

def _bb_south(bb: int) -> int:
    return bb >> 8

def _bb_east(bb: int) -> int:
    return (bb & ~chess.BB_FILE_H) << 1

def _bb_west(bb: int) -> int:
    return (bb & ~chess.BB_FILE_A) >> 1

def _north_fill(bb: int) -> int:
    bb |= (bb <<  8); bb |= (bb << 16); bb |= (bb << 32)
    return bb & _BB_ALL

def _south_fill(bb: int) -> int:
    bb |= (bb >>  8); bb |= (bb >> 16); bb |= (bb >> 32)
    return bb

def _file_fill(bb: int) -> int:
    return _north_fill(bb) | _south_fill(bb)

def _wp_attacks(bb: int) -> int:
    """Squares attacked by white pawns at positions bb."""
    n = _bb_north(bb)
    return _bb_east(n) | _bb_west(n)

def _bp_attacks(bb: int) -> int:
    """Squares attacked by black pawns at positions bb."""
    s = _bb_south(bb)
    return _bb_east(s) | _bb_west(s)

# ── helpers ───────────────────────────────────────────────────────────────────

def _is_endgame(board: chess.Board) -> bool:
    """Rough endgame detector: no queens, or very few major/minor pieces."""
    queens = (len(board.pieces(chess.QUEEN, chess.WHITE))
              + len(board.pieces(chess.QUEEN, chess.BLACK)))
    minors_and_rooks = sum(
        len(board.pieces(pt, c))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for c in (chess.WHITE, chess.BLACK)
    )
    return queens == 0 or minors_and_rooks <= 6


# ── pawn structure (bitboard) ─────────────────────────────────────────────────

def _has_passed_pawn(board: chess.Board, color: chess.Color) -> bool:
    wp = int(board.pieces_mask(chess.PAWN, color))
    ep = int(board.pieces_mask(chess.PAWN, not color))
    if not wp:
        return False
    # A pawn is passed if no enemy pawn on same or adjacent files ahead of it.
    # not_passed = south-fill (for white) of enemy pawns + their adjacent files.
    if color == chess.WHITE:
        not_passed = _south_fill(ep | _bb_east(ep) | _bb_west(ep))
    else:
        not_passed = _north_fill(ep | _bb_east(ep) | _bb_west(ep))
    return bool(wp & ~not_passed)


def _has_isolated_pawn(board: chess.Board, color: chess.Color) -> bool:
    wp = int(board.pieces_mask(chess.PAWN, color))
    if not wp:
        return False
    neighbor_files = _file_fill(_bb_east(wp) | _bb_west(wp))
    return bool(wp & ~neighbor_files)


def _has_doubled_pawn(board: chess.Board, color: chess.Color) -> bool:
    wp = int(board.pieces_mask(chess.PAWN, color))
    if not wp:
        return False
    return any(chess.popcount(wp & chess.BB_FILES[f]) >= 2 for f in range(8))


def _pawn_island_count(board: chess.Board, color: chess.Color) -> int:
    wp = int(board.pieces_mask(chess.PAWN, color))
    if not wp:
        return 0
    filled = _file_fill(wp)
    islands, in_island = 0, False
    for f in range(8):
        has = bool(filled & chess.BB_FILES[f])
        if has and not in_island:
            islands += 1
            in_island = True
        elif not has:
            in_island = False
    return islands


_QS_MASK = chess.BB_FILE_A | chess.BB_FILE_B | chess.BB_FILE_C | chess.BB_FILE_D
_KS_MASK = chess.BB_FILE_E | chess.BB_FILE_F | chess.BB_FILE_G | chess.BB_FILE_H

def _has_pawn_majority(board: chess.Board, color: chess.Color) -> bool:
    wp = int(board.pieces_mask(chess.PAWN, color))
    ep = int(board.pieces_mask(chess.PAWN, not color))
    return (chess.popcount(wp & _QS_MASK) > chess.popcount(ep & _QS_MASK) or
            chess.popcount(wp & _KS_MASK) > chess.popcount(ep & _KS_MASK))


def _has_backward_pawn(board: chess.Board, color: chess.Color) -> bool:
    """Pawn whose stop square is attacked by an enemy pawn and has no adjacent-file support."""
    wp = int(board.pieces_mask(chess.PAWN, color))
    ep = int(board.pieces_mask(chess.PAWN, not color))
    if not wp:
        return False
    if color == chess.WHITE:
        stop = _bb_north(wp)
        at_risk = _bb_south(stop & _bp_attacks(ep)) & wp
        for sq in chess.scan_forward(at_risk):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            adj = (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
            if not (wp & adj & _south_fill(chess.BB_RANKS[r])):
                return True
    else:
        stop = _bb_south(wp)
        at_risk = _bb_north(stop & _wp_attacks(ep)) & wp
        for sq in chess.scan_forward(at_risk):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            adj = (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
            if not (wp & adj & _north_fill(chess.BB_RANKS[r])):
                return True
    return False


def _has_pawn_chain(board: chess.Board, color: chess.Color) -> bool:
    """At least two friendly pawns that mutually defend each other diagonally."""
    wp = int(board.pieces_mask(chess.PAWN, color))
    if not wp:
        return False
    if color == chess.WHITE:
        return bool(wp & _wp_attacks(wp))
    else:
        return bool(wp & _bp_attacks(wp))


# ── bishop quality ────────────────────────────────────────────────────────────

def _bishop_parity(sq: int) -> int:
    """0 = dark square, 1 = light square."""
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2


def _has_bad_bishop(board: chess.Board, color: chess.Color) -> bool:
    """Bishop whose square color matches the majority of its own pawns."""
    bishops = board.pieces(chess.BISHOP, color)
    pawns   = board.pieces(chess.PAWN,   color)
    if not bishops or not pawns:
        return False
    for bsq in bishops:
        bc = _bishop_parity(bsq)
        same = sum(1 for p in pawns if _bishop_parity(p) == bc)
        if same > len(pawns) - same:
            return True
    return False


def _has_good_bishop(board: chess.Board, color: chess.Color) -> bool:
    """Bishop whose pawns are mostly on the OPPOSITE color (the bishop is mobile)."""
    bishops = board.pieces(chess.BISHOP, color)
    pawns   = board.pieces(chess.PAWN,   color)
    if not bishops or not pawns:
        return False
    for bsq in bishops:
        bc = _bishop_parity(bsq)
        opp_color_pawns = sum(1 for p in pawns if _bishop_parity(p) != bc)
        if opp_color_pawns > len(pawns) - opp_color_pawns:
            return True
    return False


def _has_bishop_pair(board: chess.Board, color: chess.Color) -> bool:
    return len(board.pieces(chess.BISHOP, color)) >= 2


# ── files and ranks ──────────────────────────────────────────────────────────

def _has_open_file(board: chess.Board, color: chess.Color) -> bool:
    """A rook of `color` is on an open file (no pawns of either color on that file)."""
    rooks = int(board.pieces_mask(chess.ROOK, color))
    if not rooks:
        return False
    pawn_files = _file_fill(int(board.pawns))
    return bool(rooks & ~pawn_files)


def _has_rook_on_seventh(board: chess.Board, color: chess.Color) -> bool:
    seventh = 6 if color == chess.WHITE else 1
    return any(chess.square_rank(sq) == seventh
               for sq in board.pieces(chess.ROOK, color))


# ── square control (bitboard) ────────────────────────────────────────────────

def _outpost_squares_bb(board: chess.Board, color: chess.Color) -> int:
    """Bitboard of squares in enemy territory that enemy pawns can never attack."""
    ep = int(board.pieces_mask(chess.PAWN, not color))
    if color == chess.WHITE:
        # Black pawns advance south; ever_attacked = all squares black pawns can
        # reach and attack as they advance southward.
        ever_attacked = _bp_attacks(_south_fill(ep))
        territory = (chess.BB_RANK_4 | chess.BB_RANK_5 |
                     chess.BB_RANK_6 | chess.BB_RANK_7)
    else:
        ever_attacked = _wp_attacks(_north_fill(ep))
        territory = (chess.BB_RANK_1 | chess.BB_RANK_2 |
                     chess.BB_RANK_3 | chess.BB_RANK_4)
    return int(territory) & ~ever_attacked


def _has_outpost(board: chess.Board, color: chess.Color) -> bool:
    """A knight or bishop occupies an outpost square."""
    outposts = _outpost_squares_bb(board, color)
    pieces = (int(board.pieces_mask(chess.KNIGHT, color)) |
              int(board.pieces_mask(chess.BISHOP, color)))
    return bool(outposts & pieces)


def _has_weak_square(board: chess.Board, color: chess.Color) -> bool:
    """The opponent has outpost squares (holes) in their position."""
    return bool(_outpost_squares_bb(board, not color))


# ── king position ─────────────────────────────────────────────────────────────

def _has_opposition(board: chess.Board) -> bool:
    """Kings are in direct opposition (two squares apart on rank, file, or diagonal)."""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return False
    df = abs(chess.square_file(wk) - chess.square_file(bk))
    dr = abs(chess.square_rank(wk) - chess.square_rank(bk))
    return (df == 0 and dr == 2) or (df == 2 and dr == 0) or (df == 2 and dr == 2)


def _has_back_rank_weakness(board: chess.Board, color: chess.Color) -> bool:
    """color's back rank is weak: king there, no escape pawns, opponent has heavy piece."""
    king_sq = board.king(color)
    if king_sq is None:
        return False
    back_rank = 0 if color == chess.WHITE else 7
    if chess.square_rank(king_sq) != back_rank:
        return False
    kf = chess.square_file(king_sq)
    luft = any(
        board.piece_at(chess.square(f, back_rank + (1 if color == chess.WHITE else -1)))
        == chess.Piece(chess.PAWN, color)
        for f in (kf - 1, kf, kf + 1)
        if 0 <= f <= 7
        and 0 <= back_rank + (1 if color == chess.WHITE else -1) <= 7
    )
    if luft:
        return False
    opp = not color
    return bool(board.pieces(chess.ROOK, opp) or board.pieces(chess.QUEEN, opp))


# ── tactical ─────────────────────────────────────────────────────────────────

_PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _material_value(board: chess.Board, color: chess.Color) -> int:
    return sum(
        _PIECE_VALUES[pt] * len(board.pieces(pt, color))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def _has_pin(board: chess.Board, color: chess.Color) -> bool:
    """color has at least one piece pinned against its own king."""
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type != chess.KING:
            if board.is_pinned(color, sq):
                return True
    return False


def _has_fork(board: chess.Board, color: chess.Board) -> bool:
    """A piece of color attacks two or more valuable opponent pieces simultaneously."""
    opp = not color
    valuable = {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING}
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.PAWN):
        for sq in board.pieces(pt, color):
            targets = sum(
                1 for attacked in board.attacks(sq)
                if (p := board.piece_at(attacked))
                and p.color == opp
                and p.piece_type in valuable
            )
            if targets >= 2:
                return True
    return False


def _has_skewer(board: chess.Board, color: chess.Color) -> bool:
    """A sliding piece of color attacks a valuable enemy piece with another enemy piece behind it."""
    opp = not color
    high_value = {chess.KING, chess.QUEEN, chess.ROOK}

    for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            for attacked_sq in board.attacks(sq):
                victim = board.piece_at(attacked_sq)
                if not victim or victim.color != opp:
                    continue
                if victim.piece_type not in high_value:
                    continue
                df = chess.square_file(attacked_sq) - chess.square_file(sq)
                dr = chess.square_rank(attacked_sq) - chess.square_rank(sq)
                norm = max(abs(df), abs(dr))
                if norm == 0:
                    continue
                df //= norm
                dr //= norm
                cf = chess.square_file(attacked_sq) + df
                cr = chess.square_rank(attacked_sq) + dr
                while 0 <= cf <= 7 and 0 <= cr <= 7:
                    behind = board.piece_at(chess.square(cf, cr))
                    if behind:
                        if behind.color == opp:
                            return True
                        break
                    cf += df
                    cr += dr
    return False


def _has_overloading(board: chess.Board, color: chess.Color) -> bool:
    """An opponent piece defends two different targets; attacking one removes the other's defence."""
    opp = not color
    targets = list(board.pieces(chess.QUEEN,  color) |
                   board.pieces(chess.ROOK,   color) |
                   board.pieces(chess.BISHOP, color) |
                   board.pieces(chess.KNIGHT, color) |
                   board.pieces(chess.PAWN,   color))
    if len(targets) < 2:
        return False
    for def_sq in chess.SQUARES:
        defender = board.piece_at(def_sq)
        if not defender or defender.color != opp:
            continue
        solely_defending = []
        for t_sq in targets:
            if board.is_attacked_by(opp, t_sq):
                defenders = board.attackers(opp, t_sq)
                if def_sq in defenders and len(defenders) == 1:
                    solely_defending.append(t_sq)
        if len(solely_defending) >= 2:
            return True
    return False


# ── structural / strategic detectors ─────────────────────────────────────────

def _has_battery(board: chess.Board, color: chess.Color) -> bool:
    """Two same-color major pieces aligned on the same file/rank or diagonal."""
    rooks   = board.pieces(chess.ROOK,   color)
    queens  = board.pieces(chess.QUEEN,  color)
    bishops = board.pieces(chess.BISHOP, color)

    heavy = list(rooks | queens)
    for i in range(len(heavy)):
        for sq2 in heavy[i + 1:]:
            sq1 = heavy[i]
            f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
            f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
            if (f1 == f2 or r1 == r2) and sq2 in board.attacks(sq1):
                return True

    diag = list(bishops | queens)
    for i in range(len(diag)):
        for sq2 in diag[i + 1:]:
            sq1 = diag[i]
            f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
            f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
            if abs(f1 - f2) == abs(r1 - r2) and sq2 in board.attacks(sq1):
                return True

    return False


def _has_blockade(board: chess.Board, color: chess.Color) -> bool:
    """A piece of `color` sits directly in front of an advanced enemy pawn."""
    opp = not color
    ep = int(board.pieces_mask(chess.PAWN, opp))
    friendly = int(board.occupied_co[color])
    if opp == chess.WHITE:
        advanced = ep & (chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7)
        block_sqs = _bb_north(advanced)
    else:
        advanced = ep & (chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4)
        block_sqs = _bb_south(advanced)
    return bool(block_sqs & friendly)


def _has_pawn_storm(board: chess.Board, color: chess.Color) -> bool:
    """2+ pawns advanced toward the opponent's king flank."""
    opp     = not color
    king_sq = board.king(opp)
    if king_sq is None:
        return False
    king_f = chess.square_file(king_sq)
    flank  = 0
    for f in range(max(0, king_f - 2), min(8, king_f + 3)):
        flank |= chess.BB_FILES[f]

    wp = int(board.pieces_mask(chess.PAWN, color))
    if color == chess.WHITE:
        advanced = wp & (chess.BB_RANK_4 | chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7)
    else:
        advanced = wp & (chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4)
    return chess.popcount(advanced & flank) >= 2


# ── new silver label detectors ────────────────────────────────────────────────

def _has_promotion(board: chess.Board, color: chess.Color) -> bool:
    """Pawn on 7th rank (one step from queening)."""
    wp = int(board.pieces_mask(chess.PAWN, color))
    return bool(wp & (chess.BB_RANK_7 if color == chess.WHITE else chess.BB_RANK_2))


def _has_king_safety(board: chess.Board, color: chess.Color) -> bool:
    """King concretely exposed: weak pawn shelter AND enemy pressure on king zone."""
    if _is_endgame(board):
        return False
    king_sq = board.king(color)
    if king_sq is None:
        return False
    opp = not color
    king_f = chess.square_file(king_sq)
    king_r = chess.square_rank(king_sq)

    # King zone: 8 neighbors + 3 squares one rank further ahead
    zone_bb = int(chess.BB_KING_ATTACKS[king_sq])
    if color == chess.WHITE and king_r <= 6:
        zone_bb |= int(chess.BB_RANKS[king_r + 1]) & (
            chess.BB_FILES[max(0, king_f - 1)] | chess.BB_FILES[king_f] |
            chess.BB_FILES[min(7, king_f + 1)]
        )
    elif color == chess.BLACK and king_r >= 1:
        zone_bb |= int(chess.BB_RANKS[king_r - 1]) & (
            chess.BB_FILES[max(0, king_f - 1)] | chess.BB_FILES[king_f] |
            chess.BB_FILES[min(7, king_f + 1)]
        )

    # Pawn shield: count shield pawns (1 per file, up to 2 ranks ahead of king)
    wp = int(board.pieces_mask(chess.PAWN, color))
    shield = 0
    for f in range(max(0, king_f - 1), min(8, king_f + 2)):
        for rank_offset in (1, 2):
            r = (king_r + rank_offset) if color == chess.WHITE else (king_r - rank_offset)
            if 0 <= r <= 7 and (wp & chess.BB_SQUARES[chess.square(f, r)]):
                shield += 1
                break  # at most 1 pawn counted per file
    if shield >= 2:
        return False  # adequate shelter

    # Count enemy non-pawn attackers in zone
    ep_nopawns = int(board.occupied_co[opp]) & ~int(board.pieces_mask(chess.PAWN, opp))
    attackers = sum(1 for sq in chess.scan_forward(ep_nopawns)
                    if board.attacks_mask(sq) & zone_bb)
    if attackers >= 2:
        return True

    # Open or semi-open file adjacent to king
    all_pawns = int(board.pawns)
    for f in range(max(0, king_f - 1), min(8, king_f + 2)):
        if not (chess.BB_FILES[f] & all_pawns):
            return True  # fully open file through king flank

    return False


def _has_trapped_piece(board: chess.Board, color: chess.Color) -> bool:
    """Non-king piece attacked by enemy with no undefended escape square."""
    opp = not color
    friendly_bb = int(board.occupied_co[color])
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            if not board.is_attacked_by(opp, sq):
                continue
            targets = board.attacks_mask(sq) & ~friendly_bb
            if not targets:
                return True  # completely hemmed in and attacked
            # No undefended escape square exists → trapped
            if all(board.is_attacked_by(opp, to_sq)
                   for to_sq in chess.scan_forward(targets)):
                return True
    return False


# ── endgame type detectors ────────────────────────────────────────────────────

def _is_rook_endgame(board: chess.Board) -> bool:
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.BISHOP, chess.KNIGHT):
            if board.pieces(pt, color):
                return False
    return bool(board.pieces(chess.ROOK, chess.WHITE) or board.pieces(chess.ROOK, chess.BLACK))


def _is_pawn_endgame(board: chess.Board) -> bool:
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            if board.pieces(pt, color):
                return False
    return True


def _is_bishop_endgame(board: chess.Board) -> bool:
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.KNIGHT):
            if board.pieces(pt, color):
                return False
    return bool(board.pieces(chess.BISHOP, chess.WHITE) or board.pieces(chess.BISHOP, chess.BLACK))


def _is_knight_endgame(board: chess.Board) -> bool:
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            if board.pieces(pt, color):
                return False
    return bool(board.pieces(chess.KNIGHT, chess.WHITE) or board.pieces(chess.KNIGHT, chess.BLACK))


def _is_queen_endgame(board: chess.Board) -> bool:
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
            if board.pieces(pt, color):
                return False
    return bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))


def _is_ocb_endgame(board: chess.Board) -> bool:
    """Opposite-colored bishop endgame — strongest static draw indicator.

    True when each side has exactly one bishop and they stand on squares of
    opposite colors, with no rooks, queens, or knights remaining.  OCB endings
    are drawn in the vast majority of cases regardless of pawn count.
    """
    wb = board.pieces(chess.BISHOP, chess.WHITE)
    bb = board.pieces(chess.BISHOP, chess.BLACK)
    if len(wb) != 1 or len(bb) != 1:
        return False
    for pt in (chess.ROOK, chess.QUEEN, chess.KNIGHT):
        if board.pieces(pt, chess.WHITE) or board.pieces(pt, chess.BLACK):
            return False
    w_sq = next(iter(wb))
    b_sq = next(iter(bb))
    return _bishop_parity(w_sq) != _bishop_parity(b_sq)


def _has_development_lead(board: chess.Board, color: chess.Color) -> bool:
    """In the opening, color has at least 2 more minor pieces developed than the opponent."""
    if board.fullmove_number > 15:
        return False

    def _developed(c: chess.Color) -> int:
        back = 0 if c == chess.WHITE else 7
        n = 0
        for pt in (chess.KNIGHT, chess.BISHOP):
            start_files = {1, 6} if pt == chess.KNIGHT else {2, 5}
            for sq in board.pieces(pt, c):
                if chess.square_rank(sq) != back or chess.square_file(sq) not in start_files:
                    n += 1
        return n

    return _developed(color) >= _developed(not color) + 2


def _has_piece_activity(board: chess.Board, color: chess.Color) -> bool:
    """color has significantly more attacking mobility than the opponent."""
    def _mobility(c: chess.Color) -> int:
        return sum(
            len(board.attacks(sq))
            for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
            for sq in board.pieces(pt, c)
        )

    my_mob  = _mobility(color)
    opp_mob = _mobility(not color)
    return my_mob > opp_mob * 1.35 and my_mob - opp_mob >= 8


def _has_square_control(board: chess.Board, color: chess.Color) -> bool:
    """color controls 3+ central squares (d4, d5, e4, e5) that the opponent does not."""
    opp    = not color
    center = [chess.D4, chess.D5, chess.E4, chess.E5]
    count  = sum(
        1 for sq in center
        if board.is_attacked_by(color, sq) and not board.is_attacked_by(opp, sq)
    )
    return count >= 3


def _has_king_activity(board: chess.Board, color: chess.Color) -> bool:
    """King is centralized in an endgame (files c–f, ranks 3–6)."""
    if not _is_endgame(board):
        return False
    king_sq = board.king(color)
    if king_sq is None:
        return False
    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)
    return 2 <= f <= 5 and 2 <= r <= 5


def _has_space_advantage(board: chess.Board, color: chess.Color) -> bool:
    """Significantly more territory: pawns advanced past the center line."""
    opp = not color
    my_space  = sum(
        max(0, chess.square_rank(sq) - 3)
        if color == chess.WHITE
        else max(0, 4 - chess.square_rank(sq))
        for sq in board.pieces(chess.PAWN, color)
    )
    opp_space = sum(
        max(0, chess.square_rank(sq) - 3)
        if opp == chess.WHITE
        else max(0, 4 - chess.square_rank(sq))
        for sq in board.pieces(chess.PAWN, opp)
    )
    return my_space >= opp_space + 4


def _has_minority_attack(board: chess.Board, color: chess.Color) -> bool:
    opp    = not color
    my_qs  = sum(1 for sq in board.pieces(chess.PAWN, color) if chess.square_file(sq) < 4)
    opp_qs = sum(1 for sq in board.pieces(chess.PAWN, opp)   if chess.square_file(sq) < 4)
    if my_qs != 2 or opp_qs != 3:
        return False
    for sq in board.pieces(chess.PAWN, color):
        if chess.square_file(sq) >= 4:
            continue
        r = chess.square_rank(sq)
        if color == chess.WHITE and r >= 3:
            return True
        if color == chess.BLACK and r <= 4:
            return True
    return False


def _has_color_complex(board: chess.Board, color: chess.Color) -> bool:
    opp       = not color
    bishops   = board.pieces(chess.BISHOP, color)
    opp_pawns = board.pieces(chess.PAWN,   opp)
    if not bishops or len(opp_pawns) < 3:
        return False
    for bsq in bishops:
        bc = _bishop_parity(bsq)
        same  = sum(1 for p in opp_pawns if _bishop_parity(p) == bc)
        other = len(opp_pawns) - same
        if same > other and same >= 3:
            return True
    return False


# ── new tactical detectors ────────────────────────────────────────────────────

def _has_x_ray(board: chess.Board, color: chess.Color) -> bool:
    """A sliding piece of `color` has a second piece on its ray past a blocker."""
    for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for src in board.pieces(pt, color):
            dirs: list[tuple[int, int]] = []
            if pt in (chess.ROOK, chess.QUEEN):
                dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if pt in (chess.BISHOP, chess.QUEEN):
                dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for df, dr in dirs:
                cf = chess.square_file(src) + df
                cr = chess.square_rank(src) + dr
                blocker_found = False
                while 0 <= cf <= 7 and 0 <= cr <= 7:
                    p = board.piece_at(chess.square(cf, cr))
                    if p is not None:
                        if blocker_found:
                            return True
                        blocker_found = True
                    cf += df
                    cr += dr
    return False


def _has_discovery(board: chess.Board, color: chess.Color) -> bool:
    """A friendly piece screens a slider from a valuable enemy target it could reveal."""
    opp = not color
    valuable = {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING}
    for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for slider in board.pieces(pt, color):
            dirs: list[tuple[int, int]] = []
            if pt in (chess.ROOK, chess.QUEEN):
                dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if pt in (chess.BISHOP, chess.QUEEN):
                dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for df, dr in dirs:
                cf = chess.square_file(slider) + df
                cr = chess.square_rank(slider) + dr
                blocker_sq: int | None = None
                while 0 <= cf <= 7 and 0 <= cr <= 7:
                    csq = chess.square(cf, cr)
                    p = board.piece_at(csq)
                    if p is not None:
                        if blocker_sq is None:
                            if p.color == color:
                                blocker_sq = csq
                            else:
                                break
                        else:
                            if p.color == opp and p.piece_type in valuable:
                                return True
                            break
                    cf += df
                    cr += dr
    return False


def _has_mating_attack(board: chess.Board, color: chess.Color) -> bool:
    """2+ non-pawn pieces of `color` attack the enemy king zone and outnumber defenders."""
    opp = not color
    king_sq = board.king(opp)
    if king_sq is None:
        return False
    king_f = chess.square_file(king_sq)
    king_r = chess.square_rank(king_sq)
    zone_bb = int(chess.BB_KING_ATTACKS[king_sq]) | (1 << king_sq)
    adj_mask = (chess.BB_FILES[max(0, king_f - 1)] | chess.BB_FILES[king_f] |
                chess.BB_FILES[min(7, king_f + 1)])
    if color == chess.WHITE and king_r >= 1:
        zone_bb |= int(chess.BB_RANKS[king_r - 1]) & adj_mask
    elif color == chess.BLACK and king_r <= 6:
        zone_bb |= int(chess.BB_RANKS[king_r + 1]) & adj_mask
    attackers = sum(
        1 for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for sq in board.pieces(pt, color)
        if board.attacks_mask(sq) & zone_bb
    )
    if attackers < 2:
        return False
    defenders = sum(
        1 for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.PAWN)
        for sq in board.pieces(pt, opp)
        if board.attacks_mask(sq) & zone_bb
    )
    return attackers > defenders


def _has_interference(board: chess.Board, color: chess.Color) -> bool:
    """color can interpose on the defense ray between an enemy slider and the piece it defends.
    Condition: enemy slider X---gap_sq(s)---enemy_piece, and we attack one of those gap squares.
    """
    opp = not color
    for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for def_sq in board.pieces(pt, opp):
            df_s = chess.square_file(def_sq)
            dr_s = chess.square_rank(def_sq)
            dirs: list[tuple[int, int]] = []
            if pt in (chess.ROOK, chess.QUEEN):
                dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if pt in (chess.BISHOP, chess.QUEEN):
                dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for df, dr in dirs:
                cf, cr = df_s + df, dr_s + dr
                gap_sqs: list[int] = []
                while 0 <= cf <= 7 and 0 <= cr <= 7:
                    csq = chess.square(cf, cr)
                    p = board.piece_at(csq)
                    if p is not None:
                        if p.color == opp and gap_sqs:
                            for gsq in gap_sqs:
                                if board.is_attacked_by(color, gsq):
                                    return True
                        break
                    gap_sqs.append(csq)
                    cf += df
                    cr += dr
    return False


def _has_initiative(board: chess.Board, color: chess.Color) -> bool:
    """color generates more winning threats (more attackers than defenders on enemy pieces)
    than the opponent generates on color's pieces."""
    opp = not color

    def _winning_threat_count(attacker: chess.Color) -> int:
        defender = not attacker
        count = 0
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(piece_type, defender):
                att = chess.popcount(board.attackers_mask(attacker, sq))
                dfs = chess.popcount(board.attackers_mask(defender, sq))
                if att > dfs:
                    count += 1
        return count

    return _winning_threat_count(color) > _winning_threat_count(opp)


def _has_prophylaxis_pos(board: chess.Board, color: chess.Color) -> bool:
    """Prophylaxis: color has an overprotected piece (≥2 more defenders than attackers)
    OR dominates a key outpost square that an enemy piece is eyeing but cannot reach."""
    opp = not color
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(piece_type, color):
            dfs = chess.popcount(board.attackers_mask(color, sq))
            att = chess.popcount(board.attackers_mask(opp, sq))
            if dfs >= att + 2:
                return True
    opp_outposts = _outpost_squares_bb(board, opp)
    for sq in chess.scan_forward(opp_outposts):
        if board.piece_at(sq):
            continue
        our_ctrl = chess.popcount(board.attackers_mask(color, sq))
        opp_ctrl = chess.popcount(board.attackers_mask(opp, sq))
        if our_ctrl > opp_ctrl:
            for opp_pt in (chess.KNIGHT, chess.BISHOP):
                for opp_sq in board.pieces(opp_pt, opp):
                    if sq in board.attacks(opp_sq):
                        return True
    return False


def _has_sacrifice(board: chess.Board, color: chess.Color) -> bool:
    """color has offered a losing exchange: a piece attacked by a less valuable enemy with
    net material loss on capture, OR is materially down ≥3 points with king-attack compensation."""
    opp = not color
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        pt_val = _PIECE_VALUES[pt]
        for sq in board.pieces(pt, color):
            for att_sq in board.attackers(opp, sq):
                att_pt = board.piece_at(att_sq).piece_type
                if _PIECE_VALUES.get(att_pt, 0) < pt_val:
                    att = chess.popcount(board.attackers_mask(opp, sq))
                    dfs = chess.popcount(board.attackers_mask(color, sq))
                    if att >= dfs:
                        return True
    our_mat = _material_value(board, color)
    opp_mat = _material_value(board, opp)
    if our_mat <= opp_mat - 3:
        king_sq = board.king(opp)
        if king_sq is not None:
            zone_bb = int(chess.BB_KING_ATTACKS[king_sq]) | (1 << king_sq)
            attackers = sum(
                1 for apt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                for s in board.pieces(apt, color)
                if board.attacks_mask(s) & zone_bb
            )
            if attackers >= 2:
                return True
    return False


def _has_clearance(board: chess.Board, color: chess.Color) -> bool:
    """A friendly non-slider blocks a friendly slider's ray to a valuable enemy piece.
    Moving the blocker would reveal the slider's attack — a clearance sacrifice."""
    opp = not color
    valuable = {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
    slider_types = {chess.BISHOP, chess.ROOK, chess.QUEEN}
    for slider_pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        for slider_sq in board.pieces(slider_pt, color):
            sf = chess.square_file(slider_sq)
            sr = chess.square_rank(slider_sq)
            dirs: list[tuple[int, int]] = []
            if slider_pt in (chess.ROOK, chess.QUEEN):
                dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if slider_pt in (chess.BISHOP, chess.QUEEN):
                dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for df, dr in dirs:
                # Walk the ray to find the first piece (the potential blocker).
                cf, cr = sf + df, sr + dr
                while 0 <= cf <= 7 and 0 <= cr <= 7:
                    blocker_sq = chess.square(cf, cr)
                    blocker = board.piece_at(blocker_sq)
                    if blocker is not None:
                        if blocker.color == color and blocker.piece_type not in slider_types:
                            # Friendly non-slider blocks this ray. Look beyond for valuable enemy.
                            cf2, cr2 = cf + df, cr + dr
                            while 0 <= cf2 <= 7 and 0 <= cr2 <= 7:
                                tgt_sq = chess.square(cf2, cr2)
                                tgt = board.piece_at(tgt_sq)
                                if tgt is not None:
                                    if tgt.color == opp and tgt.piece_type in valuable:
                                        return True
                                    break
                                cf2 += df
                                cr2 += dr
                        break  # any piece stops the ray search for this direction
                    cf += df
                    cr += dr
    return False


def _has_deflection(board: chess.Board, color: chess.Color) -> bool:
    """An enemy piece is the sole defender of a valuable enemy piece (queen/rook),
    AND we attack that defender — deflecting it exposes the protected target."""
    opp = not color
    valuable = {chess.QUEEN, chess.ROOK}
    for def_sq in chess.SQUARES:
        defender = board.piece_at(def_sq)
        if defender is None or defender.color != opp:
            continue
        if defender.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
            continue
        for tgt_sq in board.attacks(def_sq):
            tgt = board.piece_at(tgt_sq)
            if tgt is None or tgt.color != opp or tgt.piece_type not in valuable:
                continue
            defenders_of_tgt = board.attackers(opp, tgt_sq)
            if len(defenders_of_tgt) == 1 and def_sq in defenders_of_tgt:
                if board.is_attacked_by(color, def_sq):
                    return True
    return False


def _has_zwischenzug(board: chess.Board, color: chess.Color) -> bool:
    """A losing exchange is threatened against color's piece, but color can play
    a forcing check first (the in-between move) instead of immediately defending."""
    if board.turn != color:
        return False
    opp = not color
    in_danger = False
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for sq in board.pieces(pt, color):
            if chess.popcount(board.attackers_mask(opp, sq)) > chess.popcount(board.attackers_mask(color, sq)):
                in_danger = True
                break
        if in_danger:
            break
    if not in_danger:
        return False
    for move in board.legal_moves:
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        if gives_check:
            return True
    return False


def _has_double_check(board: chess.Board) -> bool:
    """The side to move is in double check (two pieces giving check simultaneously)."""
    return board.is_check() and len(board.checkers()) >= 2


def _has_zugzwang_heuristic(board: chess.Board) -> bool:
    """Endgame with ≤5 legal moves for the side to move and no pawn advances available."""
    if not _is_endgame(board):
        return False
    if board.legal_moves.count() > 5:
        return False
    for color in (chess.WHITE, chess.BLACK):
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            adv = chess.square_rank(sq) + (1 if color == chess.WHITE else -1)
            if 0 <= adv <= 7 and board.piece_at(chess.square(f, adv)) is None:
                return False
    return True


def _has_shouldering(board: chess.Board) -> bool:
    """Pawn endgame where kings are one file apart, one king cutting off the other."""
    if not _is_pawn_endgame(board):
        return False
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return False
    wkf = chess.square_file(wk)
    bkf = chess.square_file(bk)
    if abs(wkf - bkf) != 1:
        return False
    if abs(chess.square_rank(wk) - chess.square_rank(bk)) > 1:
        return False
    return bool(board.pieces(chess.PAWN, chess.WHITE) or board.pieces(chess.PAWN, chess.BLACK))


# ── main entry point ──────────────────────────────────────────────────────────

def label_position(board: chess.Board) -> frozenset[str]:
    """
    Return the set of chess concept labels provably present in this position.
    Checks both sides; a concept is labelled if present for either.
    Result is always a subset of DETECTABLE_CONCEPTS.
    """
    labels: set[str] = set()

    # ── position-wide ────────────────────────────────────────────────────────
    if _has_opposition(board) and _is_endgame(board):
        labels.add("opposition")

    if _is_pawn_endgame(board):
        labels.add("pawn_endgame")
    elif _is_rook_endgame(board):
        labels.add("rook_endgame")
    elif _is_bishop_endgame(board):
        labels.add("bishop_endgame")
    elif _is_knight_endgame(board):
        labels.add("knight_endgame")
    elif _is_queen_endgame(board):
        labels.add("queen_endgame")

    if board.is_insufficient_material() or _is_ocb_endgame(board) or _is_syzygy_draw(board):
        labels.add("drawn_position")

    if _has_double_check(board):
        labels.add("double_check")
    if _has_zugzwang_heuristic(board):
        labels.add("zugzwang")
    if _has_shouldering(board):
        labels.add("shouldering")

    for color in (chess.WHITE, chess.BLACK):
        # pawn structure
        if _has_passed_pawn(board, color):
            labels.add("passed_pawn")
        if _has_isolated_pawn(board, color):
            labels.add("isolated_pawn")
        if _has_doubled_pawn(board, color):
            labels.add("doubled_pawn")
        if _pawn_island_count(board, color) >= 2:
            labels.add("pawn_island")
        if _has_pawn_majority(board, color):
            labels.add("pawn_majority")
        if _has_backward_pawn(board, color):
            labels.add("backward_pawn")
        if _has_pawn_chain(board, color):
            labels.add("pawn_chain")
        if _has_pawn_storm(board, color):
            labels.add("pawn_storm")
        if _has_promotion(board, color):
            labels.add("promotion")

        # bishop quality
        if _has_bad_bishop(board, color):
            labels.add("bad_bishop")
        if _has_good_bishop(board, color):
            labels.add("good_bishop")
        if _has_bishop_pair(board, color):
            labels.add("bishop_pair")

        # piece placement / activity
        if _has_battery(board, color):
            labels.add("battery")
        if _has_blockade(board, color):
            labels.add("blockade")
        if _has_piece_activity(board, color) or _has_square_control(board, color):
            labels.add("piece_activity")
        if _has_development_lead(board, color):
            labels.add("development_lead")
        if _has_trapped_piece(board, color):
            labels.add("trapped_piece")

        # files / ranks
        if _has_open_file(board, color):
            labels.add("open_file")
        if _has_rook_on_seventh(board, color):
            labels.add("rook_seventh")

        # square control / king structure
        if _has_outpost(board, color):
            labels.add("outpost")
        if _has_weak_square(board, color):
            labels.add("weak_square")
        if _has_space_advantage(board, color):
            labels.add("space_advantage")

        # king
        if _has_back_rank_weakness(board, color):
            labels.add("back_rank")
        if _has_king_activity(board, color):
            labels.add("king_activity")
        if _has_king_safety(board, color):
            labels.add("king_safety")

        # tactical
        if _has_pin(board, color):
            labels.add("pin")
        if _has_fork(board, color):
            labels.add("fork")
        if _has_skewer(board, color):
            labels.add("skewer")
        if _has_overloading(board, color):
            labels.add("overloading")
        if _has_x_ray(board, color):
            labels.add("x_ray")
        if _has_discovery(board, color):
            labels.add("discovery")
        if _has_mating_attack(board, color):
            labels.add("mating_attack")
        if _has_interference(board, color):
            labels.add("interference")
        if _has_initiative(board, color):
            labels.add("initiative")
        if _has_prophylaxis_pos(board, color):
            labels.add("prophylaxis")
        if _has_sacrifice(board, color):
            labels.add("sacrifice")
        if _has_clearance(board, color):
            labels.add("clearance")
        if _has_deflection(board, color):
            labels.add("deflection")
        if _has_zwischenzug(board, color):
            labels.add("zwischenzug")

    return frozenset(labels)


# ── concept bottleneck feature vector ────────────────────────────────────────

_PER_COLOR_DETECTORS: list[tuple[str, Callable]] = [
    ("passed_pawn",      _has_passed_pawn),
    ("isolated_pawn",    _has_isolated_pawn),
    ("doubled_pawn",     _has_doubled_pawn),
    ("pawn_island",      lambda b, c: _pawn_island_count(b, c) >= 2),
    ("pawn_majority",    _has_pawn_majority),
    ("backward_pawn",    _has_backward_pawn),
    ("pawn_chain",       _has_pawn_chain),
    ("pawn_storm",       _has_pawn_storm),
    ("bad_bishop",       _has_bad_bishop),
    ("good_bishop",      _has_good_bishop),
    ("bishop_pair",      _has_bishop_pair),
    ("battery",          _has_battery),
    ("blockade",         _has_blockade),
    ("outpost",          _has_outpost),
    ("rook_seventh",     _has_rook_on_seventh),
    ("piece_activity",   lambda b, c: _has_piece_activity(b, c) or _has_square_control(b, c)),
    ("king_activity",    _has_king_activity),
    ("back_rank",        _has_back_rank_weakness),
    ("open_file",        _has_open_file),
    ("weak_square",      _has_weak_square),
    ("space_advantage",  _has_space_advantage),
    ("development_lead", _has_development_lead),
    ("pin",              _has_pin),
    ("fork",             _has_fork),
    ("skewer",           _has_skewer),
    ("overloading",      _has_overloading),
    ("x_ray",            _has_x_ray),
    ("discovery",        _has_discovery),
    ("mating_attack",    _has_mating_attack),
    ("interference",     _has_interference),
    ("initiative",       _has_initiative),
    ("prophylaxis",      _has_prophylaxis_pos),
    ("sacrifice",        _has_sacrifice),
    ("clearance",        _has_clearance),
    ("deflection",       _has_deflection),
    ("zwischenzug",      _has_zwischenzug),
]  # 36 × 2 = 72 bits

_GLOBAL_DETECTORS: list[tuple[str, Callable]] = [
    ("opposition",     lambda b: _has_opposition(b) and _is_endgame(b)),
    ("rook_endgame",   _is_rook_endgame),
    ("pawn_endgame",   _is_pawn_endgame),
    ("bishop_endgame", _is_bishop_endgame),
    ("knight_endgame", _is_knight_endgame),
    ("queen_endgame",  _is_queen_endgame),
    ("drawn_position", lambda b: b.is_insufficient_material() or _is_ocb_endgame(b) or _is_syzygy_draw(b)),
    ("double_check",   _has_double_check),
    ("zugzwang",       _has_zugzwang_heuristic),
    ("shouldering",    _has_shouldering),
]  # 10 bits

ALGO_FEATURE_SIZE: int = len(_PER_COLOR_DETECTORS) * 2 + len(_GLOBAL_DETECTORS)  # 82


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Spatial detector maps and B6 formerly-binary feature vectors
#
# These functions are wired into algo_feature_vector_v4() below.
# algo_feature_vector() (68-dim v3) is kept for the bypass channel in the net.
#
# ALGO_FEATURE_SIZE_V4 layout  (3265 dims total):
#   [0:512]     B1-B5 spatial maps — 4 concepts × 2 colors × 64 squares
#                 weak_square_map ×2 (128), outpost_map ×2 (128),
#                 backward_pawn_map ×2 (128), passed_pawn_map ×2 (128)
#   [512:642]   B6 bishop_pair        (130)
#   [642:778]   B6 development        (136)
#   [778:1162]  B6 x_ray              (384)
#   [1162:1306] B6 battery            (144)
#   [1306:1321] B6 opposition         ( 15)
#   [1321:1325] B6 zugzwang tier-1    (  4)
#   [1325:1359] B6 rook_seventh       ( 34)
#   [1359:1361] B6 drawn_position     (  2)
#   [1361:1382] B6 shouldering        ( 21)
#   [1382:1512] B6 double_check       (130)
#   [1512:1530] B6 promotion          ( 18)
#   [1530:1663] B6 bishop_endgame     (133)
#   [1663:1811] B7 king_safety        (148)
#   [1811:2067] B8 pin_vec            (256)  w_pinned+pinner, b_pinned+pinner maps
#   [2067:2323] B8 fork_vec           (256)  w/b forking+forked square maps
#   [2323:2451] B8 isolated_pawn_vec  (128)  w/b isolated pawn maps
#   [2451:2491] B8 open_file_vec      ( 40)  rook-file presence + open/semi-open flags
#   [2491:2619] B9 pawn_chain_vec     (128)  w/b pawn chain membership (base + members)
#   [2619:2749] B9 pawn_island_vec    (130)  w/b connected-pawn maps + island count norm
#   [2749:2877] B9 mating_pressure    (128)  w/b pieces-hitting-king-zone square maps
#   [2877:3005] B9 interference_vec   (128)  w/b interposition gap squares (defense-ray disruption)
#   [3005:3135] B9 initiative_vec     (130)  w/b active-threat pieces + normalized threat count
#   [3135:3265] B9 prophylaxis_vec    (130)  w/b overprotected pieces + key outpost control ratio
#   [3265:3395] B9 sacrifice_vec      (130)  w/b offered-piece squares + material deficit norm
#   [3395:3523] B9 clearance_vec      (128)  w/b blocking-piece squares (slider ray blocked)
#   [3523:3651] B9 deflection_vec     (128)  w/b deflectable-defender squares
#   [3651:3779] B9 zwischenzug_vec    (128)  w/b under-threat piece squares + pending-intermezzo sq
# ═══════════════════════════════════════════════════════════════════════════════

# ── B2: Weak square map (bitboard) ────────────────────────────────────────────

def _weak_square_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """64-element map of squares strategically weak for `color`.
    A square is weak if no friendly pawn on an adjacent file can advance to defend it.
    Marks strategically relevant territory only (ranks 4-8 for white, 1-5 for black).
    """
    wp = int(board.pieces_mask(chess.PAWN, color))
    # Defended = north fill (for white) strictly above east/west adjacent pawn positions.
    # A pawn at (f, r) can defend squares on file f±1 at ranks r+1 and above.
    if color == chess.WHITE:
        defended = (_north_fill(_bb_north(_bb_east(wp))) |
                    _north_fill(_bb_north(_bb_west(wp))))
        territory = (chess.BB_RANK_3 | chess.BB_RANK_4 | chess.BB_RANK_5 |
                     chess.BB_RANK_6 | chess.BB_RANK_7)
    else:
        defended = (_south_fill(_bb_south(_bb_east(wp))) |
                    _south_fill(_bb_south(_bb_west(wp))))
        territory = (chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 |
                     chess.BB_RANK_4 | chess.BB_RANK_5)
    weak_bb = int(territory) & ~defended
    out = np.zeros(64, dtype=np.float32)
    for sq in chess.scan_forward(weak_bb):
        out[sq] = 1.0
    return out


# ── B3: Outpost map (bitboard) ────────────────────────────────────────────────

def _outpost_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """64-element map of squares in advanced territory that are weak for the opponent."""
    opp_weak_bb = _outpost_squares_bb(board, not color)
    out = np.zeros(64, dtype=np.float32)
    if color == chess.WHITE:
        deep = opp_weak_bb & (chess.BB_RANK_4 | chess.BB_RANK_5 |
                               chess.BB_RANK_6 | chess.BB_RANK_7)
    else:
        deep = opp_weak_bb & (chess.BB_RANK_1 | chess.BB_RANK_2 |
                               chess.BB_RANK_3 | chess.BB_RANK_4)
    for sq in chess.scan_forward(deep):
        out[sq] = 1.0
    return out


# ── B4: Backward pawn map (bitboard) ─────────────────────────────────────────

def _backward_pawn_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """64-element map of backward pawn squares for `color`."""
    wp = int(board.pieces_mask(chess.PAWN, color))
    ep = int(board.pieces_mask(chess.PAWN, not color))
    out = np.zeros(64, dtype=np.float32)
    if not wp:
        return out
    if color == chess.WHITE:
        stop = _bb_north(wp)
        at_risk = _bb_south(stop & _bp_attacks(ep)) & wp
        for sq in chess.scan_forward(at_risk):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            adj = (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
            if not (wp & adj & _south_fill(chess.BB_RANKS[r])):
                out[sq] = 1.0
    else:
        stop = _bb_south(wp)
        at_risk = _bb_north(stop & _wp_attacks(ep)) & wp
        for sq in chess.scan_forward(at_risk):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            adj = (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
            if not (wp & adj & _north_fill(chess.BB_RANKS[r])):
                out[sq] = 1.0
    return out


# ── B5: Passed pawn map (bitboard) ───────────────────────────────────────────

def _passed_pawn_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """64-element map of passed pawn squares for `color` (all ranks, no threshold)."""
    wp = int(board.pieces_mask(chess.PAWN, color))
    ep = int(board.pieces_mask(chess.PAWN, not color))
    if color == chess.WHITE:
        not_passed = _south_fill(ep | _bb_east(ep) | _bb_west(ep))
    else:
        not_passed = _north_fill(ep | _bb_east(ep) | _bb_west(ep))
    passed_bb = wp & ~not_passed
    out = np.zeros(64, dtype=np.float32)
    for sq in chess.scan_forward(passed_bb):
        out[sq] = 1.0
    return out


# ── B6: Bishop pair (130 dims) ────────────────────────────────────────────────

def _bishop_pair_vec(board: chess.Board) -> np.ndarray:
    """[white_sq_map(64), black_sq_map(64), white_has_pair(1), black_has_pair(1)]"""
    out = np.zeros(130, dtype=np.float32)
    w_bs = board.pieces(chess.BISHOP, chess.WHITE)
    b_bs = board.pieces(chess.BISHOP, chess.BLACK)
    for sq in w_bs:
        out[sq] = 1.0
    for sq in b_bs:
        out[64 + sq] = 1.0
    out[128] = 1.0 if len(w_bs) >= 2 else 0.0
    out[129] = 1.0 if len(b_bs) >= 2 else 0.0
    return out


# ── B6: Development (136 dims) ────────────────────────────────────────────────

_W_KNIGHT_START = frozenset({chess.B1, chess.G1})
_W_BISHOP_START = frozenset({chess.C1, chess.F1})
_W_ROOK_START   = frozenset({chess.A1, chess.H1})
_W_QUEEN_START  = frozenset({chess.D1})
_B_KNIGHT_START = frozenset({chess.B8, chess.G8})
_B_BISHOP_START = frozenset({chess.C8, chess.F8})
_B_ROOK_START   = frozenset({chess.A8, chess.H8})
_B_QUEEN_START  = frozenset({chess.D8})

def _development_vec(board: chess.Board) -> np.ndarray:
    """[white_dev_map(64), black_dev_map(64), white_counts(4), black_counts(4)]
    counts: [knights_out/2, bishops_out/2, rooks_connected, queen_moved]
    """
    out = np.zeros(136, dtype=np.float32)
    for color, offset, k_s, b_s, r_s, q_s in (
        (chess.WHITE, 0,  _W_KNIGHT_START, _W_BISHOP_START, _W_ROOK_START, _W_QUEEN_START),
        (chess.BLACK, 64, _B_KNIGHT_START, _B_BISHOP_START, _B_ROOK_START, _B_QUEEN_START),
    ):
        kn_out = 0
        bi_out = 0
        q_moved = 0
        for sq in board.pieces(chess.KNIGHT, color):
            if sq not in k_s:
                out[offset + sq] = 1.0
                kn_out += 1
        for sq in board.pieces(chess.BISHOP, color):
            if sq not in b_s:
                out[offset + sq] = 1.0
                bi_out += 1
        for sq in board.pieces(chess.QUEEN, color):
            if sq not in q_s:
                out[offset + sq] = 1.0
                q_moved = 1
        rooks = list(board.pieces(chess.ROOK, color))
        rooks_connected = 0
        if len(rooks) == 2:
            r1, r2 = rooks
            f1, r1_r = chess.square_file(r1), chess.square_rank(r1)
            f2, r2_r = chess.square_file(r2), chess.square_rank(r2)
            if r1_r == r2_r:
                lo, hi = sorted([f1, f2])
                if all(board.piece_at(chess.square(f, r1_r)) is None for f in range(lo + 1, hi)):
                    rooks_connected = 1
            elif f1 == f2:
                lo, hi = sorted([r1_r, r2_r])
                if all(board.piece_at(chess.square(f1, r)) is None for r in range(lo + 1, hi)):
                    rooks_connected = 1
        c_base = 128 + (0 if color == chess.WHITE else 4)
        out[c_base + 0] = kn_out / 2.0
        out[c_base + 1] = bi_out / 2.0
        out[c_base + 2] = float(rooks_connected)
        out[c_base + 3] = float(q_moved)
    return out


# ── B6: X-ray attack / defend / clearance (384 dims) ─────────────────────────

def _xray_vec(board: chess.Board) -> np.ndarray:
    """[w_attack(64), w_defend(64), w_clear(64), b_attack(64), b_defend(64), b_clear(64)]"""
    out = np.zeros(384, dtype=np.float32)
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 192)):
        for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            for src in board.pieces(pt, color):
                sf, sr = chess.square_file(src), chess.square_rank(src)
                dirs: list[tuple[int, int]] = []
                if pt in (chess.ROOK, chess.QUEEN):
                    dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if pt in (chess.BISHOP, chess.QUEEN):
                    dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for df, dr in dirs:
                    cf, cr = sf + df, sr + dr
                    blocker_sq: int | None = None
                    blocker_col: chess.Color | None = None
                    while 0 <= cf <= 7 and 0 <= cr <= 7:
                        csq = chess.square(cf, cr)
                        p = board.piece_at(csq)
                        if p is not None:
                            if blocker_sq is None:
                                blocker_sq = csq
                                blocker_col = p.color
                                if blocker_col == color:
                                    out[base + 128 + csq] = 1.0
                            else:
                                if blocker_col == (not color) and p.color == (not color):
                                    out[base + csq] = 1.0
                                elif blocker_col == color and p.color == color:
                                    out[base + 64 + csq] = 1.0
                                break
                        cf += df
                        cr += dr
    return out


# ── B6: Battery (144 dims) ────────────────────────────────────────────────────

def _battery_vec(board: chess.Board) -> np.ndarray:
    """[w_file(8), b_file(8), w_diag(64), b_diag(64)]"""
    out = np.zeros(144, dtype=np.float32)
    for color, f_base, d_base in ((chess.WHITE, 0, 16), (chess.BLACK, 8, 80)):
        rooks   = board.pieces(chess.ROOK,   color)
        queens  = board.pieces(chess.QUEEN,  color)
        bishops = board.pieces(chess.BISHOP, color)
        heavy = list(rooks | queens)
        for i in range(len(heavy)):
            for j in range(i + 1, len(heavy)):
                sq1, sq2 = heavy[i], heavy[j]
                f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
                f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
                if f1 == f2 and sq2 in board.attacks(sq1):
                    out[f_base + f1] = 1.0
        diag = list(bishops | queens)
        for i in range(len(diag)):
            for j in range(i + 1, len(diag)):
                sq1, sq2 = diag[i], diag[j]
                f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
                f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
                if abs(f1 - f2) == abs(r1 - r2) and sq2 in board.attacks(sq1):
                    out[d_base + sq1] = 1.0
                    out[d_base + sq2] = 1.0
    return out


# ── B6: Opposition (15 dims) ─────────────────────────────────────────────────

def _opposition_vec(board: chess.Board) -> np.ndarray:
    """15-dim: [is_kp_endgame, kings_same_file, kings_same_rank, kings_same_diag,
                gap_one_hot(8), white_has_opp, black_has_opp, rook_pawn_draw]"""
    out = np.zeros(15, dtype=np.float32)
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return out
    is_kpe = _is_pawn_endgame(board)
    out[0] = 1.0 if is_kpe else 0.0
    wkf, wkr = chess.square_file(wk), chess.square_rank(wk)
    bkf, bkr = chess.square_file(bk), chess.square_rank(bk)
    df = abs(wkf - bkf)
    dr = abs(wkr - bkr)
    out[1] = 1.0 if df == 0 else 0.0
    out[2] = 1.0 if dr == 0 else 0.0
    out[3] = 1.0 if df == dr else 0.0
    gap = max(df, dr)
    if 0 <= gap <= 7:
        out[4 + gap] = 1.0
    facing = (df == 0 or dr == 0 or df == dr) and gap % 2 == 1
    out[12] = 1.0 if (facing and board.turn == chess.BLACK) else 0.0
    out[13] = 1.0 if (facing and board.turn == chess.WHITE) else 0.0
    rp_files = {0, 7}
    has_rp = any(chess.square_file(sq) in rp_files
                 for sq in board.pieces(chess.PAWN, chess.WHITE) |
                            board.pieces(chess.PAWN, chess.BLACK))
    near_corner = (bkf in rp_files and bkr >= 6) or (wkf in rp_files and wkr <= 1)
    out[14] = 1.0 if (is_kpe and has_rp and near_corner) else 0.0
    return out


# ── B6: Zugzwang tier-1 (4 dims) ─────────────────────────────────────────────

def _zugzwang_vec(board: chess.Board) -> np.ndarray:
    """[is_endgame, white_few_moves, black_few_moves, pawn_breaks_blocked]

    Threshold raised from ≤3 to ≤5 legal moves — 3 almost never fired in real
    positions.  5 still signals severe restriction while covering common K+P
    vs K zugzwang structures where 4-5 moves are available but all worsen.
    """
    out = np.zeros(4, dtype=np.float32)
    out[0] = 1.0 if _is_endgame(board) else 0.0
    if board.turn == chess.WHITE:
        out[1] = 1.0 if board.legal_moves.count() <= 5 else 0.0
    else:
        out[2] = 1.0 if board.legal_moves.count() <= 5 else 0.0
    breaks_exist = False
    for color in (chess.WHITE, chess.BLACK):
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            adv = chess.square_rank(sq) + (1 if color == chess.WHITE else -1)
            if 0 <= adv <= 7 and board.piece_at(chess.square(f, adv)) is None:
                breaks_exist = True
                break
        if breaks_exist:
            break
    out[3] = 0.0 if breaks_exist else 1.0
    return out


# ── B6: Rook on seventh (34 dims) ────────────────────────────────────────────

def _rook_seventh_vec(board: chess.Board) -> np.ndarray:
    """[w_rook_7th_files(8), b_rook_2nd_files(8), b_king_8th(1), w_king_1st(1),
        b_pawns_7th_files(8), w_pawns_2nd_files(8)]"""
    out = np.zeros(34, dtype=np.float32)
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        if chess.square_rank(sq) == 6:
            out[chess.square_file(sq)] = 1.0
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        if chess.square_rank(sq) == 1:
            out[8 + chess.square_file(sq)] = 1.0
    bk = board.king(chess.BLACK)
    wk = board.king(chess.WHITE)
    out[16] = 1.0 if (bk is not None and chess.square_rank(bk) == 7) else 0.0
    out[17] = 1.0 if (wk is not None and chess.square_rank(wk) == 0) else 0.0
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        if chess.square_rank(sq) == 6:
            out[18 + chess.square_file(sq)] = 1.0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        if chess.square_rank(sq) == 1:
            out[26 + chess.square_file(sq)] = 1.0
    return out


# ── B6: Drawn position (2 dims) ──────────────────────────────────────────────

def _drawn_position_vec(board: chess.Board) -> np.ndarray:
    """[is_drawn_heuristic, is_ocb_endgame]

    dim 0 — any draw detected: insufficient material, OCB endgame, or Syzygy draw
    dim 1 — opposite-colored bishop endgame specifically (strong positional draw signal)

    The halfmove clock was removed — it is already encoded in fen_to_tensor and
    carries almost no signal in puzzle/annotated-game positions (usually 0).
    """
    out = np.zeros(2, dtype=np.float32)
    ocb = _is_ocb_endgame(board)
    out[0] = 1.0 if (board.is_insufficient_material() or ocb or _is_syzygy_draw(board)) else 0.0
    out[1] = 1.0 if ocb else 0.0
    return out


# ── B6: Shouldering (21 dims) ────────────────────────────────────────────────

def _shouldering_vec(board: chess.Board) -> np.ndarray:
    """[is_kp_endgame(1), w_cuts(1), b_cuts(1), target_file_oh(8), target_rank_oh(8),
        file_delta(1), rank_delta(1)]"""
    out = np.zeros(21, dtype=np.float32)
    if not _is_pawn_endgame(board):
        return out
    out[0] = 1.0
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return out
    wkf, wkr = chess.square_file(wk), chess.square_rank(wk)
    bkf, bkr = chess.square_file(bk), chess.square_rank(bk)
    w_adv = max((chess.square_rank(sq) for sq in board.pieces(chess.PAWN, chess.WHITE)),
                default=-1)
    b_adv = max((7 - chess.square_rank(sq) for sq in board.pieces(chess.PAWN, chess.BLACK)),
                default=-1)
    if w_adv >= b_adv and w_adv >= 0:
        target_file = chess.square_file(
            max(board.pieces(chess.PAWN, chess.WHITE), key=chess.square_rank))
        target_rank = 7
    elif b_adv > w_adv and b_adv >= 0:
        target_file = chess.square_file(
            min(board.pieces(chess.PAWN, chess.BLACK), key=chess.square_rank))
        target_rank = 0
    else:
        target_file, target_rank = 3, 3
    out[3 + target_file] = 1.0
    out[11 + target_rank] = 1.0
    out[19] = (wkf - bkf) / 7.0
    out[20] = (wkr - bkr) / 7.0
    if target_rank == 7 and abs(wkf - bkf) == 1 and (7 - wkr) < (7 - bkr):
        out[1] = 1.0
    if target_rank == 0 and abs(wkf - bkf) == 1 and bkr < wkr:
        out[2] = 1.0
    return out


# ── B6: Double check (130 dims) ──────────────────────────────────────────────

def _double_check_vec(board: chess.Board) -> np.ndarray:
    """[is_double_check(1), checker_sq_map(64), checked_king_sq(64), checking_side(1)]"""
    out = np.zeros(130, dtype=np.float32)
    if not board.is_check():
        return out
    checkers = board.checkers()
    is_double = len(checkers) >= 2
    out[0] = 1.0 if is_double else 0.0
    for sq in chess.scan_forward(int(checkers)):
        out[1 + sq] = 1.0
    king_sq = board.king(board.turn)
    if king_sq is not None:
        out[65 + king_sq] = 1.0
    out[129] = 1.0 if (not board.turn) == chess.WHITE else 0.0
    return out


# ── B6: Promotion (18 dims) ──────────────────────────────────────────────────

def _promotion_vec(board: chess.Board) -> np.ndarray:
    """[w_pawn_7th_files(8), b_pawn_2nd_files(8), w_promotes_now(1), b_promotes_now(1)]"""
    out = np.zeros(18, dtype=np.float32)
    w_now = False
    b_now = False
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        r = chess.square_rank(sq)
        if r == 6:
            out[chess.square_file(sq)] = 1.0
            w_now = True
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        r = chess.square_rank(sq)
        if r == 1:
            out[8 + chess.square_file(sq)] = 1.0
            b_now = True
    out[16] = 1.0 if (w_now and board.turn == chess.WHITE) else 0.0
    out[17] = 1.0 if (b_now and board.turn == chess.BLACK) else 0.0
    return out


# ── B6: Bishop endgame (133 dims) ────────────────────────────────────────────

def _bishop_endgame_vec(board: chess.Board) -> np.ndarray:
    """[is_be(1), same_color(1), opp_color(1), w_on_light(1), b_on_light(1),
        w_bishop_map(64), b_bishop_map(64)]"""
    out = np.zeros(133, dtype=np.float32)
    if not _is_bishop_endgame(board):
        return out
    out[0] = 1.0
    w_bs = board.pieces(chess.BISHOP, chess.WHITE)
    b_bs = board.pieces(chess.BISHOP, chess.BLACK)
    w_light = any(_bishop_parity(sq) == 1 for sq in w_bs)
    w_dark  = any(_bishop_parity(sq) == 0 for sq in w_bs)
    b_light = any(_bishop_parity(sq) == 1 for sq in b_bs)
    b_dark  = any(_bishop_parity(sq) == 0 for sq in b_bs)
    same_color = (w_light and b_light and not w_dark and not b_dark) or \
                 (w_dark  and b_dark  and not w_light and not b_light)
    opp_color  = (w_light and b_dark) or (w_dark and b_light)
    out[1] = 1.0 if same_color else 0.0
    out[2] = 1.0 if opp_color  else 0.0
    out[3] = 1.0 if w_light    else 0.0
    out[4] = 1.0 if b_light    else 0.0
    for sq in w_bs:
        out[5 + sq] = 1.0
    for sq in b_bs:
        out[69 + sq] = 1.0
    return out


# ── B7: King safety (148 dims) ────────────────────────────────────────────────

def _king_safety_vec(board: chess.Board) -> np.ndarray:
    """King safety spatial map.
    [w_zone_attacks(64), b_zone_attacks(64), w_shield(6), b_shield(6),
     w_open_adj(3), b_open_adj(3), w_attacker_norm(1), b_attacker_norm(1)] = 148 dims

    zone_attacks: 1.0 for each square in the king's zone attacked by the opponent.
    shield: presence of friendly pawn at (file-1,file,file+1) × (rank+1,rank+2) ahead of king.
    open_adj: open/semi-open file score for 3 adjacent files (1.0=open, 0.5=semi-open).
    attacker_norm: distinct enemy non-pawn pieces attacking king zone / 8.0.
    """
    out = np.zeros(148, dtype=np.float32)

    # Build global attack maps once
    w_atk = 0
    b_atk = 0
    for sq in chess.scan_forward(int(board.occupied_co[chess.WHITE])):
        w_atk |= board.attacks_mask(sq)
    for sq in chess.scan_forward(int(board.occupied_co[chess.BLACK])):
        b_atk |= board.attacks_mask(sq)
    all_pawns = int(board.pawns)

    for color, zone_base, shield_base, open_base, atk_base in (
        (chess.WHITE, 0,  128, 140, 146),
        (chess.BLACK, 64, 134, 143, 147),
    ):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        opp = not color
        enemy_atk = b_atk if color == chess.WHITE else w_atk
        king_f = chess.square_file(king_sq)
        king_r = chess.square_rank(king_sq)

        # King zone: 8 neighbors + 3 squares one more rank ahead
        zone_bb = int(chess.BB_KING_ATTACKS[king_sq])
        adj_mask = (chess.BB_FILES[max(0, king_f - 1)] | chess.BB_FILES[king_f] |
                    chess.BB_FILES[min(7, king_f + 1)])
        if color == chess.WHITE and king_r <= 6:
            zone_bb |= int(chess.BB_RANKS[king_r + 1]) & adj_mask
        elif color == chess.BLACK and king_r >= 1:
            zone_bb |= int(chess.BB_RANKS[king_r - 1]) & adj_mask

        # Squares in king zone attacked by enemy
        for sq in chess.scan_forward(zone_bb & enemy_atk):
            out[zone_base + sq] = 1.0

        # Pawn shield: 3 files × 2 ranks ahead (up to 6 bits)
        wp = int(board.pieces_mask(chess.PAWN, color))
        bit = 0
        for f in range(max(0, king_f - 1), min(8, king_f + 2)):
            for rank_offset in (1, 2):
                r = (king_r + rank_offset) if color == chess.WHITE else (king_r - rank_offset)
                if 0 <= r <= 7:
                    out[shield_base + bit] = 1.0 if (wp & chess.BB_SQUARES[chess.square(f, r)]) else 0.0
                bit += 1

        # Open / semi-open files near king (3 bits)
        wp_files = _file_fill(wp)
        for bit, f in enumerate(range(max(0, king_f - 1), min(8, king_f + 2))):
            if not (chess.BB_FILES[f] & all_pawns):
                out[open_base + bit] = 1.0    # fully open
            elif not (chess.BB_FILES[f] & wp_files):
                out[open_base + bit] = 0.5    # semi-open (no friendly pawn)

        # Attacker count
        ep_nopawns = int(board.occupied_co[opp]) & ~int(board.pieces_mask(chess.PAWN, opp))
        attacker_count = sum(1 for sq in chess.scan_forward(ep_nopawns)
                             if board.attacks_mask(sq) & zone_bb)
        out[atk_base] = min(attacker_count / 8.0, 1.0)

    return out


# ── B8: Pin spatial map (256 dims) ──────────────────────────────────────────

def _pin_vec(board: chess.Board) -> np.ndarray:
    """[w_pinned(64), w_pinner(64), b_pinned(64), b_pinner(64)] = 256 dims"""
    out = np.zeros(256, dtype=np.float32)
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 128)):
        opp = not color
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == color and p.piece_type != chess.KING:
                if board.is_pinned(color, sq):
                    out[base + sq] = 1.0
                    pin_ray = int(board.pin(color, sq))
                    for psq in chess.scan_forward(pin_ray):
                        pp = board.piece_at(psq)
                        if pp and pp.color == opp:
                            out[base + 64 + psq] = 1.0
                            break
    return out


# ── B8: Fork spatial map (256 dims) ─────────────────────────────────────────

def _fork_vec(board: chess.Board) -> np.ndarray:
    """[w_forking(64), w_forked(64), b_forking(64), b_forked(64)] = 256 dims"""
    out = np.zeros(256, dtype=np.float32)
    valuable = {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING}
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 128)):
        opp = not color
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.PAWN):
            for sq in board.pieces(pt, color):
                forked = [
                    attacked
                    for attacked in chess.scan_forward(board.attacks_mask(sq))
                    if (pp := board.piece_at(attacked)) and pp.color == opp and pp.piece_type in valuable
                ]
                if len(forked) >= 2:
                    out[base + sq] = 1.0
                    for fsq in forked:
                        out[base + 64 + fsq] = 1.0
    return out


# ── B8: Isolated pawn spatial map (128 dims) ─────────────────────────────────

def _isolated_pawn_vec(board: chess.Board) -> np.ndarray:
    """[w_isolated_map(64), b_isolated_map(64)] = 128 dims"""
    out = np.zeros(128, dtype=np.float32)
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 64)):
        wp = int(board.pieces_mask(chess.PAWN, color))
        if not wp:
            continue
        neighbor_files = _file_fill(_bb_east(wp) | _bb_west(wp))
        isolated_bb = wp & ~neighbor_files
        for sq in chess.scan_forward(isolated_bb):
            out[base + sq] = 1.0
    return out


# ── B9: Pawn chain spatial map (128 dims) ────────────────────────────────────

def _pawn_chain_vec(board: chess.Board) -> np.ndarray:
    """[w_chain(64), b_chain(64)] = 128 dims.

    A pawn chain is a diagonal sequence of friendly pawns where each pawn
    (except the base) is defended by the one behind it on the adjacent file.

    chain[sq] = 1.0 if the pawn at sq is part of a chain: either it is defended
    diagonally from behind by a friendly pawn (chain member), or it is defending
    a friendly pawn diagonally in front of it (chain base).

    Works by:
      1. chain_members = pawns that are attacked by a friendly pawn from behind.
         For white: sq in wp where _wp_attacks(wp) includes sq
                    (another white pawn one rank below + adjacent file attacks sq)
      2. chain_defenders = the base pawns doing the protecting.
         Found by stepping south+east / south+west from chain_members back to wp.
    Both groups together form the full chain geometry.
    """
    out = np.zeros(128, dtype=np.float32)
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 64)):
        wp = int(board.pieces_mask(chess.PAWN, color))
        if not wp:
            continue
        if color == chess.WHITE:
            # _wp_attacks(wp) = NE/NW of each white pawn = squares they attack forward.
            # A white pawn at sq is "protected from behind" if sq in _wp_attacks(wp)
            # (meaning some white pawn one rank below & adjacent file attacks sq).
            chain_members = wp & _wp_attacks(wp)
            chain_defenders = (
                _bb_east(_bb_south(chain_members)) | _bb_west(_bb_south(chain_members))
            ) & wp
        else:
            # Black pawns advance south; protected "from behind" = from above (higher rank).
            # _bp_attacks(wp) = SE/SW of each black pawn = squares attacked going south.
            # A black pawn at sq is protected if sq in _bp_attacks(wp).
            chain_members = wp & _bp_attacks(wp)
            chain_defenders = (
                _bb_east(_bb_north(chain_members)) | _bb_west(_bb_north(chain_members))
            ) & wp
        chain_all = chain_members | chain_defenders
        for sq in chess.scan_forward(chain_all):
            out[base + sq] = 1.0
    return out


# ── B9: Pawn island map (130 dims) ───────────────────────────────────────────

def _pawn_island_vec(board: chess.Board) -> np.ndarray:
    """[w_connected(64), b_connected(64), w_island_count(1), b_island_count(1)] = 130 dims.

    A pawn island is a group of pawns on consecutive files with no pawns on
    either neighboring file.  More islands = weaker pawn structure.

    connected[sq] = 1.0 if the pawn at sq has at least one pawn on an adjacent
    file (i.e., is part of a multi-pawn island, NOT isolated).
    island_count = _pawn_island_count() / 8.0 (normalized to [0, 1]).

    Complements isolated_pawn_vec (B8) which marks the ABSENCE of neighbors;
    together they encode the full island structure per square.
    """
    out = np.zeros(130, dtype=np.float32)
    for color, base, cnt_idx in ((chess.WHITE, 0, 128), (chess.BLACK, 64, 129)):
        wp = int(board.pieces_mask(chess.PAWN, color))
        if not wp:
            continue
        neighbor_files = _file_fill(_bb_east(wp) | _bb_west(wp))
        connected_bb = wp & neighbor_files
        for sq in chess.scan_forward(connected_bb):
            out[base + sq] = 1.0
        out[cnt_idx] = _pawn_island_count(board, color) / 8.0
    return out


# ── B9: Mating pressure map (128 dims) ───────────────────────────────────────

def _mating_pressure_vec(board: chess.Board) -> np.ndarray:
    """[w_pieces_on_bk_zone(64), b_pieces_on_wk_zone(64)] = 128 dims.

    A mating attack involves multiple pieces coordinated against the enemy king.
    Each square gets 1.0 if a non-pawn piece of that color sits there AND its
    attack mask intersects the enemy king zone (9 neighbors + 3 squares one rank
    further toward the attacker).

    Shows WHERE the attacking pieces are, not just whether an attack exists.
    Complements B7 king_safety_vec (which marks which king zone SQUARES are hit).
    """
    out = np.zeros(128, dtype=np.float32)
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 64)):
        opp = not color
        king_sq = board.king(opp)
        if king_sq is None:
            continue
        king_f = chess.square_file(king_sq)
        king_r = chess.square_rank(king_sq)
        zone_bb = int(chess.BB_KING_ATTACKS[king_sq]) | (1 << king_sq)
        adj_mask = (chess.BB_FILES[max(0, king_f - 1)] | chess.BB_FILES[king_f] |
                    chess.BB_FILES[min(7, king_f + 1)])
        if color == chess.WHITE and king_r >= 1:
            zone_bb |= int(chess.BB_RANKS[king_r - 1]) & adj_mask
        elif color == chess.BLACK and king_r <= 6:
            zone_bb |= int(chess.BB_RANKS[king_r + 1]) & adj_mask
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(pt, color):
                if board.attacks_mask(sq) & zone_bb:
                    out[base + sq] = 1.0
    return out


# ── B8: Open file map (40 dims) ──────────────────────────────────────────────

def _open_file_vec(board: chess.Board) -> np.ndarray:
    """[w_rook_files(8), b_rook_files(8), fully_open(8), semi_open_w(8), semi_open_b(8)] = 40 dims"""
    out = np.zeros(40, dtype=np.float32)
    wp = int(board.pieces_mask(chess.PAWN, chess.WHITE))
    bp = int(board.pieces_mask(chess.PAWN, chess.BLACK))
    for f in range(8):
        file_bb = chess.BB_FILES[f]
        if board.pieces_mask(chess.ROOK, chess.WHITE) & file_bb:
            out[f] = 1.0
        if board.pieces_mask(chess.ROOK, chess.BLACK) & file_bb:
            out[8 + f] = 1.0
        w_pawn = bool(wp & file_bb)
        b_pawn = bool(bp & file_bb)
        if not w_pawn and not b_pawn:
            out[16 + f] = 1.0
        if not w_pawn:
            out[24 + f] = 1.0
        if not b_pawn:
            out[32 + f] = 1.0
    return out


# ── B9: Interference spatial map (128 dims) ──────────────────────────────────

def _interference_vec(board: chess.Board) -> np.ndarray:
    """[w_interf_gaps(64), b_interf_gaps(64)] = 128 dims.
    interf_gaps[sq] = 1.0 if sq lies in an empty ray between an enemy slider and the
    enemy piece it defends, AND the given side attacks sq (viable interposition square).
    """
    out = np.zeros(128, dtype=np.float32)
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 64)):
        opp = not color
        for pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            for def_sq in board.pieces(pt, opp):
                df_s = chess.square_file(def_sq)
                dr_s = chess.square_rank(def_sq)
                dirs: list[tuple[int, int]] = []
                if pt in (chess.ROOK, chess.QUEEN):
                    dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if pt in (chess.BISHOP, chess.QUEEN):
                    dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for df, dr in dirs:
                    cf, cr = df_s + df, dr_s + dr
                    gap_sqs: list[int] = []
                    while 0 <= cf <= 7 and 0 <= cr <= 7:
                        csq = chess.square(cf, cr)
                        p = board.piece_at(csq)
                        if p is not None:
                            if p.color == opp and gap_sqs:
                                for gsq in gap_sqs:
                                    if board.is_attacked_by(color, gsq):
                                        out[base + gsq] = 1.0
                            break
                        gap_sqs.append(csq)
                        cf += df
                        cr += dr
    return out


# ── B9: Initiative spatial map (130 dims) ────────────────────────────────────

def _initiative_vec(board: chess.Board) -> np.ndarray:
    """[w_active(64), b_active(64), w_threat_norm(1), b_threat_norm(1)] = 130 dims.
    active[sq] = 1.0 if the piece at sq attacks ≥1 enemy piece with more attackers
    than defenders on it. threat_norm = threat_count / 8.0 capped at 1.0.
    """
    out = np.zeros(130, dtype=np.float32)
    for color, base, cnt_idx in ((chess.WHITE, 0, 128), (chess.BLACK, 64, 129)):
        opp = not color
        threats = 0
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(pt, color):
                for tgt in chess.scan_forward(board.attacks_mask(sq)):
                    tp = board.piece_at(tgt)
                    if tp and tp.color == opp and tp.piece_type in {
                        chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT
                    }:
                        att = chess.popcount(board.attackers_mask(color, tgt))
                        dfs = chess.popcount(board.attackers_mask(opp, tgt))
                        if att > dfs:
                            out[base + sq] = 1.0
                            threats += 1
                            break
        out[cnt_idx] = min(threats / 8.0, 1.0)
    return out


# ── B9: Prophylaxis spatial map (130 dims) ───────────────────────────────────

def _prophylaxis_vec(board: chess.Board) -> np.ndarray:
    """[w_overprotect(64), b_overprotect(64), w_key_ctrl(1), b_key_ctrl(1)] = 130 dims.
    overprotect[sq] = 1.0 if a friendly piece at sq has ≥2 more defenders than attackers.
    key_ctrl = fraction of opponent's potential outpost squares dominated by this side.
    """
    out = np.zeros(130, dtype=np.float32)
    for color, base, ctrl_idx in ((chess.WHITE, 0, 128), (chess.BLACK, 64, 129)):
        opp = not color
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(pt, color):
                dfs = chess.popcount(board.attackers_mask(color, sq))
                att = chess.popcount(board.attackers_mask(opp, sq))
                if dfs >= att + 2:
                    out[base + sq] = 1.0
        opp_outposts = _outpost_squares_bb(board, opp)
        dominated = 0
        total = chess.popcount(opp_outposts) or 1
        for sq in chess.scan_forward(opp_outposts):
            if board.piece_at(sq):
                continue
            if chess.popcount(board.attackers_mask(color, sq)) > chess.popcount(board.attackers_mask(opp, sq)):
                dominated += 1
        out[ctrl_idx] = dominated / total
    return out


# ── B9: Sacrifice spatial map (130 dims) ─────────────────────────────────────

def _sacrifice_vec(board: chess.Board) -> np.ndarray:
    """[w_offered(64), b_offered(64), w_deficit_norm(1), b_deficit_norm(1)] = 130 dims.
    offered[sq] = 1.0 if piece at sq is attacked by a less valuable enemy and under-defended.
    deficit_norm = max(0, opp_material - own_material) / 9.0 capped at 1.0.
    """
    out = np.zeros(130, dtype=np.float32)
    for color, base, def_idx in ((chess.WHITE, 0, 128), (chess.BLACK, 64, 129)):
        opp = not color
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            pt_val = _PIECE_VALUES[pt]
            for sq in board.pieces(pt, color):
                for att_sq in board.attackers(opp, sq):
                    att_pt = board.piece_at(att_sq).piece_type
                    if _PIECE_VALUES.get(att_pt, 0) < pt_val:
                        att = chess.popcount(board.attackers_mask(opp, sq))
                        dfs = chess.popcount(board.attackers_mask(color, sq))
                        if att >= dfs:
                            out[base + sq] = 1.0
                            break
        our_mat = _material_value(board, color)
        opp_mat = _material_value(board, opp)
        out[def_idx] = min(max(0, opp_mat - our_mat) / 9.0, 1.0)
    return out


# ── B9: Clearance spatial map (128 dims) ─────────────────────────────────────

def _clearance_vec(board: chess.Board) -> np.ndarray:
    """[w_blocker(64), b_blocker(64)] = 128 dims.
    blocker[sq] = 1.0 if a friendly non-slider at sq blocks a friendly slider's ray
    to a valuable enemy piece — moving this piece would reveal the slider's attack.
    """
    out = np.zeros(128, dtype=np.float32)
    slider_types = {chess.BISHOP, chess.ROOK, chess.QUEEN}
    valuable = {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 64)):
        opp = not color
        for slider_pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            for slider_sq in board.pieces(slider_pt, color):
                sf = chess.square_file(slider_sq)
                sr = chess.square_rank(slider_sq)
                dirs: list[tuple[int, int]] = []
                if slider_pt in (chess.ROOK, chess.QUEEN):
                    dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if slider_pt in (chess.BISHOP, chess.QUEEN):
                    dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                for df, dr in dirs:
                    cf, cr = sf + df, sr + dr
                    while 0 <= cf <= 7 and 0 <= cr <= 7:
                        blocker_sq = chess.square(cf, cr)
                        blocker = board.piece_at(blocker_sq)
                        if blocker is not None:
                            if blocker.color == color and blocker.piece_type not in slider_types:
                                cf2, cr2 = cf + df, cr + dr
                                while 0 <= cf2 <= 7 and 0 <= cr2 <= 7:
                                    tgt_sq = chess.square(cf2, cr2)
                                    tgt = board.piece_at(tgt_sq)
                                    if tgt is not None:
                                        if tgt.color == opp and tgt.piece_type in valuable:
                                            out[base + blocker_sq] = 1.0
                                        break
                                    cf2 += df
                                    cr2 += dr
                            break
                        cf += df
                        cr += dr
    return out


# ── B9: Deflection spatial map (128 dims) ────────────────────────────────────

def _deflection_vec(board: chess.Board) -> np.ndarray:
    """[w_deflect_sq(64), b_deflect_sq(64)] = 128 dims.
    deflect_sq[sq] = 1.0 if sq is an enemy piece that is the sole defender of a
    valuable enemy piece (queen/rook), AND we attack sq (deflection target).
    """
    out = np.zeros(128, dtype=np.float32)
    valuable = {chess.QUEEN, chess.ROOK}
    for color, base in ((chess.WHITE, 0), (chess.BLACK, 64)):
        opp = not color
        for def_sq in chess.SQUARES:
            defender = board.piece_at(def_sq)
            if defender is None or defender.color != opp:
                continue
            if defender.piece_type not in (
                chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN
            ):
                continue
            for tgt_sq in board.attacks(def_sq):
                tgt = board.piece_at(tgt_sq)
                if tgt is None or tgt.color != opp or tgt.piece_type not in valuable:
                    continue
                defenders_of_tgt = board.attackers(opp, tgt_sq)
                if len(defenders_of_tgt) == 1 and def_sq in defenders_of_tgt:
                    if board.is_attacked_by(color, def_sq):
                        out[base + def_sq] = 1.0
                        break
    return out


# ── B9: Zwischenzug spatial map (128 dims) ───────────────────────────────────

def _zwischenzug_vec(board: chess.Board) -> np.ndarray:
    """[w_threatened(64), b_threatened(64)] = 128 dims.
    threatened[sq] = 1.0 if piece at sq is under losing attack AND the side to
    move has a forcing check available (the intermezzo pattern is present).
    """
    out = np.zeros(128, dtype=np.float32)
    color = board.turn
    opp = not color
    has_intermezzo = False
    for move in board.legal_moves:
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        if gives_check:
            has_intermezzo = True
            break
    base = 0 if color == chess.WHITE else 64
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for sq in board.pieces(pt, color):
            if chess.popcount(board.attackers_mask(opp, sq)) > chess.popcount(board.attackers_mask(color, sq)):
                out[base + sq] = 1.0 if has_intermezzo else 0.5
    return out


# ── Phase 4 feature vector assembly ──────────────────────────────────────────

ALGO_FEATURE_SIZE_V4: int = 3779   # 512 B1-B5 + 1151 B6 + 148 B7 + 680 B8 + 1288 B9


def algo_feature_vector_v4(fen: str) -> np.ndarray:
    """Run Phase 4 spatial detectors and return a 3779-element float32 array.

    Layout mirrors ALGO_FEATURE_SIZE_V4 comment above.
    Invalid FENs return an all-zeros vector.
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return np.zeros(ALGO_FEATURE_SIZE_V4, dtype=np.float32)

    parts: list[np.ndarray] = []
    try:
        # B1-B5 spatial maps (512)
        for color in (chess.WHITE, chess.BLACK):
            parts.append(_weak_square_map(board, color))
        for color in (chess.WHITE, chess.BLACK):
            parts.append(_outpost_map(board, color))
        for color in (chess.WHITE, chess.BLACK):
            parts.append(_backward_pawn_map(board, color))
        for color in (chess.WHITE, chess.BLACK):
            parts.append(_passed_pawn_map(board, color))

        # B6 formerly-binary (1151)
        parts.append(_bishop_pair_vec(board))      # 130
        parts.append(_development_vec(board))      # 136
        parts.append(_xray_vec(board))             # 384
        parts.append(_battery_vec(board))          # 144
        parts.append(_opposition_vec(board))       # 15
        parts.append(_zugzwang_vec(board))         # 4
        parts.append(_rook_seventh_vec(board))     # 34
        parts.append(_drawn_position_vec(board))   # 2
        parts.append(_shouldering_vec(board))      # 21
        parts.append(_double_check_vec(board))     # 130
        parts.append(_promotion_vec(board))        # 18
        parts.append(_bishop_endgame_vec(board))   # 133

        # B7 king safety (148)
        parts.append(_king_safety_vec(board))      # 148

        # B8 new tactical spatial maps (680)
        parts.append(_pin_vec(board))              # 256
        parts.append(_fork_vec(board))             # 256
        parts.append(_isolated_pawn_vec(board))    # 128
        parts.append(_open_file_vec(board))        # 40

        # B9 pawn/mating/strategic/tactical maps (1288)
        parts.append(_pawn_chain_vec(board))       # 128
        parts.append(_pawn_island_vec(board))      # 130
        parts.append(_mating_pressure_vec(board))  # 128
        parts.append(_interference_vec(board))     # 128
        parts.append(_initiative_vec(board))       # 130
        parts.append(_prophylaxis_vec(board))      # 130
        parts.append(_sacrifice_vec(board))        # 130
        parts.append(_clearance_vec(board))        # 128
        parts.append(_deflection_vec(board))       # 128
        parts.append(_zwischenzug_vec(board))      # 128
    except Exception:
        return np.zeros(ALGO_FEATURE_SIZE_V4, dtype=np.float32)

    result = np.concatenate(parts)
    assert len(result) == ALGO_FEATURE_SIZE_V4, \
        f"Phase 4 vector length mismatch: {len(result)} != {ALGO_FEATURE_SIZE_V4}"
    return result


def algo_feature_vector(fen: str) -> np.ndarray:
    """Run all structural detectors and return a 68-element float32 array (v3 bypass channel).

    Layout:
        [0:58]  29 per-color concepts × [white_bit, black_bit] interleaved
        [58:68] 10 position-wide concept bits

    Invalid FENs (variant chess, corrupt data) return an all-zeros vector.
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return np.zeros(ALGO_FEATURE_SIZE, dtype=np.float32)

    arr = np.zeros(ALGO_FEATURE_SIZE, dtype=np.float32)
    i = 0
    for _, fn in _PER_COLOR_DETECTORS:
        try:
            arr[i]     = 1.0 if fn(board, chess.WHITE) else 0.0
            arr[i + 1] = 1.0 if fn(board, chess.BLACK) else 0.0
        except Exception:
            pass
        i += 2
    for _, fn in _GLOBAL_DETECTORS:
        try:
            arr[i] = 1.0 if fn(board) else 0.0
        except Exception:
            pass
        i += 1
    return arr
