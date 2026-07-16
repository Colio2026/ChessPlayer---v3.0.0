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

from typing import Callable

import chess
import numpy as np

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
    # Piece quality / placement
    "bad_bishop",
    "good_bishop",
    "bishop_pair",
    "battery",
    "blockade",
    "outpost",
    "rook_seventh",
    "piece_activity",
    # King
    "king_activity",
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
    # Strategic
    "development_lead",
    # Endgame types (by material composition)
    "rook_endgame",
    "pawn_endgame",
    "bishop_endgame",
    "knight_endgame",
    "queen_endgame",
    "drawn_position",
})

# ── helpers ───────────────────────────────────────────────────────────────────

def _pawn_files(board: chess.Board, color: chess.Color) -> list[int]:
    return [chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)]


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


# ── pawn structure ────────────────────────────────────────────────────────────

def _has_passed_pawn(board: chess.Board, color: chess.Color) -> bool:
    opp = not color
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        # Rank threshold: only flag pawns that are meaningfully advanced
        if color == chess.WHITE and r < 4:
            continue
        if color == chess.BLACK and r > 3:
            continue
        blocked = False
        for ep in board.pieces(chess.PAWN, opp):
            ef = chess.square_file(ep)
            er = chess.square_rank(ep)
            if abs(ef - f) <= 1:
                if color == chess.WHITE and er > r:
                    blocked = True
                    break
                if color == chess.BLACK and er < r:
                    blocked = True
                    break
        if not blocked:
            return True
    return False


def _has_isolated_pawn(board: chess.Board, color: chess.Color) -> bool:
    files = set(_pawn_files(board, color))
    return any(f - 1 not in files and f + 1 not in files for f in files)


def _has_doubled_pawn(board: chess.Board, color: chess.Color) -> bool:
    from collections import Counter
    counts = Counter(_pawn_files(board, color))
    return any(v >= 2 for v in counts.values())


def _pawn_island_count(board: chess.Board, color: chess.Color) -> int:
    files = sorted(set(_pawn_files(board, color)))
    if not files:
        return 0
    islands = 1
    for i in range(1, len(files)):
        if files[i] > files[i - 1] + 1:
            islands += 1
    return islands


def _has_pawn_majority(board: chess.Board, color: chess.Color) -> bool:
    """More pawns on the queenside (files a-d) or kingside (files e-h) than the opponent."""
    opp = not color
    my_files   = _pawn_files(board, color)
    opp_files  = _pawn_files(board, opp)
    my_qs  = sum(1 for f in my_files  if f < 4)
    opp_qs = sum(1 for f in opp_files if f < 4)
    my_ks  = sum(1 for f in my_files  if f >= 4)
    opp_ks = sum(1 for f in opp_files if f >= 4)
    return my_qs > opp_qs or my_ks > opp_ks


def _has_backward_pawn(board: chess.Board, color: chess.Color) -> bool:
    """A pawn that can't be supported by another pawn and whose advance square
    is controlled by an enemy pawn."""
    pawns = board.pieces(chess.PAWN, color)
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        advance_rank = r + 1 if color == chess.WHITE else r - 1
        if not (0 <= advance_rank <= 7):
            continue
        advance_sq = chess.square(f, advance_rank)

        # Can any friendly pawn on an adjacent file support this pawn?
        can_support = False
        for adj_f in (f - 1, f + 1):
            if not (0 <= adj_f <= 7):
                continue
            for friendly_sq in pawns:
                if chess.square_file(friendly_sq) == adj_f:
                    fr = chess.square_rank(friendly_sq)
                    if color == chess.WHITE and fr <= r:
                        can_support = True
                    elif color == chess.BLACK and fr >= r:
                        can_support = True

        if not can_support:
            # Advance square attacked by enemy pawn?
            enemy_pawn_attackers = (board.attackers(not color, advance_sq)
                                    & board.pieces(chess.PAWN, not color))
            if enemy_pawn_attackers:
                return True
    return False


def _has_pawn_chain(board: chess.Board, color: chess.Color) -> bool:
    """At least two friendly pawns that mutually defend each other diagonally."""
    pawns = board.pieces(chess.PAWN, color)
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        # A pawn on (f,r) defends (f-1,r+1) and (f+1,r+1) for white
        defend_rank = r + 1 if color == chess.WHITE else r - 1
        if not (0 <= defend_rank <= 7):
            continue
        for df in (-1, 1):
            if 0 <= f + df <= 7:
                defended = chess.square(f + df, defend_rank)
                if board.piece_at(defended) == chess.Piece(chess.PAWN, color):
                    return True
    return False


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
        if same > len(pawns) - same:       # majority on bishop's color
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
    for rook_sq in board.pieces(chess.ROOK, color):
        f = chess.square_file(rook_sq)
        white_pawns_on_file = any(chess.square_file(sq) == f for sq in board.pieces(chess.PAWN, chess.WHITE))
        black_pawns_on_file = any(chess.square_file(sq) == f for sq in board.pieces(chess.PAWN, chess.BLACK))
        if not white_pawns_on_file and not black_pawns_on_file:
            return True
    return False


def _has_rook_on_seventh(board: chess.Board, color: chess.Color) -> bool:
    seventh = 6 if color == chess.WHITE else 1   # rank index (0-based)
    return any(chess.square_rank(sq) == seventh
               for sq in board.pieces(chess.ROOK, color))


# ── square control ────────────────────────────────────────────────────────────

def _outpost_squares(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    """Squares in the opponent's half that the opponent's pawns can never attack."""
    opp = not color
    enemy_pawns = board.pieces(chess.PAWN, opp)
    # Squares enemy pawns can EVER attack (on any rank)
    ever_attacked: chess.SquareSet = chess.SquareSet()
    for ep in enemy_pawns:
        ef = chess.square_file(ep)
        for adj_f in (ef - 1, ef + 1):
            if 0 <= adj_f <= 7:
                # All ranks in the direction of advance
                rank_range = range(chess.square_rank(ep) - 1, -1, -1) if opp == chess.BLACK else range(chess.square_rank(ep) + 1, 8)
                for r in rank_range:
                    ever_attacked.add(chess.square(adj_f, r))

    # Opponent's half
    start_rank = 4 if color == chess.WHITE else 0
    end_rank   = 8 if color == chess.WHITE else 4
    candidate  = chess.SquareSet(
        chess.square(f, r) for f in range(8) for r in range(start_rank, end_rank)
    )
    return candidate - ever_attacked


def _has_outpost(board: chess.Board, color: chess.Color) -> bool:
    """A knight or bishop occupies a deep outpost (rank 5+ for white, rank 4- for black)."""
    outposts = _outpost_squares(board, color)
    # Require the piece to be truly deep in enemy territory, not just over the midline
    deep_ranks = range(4, 8) if color == chess.WHITE else range(0, 4)
    deep_outposts = chess.SquareSet(
        sq for sq in outposts if chess.square_rank(sq) in deep_ranks
    )
    for pt in (chess.KNIGHT, chess.BISHOP):
        if board.pieces(pt, color) & deep_outposts:
            return True
    return False


def _has_weak_square(board: chess.Board, color: chess.Color) -> bool:
    """The opponent has outpost squares (holes) in their position."""
    return bool(_outpost_squares(board, not color))


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
    # No luft pawns
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
    # Opponent has a rook or queen
    opp = not color
    return bool(board.pieces(chess.ROOK, opp) or board.pieces(chess.QUEEN, opp))


# ── tactical ─────────────────────────────────────────────────────────────────

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
                # Trace the ray beyond the victim
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
    # Find opponent pieces that are the sole defender of two different squares/pieces
    targets = list(board.pieces(chess.QUEEN,  color) |
                   board.pieces(chess.ROOK,   color) |
                   board.pieces(chess.BISHOP, color) |
                   board.pieces(chess.KNIGHT, color) |
                   board.pieces(chess.PAWN,   color))
    if len(targets) < 2:
        return False
    # For each opponent piece, count how many of our pieces it is the ONLY defender of
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


# ── new structural / strategic detectors ─────────────────────────────────────

def _has_battery(board: chess.Board, color: chess.Color) -> bool:
    """Two same-color major pieces aligned on the same file/rank (rook battery)
    or same diagonal (bishop/queen battery), with a clear line between them."""
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
    """A piece of `color` sits directly in front of an advanced enemy pawn,
    preventing it from advancing (classical blockade)."""
    opp = not color
    for pawn_sq in board.pieces(chess.PAWN, opp):
        rank = chess.square_rank(pawn_sq)
        file = chess.square_file(pawn_sq)

        if opp == chess.WHITE:
            if rank < 4:   # not advanced enough to blockade
                continue
            block_rank = rank + 1
        else:
            if rank > 3:
                continue
            block_rank = rank - 1

        if not (0 <= block_rank <= 7):
            continue
        piece = board.piece_at(chess.square(file, block_rank))
        if piece and piece.color == color:
            return True
    return False


def _has_pawn_storm(board: chess.Board, color: chess.Color) -> bool:
    """2+ pawns advanced toward the opponent's king flank."""
    opp     = not color
    king_sq = board.king(opp)
    if king_sq is None:
        return False
    king_f = chess.square_file(king_sq)
    flank  = set(range(max(0, king_f - 2), min(8, king_f + 3)))

    advanced = 0
    for pawn_sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(pawn_sq)
        r = chess.square_rank(pawn_sq)
        if f not in flank:
            continue
        if color == chess.WHITE and r >= 4:
            advanced += 1
        elif color == chess.BLACK and r <= 3:
            advanced += 1

    return advanced >= 2


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
    """color has 2 queenside pawns vs opponent's 3 and at least one is advanced —
    the classic minority attack formation."""
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
    """A bishop controls squares of one color while the opponent has many pawns
    fixed on that same color — classic color complex weakness."""
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


# ── endgame type detectors ────────────────────────────────────────────────────

def _is_rook_endgame(board: chess.Board) -> bool:
    """Only kings, rooks, and pawns on the board."""
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.BISHOP, chess.KNIGHT):
            if board.pieces(pt, color):
                return False
    return bool(board.pieces(chess.ROOK, chess.WHITE) or board.pieces(chess.ROOK, chess.BLACK))


def _is_pawn_endgame(board: chess.Board) -> bool:
    """Only kings and pawns on the board."""
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            if board.pieces(pt, color):
                return False
    return True


def _is_bishop_endgame(board: chess.Board) -> bool:
    """Only kings, bishops, and pawns on the board."""
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.KNIGHT):
            if board.pieces(pt, color):
                return False
    return bool(board.pieces(chess.BISHOP, chess.WHITE) or board.pieces(chess.BISHOP, chess.BLACK))


def _is_knight_endgame(board: chess.Board) -> bool:
    """Only kings, knights, and pawns on the board."""
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            if board.pieces(pt, color):
                return False
    return bool(board.pieces(chess.KNIGHT, chess.WHITE) or board.pieces(chess.KNIGHT, chess.BLACK))


def _is_queen_endgame(board: chess.Board) -> bool:
    """Only kings, queens, and pawns on the board."""
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
            if board.pieces(pt, color):
                return False
    return bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))


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

    # Endgame types (mutually exclusive by material composition)
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

    if board.is_insufficient_material():
        labels.add("drawn_position")

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

        # tactical
        if _has_pin(board, color):
            labels.add("pin")
        if _has_fork(board, color):
            labels.add("fork")
        if _has_skewer(board, color):
            labels.add("skewer")
        if _has_overloading(board, color):
            labels.add("overloading")

    return frozenset(labels)


# ── concept bottleneck feature vector ────────────────────────────────────────
# Fixed-order tables used to build the algo input feature vector.
# Per-color: each concept gets two bits [white, black].
# Global: single bit for position-wide concepts.

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
]  # 26 × 2 = 52 bits

_GLOBAL_DETECTORS: list[tuple[str, Callable]] = [
    ("opposition",     lambda b: _has_opposition(b) and _is_endgame(b)),
    ("rook_endgame",   _is_rook_endgame),
    ("pawn_endgame",   _is_pawn_endgame),
    ("bishop_endgame", _is_bishop_endgame),
    ("knight_endgame", _is_knight_endgame),
    ("queen_endgame",  _is_queen_endgame),
    ("drawn_position", lambda b: b.is_insufficient_material()),
]  # 7 bits

ALGO_FEATURE_SIZE: int = len(_PER_COLOR_DETECTORS) * 2 + len(_GLOBAL_DETECTORS)  # 59


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Spatial detector maps and B6 formerly-binary feature vectors
#
# These functions are wired into algo_feature_vector_v4() below.
# algo_feature_vector() (59-dim, Phase 3) is kept intact for backward compat
# until the full pipeline re-run switches to v4.
#
# ALGO_FEATURE_SIZE_V4 layout  (1663 dims total):
#   [0:512]    B1-B5 spatial maps — 4 concepts × 2 colors × 64 squares
#   [512:642]  B6 bishop_pair        (130)
#   [642:778]  B6 development        (136)
#   [778:1162] B6 x_ray              (384)
#   [1162:1306] B6 battery           (144)
#   [1306:1321] B6 opposition        ( 15)
#   [1321:1325] B6 zugzwang tier-1   (  4)
#   [1325:1359] B6 rook_seventh      ( 34)
#   [1359:1361] B6 drawn_position    (  2)
#   [1361:1382] B6 shouldering       ( 21)
#   [1382:1512] B6 double_check      (130)
#   [1512:1530] B6 promotion         ( 18)
#   [1530:1663] B6 bishop_endgame    (133)
# ═══════════════════════════════════════════════════════════════════════════════

# ── B2: Weak square map ───────────────────────────────────────────────────────

def _weak_square_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """64-element map of squares strategically weak for `color` (B2 definition).
    A square is weak if no friendly pawn on an adjacent file can advance to defend it.
    Only marks strategically relevant territory (ranks 4-8 for white, 1-5 for black).
    """
    by_file: list[list[int]] = [[] for _ in range(8)]
    for sq in board.pieces(chess.PAWN, color):
        by_file[chess.square_file(sq)].append(chess.square_rank(sq))
    out = np.zeros(64, dtype=np.float32)
    for sq in range(64):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        if color == chess.WHITE and r < 3:
            continue
        if color == chess.BLACK and r > 4:
            continue
        defended = False
        for af in (f - 1, f + 1):
            if 0 <= af <= 7:
                for pr in by_file[af]:
                    if (color == chess.WHITE and pr < r) or (color == chess.BLACK and pr > r):
                        defended = True
                        break
            if defended:
                break
        if not defended:
            out[sq] = 1.0
    return out


# ── B3: Outpost map ───────────────────────────────────────────────────────────

def _outpost_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """64-element map of squares in advanced territory that are weak for the opponent."""
    opp_weak = _weak_square_map(board, not color)
    out = np.zeros(64, dtype=np.float32)
    for sq in range(64):
        if not opp_weak[sq]:
            continue
        r = chess.square_rank(sq)
        if (color == chess.WHITE and r >= 4) or (color == chess.BLACK and r <= 3):
            out[sq] = 1.0
    return out


# ── B4: Backward pawn map ─────────────────────────────────────────────────────

def _backward_pawn_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """64-element map of backward pawn squares for `color`."""
    pawns = board.pieces(chess.PAWN, color)
    out = np.zeros(64, dtype=np.float32)
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        adv_r = r + 1 if color == chess.WHITE else r - 1
        if not (0 <= adv_r <= 7):
            continue
        adv_sq = chess.square(f, adv_r)
        supported = False
        for adj_f in (f - 1, f + 1):
            if not (0 <= adj_f <= 7):
                continue
            for fsq in pawns:
                if chess.square_file(fsq) == adj_f:
                    fr = chess.square_rank(fsq)
                    if (color == chess.WHITE and fr <= r) or (color == chess.BLACK and fr >= r):
                        supported = True
                        break
            if supported:
                break
        if not supported:
            enemy_pawns = board.attackers(not color, adv_sq) & board.pieces(chess.PAWN, not color)
            if enemy_pawns:
                out[sq] = 1.0
    return out


# ── B5: Passed pawn map (per-square, replaces per-file encoding) ──────────────

def _passed_pawn_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """64-element map of passed pawn squares for `color` (all ranks, no threshold)."""
    opp = not color
    opp_by_file: list[list[int]] = [[] for _ in range(8)]
    for sq in board.pieces(chess.PAWN, opp):
        opp_by_file[chess.square_file(sq)].append(chess.square_rank(sq))
    out = np.zeros(64, dtype=np.float32)
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        blocked = False
        for af in (f - 1, f, f + 1):
            if not (0 <= af <= 7):
                continue
            for er in opp_by_file[af]:
                if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
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
    """[w_attack(64), w_defend(64), w_clear(64), b_attack(64), b_defend(64), b_clear(64)]
    For each sliding piece, cast rays through the first blocker:
      attack:   blocker is enemy → next enemy piece is x-ray attacked
      defend:   blocker is friendly → next friendly piece is x-ray defended
      clear:    blocker is friendly → that blocker square is a clearance opportunity
    """
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
                                    out[base + 128 + csq] = 1.0   # clearance
                            else:
                                if blocker_col == (not color) and p.color == (not color):
                                    out[base + csq] = 1.0          # x-ray attack
                                elif blocker_col == color and p.color == color:
                                    out[base + 64 + csq] = 1.0    # x-ray defend
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
    # opposition: same axis, odd gap, other side to move
    facing = (df == 0 or dr == 0 or df == dr) and gap % 2 == 1
    out[12] = 1.0 if (facing and board.turn == chess.BLACK) else 0.0  # white has opp
    out[13] = 1.0 if (facing and board.turn == chess.WHITE) else 0.0  # black has opp
    # rook-pawn draw risk: corner pawn + defending king near corner
    rp_files = {0, 7}
    has_rp = any(chess.square_file(sq) in rp_files
                 for sq in board.pieces(chess.PAWN, chess.WHITE) |
                            board.pieces(chess.PAWN, chess.BLACK))
    near_corner = (bkf in rp_files and bkr >= 6) or (wkf in rp_files and wkr <= 1)
    out[14] = 1.0 if (is_kpe and has_rp and near_corner) else 0.0
    return out


# ── B6: Zugzwang tier-1 (4 dims) ─────────────────────────────────────────────

def _zugzwang_vec(board: chess.Board) -> np.ndarray:
    """[is_endgame, white_few_moves, black_few_moves, pawn_breaks_blocked]"""
    out = np.zeros(4, dtype=np.float32)
    out[0] = 1.0 if _is_endgame(board) else 0.0
    if board.turn == chess.WHITE:
        out[1] = 1.0 if board.legal_moves.count() <= 3 else 0.0
    else:
        out[2] = 1.0 if board.legal_moves.count() <= 3 else 0.0
    # Pawn breaks blocked: no pawn can advance without capture
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
    """[insufficient_material, halfmove_clock_normalized]"""
    out = np.zeros(2, dtype=np.float32)
    out[0] = 1.0 if board.is_insufficient_material() else 0.0
    out[1] = board.halfmove_clock / 100.0
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
    # Target: most advanced own passer, else center
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
    # Cutting heuristic
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
    # checking_side: who is giving check = the side NOT to move
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


# ── Phase 4 feature vector assembly ──────────────────────────────────────────

ALGO_FEATURE_SIZE_V4: int = 1663   # 512 spatial (B1-B5) + 1151 B6


def algo_feature_vector_v4(fen: str) -> np.ndarray:
    """Run Phase 4 spatial detectors and return a 1663-element float32 array.

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
    except Exception:
        return np.zeros(ALGO_FEATURE_SIZE_V4, dtype=np.float32)

    result = np.concatenate(parts)
    assert len(result) == ALGO_FEATURE_SIZE_V4, \
        f"Phase 4 vector length mismatch: {len(result)} != {ALGO_FEATURE_SIZE_V4}"
    return result


def algo_feature_vector(fen: str) -> np.ndarray:
    """Run all structural detectors and return a 59-element float32 array.

    Layout:
        [0:52]  26 per-color concepts × [white_bit, black_bit] interleaved
        [52:59] 7 position-wide concept bits

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
