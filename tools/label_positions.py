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

import chess

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
    "pawn_weakness",
    "pawn_storm",
    "minority_attack",
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
    "square_control",
    "color_complex",
    "space_advantage",
    # Tactics (detectable from single position)
    "pin",
    "fork",
    "skewer",
    "overloading",
    # Strategic
    "development_lead",
    "endgame_technique",
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
    enemy_pawn_files = set(_pawn_files(board, opp))
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
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

def _has_open_file(board: chess.Board) -> bool:
    """At least one file with no pawns of either color."""
    for f in range(8):
        if (not any(chess.square_file(sq) == f for sq in board.pieces(chess.PAWN, chess.WHITE))
                and not any(chess.square_file(sq) == f for sq in board.pieces(chess.PAWN, chess.BLACK))):
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
    """A knight or bishop occupies an outpost square."""
    outposts = _outpost_squares(board, color)
    for pt in (chess.KNIGHT, chess.BISHOP):
        if board.pieces(pt, color) & outposts:
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
    if _is_endgame(board):
        labels.add("endgame_technique")
    if _has_opposition(board) and _is_endgame(board):
        labels.add("opposition")
    if _has_open_file(board):
        labels.add("open_file")

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
        if (_has_isolated_pawn(board, color)
                or _has_doubled_pawn(board, color)
                or _has_backward_pawn(board, color)):
            labels.add("pawn_weakness")
        if _has_pawn_storm(board, color):
            labels.add("pawn_storm")
        if _has_minority_attack(board, color):
            labels.add("minority_attack")

        # bishop
        if _has_bad_bishop(board, color):
            labels.add("bad_bishop")
        if _has_good_bishop(board, color):
            labels.add("good_bishop")
        if _has_bishop_pair(board, color):
            labels.add("bishop_pair")
        if _has_color_complex(board, color):
            labels.add("color_complex")

        # piece placement / activity
        if _has_battery(board, color):
            labels.add("battery")
        if _has_blockade(board, color):
            labels.add("blockade")
        if _has_piece_activity(board, color):
            labels.add("piece_activity")
        if _has_development_lead(board, color):
            labels.add("development_lead")

        # files / ranks
        if _has_rook_on_seventh(board, color):
            labels.add("rook_seventh")

        # square control
        if _has_outpost(board, color):
            labels.add("outpost")
        if _has_weak_square(board, color):
            labels.add("weak_square")
        if _has_square_control(board, color):
            labels.add("square_control")
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
