"""
coach/board_sentences.py
=========================
Generate board-specific coaching sentences from NimzoNet concept predictions.

Each builder examines the actual chess.Board object and produces a sentence that
names real piece types and square coordinates from the current position — not
generic advice. Falls back to '' when the concept isn't detectable on the board
so the caller can supply an action-hint instead.
"""
from __future__ import annotations

import chess


# ── Public API ────────────────────────────────────────────────────────────────

def build_board_sentences(
    concepts: list[tuple[str, float]],
    board: chess.Board,
    side: str,
    phase: str,
) -> list[str]:
    """Return up to 4 board-specific sentences for the highest-probability concepts."""
    color = chess.WHITE if side == 'white' else chess.BLACK
    sentences: list[str] = []
    seen: set[str] = set()

    for name, _ in concepts:
        if name in seen:
            continue
        seen.add(name)
        fn = _BUILDERS.get(name)
        if fn is None:
            continue
        try:
            s = fn(board, color, phase)
        except Exception:
            s = ''
        if s:
            sentences.append(s)
        if len(sentences) >= 4:
            break

    return sentences


# ── Helpers ───────────────────────────────────────────────────────────────────

def _piece_sym(piece: chess.Piece) -> str:
    return chess.piece_name(piece.piece_type).capitalize()

def _sq(sq: int) -> str:
    return chess.square_name(sq)

def _file_name(sq: int) -> str:
    return chess.FILE_NAMES[chess.square_file(sq)]

def _rank(sq: int) -> int:
    return chess.square_rank(sq) + 1

def _side_name(color: chess.Color) -> str:
    return 'White' if color == chess.WHITE else 'Black'

def _find_passed_pawns(board: chess.Board, color: chess.Color) -> list[int]:
    enemy = not color
    passed = []
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        blocked = False
        for ep_sq in board.pieces(chess.PAWN, enemy):
            ef = chess.square_file(ep_sq)
            er = chess.square_rank(ep_sq)
            if abs(ef - f) <= 1:
                if color == chess.WHITE and er > r:
                    blocked = True
                    break
                if color == chess.BLACK and er < r:
                    blocked = True
                    break
        if not blocked:
            passed.append(sq)
    passed.sort(key=lambda s: chess.square_rank(s) if color == chess.WHITE else -chess.square_rank(s),
                reverse=True)
    return passed

def _find_isolated_pawns(board: chess.Board, color: chess.Color) -> list[int]:
    pawns = list(board.pieces(chess.PAWN, color))
    files = {chess.square_file(sq) for sq in pawns}
    return [sq for sq in pawns
            if chess.square_file(sq) - 1 not in files and chess.square_file(sq) + 1 not in files]

def _find_doubled_pawns(board: chess.Board, color: chess.Color) -> list[int]:
    from collections import Counter
    pawns = list(board.pieces(chess.PAWN, color))
    file_counts = Counter(chess.square_file(sq) for sq in pawns)
    return [sq for sq in pawns if file_counts[chess.square_file(sq)] >= 2]

def _find_open_files(board: chess.Board) -> list[int]:
    all_pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
    return [f for f in range(8) if not (all_pawns & chess.BB_FILES[f])]

def _find_outpost_squares(board: chess.Board, color: chess.Color) -> list[int]:
    enemy = not color
    enemy_pawns = board.pieces(chess.PAWN, enemy)
    target_ranks = range(4, 8) if color == chess.WHITE else range(0, 4)
    result = []
    for sq in chess.SQUARES:
        if chess.square_rank(sq) not in target_ranks:
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        attackable = any(
            chess.square(ef, r + (1 if color == chess.WHITE else -1)) in enemy_pawns
            for ef in (f - 1, f + 1) if 0 <= ef < 8
        )
        if not attackable:
            result.append(sq)
    result.sort(key=lambda s: chess.square_rank(s) if color == chess.WHITE else -chess.square_rank(s),
                reverse=True)
    return result[:6]


# ── Concept builders ──────────────────────────────────────────────────────────

def _passed_pawn(board: chess.Board, color: chess.Color, phase: str) -> str:
    passed = _find_passed_pawns(board, color)
    if not passed:
        return ''
    sq = passed[0]
    promo_rank = 8 if color == chess.WHITE else 1
    dist = abs(promo_rank - _rank(sq))
    side = _side_name(color)
    if dist <= 2:
        return (f'{side}\'s passed pawn on {_sq(sq)} is {dist} square'
                f'{"s" if dist != 1 else ""} from promotion — support it or it will queen.')
    return (f'The passed pawn on {_sq(sq)} is free to advance. '
            f'Support it from behind with a rook and escort it with the king in the endgame.')

def _isolated_pawn(board: chess.Board, color: chess.Color, phase: str) -> str:
    iso = _find_isolated_pawns(board, color)
    enemy_iso = _find_isolated_pawns(board, not color)
    if enemy_iso:
        sq = enemy_iso[0]
        return (f'The opponent\'s pawn on {_sq(sq)} is isolated — no neighbour can defend it. '
                f'Pressurise it with rooks on the {_file_name(sq)}-file and a blockading piece.')
    if iso:
        sq = iso[0]
        return (f'The pawn on {_sq(sq)} is isolated. '
                f'It can be defended but never made truly strong — '
                f'avoid further exchanges that create additional isolated pawns.')
    return ''

def _doubled_pawn(board: chess.Board, color: chess.Color, phase: str) -> str:
    enemy_doubled = _find_doubled_pawns(board, not color)
    if enemy_doubled:
        sq = enemy_doubled[0]
        return (f'The opponent has doubled pawns on the {_file_name(sq)}-file. '
                f'Pressurise them with rooks — they cannot advance without cost.')
    our_doubled = _find_doubled_pawns(board, color)
    if our_doubled:
        sq = our_doubled[0]
        return (f'The doubled pawns on {_file_name(sq)}-file tie a rook to passive defence '
                f'and limit pawn-break options. Trade them off when possible.')
    return ''

def _weak_square(board: chess.Board, color: chess.Color, phase: str) -> str:
    outposts = _find_outpost_squares(board, color)
    if not outposts:
        return ''
    sq = outposts[0]
    return (f'The square {_sq(sq)} cannot be covered by an enemy pawn — '
            f'plant a knight or bishop there permanently. No enemy pawn can drive it away.')

def _outpost(board: chess.Board, color: chess.Color, phase: str) -> str:
    outpost_sqs = _find_outpost_squares(board, color)
    for sq in outpost_sqs:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type in (chess.KNIGHT, chess.BISHOP):
            return (f'The {_piece_sym(p)} on {_sq(sq)} occupies a permanent outpost — '
                    f'no enemy pawn can challenge it. Build all plans around its stability.')
    if outpost_sqs:
        sq = outpost_sqs[0]
        return (f'{_sq(sq)} is an available outpost square. '
                f'Route a knight there before the opponent closes it with a pawn.')
    return ''

def _open_file(board: chess.Board, color: chess.Color, phase: str) -> str:
    open_files = _find_open_files(board)
    if not open_files:
        return ''
    f = open_files[0]
    fn = chess.FILE_NAMES[f]
    rooks = list(board.pieces(chess.ROOK, color))
    rook_part = f' Bring the rook from {_sq(rooks[0])}.' if rooks else ''
    return (f'The {fn}-file is open.{rook_part} '
            f'A rook on {fn}{"7" if color == chess.WHITE else "2"} '
            f'dominates the seventh rank and harvests enemy pawns.')

def _bishop_pair(board: chess.Board, color: chess.Color, phase: str) -> str:
    bishops = list(board.pieces(chess.BISHOP, color))
    if len(bishops) < 2:
        return ''
    b1, b2 = sorted(bishops)
    side = _side_name(color)
    return (f'{side}\'s bishop pair on {_sq(b1)} and {_sq(b2)} controls both colour complexes. '
            f'Open the position with a pawn break to unleash their long-range power.')

def _bad_bishop(board: chess.Board, color: chess.Color, phase: str) -> str:
    for b_sq in board.pieces(chess.BISHOP, color):
        b_col = chess.square_color(b_sq)
        own_pawns_same = [sq for sq in board.pieces(chess.PAWN, color)
                          if chess.square_color(sq) == b_col]
        if len(own_pawns_same) >= 3:
            return (f'The bishop on {_sq(b_sq)} is a \'bad bishop\' — '
                    f'{len(own_pawns_same)} of your own pawns occupy its colour complex. '
                    f'Either trade it for the opponent\'s active piece or restructure the pawns.')
    return ''

def _good_bishop(board: chess.Board, color: chess.Color, phase: str) -> str:
    enemy = not color
    for b_sq in board.pieces(chess.BISHOP, color):
        b_col = chess.square_color(b_sq)
        enemy_pawns_same = [sq for sq in board.pieces(chess.PAWN, enemy)
                            if chess.square_color(sq) == b_col]
        if enemy_pawns_same:
            targets = ', '.join(_sq(sq) for sq in sorted(enemy_pawns_same)[:2])
            return (f'The bishop on {_sq(b_sq)} eyes enemy pawns on {targets}. '
                    f'Manoeuvre to keep enemy pawns fixed on its colour complex.')
    return ''

def _king_safety(board: chess.Board, color: chess.Color, phase: str) -> str:
    king_sq = board.king(color)
    if king_sq is None:
        return ''
    side = _side_name(color)
    kr, kf = chess.square_rank(king_sq), chess.square_file(king_sq)
    shield_rank = kr + (1 if color == chess.WHITE else -1)
    shield_pawns = []
    for f in range(max(0, kf - 1), min(8, kf + 2)):
        if 0 <= shield_rank < 8:
            sq = chess.square(f, shield_rank)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                shield_pawns.append(_sq(sq))
    if not shield_pawns:
        return (f'{side}\'s king on {_sq(king_sq)} has NO pawn shield — '
                f'the position in front of the king is completely open to attack.')
    if len(shield_pawns) == 1:
        return (f'{side}\'s king on {_sq(king_sq)} has only one pawn shield at '
                f'{shield_pawns[0]} — the shelter is dangerously thin.')
    return (f'{side}\'s king on {_sq(king_sq)} is shielded by pawns on '
            f'{", ".join(shield_pawns)} — maintain this structure.')

def _trapped_piece(board: chess.Board, color: chess.Color, phase: str) -> str:
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != color:
            continue
        if piece.piece_type in (chess.KING, chess.PAWN):
            continue
        n_moves = sum(1 for m in board.legal_moves if m.from_square == sq)
        if n_moves <= 1:
            return (f'The {_piece_sym(piece)} on {_sq(sq)} has only '
                    f'{n_moves} legal move{"" if n_moves == 1 else "s"} — '
                    f'trap it immediately before it finds an escape route.')
    # Check enemy trapped pieces too
    enemy = not color
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != enemy:
            continue
        if piece.piece_type in (chess.KING, chess.PAWN):
            continue
        n_moves = sum(1 for m in board.legal_moves if m.from_square == sq)
        if n_moves <= 1:
            return (f'The opponent\'s {_piece_sym(piece)} on {_sq(sq)} is nearly trapped — '
                    f'cut its escape routes with pawns and pieces before it slips away.')
    return ''

def _pin(board: chess.Board, color: chess.Color, phase: str) -> str:
    enemy = not color
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == enemy and piece.piece_type != chess.KING:
            if board.is_pinned(enemy, sq):
                king_sq = board.king(enemy)
                king_str = f' to the king on {_sq(king_sq)}' if king_sq is not None else ''
                return (f'The {_piece_sym(piece)} on {_sq(sq)} is pinned{king_str} — '
                        f'pile additional attackers onto the pinned piece; it cannot retreat.')
    return ''

def _fork(board: chess.Board, color: chess.Color, phase: str) -> str:
    enemy = not color
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.color != color or p.piece_type != chess.KNIGHT:
            continue
        attacks = chess.SquareSet(chess.BB_KNIGHT_ATTACKS[sq])
        targets = [s for s in attacks
                   if (ep := board.piece_at(s)) and ep.color == enemy
                   and ep.piece_type not in (chess.PAWN,)]
        if len(targets) >= 2:
            p1, p2 = board.piece_at(targets[0]), board.piece_at(targets[1])
            return (f'The knight on {_sq(sq)} forks the {_piece_sym(p1)} on {_sq(targets[0])} '
                    f'and the {_piece_sym(p2)} on {_sq(targets[1])} — one of them must be lost.')
    return ''

def _battery(board: chess.Board, color: chess.Color, phase: str) -> str:
    heavy = (list(board.pieces(chess.ROOK, color))
             + list(board.pieces(chess.QUEEN, color)))
    for i, sq1 in enumerate(heavy):
        for sq2 in heavy[i + 1:]:
            p1, p2 = board.piece_at(sq1), board.piece_at(sq2)
            if chess.square_file(sq1) == chess.square_file(sq2):
                fn = _file_name(sq1)
                return (f'The {_piece_sym(p1)}–{_piece_sym(p2)} battery on the {fn}-file '
                        f'delivers concentrated pressure — find the target and break through.')
            if chess.square_rank(sq1) == chess.square_rank(sq2):
                r = chess.square_rank(sq1) + 1
                return (f'The {_piece_sym(p1)}–{_piece_sym(p2)} battery controls rank {r} — '
                        f'use it to penetrate the opponent\'s back ranks.')
    return ''

def _rook_seventh(board: chess.Board, color: chess.Color, phase: str) -> str:
    seventh = 6 if color == chess.WHITE else 1
    for sq in board.pieces(chess.ROOK, color):
        if chess.square_rank(sq) == seventh:
            return (f'The rook on {_sq(sq)} dominates the seventh rank — '
                    f'keep it there to harvest pawns and cut the enemy king off from safety.')
    return ''

def _piece_activity(board: chess.Board, color: chess.Color, phase: str) -> str:
    worst_sq, worst_n = None, 100
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p or p.color != color or p.piece_type in (chess.KING, chess.PAWN):
            continue
        n = sum(1 for m in board.legal_moves if m.from_square == sq)
        if n < worst_n:
            worst_n, worst_sq = n, sq
    if worst_sq is not None and worst_n <= 3:
        p = board.piece_at(worst_sq)
        return (f'The {_piece_sym(p)} on {_sq(worst_sq)} controls only {worst_n} '
                f'square{"" if worst_n == 1 else "s"} — the least active piece in the position. '
                f'Reroute it to a central square where it controls maximum space.')
    return ''

def _back_rank(board: chess.Board, color: chess.Color, phase: str) -> str:
    enemy = not color
    king_sq = board.king(enemy)
    if king_sq is None:
        return ''
    back = 7 if enemy == chess.WHITE else 0
    if chess.square_rank(king_sq) != back:
        return ''
    heavy = (list(board.pieces(chess.ROOK, color))
             + list(board.pieces(chess.QUEEN, color)))
    if not heavy:
        return ''
    side = _side_name(enemy)
    fn = _file_name(king_sq)
    return (f'The {side} king on {_sq(king_sq)} is confined to the back rank '
            f'with no escape square — a rook or queen on the {fn}-file delivers checkmate.')

def _mating_attack(board: chess.Board, color: chess.Color, phase: str) -> str:
    king_sq = board.king(not color)
    if king_sq is None:
        return ''
    side = _side_name(not color)
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
    controlled = sum(1 for sq in zone if board.is_attacked_by(color, sq))
    return (f'The {side} king on {_sq(king_sq)} is under pressure — '
            f'{controlled} of its surrounding squares are controlled by your pieces. '
            f'Concentrate every available piece on the king zone for the final assault.')

def _development_lead(board: chess.Board, color: chess.Color, phase: str) -> str:
    if phase != 'opening':
        return ''
    enemy = not color
    home = 0 if color == chess.WHITE else 7
    our_undeveloped = [sq for sq in list(board.pieces(chess.KNIGHT, color))
                       + list(board.pieces(chess.BISHOP, color))
                       if chess.square_rank(sq) == home]
    their_undeveloped = [sq for sq in list(board.pieces(chess.KNIGHT, enemy))
                         + list(board.pieces(chess.BISHOP, enemy))
                         if chess.square_rank(sq) == (7 - home)]
    side = _side_name(color)
    if len(our_undeveloped) < len(their_undeveloped):
        diff = len(their_undeveloped) - len(our_undeveloped)
        return (f'{side} leads in development by {diff} piece{"s" if diff > 1 else ""}. '
                f'Open the position now — a development lead evaporates if the position stays closed.')
    return ''

def _promotion(board: chess.Board, color: chess.Color, phase: str) -> str:
    passed = _find_passed_pawns(board, color)
    if not passed:
        return ''
    sq = passed[0]
    promo_rank = 8 if color == chess.WHITE else 1
    dist = abs(promo_rank - _rank(sq))
    if dist <= 3:
        side = _side_name(color)
        return (f'{side}\'s pawn on {_sq(sq)} is {dist} square{"s" if dist > 1 else ""} from '
                f'queening — clear the path immediately. Every tempo here is critical.')
    return ''

def _space_advantage(board: chess.Board, color: chess.Color, phase: str) -> str:
    # Count squares controlled past centre line
    center_rank = 4 if color == chess.WHITE else 3
    our_sq = sum(1 for sq in chess.SQUARES
                 if board.is_attacked_by(color, sq)
                 and (chess.square_rank(sq) >= center_rank if color == chess.WHITE
                      else chess.square_rank(sq) <= center_rank))
    their_sq = sum(1 for sq in chess.SQUARES
                   if board.is_attacked_by(not color, sq)
                   and (chess.square_rank(sq) >= center_rank if not color == chess.WHITE
                        else chess.square_rank(sq) <= center_rank))
    if our_sq > their_sq + 6:
        side = _side_name(color)
        return (f'{side} controls {our_sq} squares in enemy territory vs the opponent\'s {their_sq}. '
                f'Use the space edge to restrict enemy piece movement and prepare a pawn break.')
    return ''

def _pawn_storm(board: chess.Board, color: chess.Color, phase: str) -> str:
    enemy_king_sq = board.king(not color)
    if enemy_king_sq is None:
        return ''
    kf = chess.square_file(enemy_king_sq)
    # Find our pawns bearing down on that flank
    storm_pawns = [sq for sq in board.pieces(chess.PAWN, color)
                   if abs(chess.square_file(sq) - kf) <= 2]
    if len(storm_pawns) >= 2:
        most_advanced = sorted(storm_pawns,
                               key=lambda s: chess.square_rank(s) if color == chess.WHITE else -chess.square_rank(s),
                               reverse=True)[0]
        return (f'{len(storm_pawns)} pawns bearing down on the enemy king-side flank — '
                f'advance the lead pawn from {_sq(most_advanced)} to crack open the shelter.')
    return ''

def _prophylaxis(board: chess.Board, color: chess.Color, phase: str) -> str:
    # Find the most advanced enemy pawn as the threat to prevent
    enemy_passed = _find_passed_pawns(board, not color)
    if enemy_passed:
        sq = enemy_passed[0]
        return (f'The opponent\'s passed pawn on {_sq(sq)} is a long-term threat — '
                f'blockade it on {_sq(sq)} with a piece before it advances further.')
    return ''

def _initiative(board: chess.Board, color: chess.Color, phase: str) -> str:
    # Simply find how many checks/captures/threats are available
    n_threats = sum(1 for m in board.legal_moves
                    if board.gives_check(m)
                    or board.is_capture(m))
    if n_threats >= 3:
        side = _side_name(color)
        return (f'{side} has {n_threats} forcing moves available (checks and captures). '
                f'Keep the initiative — every move must create a new threat so the opponent never consolidates.')
    return ''


# ── Dispatch table ────────────────────────────────────────────────────────────

_BUILDERS: dict = {
    'passed_pawn':      _passed_pawn,
    'isolated_pawn':    _isolated_pawn,
    'doubled_pawn':     _doubled_pawn,
    'weak_square':      _weak_square,
    'outpost':          _outpost,
    'open_file':        _open_file,
    'bishop_pair':      _bishop_pair,
    'bad_bishop':       _bad_bishop,
    'good_bishop':      _good_bishop,
    'king_safety':      _king_safety,
    'trapped_piece':    _trapped_piece,
    'pin':              _pin,
    'fork':             _fork,
    'battery':          _battery,
    'rook_seventh':     _rook_seventh,
    'piece_activity':   _piece_activity,
    'back_rank':        _back_rank,
    'mating_attack':    _mating_attack,
    'development_lead': _development_lead,
    'promotion':        _promotion,
    'space_advantage':  _space_advantage,
    'pawn_storm':       _pawn_storm,
    'prophylaxis':      _prophylaxis,
    'initiative':       _initiative,
}
