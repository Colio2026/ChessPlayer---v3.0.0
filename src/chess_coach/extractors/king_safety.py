"""
extractors/king_safety.py
=========================
Measures king danger for both sides independently then produces
relative MetricSignals reflecting each side's true exposure.

Relative scoring design
-----------------------
King safety is inherently relative. This extractor computes:
  - raw_exposure(side)  — absolute vulnerability of that king
  - raw_attack(side)    — how aggressively opponent pieces bear in

final score = clamp(raw_exposure + attack_pressure_bonus, 0, 1)

Components measured
-------------------
1. Pawn shield integrity on f/g/h (kingside) or a/b/c (queenside)
   - Each missing close-rank shield pawn: +0.20
   - Each missing far-rank shield pawn (only when close also empty): +0.12
2. Open files adjacent to king
   - Fully open: +0.18
   - Semi-open (own pawn gone, opponent present): +0.10
3. Attacker concentration in king zone (7 squares around king)
   - Per opponent piece bearing on zone: +0.14  (Q=1.5x, R=1.2x weight)
4. King tropism — Σ(weight/chebyshev_dist) for opponent pieces, capped at 0.15

Normalisation: score = min(raw / 1.20, 1.0)
Calibrated so starting position ≈ 0.03, Tal attack position > 0.80.
"""
from __future__ import annotations
import chess
from core.data_types import MetricSignal
from core.board_utils import get_king_zone, square_to_str

_SHIELD_MISSING_CLOSE = 0.20
_SHIELD_MISSING_FAR   = 0.12
_OPEN_FILE_FULL       = 0.18
_OPEN_FILE_SEMI       = 0.10
_ATTACKER_BASE        = 0.14
_QUEEN_MULT           = 1.5
_ROOK_MULT            = 1.2
_TROPISM_MAX          = 0.15
_NORMALISER           = 0.93   # calibrated: starting pos <0.15, Tal attack >0.80
_SHIELD_OFFSETS       = (-1, 0, 1)


def extract_king_safety(
    board: chess.Board,
    phase: str = 'middlegame',
) -> list[MetricSignal]:
    """Return [white_signal, black_signal] — both sides always emitted."""
    return [
        _compute_exposure(board, chess.WHITE, phase),
        _compute_exposure(board, chess.BLACK, phase),
    ]


def _compute_exposure(
    board: chess.Board,
    side: chess.Color,
    phase: str,
) -> MetricSignal:
    color_str = 'white' if side == chess.WHITE else 'black'
    opponent  = not side

    king_sq = board.king(side)
    if king_sq is None:
        return MetricSignal(metric_name='king_exposure', score=0.0,
                            side=color_str, cause='no_king', phase=phase)

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    king_zone = get_king_zone(board, side)

    raw          = 0.0
    cause_tags   = []
    key_squares  = []
    key_pieces   = []

    # 1. Pawn shield
    s_raw, s_causes, s_squares = _pawn_shield(board, side, king_file, king_rank)
    raw += s_raw; cause_tags.extend(s_causes); key_squares.extend(s_squares)

    # 2. Open files
    f_raw, f_causes, f_squares = _open_files(board, side, king_file, king_rank)
    raw += f_raw; cause_tags.extend(f_causes); key_squares.extend(f_squares)

    # 3. Attacker concentration
    a_raw, a_pieces = _attacker_concentration(board, opponent, king_zone)
    raw += a_raw; key_pieces.extend(a_pieces)
    if a_raw > 0:
        cause_tags.append('pieces_bearing_on_king_zone')

    # 4. King tropism
    t_raw = _tropism(board, opponent, king_sq)
    raw += t_raw
    if t_raw > 0.08:
        cause_tags.append('pieces_approaching_king')

    # Phase adjustment: endgame kings should centralise
    if phase == 'endgame':
        raw *= 0.60

    score = min(raw / _NORMALISER, 1.0)

    severity = ('critical' if score >= 0.75 else
                'high'     if score >= 0.50 else
                'moderate' if score >= 0.25 else 'mild')

    primary_cause = cause_tags[0] if cause_tags else ('king_safe' if score < 0.15 else 'king_exposure')

    return MetricSignal(
        metric_name = 'king_exposure',
        score       = round(score, 4),
        side        = color_str,
        cause       = primary_cause,
        key_squares = list(dict.fromkeys(key_squares))[:8],
        key_pieces  = list(dict.fromkeys(key_pieces))[:6],
        severity    = severity,
        fragment    = '',
        action_hint = _action_hint(side, score, key_squares, key_pieces),
        phase       = phase,
    )


def _pawn_shield(
    board: chess.Board,
    side: chess.Color,
    king_file: int,
    king_rank: int,
) -> tuple[float, list[str], list[str]]:
    raw = 0.0
    causes: list[str] = []
    missing: list[str] = []

    is_white  = (side == chess.WHITE)
    close_rank = king_rank + 1 if is_white else king_rank - 1
    far_rank   = king_rank + 2 if is_white else king_rank - 2

    # Only apply shield penalty when king is on the flank (castled area)
    on_flank = king_file <= 2 or king_file >= 5

    if not on_flank:
        return 0.0, [], []

    for df in _SHIELD_OFFSETS:
        f = king_file + df
        if not (0 <= f <= 7):
            continue

        # ── Close shield rank ─────────────────────────────────────────────
        if 0 <= close_rank <= 7:
            sq = chess.square(f, close_rank)
            p  = board.piece_at(sq)
            has_own = p is not None and p.piece_type == chess.PAWN and p.color == side
            if not has_own:
                raw += _SHIELD_MISSING_CLOSE
                sq_name = square_to_str(sq)
                if sq_name not in missing:
                    missing.append(sq_name)
                if 'missing_pawn_shield' not in causes:
                    causes.append('missing_pawn_shield')

        # ── Far shield rank (only if close also missing) ──────────────────
        if 0 <= far_rank <= 7 and 0 <= close_rank <= 7:
            close_sq = chess.square(f, close_rank)
            cp = board.piece_at(close_sq)
            close_has_own = cp is not None and cp.piece_type == chess.PAWN and cp.color == side
            if not close_has_own:
                far_sq = chess.square(f, far_rank)
                fp = board.piece_at(far_sq)
                far_has_own = fp is not None and fp.piece_type == chess.PAWN and fp.color == side
                if not far_has_own:
                    raw += _SHIELD_MISSING_FAR

    return raw, causes, missing


def _open_files(
    board: chess.Board,
    side: chess.Color,
    king_file: int,
    king_rank: int,
) -> tuple[float, list[str], list[str]]:
    raw = 0.0
    causes: list[str] = []
    squares: list[str] = []

    for df in _SHIELD_OFFSETS:
        f = king_file + df
        if not (0 <= f <= 7):
            continue

        own_pawn = any(
            board.piece_at(chess.square(f, r)) is not None and
            board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and  # type: ignore
            board.piece_at(chess.square(f, r)).color == side                 # type: ignore
            for r in range(8)
        )
        opp_pawn = any(
            board.piece_at(chess.square(f, r)) is not None and
            board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and  # type: ignore
            board.piece_at(chess.square(f, r)).color != side                 # type: ignore
            for r in range(8)
        )

        if not own_pawn and not opp_pawn:
            raw += _OPEN_FILE_FULL
            causes.append('open_file_adjacent_to_king')
            squares.append(square_to_str(chess.square(f, king_rank)))
        elif not own_pawn and opp_pawn:
            raw += _OPEN_FILE_SEMI
            causes.append('semi_open_file_adjacent_to_king')

    return raw, list(dict.fromkeys(causes)), squares


def _attacker_concentration(
    board: chess.Board,
    attacker: chess.Color,
    king_zone: list[chess.Square],
) -> tuple[float, list[str]]:
    raw = 0.0
    pieces: list[str] = []

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != attacker or p.piece_type == chess.KING:
            continue

        # Does this piece attack any square in the king zone?
        attacks_zone = any(sq in board.attackers(attacker, zone_sq) for zone_sq in king_zone)
        if attacks_zone:
            mult = (_QUEEN_MULT if p.piece_type == chess.QUEEN else
                    _ROOK_MULT  if p.piece_type == chess.ROOK  else 1.0)
            raw += _ATTACKER_BASE * mult
            desc = p.symbol().upper() + square_to_str(sq)
            if desc not in pieces:
                pieces.append(desc)

    return raw, pieces


def _tropism(
    board: chess.Board,
    attacker: chess.Color,
    king_sq: chess.Square,
) -> float:
    kf = chess.square_file(king_sq)
    kr = chess.square_rank(king_sq)
    total = 0.0

    weights = {chess.QUEEN: 1.5, chess.ROOK: 1.0, chess.BISHOP: 0.7, chess.KNIGHT: 0.7}

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != attacker or p.piece_type == chess.KING:
            continue
        dist = max(abs(chess.square_file(sq) - kf), abs(chess.square_rank(sq) - kr))
        if dist <= 5:
            total += weights.get(p.piece_type, 0.5) / max(dist, 1)

    return min(total / 20.0, _TROPISM_MAX)


def _action_hint(side: chess.Color, score: float,
                 missing_shields: list[str], attackers: list[str]) -> str:
    color = 'White' if side == chess.WHITE else 'Black'
    if score < 0.20:
        return f'{color} king is safe — maintain pawn structure'
    if missing_shields and score >= 0.50:
        return f'restore pawn shield at {missing_shields[0]} or exchange key attackers'
    if attackers and score >= 0.65:
        return f'eliminate {attackers[0]} — primary attacker bearing on king zone'
    if score >= 0.40:
        return f'{color} king needs defensive consolidation'
    return f'monitor {color.lower()} king safety — structure weakened'
