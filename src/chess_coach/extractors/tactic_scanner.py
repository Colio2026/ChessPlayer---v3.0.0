"""
extractors/tactic_scanner.py
Detect pins, forks, skewers, and discovered attacks.
Outputs tactic MetricSignals for the tactic_hints slot in CoachOutput.

Relative scoring: each tactic is a concrete threat for one side.
Signal side = the side that HAS the tactic available (can exploit it).
"""
from __future__ import annotations
import chess
from core.data_types import MetricSignal
from core.board_utils import square_to_str


_PIECE_VALUES = {
    chess.KING: 20000, chess.QUEEN: 900, chess.ROOK: 500,
    chess.BISHOP: 325, chess.KNIGHT: 300, chess.PAWN: 100,
}


def extract_tactics(
    board: chess.Board,
    phase: str = "middlegame",
) -> list[MetricSignal]:
    """
    Detect tactical patterns for both sides.
    Returns a list of tactic MetricSignals (may be empty).
    """
    signals: list[MetricSignal] = []
    for color in (chess.WHITE, chess.BLACK):
        signals.extend(_find_pins(board, color, phase))
        signals.extend(_find_forks(board, color, phase))
        signals.extend(_find_skewers(board, color, phase))
        signals.extend(_find_discoveries(board, color, phase))
    return signals


# ── Pins ──────────────────────────────────────────────────────────────────────

def _find_pins(board: chess.Board, attacker: chess.Color, phase: str) -> list[MetricSignal]:
    """
    Find pieces that are pinned to a higher-value piece or king.
    A piece is pinned if moving it would expose a more valuable piece to attack.
    """
    defender = not attacker
    signals: list[MetricSignal] = []
    side_str = "white" if attacker == chess.WHITE else "black"

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != defender:
            continue
        if p.piece_type in (chess.KING,):
            continue

        # Try removing this piece and see if the king becomes attacked
        temp = board.copy()
        temp.remove_piece_at(sq)
        king_sq_temp = temp.king(defender)
        if king_sq_temp is None:
            continue
        if temp.is_attacked_by(attacker, king_sq_temp):
            # Absolute pin confirmed — find attacker on TEMP board
            # (on original board the pinned piece blocks, so must use temp)
            pin_attackers = list(temp.attackers(attacker, king_sq_temp))
            if not pin_attackers:
                continue
            att_sq = pin_attackers[0]
            att_p = temp.piece_at(att_sq)
            if att_p is None:
                continue
            king_sq = board.king(defender)
            score = min(_PIECE_VALUES.get(p.piece_type, 100) / 900.0, 1.0)
            signals.append(MetricSignal(
                metric_name="tactic_pin", score=round(score, 4), side=side_str,
                cause="absolute_pin",
                key_squares=[square_to_str(sq), square_to_str(att_sq)],
                key_pieces=[att_p.symbol().upper() + square_to_str(att_sq),
                            p.symbol().upper() + square_to_str(sq)],
                severity="high" if p.piece_type in (chess.QUEEN, chess.ROOK) else "moderate",
                fragment="",
                action_hint=f"{att_p.symbol().upper()}{square_to_str(att_sq)} pins "
                            f"{p.symbol().upper()}{square_to_str(sq)} to the king",
                phase=phase,
            ))

    return signals[:3]  # cap at 3 pin signals


# ── Forks ─────────────────────────────────────────────────────────────────────

def _find_forks(board: chess.Board, attacker: chess.Color, phase: str) -> list[MetricSignal]:
    """
    Find pieces that attack two or more opponent pieces simultaneously.
    """
    defender = not attacker
    signals: list[MetricSignal] = []
    side_str = "white" if attacker == chess.WHITE else "black"

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != attacker:
            continue
        if p.piece_type == chess.KING:
            continue

        # Find all opponent pieces this piece attacks
        attacked_opp: list[chess.Square] = []
        for tsq in chess.SQUARES:
            tp = board.piece_at(tsq)
            if tp is None or tp.color != defender:
                continue
            if sq in board.attackers(attacker, tsq):
                attacked_opp.append(tsq)

        if len(attacked_opp) >= 2:
            # Sort by value — highest value targets first
            attacked_opp.sort(
                key=lambda s: _PIECE_VALUES.get(
                    board.piece_at(s).piece_type, 0) if board.piece_at(s) else 0,
                reverse=True,
            )
            total_value = sum(
                _PIECE_VALUES.get(board.piece_at(s).piece_type, 0)
                for s in attacked_opp if board.piece_at(s)
            )
            score = min(total_value / 1800.0, 1.0)
            key_sqs = [square_to_str(sq)] + [square_to_str(s) for s in attacked_opp[:3]]
            key_pieces = (
                [p.symbol().upper() + square_to_str(sq)] +
                [board.piece_at(s).symbol().upper() + square_to_str(s)  # type: ignore
                 for s in attacked_opp[:3] if board.piece_at(s)]
            )
            sev = "critical" if any(
                board.piece_at(s) and board.piece_at(s).piece_type in (chess.KING, chess.QUEEN)  # type: ignore
                for s in attacked_opp
            ) else "high"
            signals.append(MetricSignal(
                metric_name="tactic_fork", score=round(score, 4), side=side_str,
                cause="fork",
                key_squares=key_sqs[:5],
                key_pieces=key_pieces[:4],
                severity=sev, fragment="",
                action_hint=(f"{p.symbol().upper()}{square_to_str(sq)} forks "
                             f"{' and '.join(key_pieces[1:3])}"),
                phase=phase,
            ))

    return signals[:3]


# ── Skewers ───────────────────────────────────────────────────────────────────

def _find_skewers(board: chess.Board, attacker: chess.Color, phase: str) -> list[MetricSignal]:
    """
    Detect skewers: attacking a higher-value piece through which a lower-value
    piece sits behind on the same ray.
    """
    defender = not attacker
    signals: list[MetricSignal] = []
    side_str = "white" if attacker == chess.WHITE else "black"

    slider_types = (chess.QUEEN, chess.ROOK, chess.BISHOP)

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != attacker or p.piece_type not in slider_types:
            continue

        # Get all squares this piece attacks
        for attacked_sq in board.attacks(sq):
            target = board.piece_at(attacked_sq)
            if target is None or target.color != defender:
                continue

            # Check if there's a piece behind the target on the same ray
            df = chess.square_file(attacked_sq) - chess.square_file(sq)
            dr = chess.square_rank(attacked_sq) - chess.square_rank(sq)
            if df != 0: df = df // abs(df)
            if dr != 0: dr = dr // abs(dr)

            behind_f = chess.square_file(attacked_sq) + df
            behind_r = chess.square_rank(attacked_sq) + dr

            if not (0 <= behind_f <= 7 and 0 <= behind_r <= 7):
                continue

            behind_sq = chess.square(behind_f, behind_r)
            behind_p = board.piece_at(behind_sq)

            if (behind_p is not None and behind_p.color == defender and
                    _PIECE_VALUES.get(target.piece_type, 0) >
                    _PIECE_VALUES.get(behind_p.piece_type, 0)):
                score = min(_PIECE_VALUES.get(target.piece_type, 100) / 900.0, 1.0)
                signals.append(MetricSignal(
                    metric_name="tactic_skewer", score=round(score, 4), side=side_str,
                    cause="skewer",
                    key_squares=[square_to_str(sq), square_to_str(attacked_sq),
                                 square_to_str(behind_sq)],
                    key_pieces=[p.symbol().upper() + square_to_str(sq),
                                target.symbol().upper() + square_to_str(attacked_sq)],
                    severity="high", fragment="",
                    action_hint=(f"{p.symbol().upper()}{square_to_str(sq)} skewers "
                                 f"{target.symbol().upper()} to win "
                                 f"{behind_p.symbol().upper()} behind"),
                    phase=phase,
                ))

    return signals[:3]


# ── Discovered attacks ────────────────────────────────────────────────────────

def _find_discoveries(board: chess.Board, attacker: chess.Color, phase: str) -> list[MetricSignal]:
    """
    Find pieces whose movement would reveal an attack by a sliding piece behind them.
    """
    defender = not attacker
    signals: list[MetricSignal] = []
    side_str = "white" if attacker == chess.WHITE else "black"

    for move in board.legal_moves:
        if board.piece_at(move.from_square) is None:
            continue
        mover = board.piece_at(move.from_square)
        if mover is None or mover.color != attacker:
            continue

        # Simulate the move and see if new attacks are revealed
        temp = board.copy()
        temp.push(move)

        # Check if any new opponent pieces are now attacked that weren't before
        for tsq in chess.SQUARES:
            tp = board.piece_at(tsq)
            if tp is None or tp.color != defender:
                continue
            if tp.piece_type == chess.PAWN:
                continue  # Minor discoveries on pawns not worth flagging

            was_attacked = board.is_attacked_by(attacker, tsq)
            now_attacked = temp.is_attacked_by(attacker, tsq)

            if not was_attacked and now_attacked:
                # Newly revealed attack — is it from a piece that didn't just move?
                new_attackers = [
                    sq for sq in temp.attackers(attacker, tsq)
                    if sq != move.to_square  # exclude the moving piece
                ]
                if new_attackers:
                    rev_sq = new_attackers[0]
                    rev_p = temp.piece_at(rev_sq)
                    if rev_p and rev_p.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP):
                        score = min(_PIECE_VALUES.get(tp.piece_type, 100) / 900.0, 1.0)
                        signals.append(MetricSignal(
                            metric_name="tactic_discovery", score=round(score, 4), side=side_str,
                            cause="discovered_attack",
                            key_squares=[square_to_str(move.from_square),
                                         square_to_str(rev_sq), square_to_str(tsq)],
                            key_pieces=[mover.symbol().upper() + square_to_str(move.from_square),
                                        rev_p.symbol().upper() + square_to_str(rev_sq)],
                            severity="critical" if tp.piece_type in (chess.QUEEN, chess.KING)
                                     else "high",
                            fragment="",
                            action_hint=(f"move {mover.symbol().upper()}{square_to_str(move.from_square)} "
                                         f"to reveal {rev_p.symbol().upper()} attack on "
                                         f"{tp.symbol().upper()}{square_to_str(tsq)}"),
                            phase=phase,
                        ))
                        if len(signals) >= 3:
                            return signals

    return signals
