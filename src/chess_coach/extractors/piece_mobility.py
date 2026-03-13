"""
extractors/piece_mobility.py
Measures legal move counts, mobility ratio, bad pieces, and trend.

Relative scoring:
  ratio = my_moves / (my_moves + opp_moves)   [0.5 = equal]
  Signal fires for the side with the higher ratio.
  Bad pieces: bishops blocked by own pawns, rooks on closed files,
              pieces with < 3 legal destination squares.
  Trend: slope of ratio over last 6 positions.
"""
from __future__ import annotations
import chess
from chess_coach.core.data_types import MetricSignal
from chess_coach.core.board_utils import square_to_str, count_legal_moves


def extract_piece_mobility(
    board: chess.Board,
    history: list[chess.Board] | None = None,
    phase: str = "middlegame",
) -> list[MetricSignal]:
    """Return mobility_ratio, bad_pieces, and optionally mobility_trend."""
    w_moves = count_legal_moves(board, chess.WHITE)
    b_moves = count_legal_moves(board, chess.BLACK)
    signals = [_ratio_signal(board, w_moves, b_moves, phase)]
    signals.extend(_bad_piece_signals(board, chess.WHITE, phase))
    signals.extend(_bad_piece_signals(board, chess.BLACK, phase))
    if history and len(history) >= 2:
        signals.extend(_trend_signal(history, phase))
    return signals


def _ratio_signal(board: chess.Board, w: int, b: int, phase: str) -> MetricSignal:
    total = w + b
    if total == 0:
        score = 0.5
    else:
        score = w / total  # >0.5 = White has more mobility

    delta = w - b
    if delta > 0:
        side  = "white"
        cause = "mobility_advantage"
        hint  = f"White has {delta} more legal moves — use mobility to create threats"
    elif delta < 0:
        side  = "black"
        score = 1.0 - score
        cause = "mobility_advantage"
        hint  = f"Black has {abs(delta)} more legal moves — use mobility advantage"
    else:
        side  = "white"
        cause = "mobility_equal"
        hint  = "Mobility is equal — improve piece coordination"

    sev = ("high"     if abs(delta) > 10 else
           "moderate" if abs(delta) > 5  else "mild")
    return MetricSignal(
        metric_name="piece_mobility_ratio", score=round(score, 4),
        side=side, cause=cause, severity=sev, fragment="", action_hint=hint, phase=phase,
    )


def _bad_piece_signals(
    board: chess.Board, color: chess.Color, phase: str
) -> list[MetricSignal]:
    side_str = "white" if color == chess.WHITE else "black"
    opp      = not color
    bad: list[MetricSignal] = []

    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != color:
            continue

        sq_name = square_to_str(sq)

        # ── Blocked bishop ────────────────────────────────────────────────
        if p.piece_type == chess.BISHOP:
            sq_color = (chess.square_file(sq) + chess.square_rank(sq)) % 2
            own_pawns_blocking = 0
            for psq in board.pieces(chess.PAWN, color):
                if (chess.square_file(psq) + chess.square_rank(psq)) % 2 == sq_color:
                    own_pawns_blocking += 1
            if own_pawns_blocking >= 4:
                bad.append(MetricSignal(
                    metric_name="bad_piece", score=0.70, side=side_str,
                    cause="bad_bishop",
                    key_squares=[sq_name],
                    key_pieces=[p.symbol().upper() + sq_name],
                    severity="high",
                    fragment="",
                    action_hint=f"exchange or reroute the bad bishop on {sq_name}",
                    phase=phase,
                ))

        # ── Rook on closed file ───────────────────────────────────────────
        if p.piece_type == chess.ROOK:
            f = chess.square_file(sq)
            own_pawn  = any(board.piece_at(chess.square(f, r)) is not None and
                            board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and  # type: ignore
                            board.piece_at(chess.square(f, r)).color == color                # type: ignore
                            for r in range(8))
            opp_pawn  = any(board.piece_at(chess.square(f, r)) is not None and
                            board.piece_at(chess.square(f, r)).piece_type == chess.PAWN and  # type: ignore
                            board.piece_at(chess.square(f, r)).color == opp                  # type: ignore
                            for r in range(8))
            if own_pawn and opp_pawn:
                bad.append(MetricSignal(
                    metric_name="bad_piece", score=0.55, side=side_str,
                    cause="rook_on_closed_file",
                    key_squares=[sq_name],
                    key_pieces=["R" + sq_name],
                    severity="moderate",
                    fragment="",
                    action_hint=f"open the {chess.FILE_NAMES[f]}-file or reroute rook from {sq_name}",
                    phase=phase,
                ))

        # ── Piece with < 3 good target squares ───────────────────────────
        if p.piece_type not in (chess.KING, chess.PAWN):
            piece_moves = [
                m for m in board.legal_moves
                if m.from_square == sq
            ]
            # "Good" moves: not retreating to own half, captures or advances
            good_moves = []
            own_half_max_rank = 3 if color == chess.WHITE else 4
            for m in piece_moves:
                to_rank = chess.square_rank(m.to_square)
                if color == chess.WHITE and to_rank > own_half_max_rank:
                    good_moves.append(m)
                elif color == chess.BLACK and to_rank < own_half_max_rank + 1:
                    good_moves.append(m)
                elif board.piece_at(m.to_square) is not None:  # capture
                    good_moves.append(m)

            if len(piece_moves) < 3 and len(piece_moves) > 0:
                bad.append(MetricSignal(
                    metric_name="bad_piece", score=0.60, side=side_str,
                    cause="restricted_piece",
                    key_squares=[sq_name],
                    key_pieces=[p.symbol().upper() + sq_name],
                    severity="moderate",
                    fragment="",
                    action_hint=f"reroute {p.symbol().upper()} from {sq_name} — only {len(piece_moves)} legal moves",
                    phase=phase,
                ))

    return bad


def _mob_delta(board: chess.Board) -> float:
    w = count_legal_moves(board, chess.WHITE)
    b = count_legal_moves(board, chess.BLACK)
    total = w + b
    return (w - b) / max(total, 1)


def _trend_signal(history: list[chess.Board], phase: str) -> list[MetricSignal]:
    window = history[-6:]
    if len(window) < 2:
        return []
    deltas = [_mob_delta(b) for b in window]
    n  = len(deltas)
    mx = (n - 1) / 2.0
    my = sum(deltas) / n
    num = sum((i - mx) * (d - my) for i, d in enumerate(deltas))
    den = sum((i - mx) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    if abs(slope) < 0.03:
        return []
    ts   = min(abs(slope) / 0.10, 1.0)
    side = "white" if slope > 0 else "black"
    return [MetricSignal(
        metric_name="mobility_trend", score=round(ts, 4), side=side,
        cause="mobility_squeeze" if slope > 0 else "mobility_recovering",
        severity="high" if ts > 0.6 else "moderate", fragment="",
        action_hint="mobility gap widening — restrict opponent piece activity" if slope > 0
                    else "mobility recovering — find active squares before paralysis",
        phase=phase,
    )]
