"""
extractors/space_control.py
Measures spatial control in opponent half, split by flank, with trend tracking.

Relative scoring:
  white_qs = White squares attacked in Black half, queenside (a-d files, ranks 5-8)
  black_qs = Black squares attacked in White half, queenside (a-d files, ranks 1-4)
  delta     = (white_qs - black_qs) / 16  normalised [-1,1], converted to [0,1]
  signal side = the side with the advantage

Trend: linear regression slope over last 5 boards. Fires when |slope| > 0.05/move.
"""
from __future__ import annotations
import chess
from core.data_types import MetricSignal
from core.board_utils import square_to_str

_WHITE_OPP_RANKS = (4, 5, 6, 7)
_BLACK_OPP_RANKS = (0, 1, 2, 3)
_QS_FILES = (0, 1, 2, 3)
_KS_FILES = (4, 5, 6, 7)
_MAX_FLANK = 16
_TREND_WIN  = 5
_TREND_MIN  = 0.05


def extract_space_control(
    board: chess.Board,
    history: list[chess.Board] | None = None,
    phase: str = "middlegame",
) -> list[MetricSignal]:
    """Return space_delta_queenside, space_delta_kingside, and optionally space_trend."""
    w_qs, w_qs_sq = _attacked(board, chess.WHITE, _QS_FILES, _WHITE_OPP_RANKS)
    w_ks, w_ks_sq = _attacked(board, chess.WHITE, _KS_FILES, _WHITE_OPP_RANKS)
    b_qs, b_qs_sq = _attacked(board, chess.BLACK, _QS_FILES, _BLACK_OPP_RANKS)
    b_ks, b_ks_sq = _attacked(board, chess.BLACK, _KS_FILES, _BLACK_OPP_RANKS)

    signals = [
        _flank_signal("space_delta_queenside", w_qs, b_qs, w_qs_sq, b_qs_sq, phase),
        _flank_signal("space_delta_kingside",  w_ks, b_ks, w_ks_sq, b_ks_sq, phase),
    ]
    if history and len(history) >= 2:
        signals.extend(_trend_signal(board, history, phase))
    return signals


def _attacked(
    board: chess.Board,
    color: chess.Color,
    files: tuple,
    ranks: tuple,
) -> tuple[int, list[str]]:
    count, sqs = 0, []
    for f in files:
        for r in ranks:
            sq = chess.square(f, r)
            if board.is_attacked_by(color, sq):
                count += 1
                sqs.append(square_to_str(sq))
    return count, sqs


def _flank_signal(
    name: str,
    w: int, b: int,
    w_sqs: list[str], b_sqs: list[str],
    phase: str,
) -> MetricSignal:
    delta = w - b
    raw   = (delta / _MAX_FLANK + 1.0) / 2.0
    score = max(0.0, min(1.0, raw))
    flank = "queenside" if "queen" in name else "kingside"

    if delta > 0:
        side, key_sqs = "white", w_sqs[:6]
        cause = f"space_advantage_{flank}"
        hint  = f"White controls {flank} — advance to restrict Black options"
    elif delta < 0:
        side, key_sqs = "black", b_sqs[:6]
        score = 1.0 - score
        cause = f"space_advantage_{flank}"
        hint  = f"Black controls {flank} — maintain and expand pawn advances"
    else:
        side, key_sqs = "white", w_sqs[:3]
        cause = "space_equal"
        hint  = f"Space equal on {flank} — contest outpost squares"

    sev = "high" if abs(delta) >= 6 else "moderate" if abs(delta) >= 3 else "mild"
    return MetricSignal(
        metric_name=name, score=round(score, 4), side=side, cause=cause,
        key_squares=key_sqs, severity=sev, fragment="",
        action_hint=hint, phase=phase,
    )


def _delta_for(board: chess.Board) -> float:
    w_qs, _ = _attacked(board, chess.WHITE, _QS_FILES, _WHITE_OPP_RANKS)
    w_ks, _ = _attacked(board, chess.WHITE, _KS_FILES, _WHITE_OPP_RANKS)
    b_qs, _ = _attacked(board, chess.BLACK, _QS_FILES, _BLACK_OPP_RANKS)
    b_ks, _ = _attacked(board, chess.BLACK, _KS_FILES, _BLACK_OPP_RANKS)
    return ((w_qs + w_ks) - (b_qs + b_ks)) / (_MAX_FLANK * 2)


def _trend_signal(board: chess.Board, history: list[chess.Board], phase: str) -> list[MetricSignal]:
    window = history[-_TREND_WIN:]
    if len(window) < 2:
        return []
    deltas = [_delta_for(b) for b in window]
    n = len(deltas)
    mx = (n - 1) / 2.0
    my = sum(deltas) / n
    num = sum((i - mx) * (d - my) for i, d in enumerate(deltas))
    den = sum((i - mx) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    if abs(slope) < _TREND_MIN:
        return []
    ts = min(abs(slope) / 0.15, 1.0)
    side  = "white" if slope > 0 else "black"
    cause = "space_squeeze_tightening" if slope > 0 else "space_recovering"
    hint  = ("White squeeze tightening — push to restrict mobility"
             if slope > 0 else
             "Black recovering space — find counterplay before it's too late")
    return [MetricSignal(
        metric_name="space_trend", score=round(ts, 4), side=side, cause=cause,
        severity="high" if ts > 0.6 else "moderate", fragment="",
        action_hint=hint, phase=phase,
    )]
