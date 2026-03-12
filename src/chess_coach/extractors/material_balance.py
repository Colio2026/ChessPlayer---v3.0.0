"""
extractors/material_balance.py
Centipawn eval deficit, sacrifice delta, piece trade tracking, overextension.

Relative scoring:
  eval_deficit: signed centipawn score from each side's perspective.
  sacrifice_delta: (standard_material_value - actual_material) vs eval shift.
    Positive = gave up material but position improved or held — Tal fingerprint.
  overextension: opponent pawns past 5th rank without pawn support behind them.

Requires EvalResult from StockfishBridge for eval_deficit and sacrifice_delta.
If eval_result is None, those signals are skipped (graceful degradation).
"""
from __future__ import annotations
import chess
from core.data_types import MetricSignal
from core.board_utils import square_to_str

_PIECE_VALUES = {
    chess.QUEEN: 900, chess.ROOK: 500,
    chess.BISHOP: 325, chess.KNIGHT: 300, chess.PAWN: 100,
}
_OVEREXT_RANKS = {chess.WHITE: range(4, 7), chess.BLACK: range(1, 4)}


def extract_material_balance(
    board: chess.Board,
    eval_result=None,
    phase: str = "middlegame",
) -> list[MetricSignal]:
    """
    Returns material balance signals. eval_result is an EvalResult (may be None).
    """
    signals: list[MetricSignal] = []
    signals.extend(_material_count_signal(board, phase))
    if eval_result is not None:
        signals.extend(_eval_deficit_signals(board, eval_result, phase))
        signals.extend(_sacrifice_delta_signal(board, eval_result, phase))
    signals.extend(_overextension_signals(board, chess.WHITE, phase))
    signals.extend(_overextension_signals(board, chess.BLACK, phase))
    return signals


def _material_value(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(pt, color)) * val
        for pt, val in _PIECE_VALUES.items()
    )


def _material_count_signal(board: chess.Board, phase: str) -> list[MetricSignal]:
    w = _material_value(board, chess.WHITE)
    b = _material_value(board, chess.BLACK)
    delta = w - b
    if abs(delta) < 50:
        return []  # Within half-pawn — material equal, no signal needed

    side_str = "white" if delta > 0 else "black"
    adv = abs(delta)
    score = min(adv / 900, 1.0)  # Normalise: queen = 1.0
    sev = "critical" if adv >= 500 else "high" if adv >= 300 else "moderate"
    return [MetricSignal(
        metric_name="material_count", score=round(score, 4), side=side_str,
        cause="material_advantage",
        severity=sev, fragment="",
        action_hint=("White" if side_str == "white" else "Black") + f" is up {adv}cp — simplify to a winning endgame",
        phase=phase,
    )]


def _eval_deficit_signals(board: chess.Board, eval_result, phase: str) -> list[MetricSignal]:
    if eval_result.centipawns is None:
        return []
    cp = eval_result.centipawns
    if abs(cp) < 30:
        return []

    side_str = "black" if cp > 0 else "white"  # negative side is losing
    score = min(abs(cp) / 300.0, 1.0)
    sev = "critical" if abs(cp) >= 200 else "high" if abs(cp) >= 100 else "moderate"
    return [MetricSignal(
        metric_name="eval_deficit", score=round(score, 4), side=side_str,
        cause="eval_deficit",
        severity=sev, fragment="",
        action_hint="find defensive resources or seek immediate counterplay",
        phase=phase,
    )]


def _sacrifice_delta_signal(board: chess.Board, eval_result, phase: str) -> list[MetricSignal]:
    """
    Detect sacrifice patterns: material down but eval neutral or better.
    A positive sacrifice_delta = gave up material AND kept the eval.
    """
    if eval_result.centipawns is None:
        return []
    cp = eval_result.centipawns
    w_mat = _material_value(board, chess.WHITE)
    b_mat = _material_value(board, chess.BLACK)
    mat_delta = w_mat - b_mat  # positive = White has more material

    # Sacrifice detected if: material deficit but eval NOT proportionally worse
    # White sacrificed: mat_delta < -50 but eval >= mat_delta * 0.5 (holding despite deficit)
    # Black sacrificed: mat_delta > 50  but eval <= -mat_delta * 0.5

    threshold = 150  # minimum sacrifice to report (minor piece+)

    if mat_delta < -threshold and cp >= mat_delta * 0.4:
        # White sacrificed and is holding or better
        score = min(abs(mat_delta) / 500.0, 1.0)
        return [MetricSignal(
            metric_name="sacrifice_delta", score=round(score, 4), side="white",
            cause="material_sacrifice_holding",
            key_pieces=[], severity="high", fragment="",
            action_hint="material down but position holds — press the attack before opponent consolidates",
            phase=phase,
        )]
    elif mat_delta > threshold and cp <= -mat_delta * 0.4:
        score = min(mat_delta / 500.0, 1.0)
        return [MetricSignal(
            metric_name="sacrifice_delta", score=round(score, 4), side="black",
            cause="material_sacrifice_holding",
            key_pieces=[], severity="high", fragment="",
            action_hint="material down but position holds — press the attack before opponent consolidates",
            phase=phase,
        )]
    return []


def _overextension_signals(
    board: chess.Board, color: chess.Color, phase: str
) -> list[MetricSignal]:
    """Opponent pawns advanced past 5th rank without pawn support behind them."""
    opp = not color
    opp_pawns = list(board.pieces(chess.PAWN, opp))
    overext: list[chess.Square] = []

    # Overextended = opponent pawn past the midpoint, with no own pawn behind
    for psq in opp_pawns:
        f, r = chess.square_file(psq), chess.square_rank(psq)
        # "past midpoint" = rank 5+ for White pawns (r >= 4), rank 4- for Black
        if opp == chess.WHITE and r < 4:
            continue
        if opp == chess.BLACK and r > 3:
            continue

        # Check if there's a supporting pawn on adjacent/same file behind it
        supported = False
        opp_dir = -1 if opp == chess.WHITE else 1
        for df in (-1, 0, 1):
            bf = f + df
            if not (0 <= bf <= 7):
                continue
            for dr in range(1, 4):
                support_r = r + opp_dir * dr
                if not (0 <= support_r <= 7):
                    break
                support_sq = chess.square(bf, support_r)
                sp = board.piece_at(support_sq)
                if sp and sp.piece_type == chess.PAWN and sp.color == opp:
                    supported = True
                    break
            if supported:
                break

        if not supported:
            overext.append(psq)

    if not overext:
        return []

    side_str = "white" if color == chess.WHITE else "black"
    score = min(len(overext) / 3.0, 1.0)
    key_sqs = [square_to_str(sq) for sq in overext[:4]]
    return [MetricSignal(
        metric_name="overextension", score=round(score, 4), side=side_str,
        cause="opponent_pawn_overextended",
        key_squares=key_sqs,
        severity="high" if score > 0.6 else "moderate", fragment="",
        action_hint=f"attack overextended pawn on {key_sqs[0]} — it lacks pawn support",
        phase=phase,
    )]
