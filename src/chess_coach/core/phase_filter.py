"""
core/phase_filter.py
=====================
Classifies game phase and re-weights MetricSignals accordingly.

Phase boundaries (hard thresholds, matching board_utils.get_phase):
  opening:    material >= 50
  middlegame: 28 <= material < 50
  endgame:    material < 28

Re-weighting rationale
----------------------
Certain metrics matter more in certain phases. A missing pawn shield
is catastrophic in the middlegame but less relevant in a king-and-pawn
endgame. Passed pawns are mildly interesting in the opening but
decisive in the endgame.

Phase multipliers (applied to MetricSignal.score):
  metric                  opening  middlegame  endgame
  king_exposure           0.6      1.0         0.5
  space_delta_*           0.8      1.0         1.1
  piece_mobility_ratio    0.9      1.0         1.0
  pawn_fixedness          0.7      1.0         1.2
  passed_pawn             0.5      0.8         1.4
  material_count          1.0      1.0         1.2
  outpost_occupation      0.6      1.0         1.1
  tactic_*                1.0      1.0         0.8
  (all others)            1.0      1.0         1.0

Scores are clamped to [0.0, 1.0] after weighting.
Phase tag is injected into each MetricSignal.phase field.
"""
from __future__ import annotations
import chess
from chess_coach.core.data_types import MetricSignal
from chess_coach.core.board_utils import get_phase

# Phase multipliers: metric_prefix -> (opening, middlegame, endgame)
_WEIGHTS: dict[str, tuple[float, float, float]] = {
    'king_exposure':        (0.6, 1.0, 0.5),
    'space_delta':          (0.8, 1.0, 1.1),   # prefix match
    'space_trend':          (0.7, 1.0, 1.1),
    'piece_mobility_ratio': (0.9, 1.0, 1.0),
    'mobility_trend':       (0.9, 1.0, 1.0),
    'pawn_fixedness':       (0.7, 1.0, 1.2),
    'passed_pawn':          (0.5, 0.8, 1.4),
    'weak_pawns':           (0.7, 1.0, 1.1),
    'outpost_occupation':   (0.6, 1.0, 1.1),
    'material_count':       (1.0, 1.0, 1.2),
    'eval_deficit':         (0.8, 1.0, 1.2),
    'sacrifice_delta':      (0.9, 1.0, 0.7),
    'overextension':        (0.8, 1.0, 1.1),
    'tactic_':              (1.0, 1.0, 0.8),   # prefix match
    'bad_piece':            (0.8, 1.0, 1.0),
}
_PHASE_IDX = {'opening': 0, 'middlegame': 1, 'endgame': 2}


def apply_phase_filter(
    signals: list[MetricSignal],
    board: chess.Board,
) -> tuple[str, list[MetricSignal]]:
    """
    Classify phase and re-weight all MetricSignals.

    Parameters
    ----------
    signals : list[MetricSignal]
        Raw signals from all extractors (phase field may be stale).
    board : chess.Board
        Current position for phase classification.

    Returns
    -------
    (phase_str, weighted_signals)
        phase_str        : 'opening' | 'middlegame' | 'endgame'
        weighted_signals : new MetricSignal list with updated scores and phase tags
    """
    phase = get_phase(board)
    idx   = _PHASE_IDX[phase]

    weighted: list[MetricSignal] = []
    for sig in signals:
        mult  = _get_multiplier(sig.metric_name, idx)
        new_score = round(min(sig.score * mult, 1.0), 4)

        # Re-inject phase tag (extractor may have used a default)
        weighted.append(MetricSignal(
            metric_name = sig.metric_name,
            score       = new_score,
            side        = sig.side,
            cause       = sig.cause,
            key_squares = sig.key_squares,
            key_pieces  = sig.key_pieces,
            severity    = _severity_from_score(new_score),
            fragment    = sig.fragment,
            action_hint = sig.action_hint,
            phase       = phase,
        ))

    return phase, weighted


def _get_multiplier(metric_name: str, phase_idx: int) -> float:
    """Return the phase multiplier for this metric (prefix match)."""
    for key, weights in _WEIGHTS.items():
        if metric_name.startswith(key):
            return weights[phase_idx]
    return 1.0   # default: no adjustment


def _severity_from_score(score: float) -> str:
    if score >= 0.75: return 'critical'
    if score >= 0.50: return 'high'
    if score >= 0.25: return 'moderate'
    return 'mild'
