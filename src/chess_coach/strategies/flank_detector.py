"""
strategies/flank_detector.py
Flank/Squeeze confidence scorer — Carlsen/Karpov style.

Philosophy: Slowly suffocate. Accumulate small advantages, restrict
options, and wait for collapse. Measured in TRENDS, not snapshots.

Firing condition: space advantage growing AND bad pieces AND mobility
ratio trending in your favour.

Weighted formula:
  snapshot = space_qs*0.15 + space_ks*0.15 + mob_ratio*0.20
           + outpost*0.15  + fixedness*0.10 + bad_pieces*0.15
           + option_red*0.10

Trend multiplier: if space_trend fires for player_side, multiply
snapshot score by 1.25 (capped at 1.0). This rewards the actual
squeeze tightening vs a static snapshot advantage.

Snapshot fallback: if no history, use snapshot score at reduced
weight (multiply by 0.80) and add a 'no_history' note.
"""
from __future__ import annotations
from core.data_types import MetricSignal

_W = dict(space_qs=0.15, space_ks=0.15, mob_ratio=0.20,
          outpost=0.15, fixedness=0.10, bad_pieces=0.15,
          option_red=0.10)
_TREND_MULT     = 1.25
_NO_HISTORY_MULT = 0.80


def score_flank(
    signals: list[MetricSignal],
    player_side: str,
    history_signals: list[list[MetricSignal]] | None = None,
) -> float:
    """Return flank/squeeze confidence [0,1] from player_side's perspective."""
    opp = 'black' if player_side == 'white' else 'white'

    space_qs  = _get(signals, 'space_delta_queenside', player_side)
    space_ks  = _get(signals, 'space_delta_kingside',  player_side)
    mob_ratio = _get(signals, 'piece_mobility_ratio',  player_side)
    outpost   = _get(signals, 'outpost_occupation',    player_side)
    fixedness = _get(signals, 'pawn_fixedness',        'white')  # structural — shared
    bad_n     = sum(1 for s in signals if s.metric_name == 'bad_piece' and s.side == opp)
    bad_pieces = min(bad_n / 3.0, 1.0)
    # option_reduction proxy: opponent mobility trending down
    mob_trend_opp = _get(signals, 'mobility_trend', opp)
    option_red = mob_trend_opp

    snapshot = (space_qs  * _W['space_qs']  +
                space_ks  * _W['space_ks']  +
                mob_ratio * _W['mob_ratio']  +
                outpost   * _W['outpost']   +
                fixedness * _W['fixedness'] +
                bad_pieces* _W['bad_pieces']  +
                option_red* _W['option_red'])

    # Trend multiplier — flank is most reliable when the squeeze is tightening
    trend_active = any(
        s.metric_name == 'space_trend' and s.side == player_side
        for s in signals
    )
    if history_signals and len(history_signals) >= 2:
        if trend_active:
            snapshot = min(snapshot * _TREND_MULT, 1.0)
    else:
        # Snapshot fallback: valid but reduced confidence
        snapshot = snapshot * _NO_HISTORY_MULT

    return round(min(max(snapshot, 0.0), 1.0), 4)


def _get(signals: list[MetricSignal], metric: str, side: str) -> float:
    return max((s.score for s in signals if s.metric_name == metric and s.side == side), default=0.0)
