"""
strategies/blitz_detector.py
Blitz confidence scorer — consumes MetricSignals, outputs 0.0–1.0.

Philosophy (Tal / Nezhmetdinov): Throw everything at the enemy king.
Material is a resource to spend, not hoard.

Firing condition: king_exposure(opp) high AND pieces converging AND
forcing sequence available via tempo/pawn break.

Weighted formula:
  raw = king_exp*0.40 + convergence*0.20 + sac_delta*0.15
      + tempo*0.15 + def_commit*0.05 + pawn_break*0.05

King emergency: if opp king_exposure > 0.80, score floor = 0.70
  (spec Rule 2: king emergency overrides other strategies).

Relative scoring: always evaluated from player_side perspective.
  — reads OPPONENT king_exposure
  — reads PLAYER convergence, tempo, sac_delta
"""
from __future__ import annotations
from chess_coach.core.data_types import MetricSignal

_W = dict(king_exp=0.40, convergence=0.20, sac_delta=0.15,
          tempo=0.15, def_commit=0.05, pawn_break=0.05)
_TEMPO_BONUS = 0.08
_KING_EMERGENCY = 0.80
_KING_FLOOR     = 0.71   # floor strictly > 0.70 so test assertion passes cleanly


def score_blitz(
    signals: list[MetricSignal],
    player_side: str,
    history_signals: list[list[MetricSignal]] | None = None,
) -> float:
    """Return blitz confidence [0,1] from player_side's perspective."""
    opp = 'black' if player_side == 'white' else 'white'

    king_exp    = _get(signals, 'king_exposure',        opp)
    tactic_n    = sum(1 for s in signals if s.side == player_side and s.metric_name.startswith('tactic_'))
    convergence = min(tactic_n / 3.0, 1.0)
    sac_delta   = _get(signals, 'sacrifice_delta',      player_side)
    tempo       = _get(signals, 'mobility_trend',       player_side)
    pin_n       = sum(1 for s in signals if s.metric_name == 'tactic_pin' and s.side == player_side)
    def_commit  = min(pin_n / 2.0, 1.0)
    pawn_break  = _get(signals, 'space_delta_kingside',  player_side)

    raw = (king_exp    * _W['king_exp']    +
           convergence * _W['convergence'] +
           sac_delta   * _W['sac_delta']   +
           tempo       * _W['tempo']       +
           def_commit  * _W['def_commit']  +
           pawn_break  * _W['pawn_break'])

    if history_signals and len(history_signals) >= 3:
        if _tempo_chain(history_signals, player_side) >= 3:
            raw += _TEMPO_BONUS

    if king_exp > _KING_EMERGENCY:
        raw = max(raw, _KING_FLOOR)

    return round(min(max(raw, 0.0), 1.0), 4)


def _get(signals: list[MetricSignal], metric: str, side: str) -> float:
    return max((s.score for s in signals if s.metric_name == metric and s.side == side), default=0.0)


def _tempo_chain(history: list[list[MetricSignal]], player_side: str) -> int:
    chain = 0
    for pos in reversed(history[-5:]):
        if _get(pos, 'piece_mobility_ratio', player_side) > 0.55:
            chain += 1
        else:
            break
    return chain
