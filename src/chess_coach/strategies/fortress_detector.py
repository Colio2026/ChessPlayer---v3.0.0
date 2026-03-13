"""
strategies/fortress_detector.py
Fortress/Blockade confidence scorer — Botvinnik/Petrosian style.

Philosophy: You are objectively worse. Lock the position down, find
the wall, eliminate pawn breaks, neutralise the opponent's plan, and
wait for overextension.

Firing condition: eval_deficit exists AND pawn structure is fixed AND
opponent breakthrough potential is low.

PREREQUISITE: Fortress only fires meaningfully when the coached side
is LOSING (eval_deficit > 0). Without a deficit, this score is
deliberately suppressed — a winning side does not need a fortress.

Weighted formula:
  raw = deficit_bonus*0.35 + fixedness*0.25 + king_eq*0.15
      + trade_ratio*0.10  + low_dynamic*0.10 + overext_watch*0.05

Eval deficit bonus: eval_deficit score (from material_balance extractor)
  is the gating input. If 0 (no deficit detected), fortress score
  is capped at 0.30 regardless of other inputs.

Overextension flip: if overextension signal fires FOR the coached side
  (opponent pawns are overextended), the overext_watch component
  is treated as a counterattack opportunity, boosting the score.
"""
from __future__ import annotations
from chess_coach.core.data_types import MetricSignal

_W = dict(deficit=0.35, fixedness=0.25, king_eq=0.15,
          trade=0.10, low_dyn=0.10, overext=0.05)
_NO_DEFICIT_CAP = 0.30


def score_fortress(
    signals: list[MetricSignal],
    player_side: str,
    history_signals: list[list[MetricSignal]] | None = None,
) -> float:
    """Return fortress confidence [0,1] from player_side's perspective."""
    opp = 'black' if player_side == 'white' else 'white'

    # ── Prerequisite: eval deficit ─────────────────────────────────────────
    eval_deficit = _get(signals, 'eval_deficit', player_side)

    # Fixedness: locked structure benefits the defensive side
    fixedness = _get(signals, 'pawn_fixedness', 'white')  # structural signal

    # King safety equality: if both kings equally safe → no attack vector
    w_king_exp = _get(signals, 'king_exposure', 'white')
    b_king_exp = _get(signals, 'king_exposure', 'black')
    king_delta = abs(w_king_exp - b_king_exp)
    king_eq = 1.0 - min(king_delta * 2.0, 1.0)  # high when kings equally safe

    # Piece trade ratio: fewer pieces = less opponent firepower
    # Proxy: material_count signal shows how much material is left on board
    mat_sig = _get(signals, 'material_count', opp)
    trade = 1.0 - mat_sig  # higher when opponent has less material

    # Low dynamic potential: mobility_trend NOT active = static position
    mob_trend_opp = _get(signals, 'mobility_trend', opp)
    low_dyn = 1.0 - mob_trend_opp

    # Overextension watch: opponent pawns overextended = counterattack possible
    overext = _get(signals, 'overextension', player_side)

    raw = (eval_deficit * _W['deficit']   +
           fixedness    * _W['fixedness']  +
           king_eq      * _W['king_eq']    +
           trade        * _W['trade']      +
           low_dyn      * _W['low_dyn']    +
           overext      * _W['overext'])

    # Prerequisite gate: no deficit → cap at 0.30
    if eval_deficit == 0.0:
        raw = min(raw, _NO_DEFICIT_CAP)

    return round(min(max(raw, 0.0), 1.0), 4)


def _get(signals: list[MetricSignal], metric: str, side: str) -> float:
    return max((s.score for s in signals if s.metric_name == metric and s.side == side), default=0.0)
