"""
core/conflict_resolver.py
==========================
Applies the 5-rule priority cascade from spec Section 6.

Input:  raw scores {blitz, flank, fortress, feint} + context signals
Output: (primary, secondary|None, confidence, tie_band: bool)

Priority cascade (applied in order — earlier rules win):

  Rule 1 — Eval check:
    If eval_deficit > 1.5 (normalised: > 0.50 in our 0-1 scale),
    Fortress gets a +0.25 multiplier bonus regardless of other scores.

  Rule 2 — King emergency:
    If king_exposure(opponent) > 0.80, Blitz overrides Flank
    regardless of score difference.

  Rule 3 — Phase override:
    In endgame, Flank overrides Blitz if both score above 0.65.

  Rule 4 — Tie band:
    If primary and secondary scores are within 0.08, output BOTH
    strategies and set tie_band=True.

  Rule 5 — Feint gate:
    Feint only outputs as primary if db_confirmation=True.
    Otherwise capped at secondary note.

Fire threshold: a strategy must score > 0.65 to be considered.
Below threshold, the highest scorer is used as primary regardless.
"""
from __future__ import annotations
from dataclasses import dataclass

_FIRE_THRESHOLD = 0.65
_TIE_BAND       = 0.08
_EVAL_BONUS     = 0.25
_KING_EMERGENCY = 0.80


@dataclass
class ResolverResult:
    primary:    str
    secondary:  str | None
    confidence: float
    tie_band:   bool


def resolve(
    scores: dict[str, float],
    signals_lookup: dict,
    phase: str,
    player_side: str,
    db_confirmation: bool = False,
) -> ResolverResult:
    """
    Apply cascade rules and return the resolved strategy.

    Parameters
    ----------
    scores : dict[str, float]
        {'blitz': 0.xx, 'flank': 0.xx, 'fortress': 0.xx, 'feint': 0.xx}
    signals_lookup : dict
        Helper dict with pre-computed context:
          'eval_deficit'   : float  (0-1 normalised)
          'king_exposure'  : float  (opponent's king exposure)
        Built by strategy_engine before calling resolve().
    phase : str
        'opening' | 'middlegame' | 'endgame'
    player_side : str
        'white' | 'black'
    db_confirmation : bool
        True if the DB matcher confirmed a feint GM pattern.
    """
    s = dict(scores)  # copy — we mutate during cascade

    # ── Rule 1: Eval check → Fortress bonus ───────────────────────────────
    eval_deficit = signals_lookup.get('eval_deficit', 0.0)
    if eval_deficit > 0.50:   # > 1.5 pawns in real terms
        s['fortress'] = min(s['fortress'] + _EVAL_BONUS, 1.0)

    # ── Sort by score ──────────────────────────────────────────────────────
    ranked = sorted(s.items(), key=lambda x: x[1], reverse=True)
    primary_name, primary_score = ranked[0]
    second_name,  second_score  = ranked[1]

    # ── Rule 2: King emergency → Blitz overrides Flank ────────────────────
    rule2_fired = False
    king_exp = signals_lookup.get('king_exposure', 0.0)
    if king_exp > _KING_EMERGENCY:
        if primary_name == 'flank' and s['blitz'] >= s['flank'] * 0.75:
            primary_name, primary_score = 'blitz', s['blitz']
            remaining = [(n, v) for n, v in ranked if n != 'blitz']
            second_name, second_score = remaining[0]
            rule2_fired = True

    # ── Rule 3: Phase override → Flank beats Blitz in endgame ─────────────
    # Only applies if Rule 2 did NOT fire — Rule 2 (immediate king danger)
    # takes priority over phase-based preference.
    if phase == 'endgame' and not rule2_fired:
        if (primary_name == 'blitz' and primary_score > _FIRE_THRESHOLD and
                s['flank'] > _FIRE_THRESHOLD):
            primary_name, primary_score = 'flank', s['flank']
            second_name,  second_score  = 'blitz',  s['blitz']

    # ── Rule 5: Feint gate → feint cannot be primary without DB ───────────
    if primary_name == 'feint' and not db_confirmation:
        # Demote feint to secondary, promote the next best
        remaining = [(n, v) for n, v in ranked if n != 'feint']
        primary_name, primary_score = remaining[0]
        second_name,  second_score  = 'feint', s['feint']

    # ── Rule 4: Tie band → output both strategies ─────────────────────────
    tie = False
    secondary = None
    if abs(primary_score - second_score) <= _TIE_BAND:
        if second_score >= _FIRE_THRESHOLD * 0.85:  # secondary must be credible
            secondary = second_name
            tie = True

    return ResolverResult(
        primary    = primary_name,
        secondary  = secondary,
        confidence = round(primary_score, 4),
        tie_band   = tie,
    )
