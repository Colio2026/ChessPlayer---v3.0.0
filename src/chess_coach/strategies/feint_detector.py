"""
strategies/feint_detector.py
Feint/Misdirection confidence scorer — Petrosian/Fischer style.

Philosophy: Pretend to go left, then go right. A quiet preparatory
move that looks passive sets up a devastating threat 2–4 moves later.

Firing condition: engine_disagreement (your move ≠ engine best but
eval holds) AND latent multi-move threat being constructed AND opponent
overcommits to the wrong area.

DB DEPENDENCY (Phase 4)
-----------------------
The spec states: "Feint relies most heavily on the 6M game database
for pattern confirmation." Without the DB, feint CANNOT fire as
primary — it caps at 0.64 (below the 0.65 fire threshold).

This is the Phase 3 stub implementation. The score is computed from
available signals only. When the DB is wired in (Phase 4), the
db_confirmation parameter will lift the cap.

Weighted formula (DB-pending):
  raw = eng_disagree*0.35 + latent_threat*0.25 + opp_overcommit*0.20
      + quiet_move*0.10   + positional_tension*0.10

Cap: min(raw, 0.64) until db_confirmation=True.
"""
from __future__ import annotations
from core.data_types import MetricSignal

_W = dict(eng_disagree=0.35, latent=0.25, overcommit=0.20,
          quiet=0.10, tension=0.10)
_DB_PENDING_CAP = 0.64   # cannot fire as primary without DB confirmation


def score_feint(
    signals: list[MetricSignal],
    player_side: str,
    history_signals: list[list[MetricSignal]] | None = None,
    db_confirmation: bool = False,
) -> float:
    """
    Return feint confidence [0,1] from player_side's perspective.

    db_confirmation : bool
        Set True when Phase 4 DB matcher finds a matching GM precedent.
        Until then, score is capped at 0.64 (below fire threshold).
    """
    opp = 'black' if player_side == 'white' else 'white'

    # Engine disagreement proxy: mobility_trend NOT strongly favouring either side
    # (the position looks equal to an engine but has hidden tension)
    mob_ratio = _get(signals, 'piece_mobility_ratio', player_side)
    # Near 0.5 = equal mobility = engine sees nothing but human sees potential
    eng_disagree = 1.0 - abs(mob_ratio - 0.5) * 4.0
    eng_disagree = max(eng_disagree, 0.0)

    # Latent threat proxy: outpost available but not yet occupied
    outpost = _get(signals, 'outpost_occupation', player_side)
    # Outpost AVAILABLE (score > 0 but pieces not yet there) = preparation happening
    latent = outpost * 0.5  # modest weight — full detection needs DB

    # Opponent overcommit: opponent space on one flank only = overcommit to that flank
    opp_qs = _get(signals, 'space_delta_queenside', opp)
    opp_ks = _get(signals, 'space_delta_kingside',  opp)
    overcommit = abs(opp_qs - opp_ks)  # high when opponent is lopsided

    # Quiet move indicator: no tactical signals firing = preparatory position
    tactic_n = sum(1 for s in signals if s.side == player_side and s.metric_name.startswith('tactic_'))
    quiet = 1.0 - min(tactic_n / 3.0, 1.0)

    # Positional tension: fixed structure + latent outpost = tension held in reserve
    fixedness = _get(signals, 'pawn_fixedness', 'white')
    tension = fixedness * 0.5

    raw = (eng_disagree * _W['eng_disagree'] +
           latent       * _W['latent']       +
           overcommit   * _W['overcommit']   +
           quiet        * _W['quiet']        +
           tension      * _W['tension'])

    # DB gate: cannot fire as primary until DB confirms a GM pattern
    if not db_confirmation:
        raw = min(raw, _DB_PENDING_CAP)

    return round(min(max(raw, 0.0), 1.0), 4)


def _get(signals: list[MetricSignal], metric: str, side: str) -> float:
    return max((s.score for s in signals if s.metric_name == metric and s.side == side), default=0.0)
