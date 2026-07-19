"""
coach/narrator.py
==================
Pure slot-filling assembly. No scoring. No logic. No chess knowledge.

The narrator's job is exactly this:
    signals + strategy + phase + phrase_db + precedents → CoachOutput

It asks the phrase DB for fragments, fills the slots in order, and
packages everything into a CoachOutput. If a slot has no phrase it
falls back to the signal's own action_hint. If there are no hints
it uses a neutral stub. No decision is ever made here about what
the position means — that work is done upstream.

Design contract (spec Section 12):
    "Metrics own the facts. Language owns the words."
    "No extractor produces English. No narrator produces scores."

Headline assembly
-----------------
1. Try phrase DB for a 'headline' fragment matching strategy/phase/signals
2. Fall back to template string built from strategy name + confidence

Plan sentence assembly (4-slot ordered)
-----------------------------------------
Slot 1 diagnosis  — what is wrong
Slot 2 evidence   — why we know
Slot 3 plan       — what to do
Slot 4 urgency    — why right now

Each slot filled by the highest-priority matching phrase for the
leading MetricSignal. Slots with no phrase are skipped rather than
filled with empty strings — the output always has 2–4 sentences.

Tactic hints
------------
Sourced from the tactics table, one per tactic type, max 3 total.
"""
from __future__ import annotations

from chess_coach.core.data_types        import CoachOutput, MetricSignal, GMPrecedent
from chess_coach.core.conflict_resolver import ResolverResult
from chess_coach.database.phrase_db     import PhraseDB


_STRATEGY_NAMES: dict[str, str] = {
    # Legacy
    'blitz':            'Kingside Attack',
    'flank':            'Positional Squeeze',
    'fortress':         'Fortress Defence',
    'feint':            'Positional Feint',
    # Tier 1 ML strategies
    'mating_attack':    'Mating Attack',
    'passed_pawn':      'Passed Pawn Advance',
    'outpost':          'Outpost Occupation',
    'space_advantage':  'Space Advantage',
    'pawn_storm':       'Pawn Storm',
    'pawn_majority':    'Pawn Majority',
    'blockade':         'Blockade',
    'prophylaxis':      'Prophylaxis',
    'initiative':       'Initiative',
    'development_lead': 'Development Lead',
    'piece_activity':   'Piece Activity',
    'king_activity':    'King Centralisation',
    # Fallback
    'general':          'Positional Play',
}

_SLOT_ORDER = ('diagnosis', 'evidence', 'plan', 'urgency')


# ── Public API ────────────────────────────────────────────────────────────────

def assemble(
    result: ResolverResult,
    phase: str,
    signals: list[MetricSignal],
    player_side: str,
    phrase_db: PhraseDB,
    gm_precedents: list[GMPrecedent],
    move_flags: list[dict],
    weakness_squares: list[str],
) -> CoachOutput:
    """
    Assemble a complete CoachOutput from pre-computed parts.

    Parameters
    ----------
    result : ResolverResult
        Output of conflict_resolver.resolve() — primary/secondary strategy,
        confidence, tie_band flag.
    phase : str
        Game phase: 'opening' | 'middlegame' | 'endgame'
    signals : list[MetricSignal]
        Phase-filtered signals from all extractors.
    player_side : str
        'white' | 'black'
    phrase_db : PhraseDB
        Open phrase database. May be unavailable — narrator degrades gracefully.
    gm_precedents : list[GMPrecedent]
        0–3 GM matches from pattern_matcher.
    move_flags : list[dict]
        Pre-computed by plan_recommender.
    weakness_squares : list[str]
        Pre-computed by plan_recommender.
    """
    strategy = result.primary
    headline       = _build_headline(result, phase, signals, phrase_db)
    plan_sentences = _build_plan(strategy, phase, signals, phrase_db)
    tactic_hints   = _build_tactic_hints(signals, player_side, phrase_db)

    return CoachOutput(
        strategy_primary   = strategy,
        strategy_secondary = result.secondary,
        confidence         = result.confidence,
        phase              = phase,
        headline           = headline,
        plan_sentences     = plan_sentences,
        tactic_hints       = tactic_hints,
        move_flags         = move_flags,
        weakness_squares   = weakness_squares,
        gm_precedents      = gm_precedents,
        signal_dump        = signals,
    )


# ── Headline ──────────────────────────────────────────────────────────────────

def _build_headline(
    result: ResolverResult,
    phase: str,
    signals: list[MetricSignal],
    phrase_db: PhraseDB,
) -> str:
    """Try phrase DB 'headline' fragment first, fall back to template."""
    if phrase_db.is_available and signals:
        fragments = phrase_db.get_fragments(
            result.primary, phase, signals, max_per_slot=1
        )
        hl = fragments.get('headline', [])
        if hl and hl[0].strip():
            return hl[0]
    return _template_headline(result, phase)


def _template_headline(result: ResolverResult, phase: str) -> str:
    name = _STRATEGY_NAMES.get(result.primary, result.primary.title())
    conf = result.confidence

    if result.tie_band and result.secondary:
        sec = _STRATEGY_NAMES.get(result.secondary, result.secondary.title())
        return (f'{name} or {sec} — two plans are in tension '
                f'({phase}, {conf:.0%} confidence).')

    urgency = ('decisively indicated' if conf >= 0.80
               else 'recommended' if conf >= 0.65
               else 'suggested')
    return f'{name} is {urgency} — {phase} ({conf:.0%} confidence).'


# ── Plan sentences ────────────────────────────────────────────────────────────

def _build_plan(
    strategy: str,
    phase: str,
    signals: list[MetricSignal],
    phrase_db: PhraseDB,
) -> list[str]:
    """
    Fill 4 slots from phrase DB. Fall back to action_hints.
    Always returns 2–4 sentences to satisfy CoachOutput contract.
    """
    if phrase_db.is_available and signals:
        fragments = phrase_db.get_fragments(strategy, phase, signals)
        sentences = [s for slot in _SLOT_ORDER
                     for s in fragments.get(slot, []) if s.strip()]
        if len(sentences) >= 2:
            return sentences[:4]

    # Fallback: action_hints from top signals
    ranked = sorted(signals, key=lambda s: s.score, reverse=True)
    hints  = [s.action_hint for s in ranked if s.action_hint][:4]
    if len(hints) >= 2:
        return hints

    return [
        'The position has been analysed.',
        'Consult the signal panel for detailed metric readings.',
    ]


# ── Tactic hints ──────────────────────────────────────────────────────────────

def _build_tactic_hints(
    signals: list[MetricSignal],
    player_side: str,
    phrase_db: PhraseDB,
) -> list[str]:
    """Return 0–3 tactic hint sentences."""
    tactic_signals = sorted(
        [s for s in signals
         if s.metric_name.startswith('tactic_') and s.side == player_side],
        key=lambda s: s.score, reverse=True,
    )
    if phrase_db.is_available and tactic_signals:
        phase = tactic_signals[0].phase
        hints = phrase_db.get_tactic_hints(tactic_signals, phase, max_hints=3)
        if hints:
            return hints
    return [s.action_hint for s in tactic_signals[:3] if s.action_hint]
