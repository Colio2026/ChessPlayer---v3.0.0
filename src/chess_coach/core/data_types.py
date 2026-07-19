"""
core/data_types.py
==================
All structured data objects exchanged between layers.

Design rule (Section 12 of spec):
  - Extractors produce MetricSignal objects — never raw numbers, never English.
  - The narrator consumes MetricSignal objects — never raw scores.
  - CoachOutput is the sole output of the entire backend.
  - No layer may bypass these contracts.

Every field is documented here because these types are the API surface
that all 18 modules depend on. Changing a field name here is a
breaking change across the entire project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Literals used across all layers ──────────────────────────────────────────

SIDES      = ('white', 'black')
PHASES     = ('opening', 'middlegame', 'endgame')
SEVERITIES = ('critical', 'high', 'moderate', 'mild')
STRATEGIES = (
    # Legacy rule-based strategies (kept for backward compat with old checkpoints)
    'blitz', 'flank', 'fortress', 'feint',
    # Tier 1 ML concept strategies
    'mating_attack', 'passed_pawn', 'outpost', 'space_advantage',
    'pawn_storm', 'pawn_majority', 'blockade', 'prophylaxis',
    'initiative', 'development_lead', 'piece_activity', 'king_activity',
    # Fallback when no Tier 1 concept fires above threshold
    'general',
)


# ── MetricSignal ─────────────────────────────────────────────────────────────

@dataclass
class MetricSignal:
    """
    The atomic unit of measurement produced by every extractor.

    Layer 1 (extractors) writes these.
    Layer 2 (strategy scorers) reads them.
    Layer 5 (narrator) reads them for sentence assembly.

    Fields
    ------
    metric_name : str
        Unique identifier for this metric.
        Convention: snake_case noun phrase.
        Examples: 'king_exposure', 'space_delta', 'piece_mobility_ratio',
                  'pawn_chain_stability', 'sacrifice_delta'.

    score : float
        Normalised score in [0.0, 1.0].
        0.0 = metric is entirely absent / irrelevant.
        1.0 = metric is at maximum intensity.
        The extractors own normalisation — scorers receive clean floats.

    side : str
        Which side this signal applies to: 'white' or 'black'.
        Every signal is side-specific. A signal for White's king exposure
        is a different object from one for Black's.

    cause : str
        Machine-readable cause tag. Used by scorers and the phrase DB
        lookup to select the correct phrase template.
        Examples: 'missing_pawn_shield', 'bad_bishop', 'overextended_pawn',
                  'open_file_adjacent_to_king', 'knight_outpost_d5'.

    key_squares : list[str]
        Algebraic square names relevant to this signal.
        Examples: ['g7', 'h6', 'f5'] for a king exposure signal.
        Used by plan_recommender to populate weakness_squares in CoachOutput.
        Empty list is valid when no specific square is responsible.

    key_pieces : list[str]
        Piece descriptors relevant to this signal.
        Format: '<Color><PieceType>[<square>]'  e.g. 'Ng5', 'Qd3', 'Bc4'.
        Used by narrator to name specific pieces in sentences.
        Empty list is valid.

    severity : str
        Priority tier for sentence selection and ordering.
        'critical' — position-defining. Narrator leads with this.
        'high'     — important but not immediately decisive.
        'moderate' — supporting evidence.
        'mild'     — context only; may be omitted if space is tight.

    fragment : str
        Human-readable cause fragment sourced from the phrase database.
        NOT set by the extractor — set to '' at extraction time.
        Filled in by database/phrase_db.py before the narrator runs.
        Example: 'the g7–h6 pawn shield has been dissolved'

    action_hint : str
        What to DO about this metric. Set by the extractor because
        the extractor knows the position context.
        Example: 'g4 pawn break tears open the h-file',
                 'reroute knight via e3–d5',
                 'exchange the dark-squared bishop on h6'.

    phase : str
        Game phase at the time this signal was computed.
        Injected by core/phase_filter.py before scorers run.
        'opening' | 'middlegame' | 'endgame'
    """

    metric_name:  str
    score:        float
    side:         str
    cause:        str
    key_squares:  list[str]   = field(default_factory=list)
    key_pieces:   list[str]   = field(default_factory=list)
    severity:     str         = 'moderate'
    fragment:     str         = ''
    action_hint:  str         = ''
    phase:        str         = 'middlegame'

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"MetricSignal.score must be in [0.0, 1.0], got {self.score!r} "
                f"(metric={self.metric_name!r})"
            )
        if self.side not in SIDES:
            raise ValueError(f"MetricSignal.side must be one of {SIDES}, got {self.side!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"MetricSignal.severity must be one of {SEVERITIES}, got {self.severity!r}")
        if self.phase not in PHASES:
            raise ValueError(f"MetricSignal.phase must be one of {PHASES}, got {self.phase!r}")


# ── GMPrecedent ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GMPrecedent:
    """
    A single GM game match returned by database/pattern_matcher.py.

    The widget renders these as clickable references that load the
    GM game into the board viewer (Section 10 of spec).

    Fields
    ------
    player : str
        Name of the GM who played the position. Format: 'Lastname, Firstname'
        or 'Lastname' for universally recognised players.

    game_id : str
        Unique identifier tracing back to the PGN source.
        Format: '<filename>:<game_number_within_file>'
        Example: 'tal_games.pgn:142'

    ply : int
        Half-move number of the matched position within the game.
        Used by the widget to navigate directly to this position.

    key_move : str
        The move played by the GM from this position, in UCI format.
        Example: 'g2g4'

    annotation : str
        Human annotation of the move if present in the source PGN.
        Empty string if no annotation available.
    """

    player:     str
    game_id:    str
    ply:        int
    key_move:   str
    annotation: str = ''


# ── CoachOutput ───────────────────────────────────────────────────────────────

@dataclass
class CoachOutput:
    """
    The single output object of the entire coach backend.

    Produced by coach/narrator.py. Consumed by the PySide6 widget.
    The widget renders; it never interprets or transforms logic.

    Fields
    ------
    strategy_primary : str
        The dominant strategy: 'blitz' | 'flank' | 'fortress' | 'feint'.

    strategy_secondary : str | None
        Second strategy when within the tie-band threshold defined in
        core/conflict_resolver.py. None if primary is unambiguous.

    confidence : float
        0.0–1.0 confidence of the primary strategy classification.
        Strategies only fire in coaching output when confidence > 0.65
        (Section 4 of spec).

    phase : str
        Game phase at the time of analysis: 'opening' | 'middlegame' | 'endgame'.

    headline : str
        One assembled sentence summarising the position.
        Displayed prominently in the widget above plan_sentences.
        Example: 'White's king on g8 is critically exposed — the attack must
                  begin before Black consolidates.'

    plan_sentences : list[str]
        2–4 assembled sentences: diagnosis, evidence, plan, urgency.
        These are the main coaching text shown in the widget.

    tactic_hints : list[str]
        0–3 short tactical observations from the tactics extractor.
        Shown in a secondary / collapsible section, visually distinct
        from plan_sentences.

    move_flags : list[dict]
        Per-move flags that drive the move list heatmap.
        Each entry: { 'move': '<uci>', 'flag': '<tag>', 'strategy': '<name>' }
        Example: { 'move': 'g2g4', 'flag': 'strong_break', 'strategy': 'blitz' }

    weakness_squares : list[str]
        Algebraic square names identified as weak.
        Aggregated from key_squares across all fired MetricSignals.
        Drives the coloured overlay on the board in the widget.

    gm_precedents : list[GMPrecedent]
        0–3 GM game matches from the position database.
        Rendered as clickable references in the widget.

    signal_dump : list[MetricSignal]
        Full list of all MetricSignals computed for this position.
        Used for the debug / advanced detail panel in the widget.
        Not displayed in normal coaching view.
    """

    strategy_primary:   str
    strategy_secondary: Optional[str]
    confidence:         float
    phase:              str
    headline:           str
    plan_sentences:     list[str]
    tactic_hints:       list[str]              = field(default_factory=list)
    move_flags:         list[dict]             = field(default_factory=list)
    weakness_squares:   list[str]              = field(default_factory=list)
    gm_precedents:      list[GMPrecedent]      = field(default_factory=list)
    signal_dump:        list[MetricSignal]     = field(default_factory=list)
    # Each row: (human_label, before_pawns, after_pawns, delta_from_player_pov)
    # Populated by explainer.py when a PV breakdown is available.
    metrics_table:      list[tuple]            = field(default_factory=list)
    # Stockfish PV line — populated when coach explains a concrete line
    pv_san:             list[str]              = field(default_factory=list)
    pv_uci:             list[str]              = field(default_factory=list)
    pv_line_text:       str                    = ''   # "11. Nf3 Nc6  12. Bb5 a6"

    def __post_init__(self) -> None:
        if self.strategy_primary not in STRATEGIES:
            raise ValueError(
                f"CoachOutput.strategy_primary must be one of {STRATEGIES}, "
                f"got {self.strategy_primary!r}"
            )
        if self.strategy_secondary is not None and self.strategy_secondary not in STRATEGIES:
            raise ValueError(
                f"CoachOutput.strategy_secondary must be one of {STRATEGIES} or None, "
                f"got {self.strategy_secondary!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"CoachOutput.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if self.phase not in PHASES:
            raise ValueError(f"CoachOutput.phase must be one of {PHASES}, got {self.phase!r}")
        if not (1 <= len(self.plan_sentences) <= 8):
            raise ValueError(
                f"CoachOutput.plan_sentences must contain 1–8 sentences, "
                f"got {len(self.plan_sentences)}"
            )
