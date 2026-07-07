"""
coach/explainer.py
==================
Assembles CoachOutput by explaining WHY Stockfish recommends its PV line,
enriched with the strategic theme derived from the current game's history.

Two layers of analysis are combined:
  1. Extractor signal deltas (before/after the PV line) — shows WHAT the engine
     is doing in positional terms; bypasses SF16 NNUE no-classical-table issue.
  2. Game theme (from 6 extractors over the last 20 moves) — explains the
     strategic context of the current game.
"""
from __future__ import annotations

from chess_coach.core.data_types       import CoachOutput, GMPrecedent
from chess_coach.core.stockfish_bridge import PVExplanation


_TERM_LABELS: dict[str, str] = {
    'king_safety':  'King safety',
    'mobility':     'Mobility',
    'space':        'Space',
    'threats':      'Threats',
    'passed_pawns': 'Passed pawns',
    'material':     'Material',
    'pawns':        'Pawns',
}

_STRATEGY_VERBS: dict[str, str] = {
    'blitz':    'launch a king attack',
    'flank':    'gain positional space',
    'fortress': 'fortify and consolidate',
    'feint':    'execute a positional feint',
}

_STRATEGY_DESCRIPTIONS: dict[str, str] = {
    'blitz':    'kingside attack',
    'flank':    'flank expansion',
    'fortress': 'defensive consolidation',
    'feint':    'positional feint',
}

_PHASE_ADVICE: dict[str, str] = {
    'opening':    'In the opening, keep pieces active and the king safe.',
    'middlegame': 'In the middlegame, convert the positional advantage concretely.',
    'endgame':    'In the endgame, technical precision decides.',
}

# Terms shown in metrics table (fallback order when using EvalBreakdown)
_TABLE_TERMS: list[tuple[str, str]] = [
    ('King safety',  'king_safety'),
    ('Mobility',     'mobility'),
    ('Space',        'space'),
    ('Threats',      'threats'),
    ('Passed pawns', 'passed_pawns'),
    ('Material',     'material'),
    ('Pawns',        'pawns'),
]


def assemble(
    pv_exp:         PVExplanation,
    phase:          str,
    gm_precedents:  list[GMPrecedent],
    player_side:    str = 'white',
    game_theme:     dict | None = None,
    score_cp:       int | None = None,
    pv_signal_rows: list[tuple] | None = None,
    pv_uci:         list[str] | None = None,
    pv_line_text:   str = '',
) -> CoachOutput:
    """
    Build a CoachOutput explaining Stockfish's recommended line.

    Parameters
    ----------
    pv_exp         : Stockfish PV explanation (eval term breakdown, strategy)
    phase          : current game phase
    gm_precedents  : GM database matches
    player_side    : 'white' | 'black'
    game_theme     : output of StrategyEngine._get_game_theme(), or None
    score_cp       : Stockfish centipawn score from the engine panel (optional)
    pv_signal_rows : extractor-based (label, before, after, delta) rows;
                     used for the metrics table in place of EvalBreakdown when
                     SF16 NNUE mode returns no classical breakdown.
    """
    flip           = 1.0 if player_side == 'white' else -1.0
    headline       = _build_headline(pv_exp, game_theme, score_cp, pv_signal_rows)
    plan_sentences = _build_plan(pv_exp, phase, player_side, game_theme, score_cp, pv_signal_rows)
    tactic_hints   = _build_tactic_hints(pv_exp, score_cp, pv_signal_rows)
    metrics_table  = _build_metrics_table(pv_exp, flip, pv_signal_rows)

    if game_theme:
        strategy_primary   = game_theme['primary']
        secondary_from_gt  = game_theme.get('secondary')
        line_strategy      = pv_exp.strategy
        strategy_secondary = line_strategy if line_strategy != strategy_primary else secondary_from_gt
        confidence = 0.6 * game_theme['confidence'] + 0.4 * pv_exp.confidence
    else:
        strategy_primary   = pv_exp.strategy
        strategy_secondary = None
        confidence         = pv_exp.confidence

    confidence = max(0.0, min(1.0, confidence))

    return CoachOutput(
        strategy_primary   = strategy_primary,
        strategy_secondary = strategy_secondary,
        confidence         = confidence,
        phase              = phase,
        headline           = headline,
        plan_sentences     = plan_sentences,
        tactic_hints       = tactic_hints,
        move_flags         = [],
        weakness_squares   = [],
        gm_precedents      = gm_precedents,
        signal_dump        = [],
        metrics_table      = metrics_table,
        pv_san             = list(pv_exp.pv_san) if pv_exp.pv_san else [],
        pv_uci             = list(pv_uci) if pv_uci else [],
        pv_line_text       = pv_line_text,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mat_delta(pv_signal_rows: list[tuple] | None) -> float | None:
    """Return the Material delta from pv_signal_rows, or None if unavailable."""
    if not pv_signal_rows:
        return None
    for label, _b, _a, delta in pv_signal_rows:
        if label == 'Material':
            return delta
    return None


def _dominant_signal(pv_signal_rows: list[tuple] | None) -> tuple[str, float]:
    """Return (label, delta) for the term with the largest |delta|, or ('', 0.0)."""
    if not pv_signal_rows:
        return ('', 0.0)
    best = max(pv_signal_rows, key=lambda r: abs(r[3]))
    return best[0], best[3]


# ── Builders ──────────────────────────────────────────────────────────────────

def _build_headline(
    pv_exp:         PVExplanation,
    game_theme:     dict | None,
    score_cp:       int | None = None,
    pv_signal_rows: list[tuple] | None = None,
) -> str:
    first_move = pv_exp.pv_san[0] if pv_exp.pv_san else "the recommended line"
    score_str  = f"{score_cp / 100.0:+.2f}" if score_cp is not None else None

    def _theme_suffix() -> str:
        if not game_theme:
            return ''
        desc = _STRATEGY_DESCRIPTIONS.get(game_theme['primary'], game_theme['primary'])
        return f"  ·  Game: {desc} ({game_theme['confidence']:.0%})"

    # Material capture — detected from real piece-count delta
    mat = _mat_delta(pv_signal_rows)
    if mat is not None and mat >= 2.5:
        if mat >= 8.5:
            desc = "captures the queen"
        elif mat >= 4.5:
            desc = "wins significant material"
        else:
            desc = "wins material"
        score_part = f" ({score_str})" if score_str else ""
        return f"Stockfish plays {first_move} — {desc}{score_part}" + _theme_suffix()

    # Tactical (engine-flagged forcing sequence)
    if pv_exp.is_tactical:
        score_part = f" ({score_str})" if score_str else ""
        base = f"Stockfish plays {first_move} — forcing sequence{score_part}."
        if game_theme:
            base += f"  Game theme: {game_theme['primary']} ({game_theme['confidence']:.0%})."
        return base

    # Positional — prefer signal-row dominant term over EvalBreakdown (which is
    # zero in SF16 NNUE mode)
    dom_label, dom_delta = _dominant_signal(pv_signal_rows)
    if dom_label and abs(dom_delta) >= 0.03:
        sign  = '+' if dom_delta >= 0 else ''
        parts = [f"Stockfish plays {first_move} — {dom_label.lower()} {sign}{dom_delta:.2f}"]
        if score_str:
            parts[0] += f" ({score_str})"
        if game_theme:
            desc = _STRATEGY_DESCRIPTIONS.get(game_theme['primary'], game_theme['primary'])
            parts.append(f"Game: {desc} ({game_theme['confidence']:.0%})")
        return '  ·  '.join(parts)

    # Fallback to EvalBreakdown dominant term
    term  = _TERM_LABELS.get(pv_exp.dominant_term,
                              pv_exp.dominant_term.replace('_', ' ').title())
    d     = pv_exp.dominant_delta
    sign  = '+' if d >= 0 else ''
    parts = [f"Stockfish plays {first_move} — {term.lower()} {sign}{d:.2f}"]
    if score_str:
        parts[0] += f" ({score_str})"
    if game_theme:
        desc = _STRATEGY_DESCRIPTIONS.get(game_theme['primary'], game_theme['primary'])
        parts.append(f"Game: {desc} ({game_theme['confidence']:.0%})")
    return '  ·  '.join(parts)


def _build_plan(
    pv_exp:         PVExplanation,
    phase:          str,
    player_side:    str,
    game_theme:     dict | None,
    score_cp:       int | None = None,
    pv_signal_rows: list[tuple] | None = None,
) -> list[str]:
    sentences: list[str] = []

    mat = _mat_delta(pv_signal_rows)
    score_pawns = score_cp / 100.0 if score_cp is not None else None

    # ── Material win branch ──────────────────────────────────────────────────
    if mat is not None and mat >= 2.5:
        first_move = pv_exp.pv_san[0] if pv_exp.pv_san else "this move"
        if mat >= 8.5:
            piece = "queen"
        elif mat >= 4.5:
            piece = "rook or multiple pieces"
        else:
            piece = "piece"
        score_note = f" (evaluation: {score_pawns:+.2f} pawns)" if score_pawns is not None else ""
        sentences.append(
            f"{first_move} captures the {piece}, winning {mat:+.2f} material{score_note}."
        )
        if game_theme:
            n    = game_theme.get('n_positions', 0)
            gt   = game_theme['primary']
            desc = _STRATEGY_DESCRIPTIONS.get(gt, gt)
            sentences.append(
                f"Your game (last {n} positions) shows a {desc} theme — "
                "the material win arises from this positional pressure."
            )
        # Mention any other significant changes
        if pv_signal_rows:
            notable = [(l, d) for l, _b, _a, d in pv_signal_rows
                       if l != 'Material' and abs(d) > 0.05]
            notable.sort(key=lambda kv: abs(kv[1]), reverse=True)
            if notable:
                parts = [f"{l} {'+' if d >= 0 else ''}{d:.2f}" for l, d in notable[:2]]
                sentences.append("Secondary effects: " + ",  ".join(parts) + ".")
        sentences.append(_PHASE_ADVICE.get(phase, "Follow the engine line carefully."))
        return sentences[:8]

    # ── Tactical branch ──────────────────────────────────────────────────────
    if pv_exp.is_tactical:
        sentences.append("Stockfish sees a forcing sequence — follow the moves precisely.")
        if pv_exp.tactic_move_idx > 0:
            idx = pv_exp.tactic_move_idx - 1
            if 0 <= idx < len(pv_exp.pv_san):
                sentences.append(
                    f"Critical moment: {pv_exp.pv_san[idx]} — "
                    "this move changes the evaluation sharply."
                )
        if game_theme:
            n   = game_theme.get('n_positions', 0)
            gt  = game_theme['primary']
            desc = _STRATEGY_DESCRIPTIONS.get(gt, gt)
            sentences.append(
                f"Your game (last {n} positions) shows a {desc} pattern — "
                "tactics arise from this accumulated pressure."
            )
        sentences.append(_PHASE_ADVICE.get(phase, "Follow the engine line carefully."))
        return sentences[:6]

    # ── Positional branch ────────────────────────────────────────────────────

    # 1. Game theme sentence
    if game_theme:
        n     = game_theme.get('n_positions', 0)
        gt    = game_theme['primary']
        conf  = game_theme['confidence']
        desc  = _STRATEGY_DESCRIPTIONS.get(gt, gt)
        verb  = _STRATEGY_VERBS.get(gt, 'play this position')
        sentences.append(
            f"Your game (last {n} positions) shows a {desc} theme "
            f"({conf:.0%} confidence) — Stockfish continues to {verb}."
        )

    # 2. What the PV line achieves — use signal rows when available
    verb    = _STRATEGY_VERBS.get(pv_exp.strategy, 'improve the position')
    n_moves = len(pv_exp.pv_san)
    if pv_signal_rows:
        summary = _summarise_signal_rows(pv_signal_rows)
    else:
        summary = _summarise_deltas(pv_exp.deltas)
    sentences.append(
        f"This {n_moves}-move line aims to {verb}: {summary}."
    )

    # 3. Top 2 significant signal-row changes (beyond dominant)
    if pv_signal_rows:
        dom_label, _ = _dominant_signal(pv_signal_rows)
        notable = [(l, d) for l, _b, _a, d in pv_signal_rows
                   if l != dom_label and abs(d) > 0.05]
        notable.sort(key=lambda kv: abs(kv[1]), reverse=True)
        if notable:
            parts = [f"{l} {'+' if d >= 0 else ''}{d:.2f}" for l, d in notable[:2]]
            sentences.append(
                "Secondary changes: " + ",  ".join(parts)
                + " — see the metrics table for the full breakdown."
            )
    else:
        notable = [
            (k, v) for k, v in pv_exp.deltas.items()
            if abs(v) > 0.05 and k != pv_exp.dominant_term
        ]
        notable.sort(key=lambda kv: abs(kv[1]), reverse=True)
        if notable:
            parts = []
            for key, val in notable[:2]:
                label = _TERM_LABELS.get(key, key.replace('_', ' ').title())
                sign  = '+' if val >= 0 else ''
                parts.append(f"{label} {sign}{val:.2f}")
            sentences.append(
                "Secondary changes: " + ",  ".join(parts)
                + " — see the metrics table for the full breakdown."
            )

    # 4. Phase advice
    sentences.append(_PHASE_ADVICE.get(phase, "Follow the engine line carefully."))

    if len(sentences) < 2:
        sentences.append("Consult the signal panel for detailed readings.")
    return sentences[:8]


def _summarise_signal_rows(rows: list[tuple]) -> str:
    """One-phrase summary of the two largest delta terms from signal rows."""
    ranked = sorted(rows, key=lambda r: abs(r[3]), reverse=True)
    parts = []
    for label, _b, _a, delta in ranked[:2]:
        if abs(delta) < 0.02:
            break
        sign = '+' if delta >= 0 else ''
        parts.append(f"{label.lower()} {sign}{delta:.2f}")
    return ", ".join(parts) if parts else "small positional improvements"


def _summarise_deltas(deltas: dict[str, float]) -> str:
    """One-phrase summary of the two largest delta terms from EvalBreakdown."""
    ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
    parts = []
    for key, val in ranked[:2]:
        if abs(val) < 0.02:
            break
        label = _TERM_LABELS.get(key, key.replace('_', ' ').title())
        sign  = '+' if val >= 0 else ''
        parts.append(f"{label.lower()} {sign}{val:.2f}")
    return ", ".join(parts) if parts else "small positional improvements"


def _build_tactic_hints(
    pv_exp:         PVExplanation,
    score_cp:       int | None = None,
    pv_signal_rows: list[tuple] | None = None,
) -> list[str]:
    hints: list[str] = []
    if pv_exp.is_tactical and pv_exp.tactic_move_idx > 0:
        idx = pv_exp.tactic_move_idx - 1
        if 0 <= idx < len(pv_exp.pv_san):
            hints.append(
                f"Critical move: {pv_exp.pv_san[idx]} — "
                "changes the evaluation sharply."
            )
    # Material capture hint
    mat = _mat_delta(pv_signal_rows)
    if mat is not None and mat >= 2.5:
        first_move = pv_exp.pv_san[0] if pv_exp.pv_san else "this move"
        hints.append(
            f"{first_move} wins {mat:+.2f} material — the most forcing continuation."
        )
    elif abs(pv_exp.dominant_delta) > 1.0 and pv_exp.dominant_term == 'king_safety':
        side_label = 'defence' if pv_exp.dominant_delta > 0 else 'attack'
        hints.append(f"King {side_label} is decisive — the king position dominates this line.")
    return hints[:3]


def _build_metrics_table(
    pv_exp:         PVExplanation,
    flip:           float,
    pv_signal_rows: list[tuple] | None = None,
) -> list[tuple[str, float, float, float]]:
    """
    Build the per-term breakdown table.

    Prefers extractor-based pv_signal_rows (which work with SF16 NNUE mode).
    Falls back to EvalBreakdown fields when signal rows are unavailable.
    """
    if pv_signal_rows:
        # Signal rows are already from the player's perspective
        return [(label, before, after, delta)
                for label, before, after, delta in pv_signal_rows]

    # Fallback: EvalBreakdown (zero in SF16 NNUE, but best we can do without PV)
    rows: list[tuple[str, float, float, float]] = []
    for label, key in _TABLE_TERMS:
        before = getattr(pv_exp.eval_before, key, 0.0) * flip
        after  = getattr(pv_exp.eval_after,  key, 0.0) * flip
        delta  = pv_exp.deltas.get(key, (after - before))
        rows.append((label, before, after, delta))
    return rows
