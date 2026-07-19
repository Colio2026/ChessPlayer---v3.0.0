"""
ml/concept_signal_adapter.py
============================
Converts (concept_name, probability) output from ChessConceptClassifier
into MetricSignal objects ready for the narrator.

Tier classification
-------------------
Tier 1 — Grand Strategies   (can be strategy_primary)
Tier 2 — Positional Diagnostics
Tier 3 — Tactics
Tier 4 — Endgame type
"""
from __future__ import annotations

import chess

from chess_coach.core.data_types import MetricSignal

# ── Tier classification ───────────────────────────────────────────────────────

TIER1_CONCEPTS: frozenset[str] = frozenset({
    'mating_attack', 'passed_pawn', 'outpost', 'space_advantage',
    'pawn_storm', 'pawn_majority', 'blockade', 'prophylaxis',
    'initiative', 'development_lead', 'piece_activity', 'king_activity',
})

TIER2_CONCEPTS: frozenset[str] = frozenset({
    'king_safety', 'weak_square', 'open_file', 'isolated_pawn',
    'doubled_pawn', 'backward_pawn', 'pawn_chain', 'pawn_island',
    'rook_seventh', 'battery', 'bishop_pair', 'good_bishop', 'bad_bishop',
    'trapped_piece', 'promotion',
})

TIER3_CONCEPTS: frozenset[str] = frozenset({
    'pin', 'fork', 'skewer', 'x_ray', 'discovery', 'double_check',
    'clearance', 'deflection', 'overloading', 'zwischenzug', 'interference',
    'back_rank', 'sacrifice',
})

TIER4_CONCEPTS: frozenset[str] = frozenset({
    'rook_endgame', 'pawn_endgame', 'bishop_endgame', 'knight_endgame',
    'queen_endgame', 'drawn_position', 'shouldering', 'opposition', 'zugzwang',
})

# Tier 1 priority order for tie-breaking (index 0 = highest priority)
TIER1_PRIORITY: list[str] = [
    'mating_attack', 'passed_pawn', 'outpost', 'space_advantage',
    'pawn_storm', 'pawn_majority', 'blockade', 'prophylaxis',
    'initiative', 'development_lead', 'piece_activity', 'king_activity',
]

# ── Severity thresholds ───────────────────────────────────────────────────────

def _severity(prob: float) -> str:
    if prob >= 0.85:
        return 'critical'
    if prob >= 0.70:
        return 'high'
    if prob >= 0.55:
        return 'moderate'
    return 'mild'


# ── Action hints (shown when phrase DB has no coverage for a concept) ─────────

_ACTION_HINTS: dict[str, str] = {
    # Tier 1
    'mating_attack':    'Coordinate your pieces toward the enemy king — seek forcing checks and mating patterns.',
    'passed_pawn':      'Advance the passed pawn with piece support; clear obstacles before the opponent organises a blockade.',
    'outpost':          'Install a knight or bishop on the outpost — no enemy pawn can drive it away.',
    'space_advantage':  'Exploit your spatial edge by restricting enemy piece movement and preparing a pawn break.',
    'pawn_storm':       'Advance the pawn majority toward the enemy king flank to crack open attacking lines.',
    'pawn_majority':    'Convert the majority into a passed pawn through well-timed exchanges on the correct file.',
    'blockade':         'Fix the enemy pawn and plant a piece directly in front of it — the pawn becomes a permanent liability.',
    'prophylaxis':      'Prevent the opponent\'s key plan before it gets started — think what they want to do, then stop it.',
    'initiative':       'Keep making threats to deny the opponent time to organise — each move must force a response.',
    'development_lead': 'Cash in the development advantage now — open the position while the opponent is still uncoordinated.',
    'piece_activity':   'Improve the least active piece — find it a square where it controls maximum space.',
    'king_activity':    'Centralise the king at once — in the endgame an active king is a decisive weapon.',
    # Tier 2
    'king_safety':      'Address the king\'s exposure — shore up the pawn shelter or seek active counterplay.',
    'weak_square':      'Target the weak square complex — install a piece there that cannot be challenged by a pawn.',
    'open_file':        'Seize the open file with a rook and press for penetration.',
    'isolated_pawn':    'Blockade the isolated pawn and attack it with rooks — it cannot defend itself.',
    'doubled_pawn':     'Target the doubled pawns — they tie a rook to passive defence and limit pawn breaks.',
    'backward_pawn':    'Pressurise the backward pawn on the half-open file; it can never advance without cost.',
    'pawn_chain':       'Attack the base of the pawn chain — the whole structure collapses if the base falls.',
    'pawn_island':      'Fewer pawn islands mean fewer weaknesses — trade toward a simplified pawn structure.',
    'rook_seventh':     'The rook on the seventh rank harvests pawns and confines the enemy king.',
    'battery':          'Maintain the battery alignment — coordinated rooks or queen-bishop deliver overwhelming pressure.',
    'bishop_pair':      'Open the position to unleash the bishop pair\'s long-range dominance.',
    'good_bishop':      'Manoeuvre to keep enemy pawns on squares your bishop controls.',
    'bad_bishop':       'Trade the bad bishop or restructure pawns to free it.',
    'trapped_piece':    'Trap the piece immediately before it finds an escape route.',
    'promotion':        'Clear the queening path now — every tempo on a pawn this advanced is critical.',
    # Tier 3
    'pin':              'Exploit the pin — pile on the pinned piece or use it as a hook for a combination.',
    'fork':             'Strike with the fork — both targets cannot be saved simultaneously.',
    'skewer':           'Execute the skewer — drive the valuable piece off the line to win what stands behind it.',
    'x_ray':            'Use the x-ray — your piece exerts pressure through the opponent\'s piece to a target behind.',
    'discovery':        'Unleash the discovered attack — the piece moved opens fire from the piece behind.',
    'double_check':     'The double check forces the king to move — use this to re-direct the attack decisively.',
    'clearance':        'Clear the critical square or line so the key piece can occupy it.',
    'deflection':       'Deflect the defender away from its duty — the protected square then falls.',
    'overloading':      'Overload the defender — it cannot guard two things at once.',
    'zwischenzug':      'Insert the in-between move — do not recapture immediately when a zwischenzug wins.',
    'interference':     'Interpose a piece to cut the communication between the defender and what it defends.',
    'back_rank':        'Exploit the back-rank weakness — the king has no escape square.',
    'sacrifice':        'The sacrifice creates a long-term imbalance — accept the material deficit and play for the compensation.',
    # Tier 4
    'rook_endgame':     'In rook endings, activate the rook behind passed pawns and seek the Lucena or Philidor position.',
    'pawn_endgame':     'Count the tempi carefully — opposition and pawn races decide pawn endgames.',
    'bishop_endgame':   'Place pawns on the opposite colour from your bishop to maximise its scope.',
    'knight_endgame':   'Keep knights on the board when pawns are scattered on both wings.',
    'queen_endgame':    'Perpetual check is always in the air — centralise your queen and watch for back-rank tricks.',
    'drawn_position':   'The position is likely drawn — look for the most active try to create winning chances.',
    'shouldering':      'The king shoulder-check prevents the opponent\'s king from reaching the key square.',
    'opposition':       'Seize the opposition to control the key squares in front of the passed pawn.',
    'zugzwang':         'You are in zugzwang — any move worsens your position. Find the least evil.',
}


# ── Public API ────────────────────────────────────────────────────────────────

def adapt(
    concepts: list[tuple[str, float]],
    board:    chess.Board,
    phase:    str,
    side:     str,
) -> list[MetricSignal]:
    """
    Convert classifier output into MetricSignal objects.

    Parameters
    ----------
    concepts : list of (concept_name, probability) above the classifier threshold
    board    : position being analysed (used for key_squares extraction)
    phase    : 'opening' | 'middlegame' | 'endgame'
    side     : 'white' | 'black'  — the side being coached
    """
    signals: list[MetricSignal] = []
    for name, prob in concepts:
        # Tier 3 tactics get a tactic_ prefix so narrator._build_tactic_hints picks them up
        metric_name = f'tactic_{name}' if name in TIER3_CONCEPTS else name
        hint = _ACTION_HINTS.get(name, f'The position features {name.replace("_", " ")}.')
        signals.append(MetricSignal(
            metric_name = metric_name,
            score       = min(prob, 1.0),
            side        = side,
            cause       = name,
            severity    = _severity(prob),
            action_hint = hint,
            phase       = phase,
            key_squares = _extract_key_squares(name, board, side),
            key_pieces  = [],
            fragment    = '',
        ))
    return signals


def infer_strategy(
    concepts: list[tuple[str, float]],
    tie_gap:  float = 0.10,
) -> tuple[str, str | None, float, bool]:
    """
    Derive primary/secondary strategy from classifier output.

    Returns (primary, secondary_or_None, confidence, tie_band).
    """
    tier1 = [(name, prob) for name, prob in concepts if name in TIER1_CONCEPTS]
    if not tier1:
        return 'general', None, 0.5, False

    # Sort by probability, then by priority index for ties
    def _sort_key(pair):
        name, prob = pair
        pri = TIER1_PRIORITY.index(name) if name in TIER1_PRIORITY else 99
        return (-prob, pri)

    tier1.sort(key=_sort_key)
    primary_name, primary_prob = tier1[0]

    secondary_name = None
    tie = False
    if len(tier1) >= 2:
        sec_name, sec_prob = tier1[1]
        if primary_prob - sec_prob <= tie_gap:
            secondary_name = sec_name
            tie = True

    return primary_name, secondary_name, primary_prob, tie


# ── Key square extraction (lightweight; augments signals for board overlay) ───

def _extract_key_squares(
    concept: str,
    board:   chess.Board,
    side:    str,
) -> list[str]:
    """Return up to 3 relevant square names for a given concept."""
    color = chess.WHITE if side == 'white' else chess.BLACK
    try:
        if concept == 'passed_pawn':
            from tools.label_positions import _has_passed_pawn, _north_fill, _south_fill, _bb_east, _bb_west
            wp = int(board.pieces_mask(chess.PAWN, color))
            ep = int(board.pieces_mask(chess.PAWN, not color))
            if color == chess.WHITE:
                not_p = _south_fill(ep | _bb_east(ep) | _bb_west(ep))
            else:
                not_p = _north_fill(ep | _bb_east(ep) | _bb_west(ep))
            passed = wp & ~not_p
            return [chess.square_name(sq) for sq in chess.scan_forward(passed)][:3]

        if concept == 'outpost':
            from tools.label_positions import _outpost_squares_bb
            outposts = _outpost_squares_bb(board, color)
            pieces = (int(board.pieces_mask(chess.KNIGHT, color)) |
                      int(board.pieces_mask(chess.BISHOP, color)))
            occupied = outposts & pieces
            return [chess.square_name(sq) for sq in chess.scan_forward(occupied)][:3]

        if concept == 'king_safety':
            king_sq = board.king(color)
            if king_sq is not None:
                return [chess.square_name(king_sq)]

        if concept in ('pin', 'fork', 'skewer'):
            king_sq = board.king(not color)
            if king_sq is not None:
                return [chess.square_name(king_sq)]
    except Exception:
        pass
    return []
