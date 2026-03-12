"""
coach/plan_recommender.py
==========================
Generates move_flags and weakness_squares for the board heatmap.

Two complementary sources — both run when available, merged at the end:

Structural flags  (always available)
    Derived from MetricSignal.key_squares and MetricSignal.metric_name.
    Every legal move that lands on, vacates, or attacks a key square
    receives a flag matching the strategy that fired it.

Engine flags  (when Stockfish is available)
    Any legal move scored within _ENGINE_THRESHOLD centipawns of the
    best move is flagged as 'engine_best' or 'engine_good'.

Output format
    move_flags      list[dict]  — {move: uci, flag: str, strategy: str}
    weakness_squares list[str]  — unique algebraic squares, opponent side
"""
from __future__ import annotations

import chess
from typing import Optional

from core.data_types import MetricSignal

_ENGINE_THRESHOLD_BEST = 10    # cp — within this = 'engine_best'
_ENGINE_THRESHOLD_GOOD = 50    # cp — within this = 'engine_good'
_MAX_ENGINE_MOVES      = 5     # top N moves to evaluate
_MIN_SIGNAL_SCORE      = 0.35  # signals below this don't generate flags

# Metric → flag tag mappings
_METRIC_FLAGS: dict[str, str] = {
    'king_exposure':         'attack_target',
    'space_delta_kingside':  'kingside_break',
    'space_delta_queenside': 'queenside_break',
    'outpost_occupation':    'outpost_target',
    'passed_pawn':           'pawn_advance',
    'weak_pawns':            'weak_pawn_target',
    'overextension':         'overextension_target',
    'sacrifice_delta':       'sacrifice_square',
    'tactic_pin':            'pin_square',
    'tactic_fork':           'fork_square',
    'tactic_skewer':         'skewer_square',
    'tactic_discovery':      'discovery_square',
}


# ── Public API ────────────────────────────────────────────────────────────────

def recommend(
    board: chess.Board,
    signals: list[MetricSignal],
    player_side: str,
    strategy: str,
    stockfish_bridge=None,
) -> tuple[list[dict], list[str]]:
    """
    Generate move_flags and weakness_squares.

    Parameters
    ----------
    board : chess.Board
        Current position.
    signals : list[MetricSignal]
        Phase-filtered signals from all extractors.
    player_side : str
        'white' | 'black' — the side being coached.
    strategy : str
        Primary strategy from the conflict resolver.
    stockfish_bridge : StockfishBridge | None
        Live engine — used for engine flags if available.

    Returns
    -------
    (move_flags, weakness_squares)
    """
    move_flags        = _structural_flags(board, signals, player_side, strategy)
    weakness_squares  = _weakness_squares(signals, player_side)

    if stockfish_bridge is not None:
        engine_flags = _engine_flags(board, stockfish_bridge, strategy)
        # Merge: structural flags take priority, engine fills gaps
        flagged_moves = {f['move'] for f in move_flags}
        for ef in engine_flags:
            if ef['move'] not in flagged_moves:
                move_flags.append(ef)

    return move_flags, weakness_squares


# ── Structural flags ──────────────────────────────────────────────────────────

def _structural_flags(
    board: chess.Board,
    signals: list[MetricSignal],
    player_side: str,
    strategy: str,
) -> list[dict]:
    """Flag legal moves that interact with signal key_squares."""
    flags: list[dict] = []
    seen: set[str]    = set()

    # Collect key squares from strong-enough signals
    target_squares: dict[str, tuple[str, str]] = {}  # sq → (flag, strategy)
    opp = 'black' if player_side == 'white' else 'white'

    for sig in sorted(signals, key=lambda s: s.score, reverse=True):
        if sig.score < _MIN_SIGNAL_SCORE:
            continue
        flag_tag = _METRIC_FLAGS.get(sig.metric_name)
        if not flag_tag:
            continue
        # Own signals → plan targets; opponent signals → attack squares
        relevant = (sig.side == player_side or sig.side == opp)
        if not relevant:
            continue
        for sq in sig.key_squares:
            if sq not in target_squares:
                target_squares[sq] = (flag_tag, strategy)

    # Walk all legal moves and flag those that touch a key square
    for move in board.legal_moves:
        uci      = move.uci()
        to_sq    = chess.square_name(move.to_square)
        from_sq  = chess.square_name(move.from_square)

        hit = target_squares.get(to_sq) or target_squares.get(from_sq)
        if hit and uci not in seen:
            flag_tag, strat = hit
            flags.append({'move': uci, 'flag': flag_tag, 'strategy': strat})
            seen.add(uci)

    return flags


# ── Engine flags ──────────────────────────────────────────────────────────────

def _engine_flags(
    board: chess.Board,
    bridge,
    strategy: str,
) -> list[dict]:
    """
    Ask Stockfish for the top N moves and flag them by quality.
    Requires bridge.get_top_moves(fen, n) → list[{move: uci, score: cp}]
    Falls back gracefully if the bridge doesn't support that method.
    """
    flags: list[dict] = []
    try:
        top = bridge.get_top_moves(board.fen(), _MAX_ENGINE_MOVES)
    except (AttributeError, Exception):
        # Bridge may not have get_top_moves — degrade gracefully
        try:
            ev = bridge.get_eval(board.fen())
            best_move = ev.best_move if hasattr(ev, 'best_move') else None
            if best_move:
                flags.append({
                    'move': best_move, 'flag': 'engine_best', 'strategy': strategy
                })
        except Exception:
            pass
        return flags

    if not top:
        return flags

    best_score = top[0].get('score', 0) if top else 0

    for entry in top:
        uci   = entry.get('move', '')
        score = entry.get('score', 0)
        if not uci:
            continue
        diff = abs(score - best_score)
        if diff <= _ENGINE_THRESHOLD_BEST:
            flag = 'engine_best'
        elif diff <= _ENGINE_THRESHOLD_GOOD:
            flag = 'engine_good'
        else:
            continue
        flags.append({'move': uci, 'flag': flag, 'strategy': strategy})

    return flags


# ── Weakness squares ──────────────────────────────────────────────────────────

def _weakness_squares(
    signals: list[MetricSignal],
    player_side: str,
) -> list[str]:
    """
    Collect squares from opponent-side signals above threshold.
    Ordered by signal score descending, deduplicated, max 8.
    """
    opp = 'black' if player_side == 'white' else 'white'
    seen:    set[str]  = set()
    squares: list[str] = []

    for sig in sorted(signals, key=lambda s: s.score, reverse=True):
        if sig.side != opp or sig.score < 0.40:
            continue
        for sq in sig.key_squares:
            if sq not in seen:
                seen.add(sq)
                squares.append(sq)
            if len(squares) >= 8:
                return squares

    return squares
