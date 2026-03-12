"""
core/board_utils.py
===================
Pure python-chess helper functions used across all extractor modules.

Design rules:
  - No Stockfish calls. No Qt. No I/O.
  - Every function takes a chess.Board (or chess.pgn.Game) and returns
    a plain Python value.
  - These are stateless utilities — no class, no instance.

Phase detection
---------------
Phase is computed from material count, not move number. Move number
alone is unreliable (e.g. a Berlin endgame is 'endgame' by move 15;
a Catalan middlegame may persist to move 35).

The formula weights queens, rooks, and piece development to give a
smooth three-phase classification that matches human intuition.
"""

from __future__ import annotations

import chess
import chess.pgn


# ── Phase detection ───────────────────────────────────────────────────────────

# Material thresholds for phase classification.
# "Material score" = sum of piece values remaining (pawns excluded — they
# are structural, not material in the phase sense).
# Values: Q=9, R=5, B=3, N=3 per side.
# Max material per side (no pawns): 9 + 10 + 6 + 6 = 31 points.
# Max total: 62 points.

_OPENING_THRESHOLD    = 50   # Both sides mostly intact → opening
_MIDDLEGAME_THRESHOLD = 28   # Significant exchanges made → middlegame
                              # Below 28 → endgame

_PIECE_VALUES = {
    chess.QUEEN:  9,
    chess.ROOK:   5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
}


def _material_score(board: chess.Board) -> int:
    """
    Sum of piece values for both sides, excluding pawns and kings.
    Used exclusively for phase detection.
    """
    total = 0
    for piece_type, value in _PIECE_VALUES.items():
        total += len(board.pieces(piece_type, chess.WHITE)) * value
        total += len(board.pieces(piece_type, chess.BLACK)) * value
    return total


def get_phase(board: chess.Board) -> str:
    """
    Classify the current game phase based on material count.

    Returns
    -------
    'opening'     — Both sides largely intact; queens almost certainly on board.
    'middlegame'  — Significant exchanges have occurred but endgame not yet.
    'endgame'     — Major material reduction; king becomes active.

    Examples
    --------
    Starting position         → 'opening'   (material = 62)
    After typical development  → 'opening'   (material ≈ 56–62)
    Active middlegame          → 'middlegame' (material ≈ 28–49)
    Rook endgame               → 'endgame'   (material ≤ 27)

    Test targets from spec:
    ply 5  (early development, ~62 material) → 'opening'
    ply 20 (active middlegame, ~40 material) → 'middlegame'
    ply 42 (late game, ~20 material)         → 'endgame'
    """
    score = _material_score(board)

    if score >= _OPENING_THRESHOLD:
        return 'opening'
    elif score >= _MIDDLEGAME_THRESHOLD:
        return 'middlegame'
    else:
        return 'endgame'


# ── FEN helpers ───────────────────────────────────────────────────────────────

def get_fen(board: chess.Board) -> str:
    """
    Return the full FEN string for the current board position.

    Thin wrapper for clarity — callers import from here so the
    FEN interface is consistent across the codebase.
    """
    return board.fen()


def get_position_key(board: chess.Board) -> str:
    """
    Return a transposition-safe position key (first 4 FEN fields only).

    Excludes halfmove clock and fullmove number so positions reached
    by different move orders hash to the same key. Consistent with
    the MoveTree implementation in pgn/move_tree.py.
    """
    return ' '.join(board.fen().split()[:4])


def get_pawn_hash(board: chess.Board) -> str:
    """
    Return a hash string for the current pawn structure only.

    Used by database/pattern_matcher.py to find GM games with similar
    pawn structures regardless of piece placement.

    Implementation: Zobrist-style — XOR the bitboard positions of all
    pawns. Represented as a hex string for SQLite storage.
    """
    white_pawns = int(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = int(board.pieces(chess.PAWN, chess.BLACK))
    # Simple but effective: XOR combined with a rotation to distinguish sides
    pawn_hash = white_pawns ^ (black_pawns << 32 | black_pawns >> 32)
    return format(pawn_hash & 0xFFFFFFFFFFFFFFFF, '016x')


# ── Move history ──────────────────────────────────────────────────────────────

def get_move_history(game: chess.pgn.Game) -> list[str]:
    """
    Return the full mainline move history of a PGN game as UCI strings.

    Parameters
    ----------
    game : chess.pgn.Game
        Root node of a loaded PGN game.

    Returns
    -------
    List of UCI move strings from move 1 to the final move.
    Example: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5']
    """
    moves: list[str] = []
    node = game
    while node.variations:
        next_node = node.variations[0]  # mainline only
        if next_node.move is not None:
            moves.append(next_node.move.uci())
        node = next_node
    return moves


def get_move_history_to_node(game: chess.pgn.Game, target_ply: int) -> list[str]:
    """
    Return move history up to (and including) target_ply.

    Parameters
    ----------
    target_ply : int
        Half-move number to stop at (1-based).
    """
    return get_move_history(game)[:target_ply]


# ── Square helpers ────────────────────────────────────────────────────────────

def square_to_str(sq: chess.Square) -> str:
    """
    Convert a python-chess Square integer to an algebraic name string.

    Parameters
    ----------
    sq : chess.Square
        An integer in range 0..63 (chess.A1=0 … chess.H8=63).

    Returns
    -------
    str — e.g. 'e4', 'g7', 'h1'
    """
    return chess.square_name(sq)


def str_to_square(name: str) -> chess.Square:
    """
    Convert an algebraic square name to a python-chess Square integer.

    Raises ValueError on invalid input.
    """
    return chess.parse_square(name)


# ── King zone ─────────────────────────────────────────────────────────────────

def get_king_zone(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    """
    Return the list of squares comprising a king's zone.

    The king zone is the 7 squares surrounding the king
    (including the king's own square) that define the attack
    target area used by the Blitz detector.

    Parameters
    ----------
    board : chess.Board
    color : chess.WHITE | chess.BLACK

    Returns
    -------
    List of up to 8 chess.Square integers (fewer for corner/edge kings).

    Example
    -------
    King on g1 → zone = [f1, g1, h1, f2, g2, h2]
    """
    king_sq = board.king(color)
    if king_sq is None:
        return []  # Shouldn't happen in a legal position

    zone: list[chess.Square] = []
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f = king_file + df
            r = king_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                zone.append(chess.square(f, r))

    return zone


def get_king_zone_str(board: chess.Board, color: chess.Color) -> list[str]:
    """
    Same as get_king_zone() but returns algebraic square names.
    Convenience wrapper for use in MetricSignal.key_squares.
    """
    return [square_to_str(sq) for sq in get_king_zone(board, color)]


# ── Piece helpers ─────────────────────────────────────────────────────────────

def get_pieces_in_zone(
    board: chess.Board,
    color: chess.Color,
    zone: list[chess.Square],
) -> list[str]:
    """
    Return piece descriptors for all pieces of `color` that attack any
    square in `zone`.

    Used by the Blitz detector's piece_convergence metric to find how
    many of your pieces bear on the opponent's king zone.

    Returns
    -------
    List of strings in the format '<Piece><Square>' e.g. ['Ng5', 'Qd3']
    """
    attacking: list[str] = []
    for sq in zone:
        for attacker_sq in board.attackers(color, sq):
            piece = board.piece_at(attacker_sq)
            if piece is not None:
                descriptor = piece.symbol().upper() + chess.square_name(attacker_sq)
                if descriptor not in attacking:
                    attacking.append(descriptor)
    return attacking


def count_legal_moves(board: chess.Board, color: chess.Color) -> int:
    """
    Count the number of legal moves available for `color`.

    Used by piece_mobility_ratio. Note: temporarily pushes a null move
    if it's not `color`'s turn, to get the legal move count for the
    other side.

    If it's already `color`'s turn, returns board.legal_moves.count().
    Otherwise uses a null-move approach (not always valid — only call
    when the position is not in check for the side to move).
    """
    if board.turn == color:
        return board.legal_moves.count()

    # Estimate legal moves for the non-moving side
    # by creating a copy and flipping turn
    b = board.copy()
    b.turn = color
    # Clear en passant to avoid illegal null-move state
    b.ep_square = None
    return b.legal_moves.count()
