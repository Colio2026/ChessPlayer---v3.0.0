# board_encoder.py
# Converts a FEN string into a fixed-length float tensor for model input.
#
# Layout (1001 elements total):
#   [0:768]    Piece placement — 12 channels × 64 squares
#              channel = color*6 + piece_type,  index = channel*64 + square
#   [768]      Side to move — 1.0=white, 0.0=black
#   [769:773]  Castling rights — [WK, WQ, BK, BQ], 1.0 if available
#   [773:781]  En-passant file — one-hot over files a–h, all zeros if none
#
#   ── Phase 2 features ───────────────────────────────────────────────────
#   [781:845]  White attack map — 1.0 if white attacks that square
#   [845:909]  Black attack map — 1.0 if black attacks that square
#   [909:941]  White pawn per-file: [has_pawn, passed, isolated, doubled] × 8 files
#   [941:973]  Black pawn per-file: same layout
#   [973:981]  Open files — 1.0 if no pawns of either color
#   [981:984]  White king shelter — 3 squares directly in front of white king
#   [984:987]  Black king shelter — 3 squares directly in front of black king
#   [987:989]  Pawn advancement — normalized for [white, black]
#   [989:995]  White mobility — normalized attack count per piece type [P,N,B,R,Q,K]
#   [995:1001] Black mobility — same

from __future__ import annotations

import numpy as np
import torch
import chess

_COLORS      = [chess.WHITE, chess.BLACK]
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]

_PIECE_OFFSET  = 0
_TURN_OFFSET   = 768
_CASTLE_OFFSET = 769
_EP_OFFSET     = 773
_ATTACK_OFFSET = 781   # white [781:845], black [845:909]
_PAWN_OFFSET   = 909   # per-file flags + shelter + advancement
_MOB_OFFSET    = 989   # mobility per piece type per side

INPUT_SIZE    = 1001
MOVE_SIZE     = 128
ALGO_SIZE     = 59    # concept bottleneck bits from algo detectors (label_positions.py)
COMBINED_SIZE = INPUT_SIZE + MOVE_SIZE + ALGO_SIZE  # 1188

# Max attack squares per piece type (for normalising mobility to [0, 1])
_MOB_MAX = [2, 8, 13, 14, 27, 8]  # P  N  B  R  Q  K


# ── Phase 2 helpers ────────────────────────────────────────────────────────────

def _add_attack_maps(board: chess.Board, arr: np.ndarray) -> None:
    """Fill arr[781:909]: white/black attack maps over all 64 squares."""
    white_bb = chess.BB_EMPTY
    black_bb = chess.BB_EMPTY
    for sq in chess.scan_forward(board.occupied_co[chess.WHITE]):
        white_bb |= board.attacks_mask(sq)
    for sq in chess.scan_forward(board.occupied_co[chess.BLACK]):
        black_bb |= board.attacks_mask(sq)
    for sq in range(64):
        if (white_bb >> sq) & 1:
            arr[_ATTACK_OFFSET + sq] = 1.0
        if (black_bb >> sq) & 1:
            arr[_ATTACK_OFFSET + 64 + sq] = 1.0


def _has_passed_pawn(
    our_ranks: list[int],
    f: int,
    their_files: list[list[int]],
    color: int,
) -> bool:
    """True if any of our_ranks is a passed pawn on file f."""
    if not our_ranks:
        return False
    enemy: list[int] = []
    for af in (f - 1, f, f + 1):
        if 0 <= af <= 7:
            enemy.extend(their_files[af])
    for r in our_ranks:
        if color == chess.WHITE:
            if not any(er > r for er in enemy):
                return True
        else:
            if not any(er < r for er in enemy):
                return True
    return False


def _is_isolated(our_ranks: list[int], f: int, our_files: list[list[int]]) -> bool:
    """True if pawn(s) on file f have no friendly pawns on adjacent files."""
    if not our_ranks:
        return False
    for af in (f - 1, f + 1):
        if 0 <= af <= 7 and our_files[af]:
            return False
    return True


def _add_pawn_structure(board: chess.Board, arr: np.ndarray) -> None:
    """Fill arr[909:989]: pawn structure features."""
    w_files: list[list[int]] = [[] for _ in range(8)]
    b_files: list[list[int]] = [[] for _ in range(8)]
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        w_files[chess.square_file(sq)].append(chess.square_rank(sq))
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        b_files[chess.square_file(sq)].append(chess.square_rank(sq))

    base = _PAWN_OFFSET

    # White per-file: [has_pawn, passed, isolated, doubled] × 8
    for f in range(8):
        arr[base + f]      = 1.0 if w_files[f] else 0.0
        arr[base + 8 + f]  = 1.0 if _has_passed_pawn(w_files[f], f, b_files, chess.WHITE) else 0.0
        arr[base + 16 + f] = 1.0 if _is_isolated(w_files[f], f, w_files) else 0.0
        arr[base + 24 + f] = 1.0 if len(w_files[f]) >= 2 else 0.0
    base += 32

    # Black per-file: same layout
    for f in range(8):
        arr[base + f]      = 1.0 if b_files[f] else 0.0
        arr[base + 8 + f]  = 1.0 if _has_passed_pawn(b_files[f], f, w_files, chess.BLACK) else 0.0
        arr[base + 16 + f] = 1.0 if _is_isolated(b_files[f], f, b_files) else 0.0
        arr[base + 24 + f] = 1.0 if len(b_files[f]) >= 2 else 0.0
    base += 32

    # Open files (no pawns of either color)
    for f in range(8):
        arr[base + f] = 1.0 if not w_files[f] and not b_files[f] else 0.0
    base += 8

    # White king shelter: 3 squares directly in front of white king
    w_king = board.king(chess.WHITE)
    if w_king is not None:
        kf, kr = chess.square_file(w_king), chess.square_rank(w_king)
        if kr < 7:
            for i, sf in enumerate((kf - 1, kf, kf + 1)):
                if 0 <= sf <= 7:
                    p = board.piece_at(chess.square(sf, kr + 1))
                    arr[base + i] = 1.0 if (p and p.piece_type == chess.PAWN and p.color == chess.WHITE) else 0.0
    base += 3

    # Black king shelter: 3 squares directly behind black king
    b_king = board.king(chess.BLACK)
    if b_king is not None:
        kf, kr = chess.square_file(b_king), chess.square_rank(b_king)
        if kr > 0:
            for i, sf in enumerate((kf - 1, kf, kf + 1)):
                if 0 <= sf <= 7:
                    p = board.piece_at(chess.square(sf, kr - 1))
                    arr[base + i] = 1.0 if (p and p.piece_type == chess.PAWN and p.color == chess.BLACK) else 0.0
    base += 3

    # Pawn advancement: 0=all on starting rank, 1=all on 7th rank
    w_all = [r for ranks in w_files for r in ranks]
    b_all = [r for ranks in b_files for r in ranks]
    arr[base]     = sum(w_all) / (len(w_all) * 7) if w_all else 0.0
    arr[base + 1] = sum(7 - r for r in b_all) / (len(b_all) * 7) if b_all else 0.0


def _add_mobility(board: chess.Board, arr: np.ndarray) -> None:
    """Fill arr[989:1001]: normalised attack count per piece type per side."""
    base = _MOB_OFFSET
    for color in _COLORS:
        for i, pt in enumerate(_PIECE_TYPES):
            pieces = list(board.pieces(pt, color))
            if pieces:
                total = sum(len(board.attacks(sq)) for sq in pieces)
                arr[base] = min(total / (len(pieces) * _MOB_MAX[i]), 1.0)
            base += 1


# ── Public API ─────────────────────────────────────────────────────────────────

def fen_to_tensor(fen: str) -> torch.Tensor:
    """Encode a FEN position as a 1001-element float32 tensor."""
    board = chess.Board(fen)
    arr   = np.zeros(INPUT_SIZE, dtype=np.float32)

    # piece placement
    for color_idx, color in enumerate(_COLORS):
        for piece_idx, piece_type in enumerate(_PIECE_TYPES):
            channel = color_idx * 6 + piece_idx
            base    = _PIECE_OFFSET + channel * 64
            for sq in board.pieces(piece_type, color):
                arr[base + sq] = 1.0

    # side to move
    arr[_TURN_OFFSET] = 1.0 if board.turn == chess.WHITE else 0.0

    # castling rights
    arr[_CASTLE_OFFSET + 0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE)  else 0.0
    arr[_CASTLE_OFFSET + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    arr[_CASTLE_OFFSET + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK)  else 0.0
    arr[_CASTLE_OFFSET + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # en passant
    if board.ep_square is not None:
        arr[_EP_OFFSET + chess.square_file(board.ep_square)] = 1.0

    _add_attack_maps(board, arr)
    _add_pawn_structure(board, arr)
    _add_mobility(board, arr)

    return torch.from_numpy(arr)


def move_to_tensor(move_uci: str) -> torch.Tensor:
    """Encode a UCI move string as a 128-element float32 tensor.

    Layout: [0:64] from-square one-hot, [64:128] to-square one-hot.
    Empty string or invalid move → all zeros.
    """
    arr = np.zeros(MOVE_SIZE, dtype=np.float32)
    if move_uci and len(move_uci) >= 4:
        try:
            move = chess.Move.from_uci(move_uci)
            arr[move.from_square]    = 1.0
            arr[64 + move.to_square] = 1.0
        except Exception:
            pass
    return torch.from_numpy(arr)
