# board_encoder.py
# Converts a FEN string into a fixed-length float tensor for model input.
#
# Layout (781 elements total):
#   [0:768]   Piece placement — 12 channels × 64 squares
#             channel = color*6 + piece_type,  index = channel*64 + square
#             1.0 if that piece occupies that square, else 0.0
#   [768]     Side to move — 1.0=white, 0.0=black
#   [769:773] Castling rights — [WK, WQ, BK, BQ], 1.0 if available
#   [773:781] En-passant file — one-hot over files a–h, all zeros if none
#
# Adding side-to-move lets the net distinguish zugzwang from non-zugzwang
# in otherwise identical positions.  Castling rights help with king-safety
# and development concepts.  En-passant helps with pawn-break detection.

from __future__ import annotations

import numpy as np
import torch
import chess

_COLORS      = [chess.WHITE, chess.BLACK]
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]

# Offset constants — change these and everything downstream breaks,
# so they live here rather than scattered across the file.
_PIECE_OFFSET  = 0     # 0 … 767
_TURN_OFFSET   = 768   # 1 bit
_CASTLE_OFFSET = 769   # 4 bits  [WK, WQ, BK, BQ]
_EP_OFFSET     = 773   # 8 bits  one-hot over files a–h

INPUT_SIZE    = 781        # board-only (piece placement + turn + castling + ep)
MOVE_SIZE     = 128        # 64 (from-square) + 64 (to-square)
COMBINED_SIZE = INPUT_SIZE + MOVE_SIZE   # 909 — what the model actually receives


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Encode a FEN position as a 781-element float32 tensor."""
    board = chess.Board(fen)
    arr   = np.zeros(INPUT_SIZE, dtype=np.float32)

    # ── piece placement ───────────────────────────────────────────────────────
    for color_idx, color in enumerate(_COLORS):
        for piece_idx, piece_type in enumerate(_PIECE_TYPES):
            channel = color_idx * 6 + piece_idx
            base    = _PIECE_OFFSET + channel * 64
            for sq in board.pieces(piece_type, color):
                arr[base + sq] = 1.0

    # ── side to move ──────────────────────────────────────────────────────────
    arr[_TURN_OFFSET] = 1.0 if board.turn == chess.WHITE else 0.0

    # ── castling rights ───────────────────────────────────────────────────────
    arr[_CASTLE_OFFSET + 0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE)  else 0.0
    arr[_CASTLE_OFFSET + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    arr[_CASTLE_OFFSET + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK)  else 0.0
    arr[_CASTLE_OFFSET + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # ── en passant ────────────────────────────────────────────────────────────
    if board.ep_square is not None:
        arr[_EP_OFFSET + chess.square_file(board.ep_square)] = 1.0

    return torch.from_numpy(arr)


def move_to_tensor(move_uci: str) -> torch.Tensor:
    """Encode a UCI move string as a 128-element float32 tensor.

    Layout: [0:64] from-square one-hot, [64:128] to-square one-hot.
    Empty string or invalid move → all zeros (used for root-comment
    examples that have no associated move).
    """
    arr = np.zeros(MOVE_SIZE, dtype=np.float32)
    if move_uci and len(move_uci) >= 4:
        try:
            move = chess.Move.from_uci(move_uci)
            arr[move.from_square]      = 1.0
            arr[64 + move.to_square]   = 1.0
        except Exception:
            pass
    return torch.from_numpy(arr)
