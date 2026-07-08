# board_encoder.py
# Converts a FEN string into a fixed-length float tensor for model input.
#
# Encoding: 768-element binary vector
#   12 piece types (6 per color) × 64 squares = 768 bits
#   Index = (color * 6 + piece_type) * 64 + square
#   1.0 if that piece is on that square, else 0.0
#
# This is the same piece-placement encoding used as NNUE's input layer,
# so the board geometry is already in the form neural nets learn well from.

from __future__ import annotations

import numpy as np
import torch
import chess

_COLORS      = [chess.WHITE, chess.BLACK]
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]

INPUT_SIZE = 768


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Encode a FEN position as a 768-element float32 tensor."""
    board = chess.Board(fen)
    arr   = np.zeros(INPUT_SIZE, dtype=np.float32)
    for color_idx, color in enumerate(_COLORS):
        for piece_idx, piece_type in enumerate(_PIECE_TYPES):
            channel = color_idx * 6 + piece_idx
            base    = channel * 64
            for sq in board.pieces(piece_type, color):
                arr[base + sq] = 1.0
    return torch.from_numpy(arr)
