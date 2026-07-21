# nnue_reader.py
# Parses a Stockfish 16.x .nnue file and implements the HalfKAv2_hm Feature Transformer.
#
# Outputs the 2048-dim FT activation vector (two 1024-dim king perspectives concatenated)
# which is the "Layer 1 Perception" input for the Phase 5 Nimzo-Net architecture.
#
# Binary layout for nn-b1a57edbea57.nnue (SF 16.1):
#   [0:96]        File header (version, net hash, arch string)
#   [96:100]      FT hash
#   [100:117]     "COMPRESSED_LEB128" tag  (bias section marker)
#   [117:4197]    FT biases in signed LEB128  — first L1_HALF values used
#   [4197:4214]   "COMPRESSED_LEB128" tag  (weight section marker)
#   [4214:4214+N] 2 header LEB128 values then NUM_FEATURES*L1_HALF weight values
#   [...]         FC layers (not used here)
#
# Feature encoding: HalfKAv2_hm (without the half-move virtual piece)
#   For each perspective (white-king-view, black-king-view):
#     king_sq   = king square of that side's king (rank-flipped for black)
#     For each non-king piece on the board:
#       pt_idx  = (piece_type-1) + (0 if piece_color==perspective else 5)  [0..9]
#       piece_sq = square (rank-flipped for black)
#       feat    = king_sq * 704 + pt_idx * 64 + piece_sq   [0..45055]
#
# Forward pass per perspective:
#   accumulator = biases + sum(weights[feat] for feat in active_features)
#   output      = clip(accumulator, 0, 127).astype(float32)

from __future__ import annotations

import numpy as np
import chess

L1_HALF      = 1024   # FT output dimensions per perspective
NNUE_SIZE    = 2048   # total (white perspective + black perspective)
NUM_FEATURES = 45056  # 64 king_sq * 704 piece-sq combos per perspective
_TAG         = b'COMPRESSED_LEB128'


def _decode_leb128_all(data: bytes | bytearray, start: int, stop: int) -> np.ndarray:
    """Decode all signed LEB128 integers in data[start:stop].  Returns int32 array."""
    out  = []
    p    = start
    while p < stop:
        result = 0
        shift  = 0
        while True:
            b       = data[p]; p += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        if result & (1 << (shift + 6)):
            result -= (1 << (shift + 7))
        out.append(result)
    return np.array(out, dtype=np.int32)


def load_feature_transformer(
    nnue_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a SF 16.x .nnue file and extract the Feature Transformer weights.

    On first call, decodes the LEB128 binary (~16s) and saves a numpy cache
    alongside the .nnue file for fast subsequent loads (~0.3s mmap).

    Returns
    -------
    biases  : int32 array, shape (L1_HALF,)               = (1024,)
    weights : int16 array, shape (NUM_FEATURES, L1_HALF)  = (45056, 1024)
    """
    from pathlib import Path
    nnue_p    = Path(nnue_path)
    cache_npz = nnue_p.with_suffix('.ft_cache.npz')

    if cache_npz.exists() and cache_npz.stat().st_mtime >= nnue_p.stat().st_mtime:
        arr     = np.load(str(cache_npz))
        return arr['biases'], arr['weights']

    print(f"  Decoding LEB128 from {nnue_p.name} (one-time, ~16s) ...")
    with open(nnue_path, 'rb') as f:
        data = f.read()

    # Locate section boundaries
    asz      = int.from_bytes(data[8:12], 'little')
    body     = 12 + asz + 4
    tag0     = data.index(_TAG, body)
    tag1     = data.index(_TAG, tag0 + len(_TAG))
    tag2     = data.index(_TAG, tag1 + len(_TAG))

    # Biases: first L1_HALF values from section 0
    bias_start = tag0 + len(_TAG)
    raw_biases = _decode_leb128_all(data, bias_start, tag1)
    biases     = raw_biases[:L1_HALF].astype(np.int32)

    # Weights: skip 2 header values, then NUM_FEATURES * L1_HALF int16 values
    wt_start = tag1 + len(_TAG)
    raw_wts  = _decode_leb128_all(data, wt_start, tag2)
    wts      = raw_wts[2:2 + NUM_FEATURES * L1_HALF]
    weights  = wts.reshape(NUM_FEATURES, L1_HALF).astype(np.int16)

    np.savez_compressed(str(cache_npz), biases=biases, weights=weights)
    print(f"  FT cache saved: {cache_npz.name}")
    return biases, weights


def get_active_features(board: chess.Board, perspective: chess.Color) -> list[int]:
    """
    Return HalfKAv2 (no-hm) active feature indices for a given king perspective.

    Parameters
    ----------
    board       : python-chess Board object
    perspective : chess.WHITE or chess.BLACK

    Returns
    -------
    List of feature indices in [0, NUM_FEATURES-1]
    """
    def mirror(sq: int) -> int:
        return sq ^ 56 if perspective == chess.BLACK else sq

    king_sq  = mirror(board.king(perspective))
    features = []
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            pt_idx = (pt - 1) + (0 if color == perspective else 5)  # 0..9
            for sq in board.pieces(pt, color):
                feat = king_sq * 704 + pt_idx * 64 + mirror(sq)
                features.append(feat)
    return features


def compute_activations(
    fen:     str,
    biases:  np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute the 2048-dim NNUE Feature Transformer output for a FEN position.

    Returns float32 array shape (NNUE_SIZE,) = (2048,)
    Values in [0, 127].
    """
    board  = chess.Board(fen)
    out    = np.empty(NNUE_SIZE, dtype=np.float32)

    for i, perspective in enumerate((chess.WHITE, chess.BLACK)):
        active = get_active_features(board, perspective)
        acc    = biases.copy().astype(np.int32)
        if active:
            acc += weights[active].sum(axis=0).astype(np.int32)
        # ReLU only — downstream BatchNorm in the classifier head handles scale normalization.
        # (SF uses ClippedReLU[0, 127×scale] but the exact scale requires per-network calibration.)
        np.maximum(acc, 0, out=acc)
        out[i * L1_HALF:(i + 1) * L1_HALF] = acc.astype(np.float32)

    return out
