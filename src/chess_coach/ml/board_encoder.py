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
ALGO_SIZE     = 59          # concept bottleneck bits from algo detectors
MAX_SEQ_LEN   = 60          # half-moves of game history stored per example
GRU_HIDDEN    = 256         # GRU hidden size, appended to board features at head
STATIC_SIZE   = INPUT_SIZE + MOVE_SIZE + ALGO_SIZE   # 1188 — board+move+algo
COMBINED_SIZE = STATIC_SIZE + GRU_HIDDEN             # 1444 — board features + GRU output

# ── Phase 4 constants (kept separate until full pipeline cutover) ─────────────
# Switch active constants to these when parse → ingest → train pipeline re-runs.
#
# MOVE_SIZE_V4 = 144    per-step encoding for history_rich_to_tensor()
#   [0:64]   from_square one-hot
#   [64:128] to_square   one-hot
#   [128:134] piece_type one-hot  (PAWN=0 … KING=5)
#   [134:141] captured   one-hot  (none=0, PAWN=1 … KING=6)
#   [141]    is_check   binary
#   [142]    is_capture binary
#   [143]    color      binary (white=1)
#
# ALGO_SIZE_V4    = 1811   (from tools/label_positions.ALGO_FEATURE_SIZE_V4)
#   1663 (B1-B6) + 148 (B7 king_safety_vec) = 1811
# STATIC_SIZE_V4  = INPUT_SIZE + MOVE_SIZE + ALGO_SIZE_V4  = 2940
# COMBINED_SIZE_V4 = STATIC_SIZE_V4 + GRU_HIDDEN           = 3196
MOVE_SIZE_V4     = 144   # GRU per-step encoding only (history_rich_to_tensor)
ALGO_SIZE_V4     = 1811
STATIC_SIZE_V4   = INPUT_SIZE + MOVE_SIZE + ALGO_SIZE_V4   # 2940 (MOVE_SIZE=128 unchanged — that's current-move, not GRU)
COMBINED_SIZE_V4 = STATIC_SIZE_V4 + GRU_HIDDEN             # 3196

# Stockfish classical eval features (pre-encoded at cache-build time via build_sf_cache.py)
# 7 terms × 2 sides = 14 floats appended after v3_summary in x
# white indices 0-6: Mobility, King safety, Threats, Passed, Space, Pawns, Imbalance
# black indices 7-13: same order
# "Passed" (white=[3], black=[10]) is the key signal for passed_pawn concept quality
SF_SIZE  = 14
SF_BREAK = STATIC_SIZE_V4 + ALGO_SIZE   # 2999 — offset where SF features start in x

# Phase 4-B: spatial bottleneck + Phase 3 summary + SF classical eval features
# x layout: [board(1001), move(128), algo_v4(1811), v3_summary(59), sf(14)] = 3013
# spatial(1811) → proj(256) | v3_summary(59) + sf(14) bypass bottleneck
PROJ_SIZE_V4       = 256
COMBINED_SIZE_V4B  = INPUT_SIZE + MOVE_SIZE + PROJ_SIZE_V4 + ALGO_SIZE + SF_SIZE + GRU_HIDDEN  # 1001+128+256+59+14+256=1714

# Phase 5: NNUE Feature Transformer replaces the 1811-dim algo_v4 spatial features.
# Frozen SF16 FT weights produce a 2048-dim perception vector (1024 per king perspective).
# x layout: [nnue(2048), board_meta(13), move(128), sf_classical(14), v3_summary(59)] = 2262
# NNUE bottleneck: 2048 → 256 via learnable projection (mirrors Phase 4's spatial_proj).
# This keeps the combined input to the head at 726 (vs 2518 without bottleneck),
# preventing the 59 v3 signal dims from being drowned by 2048 raw NNUE dims.
NNUE_SIZE         = 2048   # 2 × L1_HALF (white + black king perspectives)
BOARD_META_SIZE   = 13     # side_to_move(1) + castling(4) + ep_file(8)
STATIC_SIZE_V5    = NNUE_SIZE + BOARD_META_SIZE + MOVE_SIZE + SF_SIZE  # 2048+13+128+14=2203
COMBINED_SIZE_V5  = STATIC_SIZE_V5 + ALGO_SIZE + GRU_HIDDEN            # 2203+59+256=2518

NNUE_PROJ_SIZE    = 256    # NNUE bottleneck output (same dim as Phase 4 spatial_proj)
STATIC_SIZE_V5B   = NNUE_PROJ_SIZE + BOARD_META_SIZE + MOVE_SIZE + SF_SIZE + ALGO_SIZE  # 256+13+128+14+59=470
COMBINED_SIZE_V5B = STATIC_SIZE_V5B + GRU_HIDDEN                                        # 470+256=726

# Phase 5C: NNUE bottleneck + full board tensor (restores explicit piece placement).
# x layout: [nnue(2048), board(1001), move(128), sf(14), v3(59)] = 3250 raw input
# After nnue_proj(256): [proj(256), board(1001), move(128), sf(14), v3(59)] = 1458 static
# Same combined size as Phase 4B (1714), so same head width applies.
STATIC_SIZE_V5C   = NNUE_PROJ_SIZE + INPUT_SIZE + MOVE_SIZE + SF_SIZE + ALGO_SIZE  # 256+1001+128+14+59=1458
COMBINED_SIZE_V5C = STATIC_SIZE_V5C + GRU_HIDDEN                                   # 1458+256=1714

# Phase 5D: NNUE bottleneck + algo_v4 bottleneck + full board tensor.
# Both evaluation signal (NNUE) and explicit concept features (algo_v4) feed the head.
# x layout: [nnue(2048), board(1001), move(128), algo_v4(1811), sf(14), v3(59)] = 5061 raw
# After nnue_proj(256) + spatial_proj(256):
#   [nnue_proj(256), board(1001), move(128), algo_proj(256), sf(14), v3(59)] = 1714 static
# Combined 1970 = 1714 static + 256 GRU
STATIC_SIZE_V5D   = NNUE_PROJ_SIZE + INPUT_SIZE + MOVE_SIZE + PROJ_SIZE_V4 + SF_SIZE + ALGO_SIZE  # 256+1001+128+256+14+59=1714
COMBINED_SIZE_V5D = STATIC_SIZE_V5D + GRU_HIDDEN                                                   # 1714+256=1970

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


def history_to_tensor(
    moves: list[str],
    max_len: int = MAX_SEQ_LEN,
) -> tuple[torch.Tensor, int]:
    """Encode a list of UCI move strings as a padded (max_len, MOVE_SIZE) float32 tensor.

    Takes the last max_len moves (most recent context).
    Returns (padded_tensor, actual_sequence_length).
    Zero rows represent padding; GRU will mask these via pack_padded_sequence.
    """
    moves   = list(moves)[-max_len:]   # most recent N half-moves, don't mutate
    seq_len = len(moves)
    out     = torch.zeros(max_len, MOVE_SIZE, dtype=torch.float32)
    for i, uci in enumerate(moves):
        try:
            move                    = chess.Move.from_uci(uci)
            out[i, move.from_square]     = 1.0
            out[i, 64 + move.to_square] = 1.0
        except Exception:
            pass   # malformed UCI → leave row as zeros
    return out, seq_len


def history_rich_to_tensor(
    moves: list[dict],
    max_len: int = MAX_SEQ_LEN,
) -> tuple[torch.Tensor, int]:
    """Encode a Phase 4 history_rich move list as a padded (max_len, 144) tensor.

    Each dict has keys: uci, piece (int 1-6), captured (int 1-6 or None),
    is_check (bool), color (int 1=white 0=black).
    Returns (padded_tensor, actual_sequence_length).
    """
    moves   = list(moves)[-max_len:]
    seq_len = len(moves)
    out     = torch.zeros(max_len, MOVE_SIZE_V4, dtype=torch.float32)
    for i, m in enumerate(moves):
        try:
            move = chess.Move.from_uci(m["uci"])
            out[i, move.from_square]      = 1.0   # [0:64]
            out[i, 64 + move.to_square]   = 1.0   # [64:128]
            pt = m.get("piece")
            if pt and 1 <= pt <= 6:
                out[i, 128 + pt - 1]      = 1.0   # [128:134] piece_type one-hot
            cap = m.get("captured")
            if cap and 1 <= cap <= 6:
                out[i, 134 + cap]         = 1.0   # [135:141] captured one-hot (0=none)
            else:
                out[i, 134]               = 1.0   # [134] = no capture
            out[i, 141] = 1.0 if m.get("is_check") else 0.0
            out[i, 142] = 1.0 if (cap is not None) else 0.0
            out[i, 143] = float(m.get("color", 1))
        except Exception:
            pass
    return out, seq_len


def board_meta_tensor(fen: str) -> torch.Tensor:
    """
    Encode the 13 non-piece FEN features as a float32 tensor.
    Used in Phase 5 alongside NNUE activations (which encode piece placement).

    Layout (BOARD_META_SIZE = 13):
      [0]      Side to move  — 1.0=white, 0.0=black
      [1:5]    Castling rights — [WK, WQ, BK, BQ], 1.0 if available
      [5:13]   En-passant file — one-hot over files a–h, all zeros if none
    """
    board = chess.Board(fen)
    arr   = np.zeros(BOARD_META_SIZE, dtype=np.float32)
    arr[0] = 1.0 if board.turn == chess.WHITE else 0.0
    arr[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE)  else 0.0
    arr[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    arr[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK)  else 0.0
    arr[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        arr[5 + chess.square_file(board.ep_square)] = 1.0
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
