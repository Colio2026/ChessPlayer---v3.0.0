# paths.py
# Single source of truth for all ML data file paths.
# Import from here instead of scattering Path("data/...") literals across files.
#
# Usage:
#   from src.chess_coach.ml.paths import CLASSIFIER_BEST, THRESHOLDS, ALGO_CACHE

from pathlib import Path

DATA_DIR = Path("data")

# ── Training data ─────────────────────────────────────────────────────────────
TRAINING_JSONL   = DATA_DIR / "training_raw.jsonl"

# ── Checkpoints ───────────────────────────────────────────────────────────────
CLASSIFIER_BEST  = DATA_DIR / "classifier_best.pt"
CLASSIFIER_LAST  = DATA_DIR / "classifier_last.pt"

# ── Calibration ───────────────────────────────────────────────────────────────
THRESHOLDS       = DATA_DIR / "thresholds.json"

# ── Feature caches (indexed by _ac field in training_raw.jsonl) ───────────────
ALGO_CACHE       = DATA_DIR / "algo_cache.npy"    # (N, 1811) float32  — algo_v4 spatial features
V3_CACHE         = DATA_DIR / "v3_cache.npy"      # (N,   59) float32  — binary concept flags
SF_CACHE         = DATA_DIR / "sf_cache.npy"      # (N,   14) float32  — SF classical eval
NNUE_CACHE       = DATA_DIR / "nnue_cache.npy"    # (N, 2048) float32  — SF16 FT activations
BOARD_CACHE      = DATA_DIR / "board_cache.npy"   # (N, 1001) float32  — fen_to_tensor output

# ── Engine / external ─────────────────────────────────────────────────────────
ECO_DB           = DATA_DIR / "eco_db.json"
NNUE_WEIGHTS     = DATA_DIR / "nn.nnue"            # Stockfish16 FT weights for NNUE cache builder
