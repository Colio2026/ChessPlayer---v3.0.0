# dataset.py
# PyTorch Dataset that reads from training_raw.jsonl and encodes
# examples lazily (per __getitem__ call) rather than pre-loading
# all tensors into RAM at init.
#
# Lazy loading is required because variable-length history
# tensors cannot be pre-stacked into a single fixed-size tensor.
#
# __getitem__ returns a 4-tuple: (x, hist, seq_len, y)
#   x        : float32 (STATIC_SIZE_V4 + ALGO_SIZE,) = 2999-dim Phase 4
#              float32 (STATIC_SIZE,)                 = 1188-dim Phase 3
#   hist     : float32 (MAX_SEQ_LEN, 144) padded move history (144-dim v4 / 128-dim v3)
#   seq_len  : int64 scalar               actual history length before padding
#   y        : float32 (NUM_CONCEPTS,)    multi-hot concept labels
#
# Cache files (data/algo_cache.npy, data/v3_cache.npy) are discovered
# automatically and opened lazily so each DataLoader worker gets its own
# mmap handle — the OS shares the underlying page cache, so RAM stays low.

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .board_encoder import (
    fen_to_tensor, move_to_tensor, history_to_tensor, history_rich_to_tensor,
    STATIC_SIZE, STATIC_SIZE_V4, INPUT_SIZE, MOVE_SIZE, ALGO_SIZE, ALGO_SIZE_V4,
    SF_SIZE, MAX_SEQ_LEN,
)
from .concept_vocab import CONCEPTS, CONCEPT_TO_IDX, NUM_CONCEPTS
from tools.label_positions import algo_feature_vector


class ChessConceptDataset(Dataset):
    """
    Multi-label classification dataset with lazy per-example encoding.

    Each item is (x, hist, seq_len, y):
        x        : float32 [2999]              Phase 4: board+move+algo_v4+v3
        hist     : float32 [MAX_SEQ_LEN, 144]  padded move history for GRU
        seq_len  : int64 scalar                actual history length
        y        : float32 [NUM_CONCEPTS]      multi-hot concept labels

    Only examples with at least one theme label are included.

    Parameters
    ----------
    jsonl_path  : Path to training_raw.jsonl
    split       : 'train' | 'val' | 'test'
    seed        : random seed for reproducible splits
    train_frac  : fraction of data for training (default 0.80)
    val_frac    : fraction for validation (default 0.10)
                  remainder goes to test
    phase4      : True → use 144-dim history tensors (Phase 4 architecture)
    """

    def __init__(
        self,
        jsonl_path:  str | Path  = "data/training_raw.jsonl",
        split:       str         = "train",
        seed:        int         = 42,
        train_frac:  float       = 0.80,
        val_frac:    float       = 0.10,
        phase4:      bool        = False,
    ) -> None:
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            sys.exit(f"Dataset file not found: {jsonl_path}\n"
                     "Run tools/parse_annotated_pgn.py first.")

        print(f"Loading dataset from {jsonl_path} ...", end=" ", flush=True)
        raw = []
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    ex = json.loads(line)
                    if ex.get("themes"):
                        raw.append(ex)
                except Exception:
                    pass
        print(f"{len(raw):,} labeled examples found.")

        # Reproducible shuffle + split
        rng = random.Random(seed)
        rng.shuffle(raw)
        n       = len(raw)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        splits  = {
            "train": raw[:n_train],
            "val":   raw[n_train: n_train + n_val],
            "test":  raw[n_train + n_val:],
        }
        self._raw    = splits[split]
        self._phase4 = phase4
        print(f"  {split}: {len(self._raw):,} examples  "
              f"(train {n_train:,} / val {n_val:,} / test {n - n_train - n_val:,})")

        # Cache paths stored as strings (picklable — safe for multiprocessing spawn).
        # The actual numpy memmaps are opened lazily in __getitem__, so every
        # DataLoader worker opens its own file handle; the OS shares the page cache.
        algo_path = Path("data/algo_cache.npy").resolve()
        v3_path   = Path("data/v3_cache.npy").resolve()
        sf_path   = Path("data/sf_cache.npy").resolve()
        self._algo_cache_path: str | None = str(algo_path) if algo_path.exists() else None
        self._v3_cache_path:   str | None = str(v3_path)   if v3_path.exists()   else None
        self._sf_cache_path:   str | None = str(sf_path)   if sf_path.exists()   else None
        self._algo_cache: np.ndarray | None = None   # opened on first __getitem__
        self._v3_cache:   np.ndarray | None = None
        self._sf_cache:   np.ndarray | None = None

        if self._algo_cache_path:
            print(f"  Algo cache:  {algo_path.name}  (lazy mmap, {algo_path.stat().st_size / 1e9:.2f} GB)")
        else:
            print(f"  No algo cache — run tools/build_algo_cache.py to build caches.")
        if self._v3_cache_path:
            print(f"  V3 cache:    {v3_path.name}  (lazy mmap, {v3_path.stat().st_size / 1e6:.0f} MB)")
        else:
            print(f"  No v3 cache — v3 features computed per example (slow).")
        if self._sf_cache_path:
            print(f"  SF cache:    {sf_path.name}  (lazy mmap, {sf_path.stat().st_size / 1e6:.0f} MB)")
        else:
            print(f"  No SF cache — SF classical eval features inactive (run build_sf_cache.py).")

        # Pre-build label tensor (cheap: integer lookups only, no board computation).
        print(f"  Building label index ...", end=" ", flush=True)
        ys = []
        for ex in self._raw:
            y = torch.zeros(NUM_CONCEPTS, dtype=torch.float32)
            for theme in ex["themes"]:
                idx = CONCEPT_TO_IDX.get(theme)
                if idx is not None:
                    y[idx] = 1.0
            ys.append(y)
        self._y = torch.stack(ys)
        print("done.")

    # ── Lazy mmap openers ─────────────────────────────────────────────────────

    def _get_algo_cache(self) -> np.ndarray | None:
        """Open algo_cache mmap on first access (worker-local; picklable init)."""
        if self._algo_cache is None and self._algo_cache_path:
            self._algo_cache = np.load(self._algo_cache_path, mmap_mode="r")
        return self._algo_cache

    def _get_v3_cache(self) -> np.ndarray | None:
        """Open v3_cache mmap on first access (worker-local; picklable init)."""
        if self._v3_cache is None and self._v3_cache_path:
            self._v3_cache = np.load(self._v3_cache_path, mmap_mode="r")
        return self._v3_cache

    def _get_sf_cache(self) -> np.ndarray | None:
        """Open sf_cache mmap on first access (worker-local; picklable init)."""
        if self._sf_cache is None and self._sf_cache_path:
            self._sf_cache = np.load(self._sf_cache_path, mmap_mode="r")
        return self._sf_cache

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        ex = self._raw[idx]
        y  = self._y[idx].clone()   # clone view → independent tensor

        try:
            board_t = fen_to_tensor(ex["fen"])
            move_t  = move_to_tensor(ex.get("move_uci", ""))

            ac_idx     = ex.get("_ac")
            algo_cache = self._get_algo_cache()
            v3_cache   = self._get_v3_cache()

            # V4 spatial features (ALGO_SIZE_V4 = 1811-dim) from algo_cache
            if algo_cache is not None and ac_idx is not None:
                algo_t = torch.from_numpy(np.array(algo_cache[ac_idx], dtype=np.float32))
            else:
                af = ex.get("algo_features")
                if af is not None:
                    algo_t = torch.tensor(af, dtype=torch.float32)
                else:
                    algo_t = torch.zeros(ALGO_SIZE_V4, dtype=torch.float32)

            # V3 summary (ALGO_SIZE = 59-dim) from v3_cache or computed from FEN
            if v3_cache is not None and ac_idx is not None:
                v3_t = torch.from_numpy(np.array(v3_cache[ac_idx], dtype=np.float32))
            else:
                v3_t = torch.from_numpy(algo_feature_vector(ex["fen"]))

            # SF classical eval features (SF_SIZE = 14-dim) from sf_cache if available
            sf_cache = self._get_sf_cache()
            if sf_cache is not None and ac_idx is not None:
                sf_t = torch.from_numpy(np.array(sf_cache[ac_idx], dtype=np.float32))
            else:
                sf_t = torch.zeros(SF_SIZE, dtype=torch.float32)

            x = torch.cat([board_t, move_t, algo_t, v3_t, sf_t])
        except Exception:
            x = torch.zeros(
                STATIC_SIZE_V4 + ALGO_SIZE if self._phase4 else STATIC_SIZE,
                dtype=torch.float32,
            )

        if self._phase4:
            hist_t, seq_len = history_rich_to_tensor(ex.get("history_rich", []))
        else:
            hist_t, seq_len = history_to_tensor(ex.get("history_uci", []))

        return x, hist_t, seq_len, y

    # ── pos_weight ────────────────────────────────────────────────────────────

    @property
    def pos_weight(self) -> torch.Tensor:
        """
        Per-class positive weights for BCEWithLogitsLoss.

        Phase 2 linear formula restored — (N-pos)/pos clamped at 20 correctly
        balances the ~5% positive rate per class. Cutoff lowered to 500 (was
        2000) so shouldering / drawn_position can participate in training.
        """
        n   = len(self._y)
        pos = self._y.sum(dim=0).clamp(min=1)
        w   = ((n - pos) / pos).clamp(1.0, 20.0)
        w[pos < 500] = 0.0
        return w
