# dataset.py
# PyTorch Dataset that reads from training_raw.jsonl and encodes
# examples lazily (per __getitem__ call) rather than pre-loading
# all tensors into RAM at init.
#
# Lazy loading is required for Phase 3 because variable-length history
# tensors cannot be pre-stacked into a single fixed-size tensor.
#
# __getitem__ returns a 4-tuple: (x, hist, seq_len, y)
#   x        : float32 (STATIC_SIZE,)     board+move+algo features (2792-dim v4 / 1188-dim v3)
#   hist     : float32 (MAX_SEQ_LEN, 144) padded move history (144-dim v4 / 128-dim v3)
#   seq_len  : int64 scalar               actual history length before padding
#   y        : float32 (NUM_CONCEPTS,)    multi-hot concept labels

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
    STATIC_SIZE, INPUT_SIZE, MOVE_SIZE, ALGO_SIZE, MAX_SEQ_LEN,
)
from .concept_vocab import CONCEPTS, CONCEPT_TO_IDX, NUM_CONCEPTS
from tools.label_positions import algo_feature_vector, algo_feature_vector_v4


class ChessConceptDataset(Dataset):
    """
    Multi-label classification dataset with lazy per-example encoding.

    Each item is (x, hist, seq_len, y):
        x        : float32 [STATIC_SIZE]       board+move+algo features
        hist     : float32 [MAX_SEQ_LEN, 128]  padded move history for GRU
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
    """

    def __init__(
        self,
        jsonl_path:  str | Path       = "data/training_raw.jsonl",
        split:       str              = "train",
        seed:        int              = 42,
        train_frac:  float            = 0.80,
        val_frac:    float            = 0.10,
        algo_tensor: "torch.Tensor | np.ndarray | None" = None,
        phase4:      bool             = False,   # locks history tensor to 144-dim
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
        self._raw = splits[split]
        self._phase4 = phase4
        print(f"  {split}: {len(self._raw):,} examples  "
              f"(train {n_train:,} / val {n_val:,} / test {n - n_train - n_val:,})")
        print(f"  Lazy loading enabled — examples encoded per batch during training.")

        # Algo feature cache (written by tools/build_algo_cache.py).
        # Caller passes a numpy mmap (np.load mmap_mode='r') so both datasets share
        # the same OS page-cache view without loading ~13 GB into RAM.
        if algo_tensor is not None:
            self._algo_cache = algo_tensor          # numpy mmap or tensor from caller
            self._algo_dim: int | None = algo_tensor.shape[1]
            print(f"  Algo cache shared from caller  {algo_tensor.shape}")
        else:
            cache_path = Path("data/algo_cache.npy").resolve()
            if cache_path.exists():
                print(f"  Memory-mapping algo cache ...", end=" ", flush=True)
                self._algo_cache = np.load(str(cache_path), mmap_mode="r")
                self._algo_dim = self._algo_cache.shape[1]
                print(f"done  {self._algo_cache.shape}")
            else:
                self._algo_cache = None
                self._algo_dim   = None

        # Pre-build label tensor (cheap: just integer lookups, no board computation)
        # Kept in RAM for pos_weight computation; does NOT include board features.
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

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        ex = self._raw[idx]
        y  = self._y[idx].clone()   # clone view → independent, resizable tensor

        try:
            board_t = fen_to_tensor(ex["fen"])
            move_t  = move_to_tensor(ex.get("move_uci", ""))

            ac_idx = ex.get("_ac")
            if self._algo_cache is not None and ac_idx is not None:
                row    = self._algo_cache[ac_idx]
                algo_t = torch.from_numpy(np.array(row, dtype=np.float32))
            elif self._algo_dim is not None:
                # cache present in main process but _ac missing for this example
                algo_t = torch.zeros(self._algo_dim, dtype=torch.float32)
            else:
                af = ex.get("algo_features")
                if af is not None:
                    algo_t = torch.tensor(af, dtype=torch.float32)
                else:
                    algo_t = torch.from_numpy(algo_feature_vector(ex["fen"]))

            # Phase 3 summary bits: actualized concept flags (piece IS on outpost, etc.)
            # These directly encode what the labels measure — concat alongside spatial maps
            # so the model can read the easy answer directly AND use fine-grained location info.
            v3_t = torch.from_numpy(algo_feature_vector(ex["fen"]))
            x = torch.cat([board_t, move_t, algo_t, v3_t])
        except Exception:
            algo_sz = self._algo_dim or (STATIC_SIZE - INPUT_SIZE - MOVE_SIZE)
            x = torch.zeros(INPUT_SIZE + MOVE_SIZE + algo_sz + ALGO_SIZE, dtype=torch.float32)

        if self._phase4:
            # Always 144-dim; fall back to empty rich history for old examples
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
