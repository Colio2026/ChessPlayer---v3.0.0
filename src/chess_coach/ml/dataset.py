# dataset.py
# PyTorch Dataset that reads from training_raw.jsonl and pre-encodes
# all positions to tensors at load time (one encoding per position,
# cached in RAM — ~55 MB for 67k examples).

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .board_encoder import fen_to_tensor, move_to_tensor
from .concept_vocab import CONCEPTS, CONCEPT_TO_IDX, NUM_CONCEPTS
from tools.label_positions import algo_feature_vector


class ChessConceptDataset(Dataset):
    """
    Multi-label classification dataset.

    Each item is (x, y):
        x : float32 tensor [768]   — board encoding
        y : float32 tensor [57]    — multi-hot concept labels

    Only examples with at least one theme label are included
    (unlabeled examples have no signal for the classifier).

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
        jsonl_path:  str | Path = "data/training_raw.jsonl",
        split:       str        = "train",
        seed:        int        = 42,
        train_frac:  float      = 0.80,
        val_frac:    float      = 0.10,
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
                    if ex.get("themes"):   # labeled examples only
                        raw.append(ex)
                except Exception:
                    pass
        print(f"{len(raw):,} labeled examples found.")

        # Reproducible shuffle + split
        rng = random.Random(seed)
        rng.shuffle(raw)
        n         = len(raw)
        n_train   = int(n * train_frac)
        n_val     = int(n * val_frac)
        splits    = {
            "train": raw[:n_train],
            "val":   raw[n_train: n_train + n_val],
            "test":  raw[n_train + n_val:],
        }
        subset = splits[split]
        print(f"  {split}: {len(subset):,} examples  "
              f"(train {n_train:,} / val {n_val:,} / test {n - n_train - n_val:,})")

        # Pre-encode everything into tensors (board + move concatenated)
        total = len(subset)
        print(f"  Encoding {total:,} boards ...")
        xs, ys, skipped = [], [], 0
        milestone = max(1, total // 20)   # print every 5%
        for i, ex in enumerate(subset):
            try:
                x = torch.cat([
                    fen_to_tensor(ex["fen"]),
                    move_to_tensor(ex.get("move_uci", "")),
                    torch.from_numpy(algo_feature_vector(ex["fen"])),
                ])
            except Exception:
                skipped += 1
                continue
            y = torch.zeros(NUM_CONCEPTS, dtype=torch.float32)
            for theme in ex["themes"]:
                idx = CONCEPT_TO_IDX.get(theme)
                if idx is not None:
                    y[idx] = 1.0
            xs.append(x)
            ys.append(y)
            if (i + 1) % milestone == 0 or (i + 1) == total:
                pct = (i + 1) / total
                bar = "█" * int(pct * 25)
                print(f"    [{bar:<25}] {pct:5.1%}  ({i+1:,}/{total:,})", flush=True)
        if skipped:
            print(f"  Skipped {skipped:,} examples with invalid FENs.")
        print("  Encoding done.")

        self._x = torch.stack(xs)
        self._y = torch.stack(ys)

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]

    @property
    def pos_weight(self) -> torch.Tensor:
        """
        Per-class positive weights for BCEWithLogitsLoss.
        Corrects for class imbalance — rare concepts get higher weight
        so the loss penalises missing them more.

        weight[i] = (N - pos_i) / pos_i   (clamped to [1, 100])
        """
        n   = len(self._y)
        pos = self._y.sum(dim=0).clamp(min=1)
        # Clamp lowered 25→20: bottleneck features give the model structural priors,
        # so it needs less aggressive overpenalization to fire rare concepts correctly.
        w   = ((n - pos) / pos).clamp(1.0, 20.0)
        # Zero-weight threshold lowered 3000→2000: the bottleneck features let the
        # model bootstrap sparse concepts from structural signals, so moderately rare
        # concepts (1000–2000 positives) can now receive gradient without misfiring.
        # Truly data-sparse concepts (<1000 examples: shouldering, drawn_position,
        # initiative) remain zeroed — they need more data, not more gradient.
        w[pos < 2000] = 0.0
        return w
