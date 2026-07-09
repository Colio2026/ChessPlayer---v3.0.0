#!/usr/bin/env python3
"""
train.py  —  Train the chess concept classifier
------------------------------------------------
Reads data/training_raw.jsonl, trains a 3-layer MLP to predict chess
concepts from board position, saves the best checkpoint.

Usage
-----
    python -m src.chess_coach.ml.train

    # Faster test run with 10% of data (confirms setup works)
    python -m src.chess_coach.ml.train --quick

    # Custom settings
    python -m src.chess_coach.ml.train --epochs 50 --batch-size 256 --lr 5e-4

Output
------
    data/classifier_best.pt   — best validation-loss checkpoint
    data/classifier_last.pt   — final epoch checkpoint (for resuming)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset    import ChessConceptDataset
from .classifier import ChessConceptClassifier
from .concept_vocab import CONCEPTS, NUM_CONCEPTS

CHECKPOINT_DIR  = Path("data")
LABEL_SMOOTHING = 0.05   # 0 → 0.05, 1 → 0.95; softens noisy keyword labels


def macro_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro F1 over classes that have at least one positive in this split."""
    preds   = preds.float()
    targets = targets.float()
    tp = (preds * targets).sum(0)
    fp = (preds * (1 - targets)).sum(0)
    fn = ((1 - preds) * targets).sum(0)
    has_support  = targets.sum(0) > 0
    precision    = tp / (tp + fp).clamp(min=1)
    recall       = tp / (tp + fn).clamp(min=1)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_per_class[has_support].mean().item()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── data ──────────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    print(f"\nLoading data from {data_path}")

    train_ds = ChessConceptDataset(data_path, split="train")
    val_ds   = ChessConceptDataset(data_path, split="val")

    if args.quick:
        # Use 10% of training data for a fast sanity-check run
        n = max(500, len(train_ds) // 10)
        train_ds._x = train_ds._x[:n]
        train_ds._y = train_ds._y[:n]
        print(f"  --quick mode: using {n} training examples")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=(device.type == "cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                          shuffle=False, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    model = ChessConceptClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")

    # ── loss with class-imbalance correction ──────────────────────────────────
    pos_weight = train_ds.pos_weight.to(device)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_macro_f1  = -1.0
    best_epoch     = 0
    patience_count = 0

    print(f"\n── Training  ({args.epochs} epochs, batch {args.batch_size}) "
          f"─────────────────────────────────────")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Macro F1':>8}  "
          f"{'LR':>9}  {'Time':>6}  {'Note'}")
    print("─" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, y).item()
                all_preds.append((torch.sigmoid(logits) > 0.5).cpu())
                all_targets.append(y.cpu())
        val_loss /= len(val_dl)

        preds   = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        val_f1  = macro_f1(preds, targets)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        note    = ""

        if val_f1 > best_macro_f1:
            best_macro_f1  = val_f1
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_loss":   val_loss,
                "macro_f1":   val_f1,
                "concepts":   CONCEPTS,
            }, CHECKPOINT_DIR / "classifier_best.pt")
            note = "✓ best"
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping — no improvement for {args.patience} epochs.")
                break

        print(f"{epoch:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_f1:>8.4f}  "
              f"{lr_now:>9.2e}  {elapsed:>5.1f}s  {note}")

    # Save final checkpoint regardless of whether it's the best
    torch.save({
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "val_loss":   val_loss,
        "concepts":   CONCEPTS,
    }, CHECKPOINT_DIR / "classifier_last.pt")

    print(f"\n── Done ──────────────────────────────────────────────────────────────")
    print(f"Best checkpoint : epoch {best_epoch}  macro_f1={best_macro_f1:.4f}")
    print(f"Saved to        : {CHECKPOINT_DIR / 'classifier_best.pt'}")
    print(f"\nNext step: python -m src.chess_coach.ml.evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train chess concept classifier")
    parser.add_argument("--data",       default="data/training_raw.jsonl")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=15,
                        help="Stop after N epochs with no macro F1 improvement.")
    parser.add_argument("--quick",      action="store_true",
                        help="Use 10%% of data for a fast test run.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
