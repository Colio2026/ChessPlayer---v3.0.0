#!/usr/bin/env python3
"""
evaluate.py  —  Evaluate the trained chess concept classifier
--------------------------------------------------------------
Loads data/classifier_best.pt, runs the test split, reports:
  - Micro / macro F1 scores
  - Per-concept precision / recall / F1 / support table
  - Qualitative spot-checks on 10 famous positions

Usage
-----
    python -m src.chess_coach.ml.evaluate

    # Use a different checkpoint
    python -m src.chess_coach.ml.evaluate --checkpoint data/classifier_last.pt

    # Show spot-checks only (no dataset needed)
    python -m src.chess_coach.ml.evaluate --spot-check-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .classifier    import ChessConceptClassifier
from .dataset       import ChessConceptDataset
from .concept_vocab import CONCEPTS, NUM_CONCEPTS

DEFAULT_CHECKPOINT = Path("data/classifier_best.pt")
DEFAULT_DATA       = Path("data/training_raw.jsonl")
DEFAULT_THRESHOLD  = 0.4

# Famous positions and their expected concepts (subset — model may predict more)
SPOT_CHECKS: list[dict] = [
    {
        "name": "Lucena rook ending (rook+pawn vs rook)",
        "fen":  "1K1k4/1P6/8/8/r7/8/8/R7 w - - 0 1",
        "expect": ["endgame_technique", "passed_pawn"],
    },
    {
        "name": "Philidor drawing position",
        "fen":  "4k3/8/4K3/4P3/8/8/8/4r3 b - - 0 1",
        "expect": ["endgame_technique", "opposition"],
    },
    {
        "name": "Classic pin — Ruy Lopez middlegame",
        "fen":  "r1bq1rk1/ppp2ppp/2np1n2/1B2p3/2BPP3/2N1QN2/PPP2PPP/R4RK1 b - - 0 8",
        "expect": ["pin", "piece_activity"],
    },
    {
        "name": "Knight outpost on d5",
        "fen":  "r1bqr1k1/pp1nbppp/2p1pn2/3p4/3P1B2/2NBPN2/PPQ2PPP/R4RK1 w - - 0 1",
        "expect": ["outpost"],
    },
    {
        "name": "Passed pawn race",
        "fen":  "8/P7/8/8/8/8/7p/8 w - - 0 1",
        "expect": ["passed_pawn", "king_activity"],
    },
    {
        "name": "Bad bishop — blocked pawns on dark squares",
        "fen":  "5k2/pp1bpppp/2pp4/8/8/2PP4/PP1BPPPP/5K2 w - - 0 1",
        "expect": ["bad_bishop"],
    },
    {
        "name": "Back-rank mate threat",
        "fen":  "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "expect": ["back_rank"],
    },
    {
        "name": "Rook on seventh rank",
        "fen":  "6k1/3R1ppp/6r1/8/8/8/5PPP/6K1 w - - 0 1",
        "expect": ["rook_seventh"],
    },
    {
        "name": "Isolated queen pawn (IQP) position",
        "fen":  "r1bqr1k1/pp3ppp/2n1bn2/3p4/3P4/2NBPN2/PP3PPP/R1BQR1K1 w - - 0 1",
        "expect": ["isolated_pawn", "open_file"],
    },
    {
        "name": "Zugzwang — king and pawn ending",
        "fen":  "8/8/4k3/4p3/4K3/8/8/8 w - - 0 1",
        "expect": ["zugzwang", "endgame_technique", "opposition"],
    },
]


def load_model(checkpoint_path: Path, device: torch.device) -> ChessConceptClassifier:
    model = ChessConceptClassifier().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint: epoch={epoch}  val_loss={val_loss:.4f}")
    return model


def compute_f1(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1


def evaluate_dataset(model: ChessConceptClassifier,
                     data_path: Path,
                     device: torch.device,
                     threshold: float = DEFAULT_THRESHOLD) -> None:
    test_ds = ChessConceptDataset(data_path, split="test")
    if len(test_ds) == 0:
        print("No test examples — skipping dataset evaluation.")
        return

    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    tp = torch.zeros(NUM_CONCEPTS)
    fp = torch.zeros(NUM_CONCEPTS)
    fn = torch.zeros(NUM_CONCEPTS)

    model.eval()
    with torch.no_grad():
        for x, y_true in test_dl:
            x      = x.to(device)
            logits = model(x).cpu()
            y_pred = (torch.sigmoid(logits) > threshold).float()
            tp += (y_pred * y_true).sum(dim=0)
            fp += (y_pred * (1 - y_true)).sum(dim=0)
            fn += ((1 - y_pred) * y_true).sum(dim=0)

    prec, rec, f1 = compute_f1(tp, fp, fn)

    # Micro F1
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    _, _, micro_f1 = compute_f1(micro_tp, micro_fp, micro_fn)

    # Macro F1 — only over classes with at least 1 positive in test set
    support = (tp + fn)
    has_pos = support > 0
    macro_f1 = f1[has_pos].mean() if has_pos.any() else torch.tensor(0.0)

    print(f"\n── Dataset Metrics  (threshold={threshold}) ─────────────────────────────")
    print(f"  Micro F1 : {micro_f1.item():.4f}")
    print(f"  Macro F1 : {macro_f1.item():.4f}  "
          f"(over {has_pos.sum().item()} classes with support)")

    # Per-class table (sorted by F1 descending)
    rows = []
    for i, concept in enumerate(CONCEPTS):
        rows.append((concept, prec[i].item(), rec[i].item(), f1[i].item(),
                     int(support[i].item())))
    rows.sort(key=lambda r: -r[3])

    print(f"\n{'Concept':<22}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  {'Support':>7}")
    print("─" * 52)
    for concept, p, r, f, sup in rows:
        bar   = "█" * int(f * 20)
        flag  = "  ← low" if f < 0.30 and sup > 10 else ""
        print(f"{concept:<22}  {p:5.3f}  {r:5.3f}  {f:5.3f}  {sup:7d}  {bar}{flag}")


def spot_check(model: ChessConceptClassifier,
               threshold: float = DEFAULT_THRESHOLD) -> None:
    print(f"\n── Spot Checks  (threshold={threshold}) ──────────────────────────────────")
    from .board_encoder import fen_to_tensor

    all_pass = True
    for sc in SPOT_CHECKS:
        preds    = model.predict_concepts(sc["fen"], threshold=threshold)
        pred_set = {name for name, _ in preds}
        expected = sc["expect"]
        hits     = [c for c in expected if c in pred_set]
        miss     = [c for c in expected if c not in pred_set]

        status = "PASS" if len(miss) == 0 else "MISS"
        if status == "MISS":
            all_pass = False
        top3 = ", ".join(f"{n}({p:.2f})" for n, p in preds[:5])
        print(f"\n  {status} — {sc['name']}")
        print(f"       expected : {expected}")
        print(f"       missing  : {miss or '—'}")
        print(f"       top preds: {top3 or '(none)'}")

    if all_pass:
        print("\n  All spot checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data",           default=str(DEFAULT_DATA))
    parser.add_argument("--threshold",      type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--spot-check-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run training first: python -m src.chess_coach.ml.train")
        return

    model = load_model(ckpt_path, device)

    if not args.spot_check_only:
        evaluate_dataset(model, Path(args.data), device, args.threshold)

    spot_check(model, args.threshold)


if __name__ == "__main__":
    main()
