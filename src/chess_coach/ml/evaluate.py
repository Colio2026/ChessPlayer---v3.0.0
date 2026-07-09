#!/usr/bin/env python3
"""
evaluate.py  —  Evaluate the trained chess concept classifier
--------------------------------------------------------------
Loads data/classifier_best.pt, runs the test split, reports:
  - Micro / macro F1 scores
  - Per-concept precision / recall / F1 / support table
  - Qualitative spot-checks on 10 famous positions

Flags
-----
    --calibrate        Find per-class optimal thresholds on val set,
                       save to data/thresholds.json, then evaluate with them.
                       Run this once after every retrain.

    --threshold N      Global fallback threshold (default 0.4).
                       Ignored for classes that have a calibrated threshold.

Usage
-----
    # First run after training — calibrate then evaluate
    python -m src.chess_coach.ml.evaluate --calibrate

    # Subsequent evaluations (thresholds already saved)
    python -m src.chess_coach.ml.evaluate

    # Spot-checks only
    python -m src.chess_coach.ml.evaluate --spot-check-only

    # Different checkpoint
    python -m src.chess_coach.ml.evaluate --checkpoint data/classifier_last.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .classifier    import ChessConceptClassifier
from .dataset       import ChessConceptDataset
from .concept_vocab import CONCEPTS, NUM_CONCEPTS

DEFAULT_CHECKPOINT  = Path("data/classifier_best.pt")
DEFAULT_DATA        = Path("data/training_raw.jsonl")
DEFAULT_THRESHOLD   = 0.4
THRESHOLDS_PATH     = Path("data/thresholds.json")

# Famous positions and their expected concepts
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


# ── threshold helpers ─────────────────────────────────────────────────────────

def load_thresholds(path: Path = THRESHOLDS_PATH,
                    default: float = DEFAULT_THRESHOLD) -> torch.Tensor:
    """
    Load per-class thresholds from JSON.  Falls back to `default` for any
    missing class or if the file doesn't exist.
    Returns a float32 tensor of shape [NUM_CONCEPTS].
    """
    t = torch.full((NUM_CONCEPTS,), default, dtype=torch.float32)
    if path.exists():
        data = json.loads(path.read_text())
        for i, concept in enumerate(CONCEPTS):
            if concept in data:
                t[i] = data[concept]
    return t


def save_thresholds(thresholds: dict[str, float],
                    path: Path = THRESHOLDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(thresholds, indent=2))
    print(f"Thresholds saved → {path}")


# ── calibration ───────────────────────────────────────────────────────────────

def calibrate_thresholds(model: ChessConceptClassifier,
                          data_path: Path,
                          device: torch.device) -> dict[str, float]:
    """
    Sweep thresholds 0.05–0.95 per class on the *val* split and pick the
    value that maximises F1 for each class.  Classes with no positives in
    val default to 0.5.
    """
    print("\nCalibrating per-class thresholds on val split …")
    val_ds = ChessConceptDataset(data_path, split="val")
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    all_probs  = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            probs = torch.sigmoid(model(x.to(device))).cpu()
            all_probs.append(probs)
            all_labels.append(y)

    all_probs  = torch.cat(all_probs,  dim=0)   # [N, C]
    all_labels = torch.cat(all_labels, dim=0)   # [N, C]

    grid = torch.linspace(0.05, 0.95, 19)       # step ≈ 0.05
    thresholds: dict[str, float] = {}

    for i, concept in enumerate(CONCEPTS):
        probs_i  = all_probs[:, i]
        labels_i = all_labels[:, i]
        n_pos    = int(labels_i.sum().item())

        if n_pos == 0:
            thresholds[concept] = 0.50
            continue

        best_f1 = -1.0
        best_t  = 0.50
        for t in grid:
            preds = (probs_i >= t).float()
            tp = (preds * labels_i).sum()
            fp = (preds * (1 - labels_i)).sum()
            fn = ((1 - preds) * labels_i).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = (2 * prec * rec / (prec + rec + 1e-8)).item()
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t.item()

        thresholds[concept] = round(best_t, 2)

    # Print calibration summary
    print(f"\n{'Concept':<22}  {'Threshold':>9}  {'Val positives':>13}")
    print("─" * 48)
    for i, concept in enumerate(CONCEPTS):
        n_pos = int(all_labels[:, i].sum().item())
        mark  = "  (no val data)" if n_pos == 0 else ""
        print(f"{concept:<22}  {thresholds[concept]:>9.2f}  {n_pos:>13}{mark}")

    return thresholds


# ── evaluation ────────────────────────────────────────────────────────────────

def _f1(tp: torch.Tensor, fp: torch.Tensor,
        fn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1


def evaluate_dataset(model: ChessConceptClassifier,
                     data_path: Path,
                     device: torch.device,
                     thresholds: torch.Tensor) -> None:
    test_ds = ChessConceptDataset(data_path, split="test")
    if len(test_ds) == 0:
        print("No test examples — skipping dataset evaluation.")
        return

    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    t_dev   = thresholds.to(device)   # [C] on same device as model

    tp = torch.zeros(NUM_CONCEPTS)
    fp = torch.zeros(NUM_CONCEPTS)
    fn = torch.zeros(NUM_CONCEPTS)

    model.eval()
    with torch.no_grad():
        for x, y_true in test_dl:
            x      = x.to(device)
            probs  = torch.sigmoid(model(x))          # [B, C]
            y_pred = (probs >= t_dev).float().cpu()
            y_true = y_true.float()
            tp += (y_pred * y_true).sum(dim=0)
            fp += (y_pred * (1 - y_true)).sum(dim=0)
            fn += ((1 - y_pred) * y_true).sum(dim=0)

    prec, rec, f1 = _f1(tp, fp, fn)

    micro_prec, micro_rec, micro_f1 = _f1(tp.sum(), fp.sum(), fn.sum())
    support  = tp + fn
    has_pos  = support > 0
    macro_f1 = f1[has_pos].mean() if has_pos.any() else torch.tensor(0.0)

    # Summarise threshold source
    n_calibrated = int((thresholds != DEFAULT_THRESHOLD).sum().item())
    thresh_note  = (f"{n_calibrated}/{NUM_CONCEPTS} calibrated"
                    if n_calibrated else f"global={DEFAULT_THRESHOLD}")

    print(f"\n── Dataset Metrics  ({thresh_note}) ──────────────────────────────")
    print(f"  Micro F1 : {micro_f1.item():.4f}")
    print(f"  Macro F1 : {macro_f1.item():.4f}  "
          f"(over {int(has_pos.sum().item())} classes with support)")

    # Per-class table sorted by F1 descending
    rows = [(CONCEPTS[i], prec[i].item(), rec[i].item(), f1[i].item(),
             int(support[i].item()), thresholds[i].item())
            for i in range(NUM_CONCEPTS)]
    rows.sort(key=lambda r: -r[3])

    print(f"\n{'Concept':<22}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  "
          f"{'Support':>7}  {'Thresh':>6}")
    print("─" * 62)
    for concept, p, r, f, sup, t in rows:
        bar  = "█" * int(f * 20)
        flag = "  ← low" if f < 0.30 and sup > 10 else ""
        print(f"{concept:<22}  {p:5.3f}  {r:5.3f}  {f:5.3f}  "
              f"{sup:7d}  {t:6.2f}  {bar}{flag}")


def spot_check(model: ChessConceptClassifier,
               thresholds: torch.Tensor) -> None:
    print(f"\n── Spot Checks ────────────────────────────────────────────────────────")
    from .board_encoder import fen_to_tensor, move_to_tensor
    device = next(model.parameters()).device

    all_pass = True
    for sc in SPOT_CHECKS:
        board_t = fen_to_tensor(sc["fen"])
        move_t  = move_to_tensor(sc.get("move_uci", ""))
        x       = torch.cat([board_t, move_t]).unsqueeze(0).to(device)
        probs   = torch.sigmoid(model(x)).squeeze(0).cpu()
        t_vec   = thresholds
        pred_set = {CONCEPTS[i] for i in range(NUM_CONCEPTS) if probs[i] >= t_vec[i]}

        expected = sc["expect"]
        miss     = [c for c in expected if c not in pred_set]
        status   = "PASS" if not miss else "MISS"
        if miss:
            all_pass = False

        top5 = sorted(enumerate(probs.tolist()), key=lambda x: -x[1])[:5]
        top5_str = ", ".join(f"{CONCEPTS[i]}({p:.2f})" for i, p in top5)
        print(f"\n  {status} — {sc['name']}")
        print(f"       expected : {expected}")
        print(f"       missing  : {miss or '—'}")
        print(f"       top preds: {top5_str}")

    if all_pass:
        print("\n  All spot checks passed.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data",            default=str(DEFAULT_DATA))
    parser.add_argument("--threshold",       type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--calibrate",       action="store_true",
                        help="Find optimal per-class thresholds on val set and save them.")
    parser.add_argument("--spot-check-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run training first: python -m src.chess_coach.ml.train")
        return

    # Load model
    model = ChessConceptClassifier().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint: epoch={epoch}  val_loss={val_loss:.4f}")

    data_path = Path(args.data)

    # Calibrate thresholds if requested, then save
    if args.calibrate:
        cal = calibrate_thresholds(model, data_path, device)
        save_thresholds(cal)

    # Load thresholds (calibrated file if it exists, global fallback otherwise)
    thresholds = load_thresholds(default=args.threshold)

    if not args.spot_check_only:
        evaluate_dataset(model, data_path, device, thresholds)

    spot_check(model, thresholds)


if __name__ == "__main__":
    main()
