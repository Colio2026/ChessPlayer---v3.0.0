#!/usr/bin/env python3
"""
inspect_weights.py  --  Diagnostic tool for the chess concept classifier weights
--------------------------------------------------------------------------------
Loads a checkpoint and prints:
  1. Per-layer weight statistics (mean, std, norm)
  2. Output layer norm per concept — reveals which concepts have strong/weak signal
  3. Dead neuron estimate in hidden layers
  4. BatchNorm gamma values (near-zero = layer is being suppressed)

Usage
-----
    python tools/inspect_weights.py
    python tools/inspect_weights.py --checkpoint data/classifier_last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chess_coach.ml.classifier import ChessConceptClassifier
from src.chess_coach.ml.concept_vocab import CONCEPTS, NUM_CONCEPTS

DEFAULT_CHECKPOINT = Path("data/classifier_best.pt")


def inspect(ckpt_path: Path) -> None:
    device = torch.device("cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd     = ckpt.get("state_dict", ckpt)
    is_phase5 = any(k.startswith("nnue_proj")    for k in sd)
    is_phase4 = any(k.startswith("spatial_proj") for k in sd) and not is_phase5
    model  = ChessConceptClassifier(phase4=is_phase4, phase5=is_phase5)
    model.load_state_dict(sd)
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Epoch {epoch}  |  val_loss {val_loss:.4f}\n")

    sd = model.state_dict()

    # ── 1. Per-layer weight statistics ────────────────────────────────────────
    print("=" * 65)
    print("LAYER WEIGHT STATISTICS")
    print("=" * 65)
    print(f"{'Layer':<35} {'Shape':>16}  {'Mean':>7}  {'Std':>7}  {'L2 norm':>9}")
    print("-" * 65)

    layer_map = [
        ("net.0.weight",  "L1  Linear  weight"),
        ("net.0.bias",    "L1  Linear  bias  "),
        ("net.1.weight",  "L1  BatchNorm gamma"),
        ("net.1.bias",    "L1  BatchNorm beta "),
        ("net.4.weight",  "L2  Linear  weight"),
        ("net.4.bias",    "L2  Linear  bias  "),
        ("net.5.weight",  "L2  BatchNorm gamma"),
        ("net.5.bias",    "L2  BatchNorm beta "),
        ("net.8.weight",  "Out Linear  weight"),
        ("net.8.bias",    "Out Linear  bias  "),
    ]

    for key, label in layer_map:
        if key not in sd:
            continue
        w = sd[key].float()
        print(f"  {label:<33} {str(tuple(w.shape)):>16}  "
              f"{w.mean().item():>7.4f}  {w.std().item():>7.4f}  "
              f"{w.norm().item():>9.2f}")

    # ── 2. Output layer norm per concept ─────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("OUTPUT LAYER -- Weight L2-norm per concept")
    print("(low norm = weak signal, model barely learned this concept)")
    print("=" * 65)

    W_out = sd["net.8.weight"].float()   # [NUM_CONCEPTS, hidden]
    b_out = sd["net.8.bias"].float()     # [NUM_CONCEPTS]

    norms = W_out.norm(dim=1)            # one norm per concept
    order = norms.argsort()              # weakest first

    max_norm = norms.max().item()
    print(f"\n{'Concept':<25}  {'Norm':>7}  {'Bar'}")
    print("-" * 55)
    for idx in order:
        concept = CONCEPTS[idx]
        n       = norms[idx].item()
        bar     = "#" * int(n / max_norm * 30)
        flag    = "  <- WEAK" if n < 0.3 * max_norm else ""
        print(f"  {concept:<23}  {n:>7.3f}  {bar}{flag}")

    # ── 3. Dead neuron estimate ───────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("DEAD NEURON ESTIMATE  (BatchNorm gamma near zero)")
    print("(gamma < 0.1 means the layer learned to suppress that neuron)")
    print("=" * 65)

    for bn_key, label in [("net.1.weight", "Layer 1 BatchNorm"),
                           ("net.5.weight", "Layer 2 BatchNorm")]:
        if bn_key not in sd:
            continue
        gamma = sd[bn_key].float().abs()
        dead  = int((gamma < 0.1).sum().item())
        weak  = int((gamma < 0.3).sum().item())
        total = gamma.shape[0]
        print(f"\n  {label}  ({total} neurons)")
        print(f"    gamma < 0.1 (dead)  : {dead:>4} / {total}  ({100*dead/total:.1f}%)")
        print(f"    gamma < 0.3 (weak)  : {weak:>4} / {total}  ({100*weak/total:.1f}%)")
        print(f"    gamma mean          : {gamma.mean().item():.4f}")
        print(f"    gamma std           : {gamma.std().item():.4f}")

    # ── 4. What you can tune ─────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("TUNING GUIDE")
    print("=" * 65)
    print("""
  PROBLEM                    LIKELY CAUSE              FIX
  -------------------------------------------------------------
  Concept norm is very low   Too few training examples  More data for that concept
                             Keywords not matching      Expand keyword list
                             Concept too abstract       Fold into a related concept

  Many dead neurons (>20%)   Dropout too high           Lower dropout (0.3)
                             LR too high initially      Reduce warmup LR

  Weight std very large      Overfitting                Increase dropout / weight_decay
  (L2 norm > 30 per neuron)                             More diverse training data

  Weight std very small      Underfitting               Larger hidden layers
  (L2 norm < 5 per neuron)   LR too low                 Increase LR or train longer

  Concepts bleed into each   Keywords too broad         Tighten keywords
  other (low precision)      Concepts overlap           Consider merging concepts
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect trained classifier weights"
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    inspect(ckpt_path)


if __name__ == "__main__":
    main()
