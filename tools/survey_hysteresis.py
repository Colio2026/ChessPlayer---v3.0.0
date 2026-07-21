#!/usr/bin/env python3
"""Survey the probability distribution of concept activations across diverse positions.

Samples N positions from training_raw.jsonl, runs predict_concepts(threshold=0.0)
on each, and prints a histogram of raw probabilities per concept — helping you
validate whether ACTIVATE_THRESHOLD=0.65 and HOLD_THRESHOLD=0.40 are appropriate
for Phase 4B's actual output distribution.

Usage
-----
    python tools/survey_hysteresis.py
    python tools/survey_hysteresis.py --n 2000 --checkpoint data/classifier_last.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

# Add repo root to path so src imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Survey concept probability distribution to validate hysteresis thresholds."
    )
    parser.add_argument("--checkpoint", default="data/classifier_best.pt")
    parser.add_argument("--jsonl",      default="data/training_raw.jsonl")
    parser.add_argument("--n",          type=int, default=1000,
                        help="Number of positions to sample (default: 1000)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}\nRun training first.")

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        sys.exit(f"JSONL not found: {jsonl_path}")

    from src.chess_coach.ml.classifier import ChessConceptClassifier
    from src.chess_coach.ml.concept_vocab import CONCEPTS, NUM_CONCEPTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd     = ckpt.get("state_dict", ckpt)
    is_phase5 = any(k.startswith("nnue_proj")    for k in sd)
    is_phase4 = any(k.startswith("spatial_proj") for k in sd) and not is_phase5
    model     = ChessConceptClassifier(phase4=is_phase4, phase5=is_phase5).to(device)
    model.load_state_dict(sd)
    model.eval()

    phase_tag = "Phase 5D" if is_phase5 else ("Phase 4B" if is_phase4 else "Phase 3")
    print(f"Loaded {ckpt_path.name}  ({phase_tag})")
    print(f"Sampling {args.n} positions from {jsonl_path.name} …\n")

    # Collect a random sample of FENs
    rng = random.Random(args.seed)
    with open(jsonl_path, "rb") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    sample = rng.sample(lines, min(args.n, len(lines)))

    all_probs: list[list[float]] = []   # [N, NUM_CONCEPTS]
    errors = 0
    for raw in sample:
        try:
            ex  = json.loads(raw.decode("utf-8", errors="replace"))
            fen = ex.get("fen", "")
            if not fen:
                continue
            pairs = model.predict_concepts(fen, threshold=0.0)
            prob_row = [0.0] * NUM_CONCEPTS
            prob_map = dict(pairs)
            for i, c in enumerate(CONCEPTS):
                prob_row[i] = prob_map.get(c, 0.0)
            all_probs.append(prob_row)
        except Exception:
            errors += 1

    n = len(all_probs)
    if n == 0:
        sys.exit("No valid positions sampled.")

    print(f"Sampled {n} positions  ({errors} errors)\n")

    # Compute percentiles per concept
    import statistics

    concept_stats = []
    for i, concept in enumerate(CONCEPTS):
        col    = [row[i] for row in all_probs]
        col_s  = sorted(col)
        p50    = col_s[n // 2]
        p75    = col_s[int(n * 0.75)]
        p90    = col_s[int(n * 0.90)]
        p95    = col_s[int(n * 0.95)]
        p99    = col_s[int(n * 0.99)]
        above_65 = sum(1 for v in col if v >= 0.65) / n * 100
        above_40 = sum(1 for v in col if v >= 0.40) / n * 100
        concept_stats.append((concept, p50, p75, p90, p95, p99, above_65, above_40))

    print(f"{'Concept':<22}  {'p50':>5}  {'p75':>5}  {'p90':>5}  {'p95':>5}  {'p99':>5}"
          f"  {'>0.65':>6}  {'>0.40':>6}")
    print("─" * 80)
    for concept, p50, p75, p90, p95, p99, a65, a40 in concept_stats:
        warn = ""
        if a65 < 1.0:
            warn = "  ← ACTIVATE threshold almost never reached"
        elif a65 > 30:
            warn = "  ← fires very frequently; consider raising ACTIVATE"
        print(f"{concept:<22}  {p50:5.3f}  {p75:5.3f}  {p90:5.3f}  {p95:5.3f}  {p99:5.3f}"
              f"  {a65:6.1f}%  {a40:6.1f}%{warn}")

    print(f"\nCurrent thresholds: ACTIVATE=0.65  HOLD=0.40")
    print("'>0.65' = fraction of positions where this concept would activate from cold.")
    print("'>0.40' = fraction of positions where an active concept would remain active.")
    print("\nIf most concepts show '<1%' in the '>0.65' column, ACTIVATE is too high.")
    print("If most concepts show '>20%' in the '>0.65' column, ACTIVATE is too low.")


if __name__ == "__main__":
    main()
