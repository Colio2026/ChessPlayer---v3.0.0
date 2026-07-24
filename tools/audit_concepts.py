#!/usr/bin/env python3
"""Concept false-positive / false-negative audit for the chess concept classifier.

Runs the trained classifier on the *test split* and identifies which concepts
misfire most often, then samples actual FEN positions so a human can inspect
whether the model is wrong or the label is wrong.

Usage
-----
    # Audit the 5 concepts with the worst precision (most false positives)
    python tools/audit_concepts.py

    # Audit specific concepts
    python tools/audit_concepts.py --concepts bishop_pair,clearance,initiative

    # Show more samples per concept
    python tools/audit_concepts.py --samples 20

    # Use a different checkpoint
    python tools/audit_concepts.py --checkpoint data/classifier_last.pt

Output
------
For each audited concept:
  - Precision / Recall / F1 on the test set
  - Up to N false-positive positions (model fired, label says no)
  - Up to N false-negative positions (label says yes, model missed)
  - Lichess analysis link for every position

Interpreting results
--------------------
False positive: model fires concept but label=0.  Could mean:
  (a) model is wrong — the concept is not present
  (b) label is wrong — the concept IS present but wasn't annotated

False negative: label=1 but model didn't fire.  Could mean:
  (a) model is wrong — the concept IS present but model missed it
  (b) label is wrong — the concept is not actually present in this position

A pattern of (b) cases reveals systematic labeling noise, which is a different
problem from model error.  Look for patterns: does the FP cluster around a
specific opening, piece structure, or game phase?
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from urllib.parse import quote

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


LICHESS_ANALYSIS = "https://lichess.org/analysis/"


def _lichess_url(fen: str) -> str:
    return LICHESS_ANALYSIS + quote(fen, safe="")


def _move_description(board_before, move) -> str:
    """Human-readable description of a move's character."""
    parts = []
    if board_before.is_capture(move):
        parts.append("capture")
    if board_before.gives_check(move):
        parts.append("check")
    piece = board_before.piece_at(move.from_square)
    if piece and piece.piece_type == 1:  # PAWN
        parts.append("pawn")
    return "+".join(parts) if parts else "quiet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit concept false positives/negatives on the test split."
    )
    parser.add_argument("--checkpoint", default="data/classifier_best.pt",
                        help="Path to trained checkpoint")
    parser.add_argument("--data",       default="data/training_raw.jsonl",
                        help="Path to training JSONL")
    parser.add_argument("--concepts",   default="",
                        help="Comma-separated concepts to audit (default: top 5 by FP rate)")
    parser.add_argument("--n",          type=int, default=5,
                        help="Number of concepts to audit (ignored if --concepts set)")
    parser.add_argument("--samples",    type=int, default=10,
                        help="Max positions to show per concept per error type")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for position sampling (must match training split seed)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Override global threshold (default: use data/thresholds.json)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"JSONL not found: {data_path}")

    from src.chess_coach.ml.classifier    import ChessConceptClassifier
    from src.chess_coach.ml.concept_vocab import CONCEPTS, NUM_CONCEPTS
    from src.chess_coach.ml.dataset       import ChessConceptDataset
    from src.chess_coach.ml.evaluate      import load_thresholds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt      = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd        = ckpt.get("state_dict", ckpt)
    is_phase5 = any(k.startswith("nnue_proj")    for k in sd)
    is_phase4 = any(k.startswith("spatial_proj") for k in sd) and not is_phase5
    model     = ChessConceptClassifier(phase4=is_phase4, phase5=is_phase5).to(device)
    model.load_state_dict(sd)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    phase = "Phase 5" if is_phase5 else ("Phase 4B" if is_phase4 else "Phase 3")
    print(f"Loaded {ckpt_path.name}  epoch={epoch}  ({phase})")

    thresholds = load_thresholds(default=args.threshold or 0.40)
    print(f"Thresholds: {int((thresholds != 0.40).sum())}/{NUM_CONCEPTS} calibrated")

    print(f"\nLoading test split (seed={args.seed}) ...")
    ds = ChessConceptDataset(data_path, split="test", seed=args.seed,
                             phase4=is_phase4, phase5=is_phase5)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    # --- Full test pass: collect probs and labels with dataset indices ---
    print(f"Running inference on {len(ds):,} test examples ...")
    all_probs  = []   # list of (batch_size, NUM_CONCEPTS)
    all_labels = []   # list of (batch_size, NUM_CONCEPTS)
    t_dev = thresholds.to(device)

    with torch.no_grad():
        for x, hist, seq_len, y in dl:
            probs = torch.sigmoid(
                model(x.to(device), hist.to(device), seq_len.to(device))
            ).cpu()
            all_probs.append(probs)
            all_labels.append(y)

    all_probs  = torch.cat(all_probs,  dim=0)   # [N, C]
    all_labels = torch.cat(all_labels, dim=0)   # [N, C]
    N = len(all_probs)
    print(f"Inference complete: {N:,} examples\n")

    thresholds_np = thresholds.numpy()
    preds = (all_probs.numpy() >= thresholds_np[None, :])   # [N, C] bool

    # --- Compute per-concept stats ---
    concept_stats = []
    for ci, concept in enumerate(CONCEPTS):
        tp = int(( preds[:, ci] &  (all_labels[:, ci] == 1).numpy()).sum())
        fp = int(( preds[:, ci] & ~(all_labels[:, ci] == 1).numpy()).sum())
        fn = int((~preds[:, ci] &  (all_labels[:, ci] == 1).numpy()).sum())
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        concept_stats.append({
            "concept": concept, "ci": ci,
            "tp": tp, "fp": fp, "fn": fn, "support": support,
            "prec": prec, "rec": rec, "f1": f1,
        })

    # --- Select concepts to audit ---
    if args.concepts:
        chosen = [c.strip() for c in args.concepts.split(",") if c.strip()]
        invalid = [c for c in chosen if c not in CONCEPTS]
        if invalid:
            sys.exit(f"Unknown concept(s): {invalid}\nValid: {CONCEPTS}")
        audit_stats = [s for s in concept_stats if s["concept"] in chosen]
    else:
        # Sort by number of false positives descending, only concepts that have some
        eligible = [s for s in concept_stats if s["fp"] > 0]
        eligible.sort(key=lambda s: (-s["fp"], s["prec"]))
        audit_stats = eligible[:args.n]

    # --- FP/FN index lookup helpers ---
    def _fp_indices(ci: int) -> list[int]:
        return [i for i in range(N) if preds[i, ci] and all_labels[i, ci] == 0]

    def _fn_indices(ci: int) -> list[int]:
        return [i for i in range(N) if not preds[i, ci] and all_labels[i, ci] == 1]

    rng = random.Random(args.seed)

    def _sample_and_read(indices: list[int], k: int) -> list[dict]:
        sample = rng.sample(indices, min(k, len(indices)))
        results = []
        for idx in sample:
            try:
                ex  = ds._read_example(idx)
                fen = ex.get("fen", "")
                if not fen:
                    continue
                prob_row   = all_probs[idx].tolist()
                label_row  = all_labels[idx].tolist()
                top_preds  = sorted(
                    [(CONCEPTS[i], prob_row[i]) for i in range(NUM_CONCEPTS)
                     if prob_row[i] >= thresholds_np[i]],
                    key=lambda kv: -kv[1]
                )
                true_labels = [CONCEPTS[i] for i in range(NUM_CONCEPTS) if label_row[i] == 1]
                results.append({
                    "idx":        idx,
                    "fen":        fen,
                    "move_uci":   ex.get("move_uci", ""),
                    "top_preds":  top_preds,
                    "true_labels": true_labels,
                })
            except Exception as e:
                results.append({"idx": idx, "error": str(e)})
        return results

    # --- Print report ---
    print("=" * 78)
    print("CONCEPT AUDIT REPORT")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Test split : {N:,} examples  (seed={args.seed})")
    print("=" * 78)

    for s in audit_stats:
        concept = s["concept"]
        ci      = s["ci"]
        print(f"\n{'-' * 78}")
        print(f"CONCEPT: {concept.upper()}")
        print(f"  Precision : {s['prec']:.3f}   Recall : {s['rec']:.3f}   F1 : {s['f1']:.3f}")
        print(f"  TP={s['tp']}  FP={s['fp']}  FN={s['fn']}  Support={s['support']}")
        print(f"  Threshold : {thresholds_np[ci]:.2f}")

        # False positives
        fp_idx = _fp_indices(ci)
        print(f"\n  -- False Positives ({len(fp_idx):,} total, showing {min(args.samples, len(fp_idx))}) --")
        print(f"     Model fired '{concept}' but label=0.  Is the concept present?")
        for pos in _sample_and_read(fp_idx, args.samples):
            if "error" in pos:
                print(f"    [ERR] {pos['error']}")
                continue
            prob_for_concept = all_probs[pos["idx"], ci].item()
            print(f"\n    FP  p={prob_for_concept:.3f}")
            print(f"    FEN : {pos['fen']}")
            print(f"    URL : {_lichess_url(pos['fen'])}")
            if pos["move_uci"]:
                print(f"    Move: {pos['move_uci']}")
            preds_str = ", ".join(f"{c}({p:.2f})" for c, p in pos["top_preds"][:6])
            print(f"    Model fired   : {preds_str or '(none)'}")
            print(f"    Labels say    : {', '.join(pos['true_labels']) or '(none)'}")

        # False negatives
        fn_idx = _fn_indices(ci)
        print(f"\n  -- False Negatives ({len(fn_idx):,} total, showing {min(args.samples, len(fn_idx))}) --")
        print(f"     Label=1 but model didn't fire '{concept}' at threshold {thresholds_np[ci]:.2f}.")
        for pos in _sample_and_read(fn_idx, args.samples):
            if "error" in pos:
                print(f"    [ERR] {pos['error']}")
                continue
            prob_for_concept = all_probs[pos["idx"], ci].item()
            print(f"\n    FN  p={prob_for_concept:.3f}  (threshold={thresholds_np[ci]:.2f})")
            print(f"    FEN : {pos['fen']}")
            print(f"    URL : {_lichess_url(pos['fen'])}")
            if pos["move_uci"]:
                print(f"    Move: {pos['move_uci']}")
            preds_str = ", ".join(f"{c}({p:.2f})" for c, p in pos["top_preds"][:6])
            print(f"    Model fired   : {preds_str or '(none)'}")
            print(f"    Labels say    : {', '.join(pos['true_labels']) or '(none)'}")

    print(f"\n{'=' * 78}")
    print("Overall test metrics for audited concepts:")
    print(f"{'Concept':<22}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  {'FP':>6}  {'FN':>6}  {'Support':>7}")
    print("-" * 70)
    for s in audit_stats:
        print(f"{s['concept']:<22}  {s['prec']:5.3f}  {s['rec']:5.3f}  {s['f1']:5.3f}"
              f"  {s['fp']:6d}  {s['fn']:6d}  {s['support']:7d}")
    print()


if __name__ == "__main__":
    main()
