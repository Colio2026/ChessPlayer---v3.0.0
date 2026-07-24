#!/usr/bin/env python3
"""Export test-set positions as a labeled PGN for manual review.

Each position becomes a PGN "game" with headers describing whether it is a
True Positive, False Positive, or False Negative for a given concept.  Open
the output in any PGN viewer (Lichess, ChessBase, Arena, etc.) and step
through positions to determine whether the model is wrong or the label is wrong.

Usage
-----
    # FP + FN for all concepts, 3 positions each, sorted worst-F1 first
    python tools/export_truth_positions.py

    # 10 positions per category for specific concepts
    python tools/export_truth_positions.py --concepts x_ray,interference,pawn_chain --samples 10

    # Include true positives as well
    python tools/export_truth_positions.py --categories tp,fp,fn

    # Only the worst 10 concepts by test-set F1
    python tools/export_truth_positions.py --n 10

    # Custom output path
    python tools/export_truth_positions.py --output results/my_review.pgn

PGN headers per position
------------------------
    [Event   "x_ray - False Positive"]
    [Site    "https://lichess.org/analysis/<FEN>"]
    [Concept "x_ray"]
    [Type    "FP"]          TP / FP / FN
    [Score   "0.985"]       model probability for this concept
    [Thresh  "0.80"]        calibrated threshold
    [Labels  "outpost, bishop_pair"]
    [Fired   "outpost(1.00), x_ray(0.99), ..."]

Output
------
    results/truth_positions_YYYY-MM-DD.pgn  (default)
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chess

LICHESS_ANALYSIS = "https://lichess.org/analysis/"


def _lichess_url(fen: str) -> str:
    return LICHESS_ANALYSIS + quote(fen, safe="")


def _pgn_move(fen: str, move_uci: str) -> str:
    """Return a properly numbered SAN move for the given FEN and UCI move."""
    try:
        board = chess.Board(fen)
        move  = chess.Move.from_uci(move_uci)
        san   = board.san(move)
        n     = board.fullmove_number
        return f"{n}. {san}" if board.turn == chess.WHITE else f"{n}... {san}"
    except Exception:
        return move_uci


def _esc(s: str) -> str:
    """Escape backslashes and quotes for PGN string tokens."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _write_game(
    fh,
    concept:     str,
    result_type: str,       # "TP" | "FP" | "FN"
    fen:         str,
    move_uci:    str,
    score:       float,
    threshold:   float,
    labels:      list[str],
    fired:       list[tuple[str, float]],
    today:       str,
) -> None:
    type_label = {"TP": "True Positive", "FP": "False Positive", "FN": "False Negative"}[result_type]
    labels_str = ", ".join(labels)  if labels else "(none)"
    fired_str  = ", ".join(f"{c}({p:.2f})" for c, p in fired[:8]) if fired else "(none)"

    comment = (
        f"{result_type} | concept={concept} | score={score:.3f} | thresh={threshold:.2f}  "
        f"Labels: {labels_str}  Fired: {fired_str}"
    )
    if move_uci:
        comment += f"  Move: {move_uci}"

    fh.write(f'[Event "{_esc(concept)} - {type_label}"]\n')
    fh.write(f'[Site "{_esc(_lichess_url(fen))}"]\n')
    fh.write(f'[Date "{today}"]\n')
    fh.write( '[White "?"]\n')
    fh.write( '[Black "?"]\n')
    fh.write( '[Result "*"]\n')
    fh.write( '[SetUp "1"]\n')
    fh.write(f'[FEN "{_esc(fen)}"]\n')
    fh.write( '[Annotator "Coach Nimzowitsch Audit"]\n')
    fh.write(f'[Concept "{concept}"]\n')
    fh.write(f'[Type "{result_type}"]\n')
    fh.write(f'[Score "{score:.3f}"]\n')
    fh.write(f'[Thresh "{threshold:.2f}"]\n')
    fh.write(f'[Labels "{_esc(labels_str)}"]\n')
    fh.write(f'[Fired "{_esc(fired_str)}"]\n')
    fh.write(f'\n{{ {_esc(comment)} }}\n')
    if move_uci:
        pgn_mv = _pgn_move(fen, move_uci)
        fh.write(f"{pgn_mv} *\n")
    else:
        fh.write("*\n")
    fh.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export truth-position PGN for manual concept review."
    )
    parser.add_argument("--checkpoint",  default="data/classifier_best.pt",
                        help="Trained classifier checkpoint")
    parser.add_argument("--data",        default="data/training_raw.jsonl")
    parser.add_argument("--output",      default=None,
                        help="Output PGN path (default: results/truth_positions_YYYY-MM-DD.pgn)")
    parser.add_argument("--concepts",    default="all",
                        help="'all', or comma-separated concept names")
    parser.add_argument("--n",           type=int, default=None,
                        help="Export the N worst concepts by test-set F1 (overridden by --concepts)")
    parser.add_argument("--categories",  default="fp,fn",
                        help="Result types to export: any of tp,fp,fn (default: fp,fn)")
    parser.add_argument("--samples",     type=int, default=3,
                        help="Positions per concept per category (default: 3)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed — must match training split seed")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Data not found: {data_path}")

    today    = date.today().strftime("%Y.%m.%d")
    out_path = Path(args.output) if args.output else \
        Path(f"results/truth_positions_{date.today().strftime('%Y-%m-%d')}.pgn")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    categories = {c.strip().lower() for c in args.categories.split(",") if c.strip()}
    bad_cats   = categories - {"tp", "fp", "fn"}
    if bad_cats:
        sys.exit(f"Unknown categories: {bad_cats}  (valid: tp, fp, fn)")

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
    print(f"Loaded {ckpt_path.name}  epoch={epoch}")

    thresholds    = load_thresholds(default=0.40)
    thresholds_np = thresholds.numpy()

    ds = ChessConceptDataset(data_path, split="test", seed=args.seed,
                             phase4=is_phase4, phase5=is_phase5)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    print(f"Running inference on {len(ds):,} test examples ...")
    all_probs:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for x, hist, seq_len, y in dl:
            probs = torch.sigmoid(
                model(x.to(device), hist.to(device), seq_len.to(device))
            ).cpu()
            all_probs.append(probs)
            all_labels.append(y)

    all_probs_t  = torch.cat(all_probs,  dim=0)   # [N, C]
    all_labels_t = torch.cat(all_labels, dim=0)   # [N, C]
    N            = len(all_probs_t)
    preds        = all_probs_t.numpy() >= thresholds_np[None, :]
    probs_np     = all_probs_t.numpy()
    labels_np    = all_labels_t.numpy()
    print(f"Inference complete: {N:,} examples\n")

    # -- Per-concept F1 for sorting / filtering --------------------------------
    def _f1(ci: int) -> float:
        tp = int(( preds[:, ci] &  (labels_np[:, ci] == 1)).sum())
        fp = int(( preds[:, ci] & ~(labels_np[:, ci] == 1)).sum())
        fn = int((~preds[:, ci] &  (labels_np[:, ci] == 1)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # -- Resolve concept list -------------------------------------------------
    if args.concepts.strip().lower() != "all":
        concept_list = [c.strip() for c in args.concepts.split(",") if c.strip()]
        bad_c = [c for c in concept_list if c not in CONCEPTS]
        if bad_c:
            sys.exit(f"Unknown concepts: {bad_c}")
    else:
        # Sort by F1 ascending (worst first) unless user requested specific concepts
        concept_list = sorted(CONCEPTS, key=lambda c: _f1(CONCEPTS.index(c)))
        if args.n is not None:
            concept_list = concept_list[:args.n]

    rng = random.Random(args.seed)

    def _sample(indices: list[int], k: int) -> list[int]:
        return rng.sample(indices, min(k, len(indices)))

    total_written = 0
    print(f"Writing positions for {len(concept_list)} concepts "
          f"({'+'.join(sorted(categories)).upper()}, "
          f"{args.samples} per category) ...")

    with open(out_path, "w", encoding="utf-8") as fh:
        for concept in concept_list:
            ci  = CONCEPTS.index(concept)
            thr = float(thresholds_np[ci])

            buckets: dict[str, list[int]] = {
                "tp": [i for i in range(N) if  preds[i, ci] and labels_np[i, ci] == 1],
                "fp": [i for i in range(N) if  preds[i, ci] and labels_np[i, ci] == 0],
                "fn": [i for i in range(N) if not preds[i, ci] and labels_np[i, ci] == 1],
            }
            concept_written = 0

            for cat in ("tp", "fp", "fn"):
                if cat not in categories:
                    continue
                for idx in _sample(buckets[cat], args.samples):
                    try:
                        ex = ds._read_example(idx)
                    except Exception:
                        continue
                    fen      = ex.get("fen", "")
                    move_uci = ex.get("move_uci", "")
                    if not fen:
                        continue

                    score  = float(probs_np[idx, ci])
                    labels = [CONCEPTS[j] for j in range(NUM_CONCEPTS) if labels_np[idx, j] == 1]
                    fired  = sorted(
                        [(CONCEPTS[j], float(probs_np[idx, j]))
                         for j in range(NUM_CONCEPTS)
                         if probs_np[idx, j] >= thresholds_np[j]],
                        key=lambda kv: -kv[1],
                    )
                    _write_game(fh, concept, cat.upper(), fen, move_uci,
                                score, thr, labels, fired, today)
                    total_written  += 1
                    concept_written += 1

            f1_val = _f1(ci)
            print(f"  {concept:<22}  F1={f1_val:.3f}  {concept_written} positions written")

    print(f"\n{total_written} positions total -> {out_path}")
    print("Open in Lichess, ChessBase, or any PGN reader.")
    print("Each game header shows Concept / Type (TP/FP/FN) / Score / Labels / Fired.")


if __name__ == "__main__":
    main()
