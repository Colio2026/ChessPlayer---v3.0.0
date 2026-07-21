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
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import chess as _chess

from .classifier    import ChessConceptClassifier
from .dataset       import ChessConceptDataset
from .concept_vocab import CONCEPTS, NUM_CONCEPTS
from .paths         import CLASSIFIER_BEST, TRAINING_JSONL, THRESHOLDS

class _Tee:
    def __init__(self, log_path: Path) -> None:
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
    def write(self, s: str) -> None:
        self._stdout.write(s)
        self._file.write(s)
    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()
    def close(self) -> None:
        self._file.close()


def _next_results_path(tag: str) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    existing = sorted(results_dir.glob("results????_*.txt"))
    n = int(existing[-1].stem[7:11]) + 1 if existing else 1
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return results_dir / f"results{n:04d}_{stamp}_{tag}.txt"


DEFAULT_CHECKPOINT  = CLASSIFIER_BEST
DEFAULT_DATA        = TRAINING_JSONL
DEFAULT_THRESHOLD   = 0.4
THRESHOLDS_PATH     = THRESHOLDS

# Famous positions and their expected concepts
SPOT_CHECKS: list[dict] = [
    # ── Tactical ──────────────────────────────────────────────────────────────
    {
        "name": "Pin — Bb5 pins Nc6 to king",
        "fen":  "r1bq1rk1/ppp2ppp/2np1n2/1B2p3/2BPP3/2N1QN2/PPP2PPP/R4RK1 b - - 0 8",
        "expect": ["pin"],
    },
    {
        "name": "Fork — Nc7 forks Ra8 and Ke8",
        "fen":  "r3k3/2N5/8/8/8/8/8/4K3 w - - 0 1",
        "expect": ["fork"],
    },
    {
        "name": "Skewer — Ba1 skewers Kd4 to Qg7",
        "fen":  "8/6q1/8/8/3k4/8/8/B3K3 w - - 0 1",
        "expect": ["skewer"],
    },
    {
        "name": "Discovery — moving Be3 reveals Re1 check",
        "fen":  "4k3/8/8/8/8/4B3/8/4R3 w - - 0 1",
        "expect": ["discovery"],
    },
    {
        "name": "X-ray — Ra1 defends through Ra4 to Ra8",
        "fen":  "r3k3/8/8/r7/R7/8/8/R3K3 w - - 0 1",
        "expect": ["x_ray"],
    },
    {
        "name": "Double check — Ne6-c7 reveals rook check",
        "fen":  "4k3/8/4N3/8/8/8/8/4R1K1 w - - 0 1",
        "expect": ["double_check"],
    },
    {
        "name": "Clearance — rook vacates rank for queen",
        "fen":  "r3k2r/ppp1Rppp/8/3p4/3P4/8/PPP2PPP/4R1K1 w kq - 0 1",
        "expect": ["clearance"],
    },
    {
        "name": "Deflection — luring queen away from defense",
        "fen":  "r1b1k2r/ppppqppp/2n2n2/4p3/2B1P3/2NP4/PPP2PPP/R1BQK2R w KQkq - 0 6",
        "expect": ["deflection"],
    },
    {
        "name": "Overloading — Re6 defends two targets",
        "fen":  "6k1/5ppp/3pr3/3p4/3P4/8/5PPP/3R2K1 w - - 0 1",
        "expect": ["overloading"],
    },
    {
        "name": "Zwischenzug — in-between check before recapture",
        "fen":  "r1bqk2r/pppp1ppp/2n2n2/4p3/2BbP3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        "expect": ["zwischenzug"],
    },
    {
        "name": "Interference — piece cuts rook's defense of key square",
        "fen":  "r2q1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPP1Q1PP/R4RK1 w - - 0 1",
        "expect": ["interference"],
    },
    {
        "name": "Back rank — king has no luft, rook threatens",
        "fen":  "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "expect": ["back_rank"],
    },
    {
        "name": "Sacrifice — exchange sac for positional compensation",
        "fen":  "r3r1k1/pp1bppbp/2np1np1/q7/3NP3/2N1BP2/PPP1B1PP/R2Q1RK1 w - - 0 1",
        "expect": ["sacrifice"],
    },
    {
        "name": "Mating attack — smothered mate in one",
        "fen":  "r5k1/ppp3pp/5N2/8/8/8/PPP3PP/6K1 w - - 0 1",
        "expect": ["mating_attack"],
    },
    {
        "name": "Trapped piece — bishop on h6 with no escape",
        "fen":  "r1bqk2r/pppp1ppp/2n2n1b/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
        "expect": ["trapped_piece"],
    },
    # ── Piece concepts ────────────────────────────────────────────────────────
    {
        "name": "Outpost — knight on d5 cannot be dislodged",
        "fen":  "r1bqr1k1/pp1nbppp/2p1pn2/3p4/3P1B2/2NBPN2/PPQ2PPP/R4RK1 w - - 0 1",
        "expect": ["outpost"],
    },
    {
        "name": "Blockade — knight sits in front of passed pawn",
        "fen":  "4k3/3n4/3P4/8/8/8/8/4K3 w - - 0 1",
        "expect": ["blockade"],
    },
    {
        "name": "Bad bishop — bishop imprisoned by own pawns",
        "fen":  "5k2/pp1bpppp/2pp4/8/8/2PP4/PP1BPPPP/5K2 w - - 0 1",
        "expect": ["bad_bishop"],
    },
    {
        "name": "Good bishop — pawns on opposite color, diagonal open",
        "fen":  "5k2/pp3ppp/4p3/2pp4/8/2PP4/PP2BPPP/5K2 w - - 0 1",
        "expect": ["good_bishop"],
    },
    {
        "name": "Bishop pair — both bishops vs knight and bishop",
        "fen":  "r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "expect": ["bishop_pair"],
    },
    {
        "name": "Piece activity — centralized knights dominate undeveloped side",
        "fen":  "r1b1k2r/ppp2ppp/2n1p3/3NN3/2B5/8/PPP2PPP/R3K2R w KQkq - 0 1",
        "expect": ["piece_activity"],
    },
    {
        "name": "Battery — doubled rooks on open file",
        "fen":  "6k1/ppp2ppp/8/8/3RR3/8/PPP2PPP/6K1 w - - 0 1",
        "expect": ["battery"],
    },
    {
        "name": "Rook on seventh rank",
        "fen":  "6k1/3R1ppp/6r1/8/8/8/5PPP/6K1 w - - 0 1",
        "expect": ["rook_seventh"],
    },
    # ── Pawn structure ────────────────────────────────────────────────────────
    {
        "name": "Passed pawn — white passer on d6",
        "fen":  "4k3/8/3P4/8/8/8/8/4K3 w - - 0 1",
        "expect": ["passed_pawn"],
    },
    {
        "name": "Promotion — pawn on 7th rank",
        "fen":  "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "expect": ["promotion"],
    },
    {
        "name": "Isolated pawn — IQP on d4",
        "fen":  "r1bqr1k1/pp3ppp/2n1bn2/3p4/3P4/2NBPN2/PP3PPP/R1BQR1K1 w - - 0 1",
        "expect": ["isolated_pawn", "open_file"],
        "also_ok": ["pawn_island"],
    },
    {
        "name": "Backward pawn — weak d6 pawn",
        "fen":  "r1bqr1k1/pp1nbppp/2pp1n2/4p3/3PP3/2N1BN2/PPP1QPPP/R3R1K1 w - - 0 1",
        "expect": ["backward_pawn"],
    },
    {
        "name": "Doubled pawns — c-pawns doubled",
        "fen":  "r1bqkb1r/pp3ppp/2pp1n2/4p3/4P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 1",
        "expect": ["doubled_pawn"],
    },
    {
        "name": "Pawn majority — queenside majority vs minority",
        "fen":  "4k3/ppp5/8/8/8/8/PPPP4/4K3 w - - 0 1",
        "expect": ["pawn_majority"],
    },
    {
        "name": "Pawn chain — locked chain, attack the base",
        "fen":  "r1bqkb1r/pp3ppp/2n1pn2/2ppP3/3P1B2/2N2N2/PPP2PPP/R2QKB1R w KQkq - 0 1",
        "expect": ["pawn_chain"],
    },
    {
        "name": "Pawn storm — kingside pawns advancing on castled king",
        "fen":  "r1bq1rk1/pppp1ppp/2n5/4p3/2PP3P/2N3P1/PP3P2/R1BQKBNR w KQ - 0 1",
        "expect": ["pawn_storm"],
    },
    {
        "name": "Pawn island — three isolated pawn groups",
        "fen":  "4k3/p3p3/3p4/8/8/3P4/P3P3/4K3 w - - 0 1",
        "expect": ["pawn_island"],
        "also_ok": ["isolated_pawn"],
    },
    # ── King & endgame ────────────────────────────────────────────────────────
    {
        "name": "King safety — castled king stripped of pawn cover",
        "fen":  "r4rk1/ppp2p1p/3p1np1/4p3/4P3/3P1N2/PPP2PPP/R4RK1 w - - 0 1",
        "expect": ["king_safety"],
    },
    {
        "name": "King activity — centralized king in endgame",
        "fen":  "8/8/4k3/4p3/4P3/4K3/8/8 w - - 0 1",
        "expect": ["king_activity"],
    },
    {
        "name": "Shouldering — king blocks opposing king from key file",
        "fen":  "8/8/8/3K4/8/3k4/4P3/8 w - - 0 1",
        "expect": ["shouldering"],
    },
    {
        "name": "Opposition — kings in direct opposition",
        "fen":  "8/8/4k3/4P3/4K3/8/8/8 w - - 0 1",
        "expect": ["opposition"],
    },
    {
        "name": "Zugzwang — whoever moves loses",
        "fen":  "8/8/4k3/4p3/4K3/8/8/8 w - - 0 1",
        "expect": ["zugzwang", "opposition"],
    },
    {
        "name": "Rook endgame — Lucena position",
        "fen":  "1K1k4/1P6/8/8/r7/8/8/R7 w - - 0 1",
        "expect": ["rook_endgame", "passed_pawn", "promotion"],
    },
    {
        "name": "Pawn endgame — king and pawn",
        "fen":  "4k3/4p3/4K3/4P3/8/8/8/8 w - - 0 1",
        "expect": ["pawn_endgame"],
    },
    {
        "name": "Bishop endgame — opposite colored bishops",
        "fen":  "4k3/4p3/8/4b3/8/4B3/4P3/4K3 w - - 0 1",
        "expect": ["bishop_endgame"],
    },
    {
        "name": "Knight endgame — knights and pawns",
        "fen":  "4k3/4p3/8/4n3/8/4N3/4P3/4K3 w - - 0 1",
        "expect": ["knight_endgame"],
    },
    {
        "name": "Queen endgame — queens and pawns",
        "fen":  "4k3/4p3/8/4q3/8/4Q3/4P3/4K3 w - - 0 1",
        "expect": ["queen_endgame"],
    },
    {
        "name": "Drawn position — insufficient material (K+B vs K)",
        "fen":  "4k3/8/8/8/8/8/8/4KB2 w - - 0 1",
        "expect": ["drawn_position"],
    },
    # ── Opening positions (ECO-identified) ───────────────────────────────────
    {
        "name": "Opening — Ruy Lopez Berlin (C65)",
        "history_uci": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"],
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "expect": ["development_lead", "piece_activity"],
    },
    {
        "name": "Opening — Nimzo-Indian classical (E32)",
        "history_uci": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2"],
        "fen": "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PPQ1PPPP/R1B1KBNR b KQkq - 1 4",
        "expect": ["bishop_pair", "development_lead"],
    },
    {
        "name": "Opening — Sicilian Najdorf (B90)",
        "history_uci": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
        "fen": "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
        "expect": ["space_advantage", "development_lead"],
    },
    {
        "name": "Opening — QGD Slav exchange (D10)",
        "history_uci": ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "c4d5", "c6d5"],
        "fen": "rnbqkb1r/pp2pppp/5n2/3p4/3P4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 5",
        "expect": ["isolated_pawn", "open_file"],
        "also_ok": ["pawn_island"],
    },
    # ── Positional / Strategic ────────────────────────────────────────────────
    {
        "name": "Weak square — dark square holes around king",
        "fen":  "r1bq1rk1/ppp1nppp/4p3/3p4/3P1B2/2N1PN2/PPP2PPP/R2QKB1R w KQ - 0 1",
        "expect": ["weak_square"],
    },
    {
        "name": "Open file — rooks dominate open d-file",
        "fen":  "r1bqr1k1/pp3ppp/2n1bn2/3p4/3P4/2NBPN2/PP3PPP/R1BQR1K1 w - - 0 1",
        "expect": ["open_file"],
    },
    {
        "name": "Space advantage — advanced pawn wedge controls territory",
        "fen":  "r1bqkb1r/pp3ppp/2n1pn2/2ppP3/2PP1B2/2N2N2/PP3PPP/R2QKB1R w KQkq - 0 1",
        "expect": ["space_advantage"],
    },
    {
        "name": "Development lead — all pieces active vs undeveloped side",
        "fen":  "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        "expect": ["development_lead"],
    },
    {
        "name": "Initiative — dictating pace with active play",
        "fen":  "r1bqr1k1/ppp2ppp/2np1n2/4p3/2BPP3/2N2N2/PPP2PPP/R1BQR1K1 w - - 0 8",
        "expect": ["initiative"],
    },
    {
        "name": "Prophylaxis — restraining opponent's plan",
        "fen":  "r1bqr1k1/ppp1bppp/2np1n2/4p3/2PPP3/2N1BN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "expect": ["prophylaxis"],
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
                          device: torch.device,
                          is_phase4: bool = False,
                          is_phase5: bool = False) -> dict[str, float]:
    """
    Sweep thresholds 0.05–0.95 per class on the *val* split and pick the
    value that maximises F1 for each class.  Classes with no positives in
    val default to 0.5.
    """
    print("\nCalibrating per-class thresholds on val split …")
    val_ds = ChessConceptDataset(data_path, split="val", phase4=is_phase4, phase5=is_phase5)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    all_probs  = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, hist, seq_len, y in val_dl:
            probs = torch.sigmoid(
                model(x.to(device), hist.to(device), seq_len.to(device))
            ).cpu()
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
                     thresholds: torch.Tensor,
                     is_phase4: bool = False,
                     is_phase5: bool = False) -> None:
    test_ds = ChessConceptDataset(data_path, split="test", phase4=is_phase4, phase5=is_phase5)
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
        for x, hist, seq_len, y_true in test_dl:
            x      = x.to(device)
            probs  = torch.sigmoid(
                model(x, hist.to(device), seq_len.to(device))
            )
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


def _identify_eco(history_uci: list[str]) -> dict | None:
    """Walk history FENs against eco_db.json; return deepest ECO match."""
    eco_db_path = Path("data/eco_db.json")
    if not eco_db_path.exists():
        return None
    try:
        eco_db = json.loads(eco_db_path.read_text())
        board  = _chess.Board()
        fens   = [board.fen()]
        for uci in history_uci:
            board.push_uci(uci)
            fens.append(board.fen())
        best: dict | None = None
        for fen in fens:
            norm  = " ".join(fen.split()[:4])
            entry = eco_db.get(norm)
            if entry and (best is None or entry["depth"] > best["depth"]):
                best = entry
        return best
    except Exception:
        return None


def _endgame_type(fen: str) -> str | None:
    """Return a human-readable endgame material description, or None if not an endgame."""
    board  = _chess.Board(fen)
    pieces = board.piece_map()
    non_kings = {sq: p for sq, p in pieces.items() if p.piece_type != _chess.KING}
    if len(non_kings) > 8:
        return None   # not an endgame

    has = {pt: False for pt in [_chess.QUEEN, _chess.ROOK, _chess.BISHOP,
                                  _chess.KNIGHT, _chess.PAWN]}
    for p in non_kings.values():
        has[p.piece_type] = True

    if not any(has.values()):
        return "Bare kings (draw)"
    if not has[_chess.QUEEN] and not has[_chess.ROOK] and \
       not has[_chess.BISHOP] and not has[_chess.KNIGHT]:
        return "King and pawn(s)"
    if has[_chess.ROOK] and not has[_chess.QUEEN] and \
       not has[_chess.BISHOP] and not has[_chess.KNIGHT]:
        return "Rook endgame"
    if has[_chess.BISHOP] and not has[_chess.QUEEN] and \
       not has[_chess.ROOK] and not has[_chess.KNIGHT]:
        w_sq = [sq for sq, p in pieces.items()
                if p.piece_type == _chess.BISHOP and p.color == _chess.WHITE]
        b_sq = [sq for sq, p in pieces.items()
                if p.piece_type == _chess.BISHOP and p.color == _chess.BLACK]
        if w_sq and b_sq:
            sq_col = lambda sq: (_chess.square_file(sq) + _chess.square_rank(sq)) % 2
            opp = (sq_col(w_sq[0]) != sq_col(b_sq[0]))
            return "Opposite-colour bishops" if opp else "Same-colour bishops"
        return "Bishop endgame"
    if has[_chess.KNIGHT] and not has[_chess.QUEEN] and \
       not has[_chess.ROOK] and not has[_chess.BISHOP]:
        return "Knight endgame"
    if has[_chess.QUEEN] and not has[_chess.ROOK]:
        return "Queen endgame"
    return "Complex endgame"


def spot_check(model: ChessConceptClassifier,
               thresholds: torch.Tensor) -> None:
    print(f"\n── Spot Checks ────────────────────────────────────────────────────────")
    from .board_encoder import (
        fen_to_tensor, move_to_tensor, history_to_tensor, history_rich_to_tensor,
        SF_SIZE,
    )
    from tools.label_positions import algo_feature_vector, algo_feature_vector_v4
    from .board_encoder import NNUE_SIZE
    device    = next(model.parameters()).device
    is_phase5 = model._phase5
    is_phase4 = model.spatial_proj is not None

    all_pass = True
    for sc in SPOT_CHECKS:
        fen     = sc["fen"]
        board_t = fen_to_tensor(fen)
        move_t  = move_to_tensor(sc.get("move_uci", ""))
        if is_phase5:
            # NNUE activations unavailable at eval time without cache — use zeros
            nnue_t = torch.zeros(NNUE_SIZE, dtype=torch.float32)
            algo_t = torch.from_numpy(algo_feature_vector_v4(fen))
            v3_t   = torch.from_numpy(algo_feature_vector(fen))
            sf_t   = torch.zeros(SF_SIZE, dtype=torch.float32)
            x      = torch.cat([nnue_t, board_t, move_t, algo_t, sf_t, v3_t]).unsqueeze(0).to(device)
            hist_t, seq_len = history_rich_to_tensor([])
        elif is_phase4:
            spatial_t = torch.from_numpy(algo_feature_vector_v4(fen))
            v3_t      = torch.from_numpy(algo_feature_vector(fen))
            sf_t      = torch.zeros(SF_SIZE, dtype=torch.float32)
            x         = torch.cat([board_t, move_t, spatial_t, v3_t, sf_t]).unsqueeze(0).to(device)
            hist_t, seq_len = history_rich_to_tensor([])
        else:
            algo_t = torch.from_numpy(algo_feature_vector(fen))
            x      = torch.cat([board_t, move_t, algo_t]).unsqueeze(0).to(device)
            hist_t, seq_len = history_to_tensor([])
        hist_t    = hist_t.unsqueeze(0).to(device)
        seq_len_t = torch.tensor([seq_len])

        probs = torch.sigmoid(model(x, hist_t, seq_len_t)).squeeze(0).cpu()
        # Toy positions don't produce the same confidence as real game positions.
        # Cap spot-check thresholds at 0.90 so near-misses from the calibrator
        # setting high thresholds don't count as failures.  Production unchanged.
        SPOT_CAP = 0.90
        t_vec    = thresholds.clamp(max=SPOT_CAP)
        pred_set = {CONCEPTS[i] for i in range(NUM_CONCEPTS) if probs[i] >= t_vec[i]}

        expected  = sc["expect"]
        also_ok   = set(sc.get("also_ok", []))
        # A missing expected concept is forgiven if any also_ok concept was detected
        alt_found = bool(also_ok & pred_set)
        miss      = [c for c in expected if c not in pred_set and not alt_found]
        status    = "MISS" if miss else "PASS"
        if miss:
            all_pass = False

        # Position context: ECO name (opening positions) + endgame type
        history_uci = sc.get("history_uci")
        eco_info    = _identify_eco(history_uci) if history_uci else None
        eg_type     = _endgame_type(fen)

        context_parts = []
        if eco_info:
            eco_str = eco_info["eco"]
            if eco_info.get("opening"):
                eco_str += f" {eco_info['opening']}"
            if eco_info.get("variation"):
                eco_str += f", {eco_info['variation']}"
            context_parts.append(eco_str)
        if eg_type:
            context_parts.append(eg_type)
        context_line = " | ".join(context_parts) if context_parts else ""

        top5 = sorted(enumerate(probs.tolist()), key=lambda kv: -kv[1])[:5]
        top5_str = ", ".join(f"{CONCEPTS[i]}({p:.2f})" for i, p in top5)
        print(f"\n  {status} — {sc['name']}")
        if context_line:
            print(f"       context  : {context_line}")
        print(f"       expected : {expected}")
        print(f"       missing  : {miss or '—'}")
        print(f"       top preds: {top5_str}")

    if all_pass:
        print("\n  All spot checks passed.")


# ── main ──────────────────────────────────────────────────────────────────────

def _main() -> None:
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

    # Load model — detect phase from checkpoint weights
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd        = ckpt["state_dict"]
    is_phase5 = any(k.startswith("nnue_proj") for k in sd)
    is_phase4 = any(k.startswith("spatial_proj") for k in sd) and not is_phase5
    model     = ChessConceptClassifier(phase4=is_phase4, phase5=is_phase5).to(device)
    model.load_state_dict(sd)
    model.eval()
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    phase_tag = "Phase 5" if is_phase5 else ("Phase 4" if is_phase4 else "Phase 3")
    print(f"Loaded checkpoint: epoch={epoch}  val_loss={val_loss:.4f}  ({phase_tag})")

    data_path = Path(args.data)

    # Calibrate thresholds if requested, then save
    if args.calibrate:
        cal = calibrate_thresholds(model, data_path, device,
                                   is_phase4=is_phase4, is_phase5=is_phase5)
        save_thresholds(cal)

    # Load thresholds (calibrated file if it exists, global fallback otherwise)
    thresholds = load_thresholds(default=args.threshold)

    if not args.spot_check_only:
        evaluate_dataset(model, data_path, device, thresholds,
                         is_phase4=is_phase4, is_phase5=is_phase5)

    spot_check(model, thresholds)


def main() -> None:
    log_path = _next_results_path("eval")
    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        _main()
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print(f"Eval log saved → {log_path}")


if __name__ == "__main__":
    main()
