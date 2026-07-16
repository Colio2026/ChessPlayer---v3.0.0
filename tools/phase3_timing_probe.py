#!/usr/bin/env python3
"""
Phase 3 Timing Probe
====================
Measures two bottlenecks before committing to the GRU architecture:

  A) Parse throughput  — re-parsing PGNs with full move history extraction
  B) Model throughput  — GRU forward+backward pass per batch

Run from project root:  python tools/phase3_timing_probe.py

Writes results to: docs/Training_results/phase3_probe_YYYY-MM-DD.txt
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import chess
import chess.pgn
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ── config ─────────────────────────────────────────────────────────────────────

PGN_DIR         = Path("data/annotated_pgns")
RESULTS_DIR     = Path("docs/Training_results")
N_PARSE_TARGET  = 1000     # labeled examples to time during parse
N_MODEL_BATCHES = 100      # forward+backward passes to time
BATCH_SIZE      = 512
MAX_SEQ_LEN     = 60       # max history steps to feed GRU (caps memory)
BOARD_DIM       = 1188     # Phase 2 combined board features (unchanged)
MOVE_DIM        = 128      # one-hot from-sq + to-sq per history step
GRU_HIDDEN      = 256
NUM_CONCEPTS    = 50

# Extrapolation anchors (from Phase 2 measurements)
TOTAL_LABELED_PGN    = 86_434    # labeled PGN examples from last parse run
TOTAL_EXAMPLES       = 949_553   # total training examples (Phase 2 dataset)
EPOCHS               = 100
PHASE2_SECS_PER_EPOCH = 30.0     # confirmed by user during Phase 2


# ── minimal keyword check (probe only, not the full CONCEPT_KEYWORDS map) ────

_BOARD_MARKER_RE = re.compile(r'\[%[^\]]+\]')

def clean_comment(raw: str) -> str:
    c = _BOARD_MARKER_RE.sub(' ', raw)
    return re.sub(r'\s+', ' ', c).strip()

_PROBE_KEYWORDS = [
    "pin", "fork", "skewer", "sacrifice", "discovered", "clearance",
    "deflect", "zwischenzug", "checkmate", "mating", "outpost", "initiative",
    "overload", "trapped", "interfer", "x-ray", "double check", "battery",
    "passed pawn", "open file", "bishop pair", "bad bishop", "blockade",
]

def has_theme(comment: str) -> bool:
    text = comment.lower()
    return any(kw in text for kw in _PROBE_KEYWORDS)


# ── Part A: Parse timing ──────────────────────────────────────────────────────

def time_parse(n_target: int = N_PARSE_TARGET) -> dict:
    """
    Walk annotated PGNs, extract n_target labeled examples WITH full move
    history. Returns timing and history-length statistics.
    """
    pgn_files = sorted(PGN_DIR.rglob("*.pgn"))
    # Exclude the ECO reference file — it has no game comments
    pgn_files = [p for p in pgn_files if "ECO_code_openings" not in str(p)]

    print(f"  Scanning {len(pgn_files)} PGN files for {n_target:,} labeled examples ...")

    start       = time.perf_counter()
    count       = 0
    hist_lens   = []
    skipped     = 0

    for pgn_path in pgn_files:
        if count >= n_target:
            break
        try:
            with open(pgn_path, encoding="utf-8", errors="replace") as fh:
                while count < n_target:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break

                    # ECO code available from header — free pickup
                    # eco = game.headers.get("ECO", None)

                    board = game.board()
                    node  = game
                    while node.variations and count < n_target:
                        node = node.variations[0]

                        # Capture history BEFORE this move (what GRU will consume)
                        history_uci = [m.uci() for m in board.move_stack]

                        board.push(node.move)

                        comment = clean_comment(node.comment)
                        if len(comment) >= 25 and has_theme(comment):
                            hist_lens.append(len(history_uci))
                            count += 1
        except Exception:
            skipped += 1

    elapsed = time.perf_counter() - start
    rate    = count / elapsed if elapsed > 0 else 0

    return {
        "elapsed_sec":    elapsed,
        "examples":       count,
        "rate_per_sec":   rate,
        "avg_hist_len":   mean(hist_lens) if hist_lens else 0.0,
        "median_hist":    median(hist_lens) if hist_lens else 0.0,
        "max_hist_len":   max(hist_lens) if hist_lens else 0,
        "skipped_files":  skipped,
    }


# ── Part B: Model timing ──────────────────────────────────────────────────────

class ProbeModel(nn.Module):
    """
    Phase 3 candidate architecture.
    Board features (1188-dim, Phase 2 unchanged) +
    GRU over move history (variable length, 128-dim per step) →
    concept head (50 classes).
    """
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(MOVE_DIM, GRU_HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(BOARD_DIM + GRU_HIDDEN, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, NUM_CONCEPTS),
        )

    def forward(
        self,
        board_x:  torch.Tensor,   # (B, 1188)
        move_seq: torch.Tensor,   # (B, max_seq_len, 128) padded
        seq_lens: torch.Tensor,   # (B,) actual lengths (0 = no history)
    ) -> torch.Tensor:
        safe_lens = seq_lens.clamp(min=1).cpu()
        packed    = pack_padded_sequence(
            move_seq, safe_lens, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)   # (1, B, 256)
        gru_out   = hidden.squeeze(0)  # (B, 256)

        # Zero out GRU output for examples with no history
        no_hist_mask = (seq_lens == 0).float().unsqueeze(1).to(gru_out.device)
        gru_out      = gru_out * (1 - no_hist_mask)

        combined = torch.cat([board_x, gru_out], dim=1)
        return self.head(combined)


def time_model(n_batches: int = N_MODEL_BATCHES, avg_hist: float = 20.0) -> dict:
    """
    Run n_batches of mixed forward+backward passes (simulated training step).
    Mix: ~10% game examples with history, ~90% puzzle examples with none.
    Returns timing statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ProbeModel().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit   = nn.BCEWithLogitsLoss()

    n_with_hist = max(1, int(BATCH_SIZE * 0.10))  # ~51 game examples per batch
    n_no_hist   = BATCH_SIZE - n_with_hist

    timings = []
    print(f"  Running {n_batches} batches on {device} "
          f"(batch={BATCH_SIZE}, ~{n_with_hist} w/history, ~{n_no_hist} puzzles) ...")

    for i in range(n_batches):
        board_x  = torch.randn(BATCH_SIZE, BOARD_DIM, device=device)
        y_true   = (torch.rand(BATCH_SIZE, NUM_CONCEPTS, device=device) > 0.97).float()
        move_seq = torch.zeros(BATCH_SIZE, MAX_SEQ_LEN, MOVE_DIM, device=device)
        seq_lens = torch.zeros(BATCH_SIZE, dtype=torch.long)

        for j in range(n_with_hist):
            length = int(torch.randint(5, int(avg_hist) + 10, (1,)).item())
            length = min(length, MAX_SEQ_LEN)
            move_seq[j, :length] = torch.randn(length, MOVE_DIM, device=device)
            seq_lens[j]          = length

        t0 = time.perf_counter()
        opt.zero_grad()
        logits = model(board_x, move_seq, seq_lens)
        loss   = crit(logits, y_true)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{n_batches}  avg {mean(timings)*1000:.1f} ms/batch",
                  flush=True)

    return {
        "device":           str(device),
        "batch_size":       BATCH_SIZE,
        "avg_ms_per_batch": mean(timings) * 1000,
        "median_ms":        median(timings) * 1000,
        "min_ms":           min(timings) * 1000,
        "max_ms":           max(timings) * 1000,
        "pct_with_hist":    n_with_hist / BATCH_SIZE * 100,
    }


# ── Extrapolation ─────────────────────────────────────────────────────────────

def extrapolate(parse_stats: dict, model_stats: dict) -> dict:
    parse_rate     = parse_stats["rate_per_sec"]
    parse_full_min = (TOTAL_LABELED_PGN / parse_rate / 60) if parse_rate > 0 else float("inf")

    ms_per_batch    = model_stats["avg_ms_per_batch"]
    batches_per_ep  = TOTAL_EXAMPLES / BATCH_SIZE
    epoch_secs      = batches_per_ep * ms_per_batch / 1000
    total_train_hrs = epoch_secs * EPOCHS / 3600
    slowdown        = epoch_secs / PHASE2_SECS_PER_EPOCH

    return {
        "parse_full_min":   parse_full_min,
        "epoch_min":        epoch_secs / 60,
        "total_train_hrs":  total_train_hrs,
        "slowdown_vs_p2":   slowdown,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(parse: dict, model: dict, ext: dict) -> str:
    parse_ok = ext["parse_full_min"] < 120
    epoch_ok = ext["epoch_min"] < 5
    slow_ok  = ext["slowdown_vs_p2"] < 10
    go       = parse_ok and epoch_ok and slow_ok

    lines = [
        "=" * 65,
        "  PHASE 3 TIMING PROBE RESULTS",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "A) PARSE THROUGHPUT  (full history extraction)",
        "-" * 65,
        f"  Examples timed      : {parse['examples']:,}",
        f"  Elapsed             : {parse['elapsed_sec']:.1f} sec",
        f"  Throughput          : {parse['rate_per_sec']:.0f} labeled examples/sec",
        f"  Avg history length  : {parse['avg_hist_len']:.1f} half-moves",
        f"  Median history      : {parse['median_hist']:.1f} half-moves",
        f"  Max history seen    : {parse['max_hist_len']} half-moves",
        f"  Skipped files       : {parse['skipped_files']}",
        "",
        f"  Extrapolated: full {TOTAL_LABELED_PGN:,} labeled examples = "
        f"{ext['parse_full_min']:.1f} min",
        "",
        "B) MODEL THROUGHPUT  (GRU forward+backward, simulated training)",
        "-" * 65,
        f"  Device              : {model['device']}",
        f"  Batch size          : {model['batch_size']}",
        f"  % w/ move history   : {model['pct_with_hist']:.0f}%  (~10% game, 90% puzzle)",
        f"  Avg ms/batch        : {model['avg_ms_per_batch']:.1f} ms",
        f"  Median ms/batch     : {model['median_ms']:.1f} ms",
        f"  Min/max ms          : {model['min_ms']:.1f} / {model['max_ms']:.1f} ms",
        "",
        f"  Extrapolated epoch  ({TOTAL_EXAMPLES:,} examples): "
        f"{ext['epoch_min']:.1f} min",
        f"  Extrapolated run    ({EPOCHS} epochs):              "
        f"{ext['total_train_hrs']:.1f} hrs",
        f"  Slowdown vs Phase 2 (30s/epoch):                "
        f"{ext['slowdown_vs_p2']:.1f}x",
        "",
        "C) GO / NO-GO",
        "-" * 65,
        f"  Parse full dataset < 2 hrs  : {'✓  GO' if parse_ok else '✗  NO-GO'}",
        f"  Epoch time < 5 min          : {'✓  GO' if epoch_ok else '✗  NO-GO'}",
        f"  Slowdown vs Phase 2 < 10x   : {'✓  GO' if slow_ok else '✗  NO-GO'}",
        "",
        "  OVERALL: " + ("✓  PROCEED TO PHASE 3 IMPLEMENTATION"
                         if go else "✗  REVIEW CONSTRAINTS BEFORE PROCEEDING"),
        "",
        "D) NOTES / OBSERVATIONS",
        "-" * 65,
        "  (add observations here after reviewing results)",
        "",
        "=" * 65,
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not PGN_DIR.exists():
        sys.exit(f"PGN directory not found: {PGN_DIR}\nRun from project root.")

    today    = datetime.now().strftime("%Y-%m-%d")
    out_path = RESULTS_DIR / f"phase3_probe_{today}.txt"

    print("\n" + "=" * 65)
    print("  PHASE 3 TIMING PROBE")
    print("=" * 65)

    print("\n[A] Parse timing ...")
    parse_stats = time_parse()

    print(f"\n    Done: {parse_stats['examples']:,} examples in "
          f"{parse_stats['elapsed_sec']:.1f}s  "
          f"(avg history: {parse_stats['avg_hist_len']:.1f} moves)")

    print("\n[B] Model timing ...")
    model_stats = time_model(avg_hist=max(parse_stats["avg_hist_len"], 5.0))

    print("\n[C] Extrapolating ...")
    ext = extrapolate(parse_stats, model_stats)

    report = build_report(parse_stats, model_stats, ext)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
