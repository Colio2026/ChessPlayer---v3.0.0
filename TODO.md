# ChessPlayer v3 — Coach Intelligence System Design
## Vision, Architecture & Build Manual

---

# ⚡ PHASE 4 — ACTIVE ENGINEERING PLAN

**Goal:** calibrated macro F1 > 0.50  
**Baseline:** Phase 3 = 0.4733  |  Phase 4-A1 = failed (overfit epoch 7)

## Why A1 Failed

Phase 4-A1 used hidden1=2048, hidden2=1024 (8.7M params).
Val loss bottomed at **0.6356 epoch 7**, then rose.
Phase 3 bottomed at **0.5608 epoch 9** — Phase 4-A1 is *worse at its own minimum*.

Root cause: 2048/1024 head is wide enough to memorize 1663-dim noise before it learns signal.
Fix: cut head width until the model is forced to generalize.

---

## A Sample's Path Through the Data

```
INPUT
  fen          "r1bqk2r/pp2bppp/2n1pn2/3p4/..."
  move_uci     "d2d4"
  history_rich [{"from":"e2","to":"e4","piece":"P",...}, ...]   <- up to 60 moves
  _ac          417382                                            <- index into algo_cache.npy

ENCODERS
  fen_to_tensor(fen)
    piece placement  768-dim  (12 planes x 64 squares)
    side to move       1-dim
    castling rights    4-dim
    en passant sq      8-dim
    attack maps      128-dim
    pawn structure    80-dim
    mobility          12-dim
    ─────────────────────────
    board_t         1001-dim  float32

  move_to_tensor(move_uci)
    from-square one-hot   64-dim
    to-square one-hot     64-dim
    ──────────────────────────
    move_t              128-dim  float32

  algo_cache.npy[_ac]                         <- binary memmap, not JSON
    weak square maps     128-dim
    outpost maps         128-dim
    backward pawn maps   128-dim
    passed pawn maps     128-dim
    (reserved B5)        128-dim
    bishop pair/dev/xray/battery/misc  1023-dim
    ────────────────────────────────────────
    algo_t             1663-dim  float32

  torch.cat([board_t, move_t, algo_t])
    static_x           2792-dim  float32     <- STATIC_SIZE_V4

  history_rich_to_tensor(history_rich)
    per step: from(64)+to(64)+piece(6)+captured(7)+check(1)+cap(1)+color(1) = 144-dim
    padded to 60 x 144
    seq_len = actual history length (0 if missing)
    ──────────────────────────────────────────────────────
    hist_t   [60, 144]  float32
    seq_len  int

MODEL
  GRU(input=144, hidden=256, batch_first=True)
    reads hist_t, returns last hidden state
    gru_h   256-dim  float32
    (zeroed if seq_len == 0)

  torch.cat([static_x, gru_h])
    combined  3048-dim                       <- COMBINED_SIZE_V4

  MLP HEAD
    Linear(3048 -> hidden1) -> BatchNorm -> ReLU -> Dropout
    Linear(hidden1 -> hidden2) -> BatchNorm -> ReLU -> Dropout
    Linear(hidden2 -> 53)        <- one logit per concept

OUTPUT
  Training:   BCEWithLogitsLoss(logits, y, pos_weight=...)
  Inference:  sigmoid(logits) > per-class threshold  ->  concept set
```

---

## Layer Width Spider Map

Each block represents ~100 dims. Wider = more parameters in that layer.

```
                 static_x    combined    hidden1    hidden2    out
                ──────────  ──────────  ─────────  ────────  ──

Phase 3         [■■■■■■■■■■■         ][■■■■■■■■■■■■■■  ][■■■■■■■■][■■■■   ][■]
3.7M params ✓    1188                  1444              1536       768      53

Phase 4-A1      [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■][■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■][■■■■■■■■■■■■■■■■■■■■][■■■■■■■■■■][■]
8.7M params ✗    2792                              3048             2048           1024                   53
overfit ep7     ^--- more input is good ---^     ^--- these two are WAY too wide for the signal quality ---^

Phase 4-A2      [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■][■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■][■■■■■■■■■■][■■■■■  ][■]
~5.1M   TRY 1    2792                              3048             1024            512         53

Phase 4-A3      [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■][■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■][■■■■■■■■■■■■■■■][■■■■■■ ][■]
~6.6M   TRY 2    2792                              3048             1536             768          53
(Phase 3 proportions, Phase 4 input)

Phase 4-B       algo(1663)->proj(256)+board+move -> combined -> head
~3.8M   TRY 3    1663->256   1001+128  -> 1641 + GRU(256) = 1897   1024       512    53
(spatial bottleneck before concat — forces compression of noisy 1663-dim)
```

---

## Experiment Queue

### 4-A2 — TRY THIS FIRST (~5.1M params)

One change in classifier.py. Cut head so the model can't memorize noise.

```python
# classifier.py — inside `if phase4:` block
hidden1  = 1024    # was 2048
hidden2  = 512     # was 1024
dropout  = 0.50    # was 0.40
dropout2 = 0.35    # was 0.20
```

Run: `python -m src.chess_coach.ml.train --phase4`

Stop condition: if val loss min > 0.60 AND F1 stalls < 0.38 by epoch 12 → move to A3.

---

### 4-A3 — Phase 3 proportions, Phase 4 input (~6.6M params)

Same hidden widths as Phase 3 (proven). Just benefits from 2792-dim input.

```python
# classifier.py — inside `if phase4:` block
hidden1  = 1536    # same as Phase 3
hidden2  = 768     # same as Phase 3
dropout  = 0.40    # same as Phase 3
dropout2 = 0.20    # same as Phase 3
```

---

### 4-B — Spatial Bottleneck (~3.8M params)

Bigger change: add Linear(1663->256) before the main concat.
Forces the model to compress noisy spatial features into a structured representation first.

```python
# classifier.py __init__  (new attribute)
self.spatial_proj = nn.Sequential(
    nn.Linear(1663, 256),
    nn.ReLU(),
    nn.Dropout(0.30),
)

# forward():
board_move_t = x[:, :INPUT_SIZE + MOVE_SIZE]    # 1129
algo_t       = x[:, INPUT_SIZE + MOVE_SIZE:]    # 1663
algo_proj    = self.spatial_proj(algo_t)         # 256
x_in = torch.cat([board_move_t, algo_proj, gru_h], dim=1)  # 1129+256+256=1641
# then hidden1=1024, hidden2=512
```

---

### 4-C — Lower LR (pair with any above, no arch change)

```python
python -m src.chess_coach.ml.train --phase4 --lr 5e-4   # was 1e-3
```

### 4-D — Stronger weight decay (pair with any above)

```python
# train.py optimizer line:
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.5e-2)
# was 6e-3
```

### 4-E — More patience (only if runs look promising but plateau)

```python
# train.py --patience default: 25  (was 15)
```

---

## Results Tracking

| Run      | Params | Val loss min | F1 uncal | F1 cal | weak_sq | outpost | back_pawn | note            |
|----------|--------|-------------|----------|--------|---------|---------|-----------|-----------------|
| Phase 3  | 3.7M   | 0.5608      | ~0.42    | 0.4733 | <0.30   | <0.30   | <0.30     | baseline ✓      |
| 4-A1     | 8.7M   | 0.6356      | 0.36     | —      | —       | —       | —         | overfit ep7 ✗   |
| 4-A2     | 5.1M   | …           | …        | …      | …       | …       | …         | next run        |
| 4-A3     | 6.6M   | …           | …        | …      | …       | …       | …         | if A2 bad       |
| 4-B      | 3.8M   | …           | …        | …      | …       | …       | …         | bottleneck      |
| target   | —      | < 0.55      | > 0.44   | > 0.50 | > 0.35  | > 0.35  | > 0.35    |                 |

## Recommended Run Order

1. 4-A2 alone (1024/512, dropout 0.50/0.35)
2. If A2 val loss min beats Phase 3 → try 4-A2 + 4-C (lower LR)
3. If A2 still below Phase 3 → try 4-A3 (1536/768)
4. If both plateau near Phase 3 → 4-B (spatial bottleneck projection)
5. Add 4-D (weight decay) to whichever config looks most promising

---

> **Goal**: A fully local chess coach that explains positions the way a chess master author
> would — in Nimzowitsch's voice, referencing real chess ideas, trained on the literature
> and games of the world's greatest players.  No cloud dependency, no API calls, runs on
> your machine.

---

## Table of Contents

1. [What We Are Building](#1-what-we-are-building)
2. [Why the Current System Has a Ceiling](#2-why-the-current-system-has-a-ceiling)
3. [The Target Architecture](#3-the-target-architecture)
4. [Neural Net Primer — How This Works](#4-neural-net-primer--how-this-works)
5. [Data Sources & What to Collect](#5-data-sources--what-to-collect)
6. [Data Processing Pipeline](#6-data-processing-pipeline)
7. [Model Designs — Simple to Advanced](#7-model-designs--simple-to-advanced)
8. [Training Guide — Step by Step](#8-training-guide--step-by-step)
9. [Evaluation — How to Know It Works](#9-evaluation--how-to-know-it-works)
10. [Integration with ChessPlayer](#10-integration-with-chessplayer)
11. [Phased Roadmap](#11-phased-roadmap)
12. [File & Folder Layout for ML Work](#12-file--folder-layout-for-ml-work)

---

## 1. What We Are Building

The coach must do three things a human chess master does when annotating a game:

**Understand**: Look at the position and know what is happening.
> "Black's knight has no good square. The d5 outpost is controlled by White's c-pawn,
> and e5 is covered twice. The knight must retreat to the back rank."

**Explain**: Connect that understanding to chess theory — name the idea.
> "This is the bad piece in its classical form — Nimzowitsch called it the 'problem child'
> of the position. Its immobility infects the whole queenside."

**Advise**: Tell the player what to do and why right now, not later.
> "Trade the knight before it becomes a permanent liability. 17. Nd7 offers the exchange
> of this piece for White's strong bishop. Delay and you lose without counterplay."

The current system can approximate the third (advise) because Stockfish gives us the move. It cannot do the second (name the idea) and only weakly does the first (understand) because the extractors are unreliable approximations. Everything in this document is about building those two missing layers properly.

---

## 2. Why the Current System Has a Ceiling

### The Extractor Problem

The six extractors (`king_safety`, `space_control`, `piece_mobility`, `pawn_structure`, `material_balance`, `tactic_scanner`) are hand-coded approximations of things Stockfish already computes, and computes better. They were necessary to get a working pipeline but they introduce two failure modes:

1. **False positives** — the extractor fires and the signal is wrong (e.g. labels a pawn "backward" when the structure is actually fine by chess standards)
2. **Missed patterns** — the extractor doesn't recognise a blockade on d5 as a blockade because nobody wrote that rule

Stockfish's classical eval (`eval` command: material, mobility, king safety, threats, space, passed pawns) is the correct version of what the extractors estimate. When we have those classical terms, we should use them directly and not the extractors.

**What the extractors ARE useful for**: pre-programmed tactical pattern detection (pins, forks, skewers, discoveries). These are deterministic rules, they fire reliably, and Stockfish's `eval` command doesn't give you "there is a pin on the e-file right now." Keep the tactic scanner. Consider retiring the rest as primary signals.

### The Phrase Database Problem

The phrase DB (`chess_coach.db`) is a slot-filling system: a signal fires → fetch a phrase with matching tags → insert it into a template. This works and produces grammatically correct chess advice. But:

- Every position in the Sicilian Najdorf gets the same 5 phrases as a random middlegame with a backward pawn
- The phrases have no memory of what happened 3 moves ago
- There is no way to say "this specific knight on d5 is strong because it was blockaded there by the c-pawn push on move 14" — the phrase system has no positional history

The phrase DB is the right short-term approach. The ML system replaces it in the long run. The two can coexist during the transition.

---

## 3. The Target Architecture

The coach has three layers. Build them in order — each one works on its own and improves what came before.

```
┌─────────────────────────────────────────────────────────────────┐
│                         LAYER 1: PERCEPTION                     │
│                                                                 │
│  Board state (FEN)  +  Stockfish signals  →  Feature vector    │
│                                                                 │
│  • SF NNUE score (already have)                                 │
│  • SF classical eval breakdown (already have, UseNNUE=false)    │
│  • Piece placement encoding (768 bits, see §7)                  │
│  • Tactic scanner results (keep — these are rule-based & right) │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Feature vector  (~800 numbers)
┌──────────────────────────────▼──────────────────────────────────┐
│                    LAYER 2: UNDERSTANDING                       │
│                                                                 │
│  Feature vector  →  Chess concept labels  (the neural net)      │
│                                                                 │
│  Concepts: outpost, bad_piece, blockade, pawn_majority,         │
│  open_file, battery, bishop_pair, rook_on_7th, zugzwang,        │
│  king_safety_deficit, isolated_pawn, pawn_chain, overprotection │
│                                                                 │
│  Output: probability over ~80 chess concepts                    │
│  e.g. {"outpost": 0.91, "bad_piece": 0.78, "blockade": 0.82}   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Concept probabilities
┌──────────────────────────────▼──────────────────────────────────┐
│                    LAYER 3: LANGUAGE                            │
│                                                                 │
│  Concept labels + board context  →  Explanation text           │
│                                                                 │
│  Option A (near term): retrieval — find the most similar        │
│    annotated position in the literature index, return its text  │
│                                                                 │
│  Option B (long term): fine-tuned small LLM — generates         │
│    Nimzowitsch-voice prose conditioned on concept labels        │
└─────────────────────────────────────────────────────────────────┘
```

The key insight: **Layer 2 is trained on annotated literature.** The neural net learns what "outpost" means not because we programmed it, but because in 47,000 annotated positions where a master wrote the word "outpost", the board features look a certain way. The net learns the statistical correlation between board geometry and chess vocabulary.

---

## 4. Neural Net Primer — How This Works

This section explains the concepts from scratch. Skip what you know.

### What a Neural Net Is

A neural net is a mathematical function with millions of tunable parameters (called **weights**). You give it an input (a list of numbers describing the board) and it produces an output (a list of numbers representing concept probabilities). The function itself is a series of matrix multiplications and simple nonlinear operations stacked in layers.

```
Input layer          Hidden layers          Output layer
(board features)     (learned filters)      (concept scores)

[0, 1, 0, 1, ...]  → [h₁, h₂, ... h₁₂₈] → [0.91, 0.78, 0.03, ...]
   768 numbers           128 numbers             80 numbers
```

### What Training Does

Training = finding the weights that make the function output the RIGHT answer for your training examples.

You start with random weights (the net outputs garbage). You show it a position and it outputs wrong concept scores. You compute how wrong it was — that's the **loss** (a single number: lower = better). You then use calculus (backpropagation) to nudge every weight slightly in the direction that reduces the loss. Repeat 100,000 times. The weights converge to values that produce correct outputs.

This is called **gradient descent**. You don't need to understand the calculus — PyTorch does it automatically.

### What "Training Data" Is

A list of (input, correct_output) pairs. For Layer 2:

```python
# One training example
{
    "input":  [0, 1, 0, 0, ...],   # 768 board features + ~10 SF eval numbers
    "output": [1, 1, 0, 0, 0, 1, ...]  # 1 = this concept is present, 0 = absent
    #           ↑  ↑              ↑
    #      outpost  bad_piece  blockade
}
```

The "correct output" comes from the annotated literature: if the master wrote "outpost" in the comment for this position, then `outpost = 1`.

### What Overfitting Is

The net memorises your training examples instead of learning the underlying pattern. You catch this by testing on examples the net never saw during training — if it works on training data but fails on test data, it has overfit. The fix is more data, regularisation (dropout), or a simpler model.

### The Tools You Will Use

```
Python               — you already have this
PyTorch              — the neural net framework (pip install torch)
python-chess         — board encoding (already in the project)
scikit-learn         — utilities, evaluation metrics
pandas               — data manipulation
sqlite3              — storage (already in the project)
```

Nothing else is required for Phase 1 and 2.

---

## 5. Data Sources & What to Collect

### Primary Sources (best quality)

| Source | Format | What You Get | Volume |
|--------|--------|-------------|--------|
| Annotated PGN files of classic games | `.pgn` with `{ comment }` | (position, master_annotation) pairs | 500–50k positions per file |
| Chess books (Nimzowitsch, Kasparov, Silman) | Text + diagrams | (position description, theoretical explanation) pairs | 1k–10k per book |
| ChessBase annotated games (exported PGN) | `.pgn` | Same as above | Millions total |
| Lichess Studies (CC licensed) | `.pgn` | User + master annotations | ~100k studies |
| Lichess puzzle database | `.csv` | (FEN, best_move, themes[]) | 4.5M puzzles |

### How to Get Annotated PGNs

1. **Magnus Carlsen games**: Download from [pgnmentor.com](https://www.pgnmentor.com/files.html) — many have GM annotations
2. **Classic annotated games**: "My Great Predecessors" series, "My System" game examples, "Chess Fundamentals" by Capablanca — these exist as PGN files online
3. **Lichess game export**: Any Lichess study can be exported as PGN with annotations
4. **Your existing library**: The `data/` folder already indexes PGNs — if any have `{ comment }` blocks, they are already training data

### How to Process a Book

A chess book is harder than a PGN because you need to extract positions AND their explanatory text. The workflow:

1. Get the book as a PGN with annotations (many classic books have been digitised into annotated PGN form by enthusiasts — search GitHub and chessgames.com)
2. If text-only: manually create a small high-quality set. Even 500 deeply annotated examples from "My System" will teach the net more than 5,000 poorly labelled ones.
3. The quality of training data matters more than quantity at this scale.

### What "Themes" Look Like

From the Lichess puzzle CSV:
```
FEN,Moves,Rating,Themes
r1bqkb1r/ppp2ppp/2n2n2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq,e1g1,1456,"fork pin mateIn2"
```

From an annotated PGN:
```pgn
22. Nd5 { The knight reaches its ideal outpost. Black has no way to challenge
it — the c7-pawn cannot advance because of Nxc7, and the bishop swap
would only activate White's remaining pieces. This is the blockade Nimzowitsch
described: the passer is stopped not by a pawn but by a piece sitting
in front of it, deriving energy from its immobility. } 22... Rf8
```

You extract: `FEN at move 22` + themes: `["outpost", "blockade", "piece_activity"]` (extracted by keyword matching the annotation text) + the full annotation text as the target for the language layer.

---

## 6. Data Processing Pipeline

Build this as a standalone script before touching any ML code.

### Step 1: Parse Annotated PGN

```python
# tools/parse_annotated_pgn.py
import chess.pgn
import io
import json

# Chess concept keywords — map words in annotations to theme labels
CONCEPT_KEYWORDS = {
    "outpost":        ["outpost", "outpost square", "ideal square"],
    "blockade":       ["blockade", "blockader", "sitting in front"],
    "bad_piece":      ["bad bishop", "bad piece", "problem child", "shut in"],
    "open_file":      ["open file", "half-open", "rook on the"],
    "bishop_pair":    ["bishop pair", "two bishops", "bishop advantage"],
    "pawn_majority":  ["pawn majority", "majority", "queenside majority"],
    "passed_pawn":    ["passed pawn", "passer", "queening"],
    "isolated_pawn":  ["isolated", "isolani", "IQP"],
    "backward_pawn":  ["backward pawn", "backward", "cannot advance"],
    "battery":        ["battery", "queen and rook", "doubling on"],
    "overprotection": ["overprotect", "overprotection", "over-protect"],
    "zugzwang":       ["zugzwang", "compulsion", "must move"],
    "king_activity":  ["king march", "king to the center", "active king"],
    "rook_seventh":   ["seventh rank", "on the seventh", "seventh"],
    "pawn_chain":     ["pawn chain", "chain", "head of the chain"],
    "weakness":       ["weakness", "weak square", "weak pawn"],
}

def extract_themes(annotation_text: str) -> list[str]:
    """Extract chess concept labels from annotation text by keyword matching."""
    text = annotation_text.lower()
    themes = []
    for theme, keywords in CONCEPT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            themes.append(theme)
    return themes

def parse_pgn_file(pgn_path: str) -> list[dict]:
    """Parse an annotated PGN file into a list of training examples."""
    examples = []
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            node = game
            while node.variations:
                node = node.variations[0]
                board.push(node.move)
                comment = node.comment.strip()
                if len(comment) < 30:
                    # Skip trivial comments like "!" or "?"
                    continue
                themes = extract_themes(comment)
                if not themes:
                    # Even without explicit theme labels, the annotation text
                    # is still valuable for the language layer
                    pass
                examples.append({
                    "fen":        board.fen(),
                    "move_played": node.move.uci(),
                    "annotation": comment,
                    "themes":     themes,
                    "source":     str(pgn_path),
                    "game_phase": _get_phase(board),
                })
    return examples

def _get_phase(board: chess.Board) -> str:
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    pieces = len(board.piece_map())
    if len(board.move_stack) < 14:
        return "opening"
    elif queens == 0 or pieces < 14:
        return "endgame"
    return "middlegame"

if __name__ == "__main__":
    import sys, json
    examples = parse_pgn_file(sys.argv[1])
    print(f"Extracted {len(examples)} annotated positions")
    with open("data/training_raw.jsonl", "w") as out:
        for ex in examples:
            out.write(json.dumps(ex) + "\n")
```

Run: `python tools/parse_annotated_pgn.py "data/My_System_Games.pgn"`

### Step 2: Enrich with Stockfish Features

```python
# tools/enrich_with_sf.py
"""
Add Stockfish classical eval features to each training example.
This runs SF in non-NNUE mode to get the interpretable breakdown.
Slow — run overnight on your full dataset.
"""
import json
import subprocess
import chess

SF_PATH = "assets/engines/stockfish-windows-x86-64-avx2/stockfish/stockfish.exe"

def get_classical_eval(fen: str) -> dict:
    """Run SF with UseNNUE=false, return classical eval breakdown."""
    proc = subprocess.Popen(
        [SF_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    # Disable NNUE to get classical eval terms
    proc.stdin.write("setoption name Use NNUE value false\n")
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write("eval\n")
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    output, _ = proc.communicate(timeout=10)

    # Parse the classical eval output
    # (reuse your existing stockfish_bridge parsing logic)
    result = {}
    for line in output.splitlines():
        for term in ["Material", "Mobility", "King safety", "Threats",
                     "Passed", "Space", "Imbalance"]:
            if line.strip().startswith(term):
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        result[term.lower().replace(" ", "_")] = float(parts[2].strip())
                    except ValueError:
                        pass
    return result

# Enrich the raw dataset
with open("data/training_raw.jsonl") as f:
    examples = [json.loads(line) for line in f]

enriched = []
for i, ex in enumerate(examples):
    if i % 100 == 0:
        print(f"{i}/{len(examples)}")
    try:
        sf_features = get_classical_eval(ex["fen"])
        ex["sf_classical"] = sf_features
    except Exception as e:
        ex["sf_classical"] = {}
    enriched.append(ex)

with open("data/training_enriched.jsonl", "w") as out:
    for ex in enriched:
        out.write(json.dumps(ex) + "\n")
```

### Step 3: Encode Boards as Tensors

```python
# ml/board_encoder.py
import chess
import torch

PIECE_TYPES  = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]
COLORS       = [chess.WHITE, chess.BLACK]
NUM_FEATURES = 64 * 12   # 64 squares × 12 piece types (6 per color) = 768

def board_to_tensor(fen: str) -> torch.Tensor:
    """
    Encode a board position as a 768-element float tensor.
    Each element is 1.0 if a specific piece is on a specific square, 0.0 otherwise.
    
    Layout: [WP_a1, WP_b1, ..., BK_h8]  (white pieces first)
    This is the same representation NNUE uses as its input layer.
    """
    board  = chess.Board(fen)
    tensor = torch.zeros(NUM_FEATURES)
    for color_idx, color in enumerate(COLORS):
        for piece_idx, piece_type in enumerate(PIECE_TYPES):
            channel = color_idx * 6 + piece_idx
            for sq in board.pieces(piece_type, color):
                tensor[channel * 64 + sq] = 1.0
    return tensor

def sf_features_to_tensor(sf_classical: dict) -> torch.Tensor:
    """
    Encode Stockfish classical eval features as a small tensor.
    Returns zeros if SF features are unavailable (e.g. SF16 NNUE-only mode).
    """
    keys = ["material", "mobility", "king_safety", "threats",
            "passed", "space", "imbalance"]
    return torch.tensor([sf_classical.get(k, 0.0) for k in keys], dtype=torch.float32)

def encode_example(example: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine board tensor + SF features into a single input tensor.
    Returns (input_tensor [775], themes_tensor [NUM_CONCEPTS]).
    """
    board_enc = board_to_tensor(example["fen"])
    sf_enc    = sf_features_to_tensor(example.get("sf_classical", {}))
    x = torch.cat([board_enc, sf_enc])   # 768 + 7 = 775 features

    # Build the multi-hot theme label vector
    from ml.concept_vocab import CONCEPTS
    y = torch.zeros(len(CONCEPTS))
    for theme in example.get("themes", []):
        if theme in CONCEPTS:
            y[CONCEPTS.index(theme)] = 1.0

    return x, y
```

```python
# ml/concept_vocab.py
# The full list of chess concepts the model will learn.
# Add to this list as your data reveals new concepts.
CONCEPTS = [
    # Piece concepts
    "outpost", "bad_piece", "good_bishop", "bad_bishop", "bishop_pair",
    "rook_seventh", "rook_open_file", "battery", "overprotection",
    # Pawn concepts
    "passed_pawn", "isolated_pawn", "backward_pawn", "doubled_pawn",
    "pawn_majority", "pawn_chain", "pawn_break", "pawn_weakness",
    "pawn_storm", "pawn_shield",
    # Structural concepts
    "blockade", "open_file", "half_open_file", "weak_square",
    "strong_square", "color_complex",
    # Dynamic concepts
    "piece_activity", "king_safety", "king_activity", "initiative",
    "zugzwang", "tempo", "prophylaxis",
    # Strategic themes
    "minority_attack", "positional_sacrifice", "exchange_sacrifice",
    "simplification", "space_advantage", "fortification",
    # Tactical themes (overlap with tactic scanner)
    "pin", "fork", "skewer", "discovered_attack", "deflection",
    "zwischenzug", "overloading",
]
```

---

## 7. Model Designs — Simple to Advanced

### Model A: Theme Classifier (Build This First)

A 3-layer feedforward network. Takes 775 numbers, outputs 70+ concept probabilities. This is the simplest possible neural net beyond a linear model and is the right starting point.

```python
# ml/theme_classifier.py
import torch
import torch.nn as nn

class ThemeClassifier(nn.Module):
    """
    Multi-label classifier: board features → chess concept probabilities.
    
    Architecture: 3-layer MLP with dropout.
    Input:  775 features (768 board encoding + 7 SF eval terms)
    Output: probabilities over NUM_CONCEPTS chess concepts
    """
    def __init__(self, input_size: int = 775, num_concepts: int = 70,
                 hidden_size: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_concepts),
            # No sigmoid here — use BCEWithLogitsLoss which is numerically
            # more stable and includes sigmoid internally
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> list[str]:
        """Return concept names with probability > threshold."""
        from ml.concept_vocab import CONCEPTS
        with torch.no_grad():
            logits = self.forward(x.unsqueeze(0))   # add batch dimension
            probs  = torch.sigmoid(logits).squeeze()
        return [CONCEPTS[i] for i, p in enumerate(probs) if p > threshold]
```

Why 3 layers? Enough to learn non-linear relationships between piece placement and chess concepts, but simple enough to train on a modest dataset and interpret what went wrong when it fails.

Why dropout? Forces the network to not rely on any single feature, which reduces overfitting. During inference it's turned off (model.eval()).

### Model B: Retrieval System (No Training Required)

Before the classifier is trained, you can already build a useful retrieval layer: encode every annotated position as its board tensor, build a searchable index, and at inference time find the K most similar positions and return their annotations.

```python
# ml/retrieval.py
"""
Position retrieval: find the most similar annotated positions to a query.
No training required — just encode and index your data.
"""
import torch
import torch.nn.functional as F
import json
from pathlib import Path

class AnnotationRetriever:
    def __init__(self, data_path: str = "data/training_enriched.jsonl"):
        self._index:       list[torch.Tensor] = []
        self._annotations: list[str]          = []
        self._themes:      list[list[str]]    = []
        self._build_index(data_path)

    def _build_index(self, path: str) -> None:
        from ml.board_encoder import board_to_tensor
        print("Building retrieval index...")
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                if not ex.get("annotation") or len(ex["annotation"]) < 50:
                    continue
                self._index.append(board_to_tensor(ex["fen"]))
                self._annotations.append(ex["annotation"])
                self._themes.append(ex.get("themes", []))
        self._matrix = torch.stack(self._index)  # (N, 768)
        # L2-normalise for cosine similarity
        self._matrix = F.normalize(self._matrix, dim=1)
        print(f"Index contains {len(self._annotations)} annotated positions.")

    def query(self, fen: str, k: int = 3) -> list[dict]:
        """Return the k most similar annotated positions and their annotations."""
        from ml.board_encoder import board_to_tensor
        q = F.normalize(board_to_tensor(fen).unsqueeze(0), dim=1)
        sims = (self._matrix @ q.T).squeeze()      # (N,) cosine similarities
        topk = sims.topk(k)
        return [
            {
                "similarity":  topk.values[i].item(),
                "annotation":  self._annotations[topk.indices[i]],
                "themes":      self._themes[topk.indices[i]],
            }
            for i in range(k)
        ]
```

This immediately makes the coach smarter with zero training: when asked to explain a position, it finds the 3 most geometrically similar annotated positions and their explanations. You can use those annotations as-is, or as context for generating a blended explanation.

### Model C: Fine-Tuned Language Model (The End Goal)

Once you have ~20k high-quality (position_description + concept_labels → annotation_text) pairs, you can fine-tune a small language model to generate Nimzowitsch-style prose.

The best candidate: **Phi-3 Mini** (3.8B params, runs on 8GB VRAM, MIT licensed) or **Qwen 2.5 0.5B** (runs on CPU, ~1GB, genuinely useful).

Fine-tuning means: take a pre-trained LLM that already knows English, show it thousands of (input, target_annotation) pairs, nudge its weights so it generates chess prose in the right style. It learns chess language from your data, not from its general training.

Input format for fine-tuning:
```
### Position
White to move. Material is balanced. White has a knight on d5 supported
by the c4-pawn. Black's c7-pawn is backward on a half-open file.
SF eval: mobility +0.45, space +0.33, threats +0.12.
Concepts: outpost, blockade, backward_pawn, open_file.

### Annotation
```
Target output:
```
The knight on d5 has found its home. Nimzowitsch called this piece "the
blockader" — it sits in front of the backward c7-pawn, preventing its
advance and simultaneously attacking e7. Black's queenside is constricted;
his pieces have no good squares. White should resist the temptation to
trade this knight. Every exchange of the blockader relieves the pressure
that is slowly strangling Black's position.
```

This is achievable with ~50k such pairs and standard fine-tuning tools (HuggingFace Trainer, QLoRA for memory efficiency).

---

## 8. Training Guide — Step by Step

### Prerequisites

```bash
pip install torch torchvision          # PyTorch (CPU version is fine to start)
pip install scikit-learn pandas        # Evaluation utilities
pip install python-chess               # Already installed
```

For Model C later:
```bash
pip install transformers datasets peft bitsandbytes  # HuggingFace fine-tuning
```

### Step 1: Build Your Dataset

```bash
# Parse all your annotated PGNs
python tools/parse_annotated_pgn.py data/nimzowitsch_games.pgn >> data/training_raw.jsonl
python tools/parse_annotated_pgn.py data/carlsen_annotated.pgn >> data/training_raw.jsonl
python tools/parse_annotated_pgn.py data/my_system_examples.pgn >> data/training_raw.jsonl

# Check how many examples you have
wc -l data/training_raw.jsonl

# Enrich with SF classical features (run overnight)
python tools/enrich_with_sf.py
```

**Minimum viable dataset**: 5,000 examples with at least 1 theme label each.
**Good dataset**: 50,000+ examples, multiple themes per example, diverse sources.

### Step 2: Split into Train / Validation / Test

Never evaluate on data you trained on. The split is always:
- **Train**: 80% — the model sees these during training
- **Validation**: 10% — you check this during training to catch overfitting
- **Test**: 10% — you check this ONCE at the very end to get your final score

```python
# ml/dataset.py
import json
import random
import torch
from torch.utils.data import Dataset
from ml.board_encoder import encode_example

class ChessAnnotationDataset(Dataset):
    def __init__(self, jsonl_path: str, split: str = "train",
                 seed: int = 42):
        with open(jsonl_path) as f:
            all_examples = [json.loads(line) for line in f
                            if json.loads(line).get("themes")]
        random.seed(seed)
        random.shuffle(all_examples)
        n = len(all_examples)
        if split == "train":
            self.examples = all_examples[:int(0.8 * n)]
        elif split == "val":
            self.examples = all_examples[int(0.8 * n):int(0.9 * n)]
        else:
            self.examples = all_examples[int(0.9 * n):]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return encode_example(self.examples[idx])
```

### Step 3: Train the Classifier

```python
# ml/train_classifier.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.dataset import ChessAnnotationDataset
from ml.theme_classifier import ThemeClassifier
from ml.concept_vocab import CONCEPTS

# Hyperparameters — these are sensible starting values, tune later
EPOCHS      = 30
BATCH_SIZE  = 64
LR          = 1e-3
HIDDEN_SIZE = 512
DROPOUT     = 0.3

# Load data
train_ds = ChessAnnotationDataset("data/training_enriched.jsonl", split="train")
val_ds   = ChessAnnotationDataset("data/training_enriched.jsonl", split="val")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ThemeClassifier(num_concepts=len(CONCEPTS),
                         hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(device)

# Loss and optimiser
# BCEWithLogitsLoss is the standard loss for multi-label classification
# pos_weight handles class imbalance (some concepts are rare)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn    = nn.BCEWithLogitsLoss()
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward()          # compute gradients (PyTorch does the calculus)
        optimizer.step()         # nudge weights in the right direction
        train_loss += loss.item()
    scheduler.step()

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            val_loss += loss_fn(model(x), y).item()

    train_loss /= len(train_dl)
    val_loss   /= len(val_dl)
    print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "data/theme_classifier_best.pt")
        print(f"  ✓ Saved new best checkpoint")

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
```

Run: `python ml/train_classifier.py`

Watch the output: `train_loss` should decrease steadily. `val_loss` should also decrease, and not diverge far from `train_loss`. If `val_loss` starts going UP while `train_loss` keeps going DOWN: overfitting — either get more data or increase dropout.

### Step 4: Evaluate

```python
# ml/evaluate.py
from sklearn.metrics import f1_score, classification_report
import torch
from ml.theme_classifier import ThemeClassifier
from ml.dataset import ChessAnnotationDataset
from ml.concept_vocab import CONCEPTS
from torch.utils.data import DataLoader

model = ThemeClassifier(num_concepts=len(CONCEPTS))
model.load_state_dict(torch.load("data/theme_classifier_best.pt"))
model.eval()

test_ds = ChessAnnotationDataset("data/training_enriched.jsonl", split="test")
test_dl = DataLoader(test_ds, batch_size=64)

all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_dl:
        preds = (torch.sigmoid(model(x)) > 0.5).float()
        all_preds.append(preds)
        all_labels.append(y)

preds  = torch.cat(all_preds).numpy()
labels = torch.cat(all_labels).numpy()

# F1 score: the key metric for multi-label classification
# 1.0 = perfect, 0.0 = no better than random
print(f"Macro F1: {f1_score(labels, preds, average='macro', zero_division=0):.3f}")
print(f"Micro F1: {f1_score(labels, preds, average='micro', zero_division=0):.3f}")
print()
print(classification_report(labels, preds,
                             target_names=CONCEPTS, zero_division=0))
```

What to aim for:
- **Macro F1 > 0.40**: reasonable with 5k examples, means the net is learning something real
- **Macro F1 > 0.65**: good — usable in production
- **Macro F1 > 0.80**: excellent — will make the coach noticeably better than rule-based

---

## 9. Evaluation — How to Know It Works

Beyond F1 score, do qualitative checks on real positions:

```python
# Manual test
from ml.board_encoder import board_to_tensor
from ml.theme_classifier import ThemeClassifier
from ml.concept_vocab import CONCEPTS
import torch

model = ThemeClassifier(num_concepts=len(CONCEPTS))
model.load_state_dict(torch.load("data/theme_classifier_best.pt"))

# Famous blockade position — Nimzowitsch vs Systemsson 1927
fen = "r2qr1k1/1b2bppp/pp1ppn2/8/3NP3/2NB4/PPP2PPP/R1BQR1K1 w - - 4 14"
concepts = model.predict(board_to_tensor(fen))
print("Detected concepts:", concepts)
# Expected: ["outpost", "piece_activity", "space_advantage"]

# Bad bishop position
fen2 = "6k1/p4ppp/1pb5/4p3/4P3/1P4P1/P4P1P/2B3K1 w - - 0 1"
concepts2 = model.predict(board_to_tensor(fen2))
print("Detected concepts:", concepts2)
# Expected: ["bad_bishop", "pawn_chain", "color_complex"]
```

If the model gets these wrong, it's either: (a) not enough training examples for this concept, (b) the keyword extraction in the parser missed these examples, or (c) the model needs more capacity (increase `hidden_size`).

---

## 10. Integration with ChessPlayer

Once the classifier is trained, plug it into `strategy_engine.py` as a replacement for the extractor outputs:

```python
# In strategy_engine.py — new method

def _classify_concepts(
    self, board: chess.Board, sf_classical: dict
) -> list[tuple[str, float]]:
    """
    Run the trained theme classifier and return (concept, probability) pairs.
    Falls back to empty list if the model is not loaded.
    """
    if self._theme_classifier is None:
        return []
    from ml.board_encoder import board_to_tensor, sf_features_to_tensor
    import torch
    x = torch.cat([
        board_to_tensor(board.fen()),
        sf_features_to_tensor(sf_classical),
    ])
    with torch.no_grad():
        probs = torch.sigmoid(self._theme_classifier(x.unsqueeze(0))).squeeze()
    from ml.concept_vocab import CONCEPTS
    return [
        (CONCEPTS[i], probs[i].item())
        for i in range(len(CONCEPTS))
        if probs[i] > 0.4
    ]
```

The retrieval layer slots into `explainer.py`:

```python
# In explainer.py — use retrieval when concepts don't map to a good template
from ml.retrieval import AnnotationRetriever

_retriever = None   # lazy-load on first call

def _get_retrieved_annotation(fen: str, concepts: list[str]) -> str | None:
    global _retriever
    if _retriever is None:
        _retriever = AnnotationRetriever()
    results = _retriever.query(fen, k=3)
    if not results or results[0]["similarity"] < 0.85:
        return None   # not similar enough, fall back to templates
    return results[0]["annotation"]
```

---

## 11. Phased Roadmap

### Phase 1 — Data Collection (weeks 1–4)

- [ ] Write `tools/parse_annotated_pgn.py` (§6 Step 1)
- [ ] Collect and parse: Nimzowitsch games, Carlsen annotated games, My System examples, Silman positions, opening study PGNs
- [ ] Target: **10,000+ annotated positions** in `data/training_raw.jsonl`
- [ ] Manual review: check 100 random examples, fix keyword extractor for missed themes
- [ ] Run SF enrichment on full dataset (`tools/enrich_with_sf.py`)
- [ ] Deliverable: `data/training_enriched.jsonl` with themes and SF features

### Phase 2 — Theme Classifier (weeks 5–8)

- [ ] Write `ml/board_encoder.py`, `ml/concept_vocab.py`, `ml/dataset.py`
- [ ] Write `ml/theme_classifier.py`
- [ ] Write `ml/train_classifier.py` and `ml/evaluate.py`
- [ ] Train first model, check F1 on test set
- [ ] If F1 < 0.40: audit training data, expand keyword extractor
- [ ] If F1 > 0.50: integrate into `strategy_engine.py`
- [ ] Deliverable: `data/theme_classifier_best.pt` with F1 > 0.50

### Phase 3 — Retrieval Layer (weeks 9–12)

- [ ] Write `ml/retrieval.py`
- [ ] Build retrieval index from enriched dataset
- [ ] Integrate into `explainer.py` as fallback when template quality is low
- [ ] Qualitative test: does the retrieved annotation feel relevant?
- [ ] Deliverable: retrieval working in production; coach uses it for ~30% of positions

### Phase 4 — Language Model Fine-tuning (months 4–6)

- [ ] Expand dataset to **50,000+ examples** with full annotation text
- [ ] Format dataset for LLM fine-tuning (§7 Model C)
- [ ] Choose model: Phi-3 Mini (GPU) or Qwen 0.5B (CPU)
- [ ] Fine-tune using QLoRA (memory-efficient fine-tuning for low VRAM)
- [ ] Evaluate: do generated annotations sound like chess masters?
- [ ] Integrate as `explainer.generate(fen, concepts)` replacing template lookup
- [ ] Deliverable: fine-tuned local model running fully offline in ChessPlayer

### Phase 5 — Voice & Style (months 7+)

- [ ] Add Nimzowitsch source material specifically to fine-tuning data
- [ ] Tune style via system prompt engineering during fine-tuning
- [ ] Add game history context (what happened in the last 5 moves influences tone)
- [ ] Continuous data collection: add high-quality positions from play sessions

---

## 12. File & Folder Layout for ML Work

```
ChessPlayer - v3.0.0/
├── src/
│   ├── chess_coach/
│   │   ├── coach/
│   │   │   ├── explainer.py          ← integrate retrieval & LM here
│   │   ├── core/
│   │   │   ├── strategy_engine.py    ← integrate classifier here
│   │   └── ...
│   └── ...
│
├── ml/                               ← NEW: all ML code lives here
│   ├── board_encoder.py              ← FEN → tensor
│   ├── concept_vocab.py              ← CONCEPTS list
│   ├── dataset.py                    ← PyTorch Dataset
│   ├── theme_classifier.py           ← Model A: MLP classifier
│   ├── retrieval.py                  ← Model B: nearest-neighbour
│   ├── train_classifier.py           ← training loop
│   └── evaluate.py                   ← F1 evaluation
│
├── tools/                            ← NEW: data processing scripts
│   ├── parse_annotated_pgn.py        ← PGN → training_raw.jsonl
│   └── enrich_with_sf.py            ← add SF classical features
│
├── data/
│   ├── index.sqlite                  ← existing game browser index
│   ├── chess_coach.db                ← existing phrase database
│   ├── training_raw.jsonl            ← NEW: parsed positions + annotations
│   ├── training_enriched.jsonl       ← NEW: + SF classical features
│   └── theme_classifier_best.pt      ← NEW: trained model checkpoint
│
└── TODO.md                           ← this document
```

---

## Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Keep tactic scanner extractors | They are rule-based and correct when they fire. The net doesn't replace deterministic rules. |
| Retire eval extractors as primary signals | SF classical eval is the correct version of what they compute. Use SF directly. |
| Start with MLP classifier, not Transformer | 3-layer MLP is learnable, interpretable when it fails, trainable on 5k examples. Transformer needs 50k+. |
| Multi-label, not single-label | Any position can have multiple concepts simultaneously (outpost + backward pawn + open file all at once). |
| Retrieval before generation | Retrieval requires zero training. It immediately improves the coach using your actual literature. Build it first. |
| Local model only, no API calls | Matches the project's no-cloud-dependency requirement. Phi-3 Mini runs on 8GB VRAM, Qwen 0.5B runs on CPU. |
| Annotated PGN as primary training source | Richest annotation density, direct (position, explanation) pairing, processable with existing python-chess tooling. |
