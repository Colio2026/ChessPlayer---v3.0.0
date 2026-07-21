# ChessPlayer v3.0.0

A desktop chess analysis and coaching application for Windows. Browse large PGN game databases, analyse positions with a UCI engine, and receive structured strategic coaching — all running locally with no cloud dependency.

The coaching layer is actively evolving from a deterministic phrase-based system toward a fully trained neural network that understands 49 chess concepts by name, trained on millions of master positions and puzzle databases.

---

## Table of Contents

1. [Features](#features)
2. [Screenshots](#screenshots)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Stockfish Setup](#stockfish-setup)
6. [Running the App](#running-the-app)
7. [Configuration](#configuration)
8. [Using the App](#using-the-app)
   - [Game Browser](#game-browser)
   - [Board & PGN Panel](#board--pgn-panel)
   - [Engine Panel](#engine-panel)
   - [Variations Panel](#variations-panel)
   - [Chess Coach](#chess-coach)
9. [Indexing PGN Files](#indexing-pgn-files)
10. [Project Architecture](#project-architecture)
    - [Directory Layout](#directory-layout)
    - [Module Map](#module-map)
    - [Dependency Rules](#dependency-rules)
    - [Signal Flow](#signal-flow)
11. [Coach Nimzowitsch — The Neural Network](#coach-nimzowitsch--the-neural-network)
    - [The 49 Concepts](#the-49-concepts)
    - [Model Architecture](#model-architecture)
    - [Training Pipeline](#training-pipeline)
    - [Data Sources](#data-sources)
    - [Algorithmic Detectors](#algorithmic-detectors)
    - [Roadmap](#roadmap)
12. [Training the Coach](#training-the-coach)
13. [The Deterministic Coach (Legacy)](#the-deterministic-coach-legacy)
14. [Database Schema](#database-schema)
15. [Configuration System](#configuration-system)
16. [Development Scripts](#development-scripts)
17. [Contributing](#contributing)

---

## Features

- **PGN Game Browser** — load any `.pgn` file or directory; filter by player, event, opening, date, and ECO code; paginated with lazy loading for databases of any size
- **Interactive Board** — drag-and-drop piece moves, full variation tree support, promote/demote variations, inline move comments
- **Engine Analysis** — Stockfish UCI integration with multi-PV evaluation, animated eval bar, and best-move arrows; runs on a background thread so the UI stays responsive
- **Continuation Statistics** — see how often a position arises in your loaded library and what the top continuations are, powered by an O(1) position-tree lookup
- **Chess Coach (Deterministic)** — Nimzowitsch-style strategic coaching: classifies each position into one of four strategies (Blitz, Flank, Fortress, Feint) and generates natural-language guidance assembled from a curated phrase database
- **Coach Nimzowitsch (Neural Network)** — a 49-class multi-label classifier (Phase 4B, 3.08M parameters) trained to identify chess concepts by name from board position, move history, and 1,811-dim geometric heuristics; trained on 1.5M+ examples from master games, Lichess puzzles, and algorithmically-labelled positions; best validation Macro F1: 0.56
- **Offline-first** — all analysis, coaching, and database queries run locally

---

## Screenshots

*Screenshots will be updated when the ML coaching UI is integrated.*

**PGN Editor — game browser, interactive board, and engine analysis**

![PGN Editor](docs/screenshots/ChessPNGEditorAlpha.png)

**Chess Coach — strategic coaching panel with Stockfish evaluation terms**

![Chess Coach](docs/screenshots/ChessCoachAlpha.png)

**Stockfish Integration — live evaluation bar and multi-PV analysis**

![Stockfish Integration](docs/screenshots/StockFishIntegration.png)

---

## Requirements

- **Python** 3.11 or 3.12
- **Windows** 10 / 11 (64-bit)
- **Stockfish** binary — see [Stockfish Setup](#stockfish-setup)

Python packages (installed via `pip`):

| Package | Version | Purpose |
|---|---|---|
| PySide6 | `>=6.8, <6.9` | GUI framework |
| python-chess | `>=1.11, <2.0` | Board logic, PGN parsing |
| PyYAML | `>=6.0.2, <7.0` | Configuration |
| torch | `>=2.0` | Neural network training and inference |
| numpy | `>=1.24` | Tensor utilities |

The `torch` and `numpy` packages are only required for the ML coaching pipeline (`src/chess_coach/ml/` and `tools/`). The main application runs without them if the coach ML components are not used.

---

## Installation

```powershell
# 1. Clone the repository
git clone https://github.com/TheSladTosser/ChessPlayer.git
cd "ChessPlayer - v3.0.0"

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Stockfish Setup

ChessPlayer does not bundle a Stockfish binary. You need to download it separately.

1. Go to [https://stockfishchess.org/download/](https://stockfishchess.org/download/) and download the Windows build for your CPU (AVX2 is recommended for modern hardware).
2. Extract the `.exe` to:
   ```
   assets/engines/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe
   ```
3. If you use a different filename or path, update `engine.path` in `config/default.yaml` (or your user config) to point at the actual binary.

Any UCI-compatible engine will work — the path in config is the only coupling.

---

## Running the App

```powershell
cd src\chessplayer
python main.py
```

Or from the repo root:

```powershell
.\scripts\run_dev.ps1
```

**CLI flags:**

| Flag | Effect |
|---|---|
| *(none)* | Launch the GUI |
| `--index` | Rebuild the index for all configured sources, then exit |
| `--index-source <path>` | Index a single `.pgn` file or directory, then exit |

---

## Configuration

Configuration is a three-layer merge system.

| Layer | File | Purpose |
|---|---|---|
| 1 — Defaults | `config/default.yaml` | Canonical defaults, never edited at runtime |
| 2 — Dev overrides | `config/dev.yaml` | Local developer overrides (gitignored) |
| 3 — User overrides | `%APPDATA%\CHESSPLAYER\config.yaml` | Auto-saved user preferences |

Key settings in `config/default.yaml`:

```yaml
engine:
  enabled_on_start: false
  path: "assets/engines/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"

pgn_sources:
  active_source:
    type: "archive_file"
    path: "data/Carlsen.pgn"

coach:
  pgn_source: "data/Carlsen.pgn"
  phrase_db: "data/chess_coach.db"
  movetime_ms: 500
```

See [config/README_config.md](config/README_config.md) for the full schema.

---

## Using the App

### Game Browser

The left panel is the game archive. Filter by White, Black, Event, Site, ECO, or date range. Click any row to load that game onto the board. The table lazy-loads from SQLite so multi-gigabyte databases open instantly.

### Board & PGN Panel

- **Drag pieces** to make moves. Promotion prompts appear automatically.
- **Click moves in the PGN panel** to navigate to that ply.
- **Arrow keys** step forward/back through the mainline.
- Right-click a variation move to promote it to the mainline.
- Click the comment icon on a move to add or edit a PGN comment.

### Engine Panel

Toggle Stockfish on/off with the toolbar button. When active, shows centipawn score, animated eval bar, best move (highlighted on board), and principal variation lines.

### Variations Panel

Shows how often the current position appears in your loaded library and what moves were played from it, with win/draw/loss percentages. Uses a pre-built move tree for O(1) lookups.

### Chess Coach

The Coach tab analyses the current position and produces a strategic recommendation. The current live system is the deterministic coach (see [The Deterministic Coach](#the-deterministic-coach-legacy)). The neural network coach is in training and will be integrated into this panel.

---

## Indexing PGN Files

On first run, ChessPlayer indexes whatever PGN source is set in config. Indexing records a byte offset per game into SQLite — it does **not** copy game text into the database. A 4 GB `.pgn` file indexes in a few minutes and opens games in milliseconds.

To add a new source:

1. Drop your `.pgn` file into `data/`.
2. Update `pgn_sources.active_source.path` in `config/default.yaml`.
3. Run with `--index` or let auto-indexing run on next launch.

The move tree for continuation statistics is built separately and cached in `data/trees/` as a gzip-pickled file. It rebuilds automatically when the source changes.

---

## Project Architecture

### Directory Layout

```
ChessPlayer - v3.0.0/
|
+-- config/                     # YAML configuration (3-layer merge)
|   +-- default.yaml
|   +-- dev.yaml                # gitignored, local overrides
|   +-- README_config.md
|
+-- data/                       # Runtime data (gitignored)
|   +-- index.sqlite            # Game metadata + byte offsets
|   +-- chess_coach.db          # Deterministic coaching phrase database
|   +-- Carlsen.pgn             # Default game library
|   +-- Caissabase.pgn          # Master game database for ML training
|   +-- lichess_elite_2020-10.pgn
|   +-- lichess_db_puzzle.csv   # Lichess 6M puzzle database
|   +-- annotated_pgns/         # Annotated PGN training material
|   |   +-- Raw_pgn/            # Manually curated annotated games
|   |   +-- lichess_studies/    # Concept-organised Lichess studies
|   +-- training_raw.jsonl      # Assembled training dataset (generated)
|   +-- classifier_best.pt      # Best model checkpoint (generated)
|   +-- thresholds.json         # Per-class calibrated thresholds (generated)
|   +-- trees/                  # MoveTree cache files (.pkl.gz)
|
+-- assets/
|   +-- pieces/                 # PNG piece images
|   +-- engines/                # Place your Stockfish binary here
|   +-- scraping/               # Lichess data collection scripts
|   +-- ui.qss                  # Qt stylesheet
|
+-- src/
|   +-- chessplayer/            # Main application package
|   |   +-- app.py
|   |   +-- core/               # Chess state machine (no Qt)
|   |   +-- pgn/                # Database, indexer, move tree
|   |   +-- engine/             # UCI engine wrapper
|   |   +-- config/             # Config loader
|   |   +-- utils/
|   |   +-- ui/                 # All PySide6 widgets
|   |       +-- qml/BoardView.qml
|   |       +-- main_window/window.py
|   +-- chess_coach/            # Coaching backend
|       +-- core/               # Data types + strategy engine
|       +-- extractors/         # Six position-metric extractors
|       +-- strategies/         # Four strategy detectors
|       +-- database/           # Phrase DB, pattern matcher, PGN indexer
|       +-- ml/                 # Neural network coach
|           +-- concept_vocab.py    # 49 concept labels (stable, order matters)
|           +-- board_encoder.py    # FEN -> 1001-dim tensor
|           +-- dataset.py          # JSONL loader + train/val/test split
|           +-- classifier.py       # MLP model definition
|           +-- train.py            # Training loop
|           +-- evaluate.py         # Threshold calibration + evaluation
|
+-- tools/                      # ML data pipeline scripts
|   +-- parse_annotated_pgn.py  # Extracts examples from annotated PGNs
|   +-- ingest_lichess_csv.py   # Processes Lichess 6M puzzle CSV
|   +-- ingest_game_database.py # Scans master game PGNs algorithmically
|   +-- label_positions.py      # 32 algorithmic concept detectors
|   +-- inspect_weights.py      # Post-training weight analysis
|
+-- retrain.ps1                 # Full pipeline: data -> train -> evaluate
+-- scripts/                    # run_dev.ps1, run_dev.sh, build_windows.ps1
```

### Module Map

| Module | Role |
|---|---|
| `core/pgn_edit.py` | Central chess controller — owns all move/navigation/annotation logic |
| `core/game_session.py` | `GameSession` — undo/redo wrapper around `chess.Board` |
| `pgn/store.py` | `PgnStore` — SQLite query layer for game metadata |
| `pgn/indexer.py` | PGN scanner, writes byte offsets to SQLite |
| `pgn/move_tree.py` | `MoveTree` — position-frequency tree for O(1) continuation lookup |
| `engine/uci_engine.py` | Stockfish subprocess controller, runs on QThread |
| `config/loader.py` | Three-layer config merge |
| `ui/main_window/window.py` | `MainWindow` — central coordinator, owns all signals |
| `chess_coach/ml/concept_vocab.py` | Canonical ordered list of all 49 concept labels |
| `chess_coach/ml/board_encoder.py` | `fen_to_tensor()` (1001-dim) + `move_to_tensor()` (128-dim) + `COMBINED_SIZE_V4B` (1714) |
| `chess_coach/ml/classifier.py` | `ChessConceptClassifier` — Phase 4B: spatial bottleneck + GRU(256) + 1024/512 MLP |
| `chess_coach/ml/train.py` | Training loop with macro F1 early stopping |
| `chess_coach/ml/evaluate.py` | Per-class threshold calibration, spot checks |
| `tools/label_positions.py` | `label_position(board)` — 32 algorithmic concept detectors |
| `tools/ingest_lichess_csv.py` | Lichess CSV ingestion with tag mapping + algo detection |
| `tools/ingest_game_database.py` | Master game database scanner, algo detection only |
| `tools/parse_annotated_pgn.py` | Keyword-based concept extraction from annotated PGNs |

### Dependency Rules

Qt is never imported outside of `ui/`.

```
ui/          -> core/, pgn/, engine/, config/, utils/
core/        -> (none - pure chess logic)
pgn/         -> utils/
engine/      -> (none - subprocess wrapper)
chess_coach/ -> core/ only
chess_coach/ml/ -> (none - standalone, numpy/torch only)
tools/       -> src/chess_coach/ml/ (concept_vocab, board_encoder)
```

### Signal Flow

```
User drags piece on QML board
  +-- BoardBridge.attemptMove(from_sq, to_sq)
       +-- PgnEditor.try_user_move()
            +-- PgnEditor updates board + PGN node
                 +-- bridge.moveMade.emit(san)
                      +-- MainWindow._on_position_changed()
                           +-- pgn_panel.refresh()
                           +-- variations_panel.refresh(prefix_uci)
                           +-- engine_panel.trigger_analysis()
                           +-- coachRequested.emit(fen, prefix_uci)
                                +-- MainWindow._on_coach_help_requested()
                                     +-- StrategyEngine.analyse(board, side)
                                          +-- CoachOutput -> CoachPanel
```

---

## Coach Nimzowitsch — The Neural Network

The central long-term project within ChessPlayer is **Coach Nimzowitsch**: a neural network trained to identify 49 chess concepts by name from a board position and the key move being played. This gives the coach a rich, human-readable vocabulary it can use to explain what's happening in a position — not just *what* Stockfish recommends but *why* it's good in terms a student can understand and remember.

### The 49 Concepts

The full concept vocabulary, in output-neuron order (fixed — adding new concepts must go at the end only, or all saved checkpoints are invalidated):

**Tactical** (15)
`pin` · `fork` · `skewer` · `discovery` · `x_ray` · `double_check` · `clearance` · `deflection` · `overloading` · `zwischenzug` · `interference` · `back_rank` · `sacrifice` · `mating_attack` · `trapped_piece`

**Piece concepts** (8)
`outpost` · `blockade` · `bad_bishop` · `good_bishop` · `bishop_pair` · `piece_activity` · `battery` · `rook_seventh`

**Pawn structure** (9)
`passed_pawn` · `promotion` · `isolated_pawn` · `backward_pawn` · `doubled_pawn` · `pawn_majority` · `pawn_chain` · `pawn_storm` · `pawn_island`

**King & endgame** (11)
`king_safety` · `king_activity` · `shouldering` · `opposition` · `zugzwang` · `rook_endgame` · `pawn_endgame` · `bishop_endgame` · `knight_endgame` · `queen_endgame` · `drawn_position`

**Positional / Strategic** (6)
`weak_square` · `open_file` · `space_advantage` · `development_lead` · `initiative` · `prophylaxis`

### Model Architecture

**Phase 4B (current)**

```
Raw static input: 3,013-dim
  - 1,001  board encoding  (12×64 piece channels, attack maps, pawn structure, king shelter, mobility)
  -   128  move one-hot    (64 from-square + 64 to-square — the key move being played)
  - 1,811  algo_v4 features  (explicit geometric chess heuristics — primary signal source)
  -    59  v3 concept bits   (binary algorithmic concept flags — bypass the spatial bottleneck)
  -    14  SF classical eval  (Mobility, King safety, Threats, Passed, Space, Pawns, Imbalance × 2 sides)

Spatial bottleneck:  Linear(1811, 256) → ReLU → Dropout(0.3)   # compresses algo_v4

GRU:  144-dim per-step history  →  256-dim context
      (encodes piece type, capture, check, and side-to-move for each prior half-move)

Combined (post-projection):  1,714-dim
  = board(1001) + move(128) + spatial_proj(256) + v3(59) + sf(14) + gru(256)

MLP head:
  Hidden 1:  Linear(1714, 1024) → BatchNorm → ReLU → Dropout(0.4)
  Hidden 2:  Linear(1024,  512) → BatchNorm → ReLU → Dropout(0.2)
  Output:    Linear( 512,   49) → per-class sigmoid  (BCEWithLogitsLoss)

Parameters: 3.08M
Best validation Macro F1: 0.56
```

**Training details:**
- Loss: `BCEWithLogitsLoss` with per-class `pos_weight` clamped to (1.0, 20.0); label smoothing ε=0.05
- Optimiser: AdamW, weight decay 6e-3, initial LR 1e-3 with cosine annealing over 100 epochs
- Early stopping: macro F1 on validation set, patience 10 epochs
- Thresholds: calibrated per class on the validation set after training, saved to `data/thresholds.json`

**Why multi-label:** Most positions have multiple concepts simultaneously present. A position can be a `pin` + `battery` + `pawn_storm` + `king_safety` threat all at once. Single-label classification would lose this richness.

**Why the move feature:** The key move often determines the concept. A rook move to e7 could be `rook_seventh`, `battery`, or `clearance` depending on context. The 128-dim move one-hot gives the model the move being played alongside the static board, dramatically reducing ambiguity for tactical concepts.

### Training Pipeline

The full pipeline is orchestrated by `retrain.ps1`. Three data sources are assembled into `data/training_raw.jsonl` and then trained on:

```
1. parse_annotated_pgn.py  <-- data/annotated_pgns/
      |
      | Keyword matching on { comment } blocks.
      | Primary source for dynamic/meta concepts that require human prose
      | to identify: tempo, combination, fortification, coordination, etc.
      |
      v
2. ingest_lichess_csv.py   <-- data/lichess_db_puzzle.csv  (6M puzzles)
      |
      | Maps Lichess theme tags to our concept vocab.
      | ALSO runs all 32 algorithmic detectors on every puzzle position,
      | filling structural concepts Lichess doesn't explicitly tag.
      | Applies a per-concept cap (default 50k) to prevent common
      | structural concepts from drowning out rare tactical ones.
      |
      v
3. ingest_game_database.py <-- data/Caissabase.pgn, Carlsen.pgn, etc.
      |
      | Pure algorithmic detection on master game positions.
      | Samples every 5th move from each game.
      | Provides rich, unbiased positional examples for structural
      | concepts that puzzles under-represent.
      |
      v
data/training_raw.jsonl
      |
      v
train.py  -->  classifier_best.pt
      |
      v
evaluate.py  -->  thresholds.json
```

### Data Sources

| Source | Examples (approx) | Concepts covered |
|---|---|---|
| Annotated PGNs (`data/annotated_pgns/`) | ~50k labeled | Dynamic/meta concepts, all 49 via keyword |
| Lichess puzzle CSV (6M puzzles) | ~830k | 40+ concepts hit 50k cap |
| Master game databases (Caissabase etc.) | ~700k+ | Structural/strategic concepts via algo detectors |

Total training examples: **1.59M** (80/10/10 train/val/test split)

### Algorithmic Detectors

`tools/label_positions.py` contains 32 position-level detectors that return concept labels directly from a `chess.Board` object. These fire on every position in the Lichess CSV and game database pipelines, providing high-precision labels at scale without requiring human annotation.

**Detectable from a single position:**

| Category | Concepts |
|---|---|
| Pawn structure | `passed_pawn` `isolated_pawn` `doubled_pawn` `pawn_island` `pawn_majority` `backward_pawn` `pawn_chain` `pawn_storm` |
| Piece quality | `bad_bishop` `good_bishop` `bishop_pair` `battery` `blockade` `outpost` `rook_seventh` `piece_activity` |
| King | `king_safety` `king_activity` `back_rank` `opposition` `shouldering` |
| Squares / files | `open_file` `weak_square` `space_advantage` |
| Endgame | `rook_endgame` `pawn_endgame` `bishop_endgame` `knight_endgame` `queen_endgame` `drawn_position` `zugzwang` |
| Tactics | `pin` `fork` `skewer` `x_ray` `overloading` |
| Strategic | `development_lead` `promotion` |

**Not algorithmically detectable** (require move history or prose): `discovery`, `double_check`, `clearance`, `deflection`, `zwischenzug`, `interference`, `sacrifice`, `mating_attack`, `trapped_piece`, `initiative`, `prophylaxis`. These come exclusively from Lichess tags and annotated PGN keyword matching.

### Roadmap

The coach has been through four architectural phases. Phase 4B is the current production model. Full experiment history with per-run numbers, hypotheses, and lessons is in [`docs/experiments.md`](docs/experiments.md).

**Phase 1 (superseded)** — 1,188-dim static features, MLP baseline. Macro F1: ~0.30. Proof of concept only.

**Phase 2 (superseded)** — Scaled MLP, ~1M examples. Macro F1: ~0.51. Good recall, poor precision — overfires.

**Phase 4B (current)** — 3,013-dim raw input → 1,714-dim combined after spatial bottleneck + GRU(256). Spatial bottleneck (Linear 1811→256) compresses 1,811 explicit geometric heuristics into a 256-dim representation. GRU reads per-move (piece, capture, check, color) over up to 60 prior half-moves. Macro F1: **0.5614** (calibrated, 49 concepts). Champion run on 1.23M examples; retrain in progress on 1.59M.

**Phase 5 variants (explored, abandoned)** — Three variants (5, 5C, 5D) substituted or augmented with Stockfish16 NNUE Feature Transformer activations (2,048-dim). All stalled at F1 ≤ 0.47, with pure NNUE runs at 0.33–0.35, due to task misalignment: NNUE represents centipawn evaluation, not concept identity. See [`docs/phase5_nnue_integration_plan.md`](docs/phase5_nnue_integration_plan.md) (ABANDONED) and [`docs/experiments.md`](docs/experiments.md) §Lessons.

**Next: Application integration**
The Phase 4B classifier is ready for integration into the live chess coaching panel. The intended flow:
1. User makes a move in the game browser.
2. `coach.analyze(fen, history_uci=...)` runs concept classification with GRU history.
3. Concepts above their calibrated thresholds (with Schmitt-trigger hysteresis) drive RAG retrieval.
4. The Nimzowitsch phrase database returns natural-language explanation keyed to active concept labels.
5. NNUE evaluation gates the coaching output: only surface concepts that make sense given Stockfish's assessment of the position.

---

## Training the Coach

Prerequisites: `torch`, `numpy`, `python-chess` installed. The Lichess puzzle CSV and at least one master game database must be in `data/`.

**Full pipeline (from repo root):**

```powershell
.\retrain.ps1
```

The script runs all six stages in order, stopping on any failure:

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `parse_annotated_pgn.py` | `data/annotated_pgns/` | `training_raw.jsonl` |
| 2 | `ingest_lichess_csv.py` | `data/lichess_db_puzzle.csv` | appended to `training_raw.jsonl` |
| 3 | `ingest_game_database.py` | `data/Caissabase.pgn` etc. | appended to `training_raw.jsonl` |
| 4 | `train.py` | `training_raw.jsonl` | `classifier_best.pt` |
| 5 | `evaluate.py` | `classifier_best.pt` | `thresholds.json` + evaluation report |
| 6 | `inspect_weights.py` | `classifier_best.pt` | Weight norm report per concept |

**Running individual steps:**

```powershell
# Dry run: see label distribution from Lichess CSV without writing
python tools/ingest_lichess_csv.py --input data/lichess_db_puzzle.csv --count-only --limit 50000

# Scan specific game databases for under-represented concepts only
python tools/ingest_game_database.py --input data/Carlsen.pgn --target battery,blockade,minority_attack --append

# Training only (if training_raw.jsonl already exists)
python -m src.chess_coach.ml.train

# Evaluate + calibrate thresholds on existing checkpoint
python -m src.chess_coach.ml.evaluate
```

**JSONL format** — each training example:
```json
{
  "fen":      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
  "move_uci": "f3g5",
  "themes":   ["pin", "fork", "battery"],
  "comment":  "The knight move creates a fork threat while exploiting the pin...",
  "phase":    "opening"
}
```

---

## The Deterministic Coach (Legacy)

The original coaching system remains the live implementation while the neural network is being trained. It is fully functional and is the system currently running in the application.

### How it works

```
chess.Board
    |
    v
1. Extractors (6 modules) -> list[MetricSignal]
    |
    v
2. Phase filter -> re-weight signals for opening/middlegame/endgame
    |
    v
3. Strategy scoring -> four detectors score 0.0-1.0 each
    |                  blitz / flank / fortress / feint
    v
4. Conflict resolver -> 5-rule priority cascade picks primary + confidence
    |
    v
5. Plan recommender -> selects weakness squares and move flags
    |
    v
6. GM precedents -> PatternMatcher queries coach_positions table
    |
    v
7. Narrator -> slot-fills phrase templates -> CoachOutput
```

### The four strategies

| Strategy | What it means |
|---|---|
| **Blitz** | Direct assault on an exposed king — concentrate attackers, open lines, strike before defence consolidates |
| **Flank** | Positional squeeze — restrict mobility, occupy outposts, cramp the opponent until the position collapses |
| **Fortress** | Blockade defence — when objectively worse, seal open files, fix the pawn structure, hold the wall |
| **Feint** | Wing misdirection — hold tension deliberately, bait a commitment to one wing, then switch |

A strategy must score above **0.65** to be considered "fired." Two strategies within 0.08 of each other are both surfaced with the tension between them noted. The Feint gate prevents misdirection from being named as primary without GM pattern database confirmation.

### Extractors

| Extractor | Signals |
|---|---|
| `king_safety.py` | `king_exposure`, `sacrifice_delta` |
| `space_control.py` | `space_delta_queenside`, `space_delta_kingside` |
| `piece_mobility.py` | `piece_mobility_ratio` |
| `pawn_structure.py` | `pawn_fixedness`, `weak_pawns`, `passed_pawn` |
| `material_balance.py` | `material_imbalance`, `eval_deficit` |
| `tactic_scanner.py` | `tactic_pin`, `tactic_fork`, `tactic_skewer`, `tactic_discovery` |

### Phrase database

`data/chess_coach.db` contains phrase templates attributed to chapters of *My System* (Aaron Nimzowitsch, 1925). The narrator fills five placeholders (`{square}`, `{file}`, `{piece}`, `{side}`, `{target}`) from the live `MetricSignal` data at query time. No extractor ever generates English; no narrator ever generates scores. This separation is a hard architectural rule.

---

## Database Schema

### `data/index.sqlite` — game index

```sql
CREATE TABLE sources (
    source_id    INTEGER PRIMARY KEY,
    type         TEXT,           -- 'archive_file' or 'directory'
    path         TEXT
);

CREATE TABLE games (
    game_id      INTEGER PRIMARY KEY,
    source_id    INTEGER,
    pgn_path     TEXT,
    offset_bytes INTEGER,        -- byte offset of this game in the file
    white        TEXT,
    black        TEXT,
    result       TEXT,
    event        TEXT,
    site         TEXT,
    date         TEXT,
    eco          TEXT,
    opening      TEXT
);
```

Full game text is never duplicated. `PgnStore.open_game_pgn_text(game_id)` seeks to `offset_bytes` and parses one game on demand.

### `data/trees/<sha1>.pkl.gz` — move tree cache

A gzip-pickled `MoveTree` keyed by the source file's SHA-1. Position keys are the first four FEN fields so transpositions hash to the same node.

### `data/training_raw.jsonl` — ML training dataset

Generated by the data pipeline. One JSON object per line in the format described in [Training the Coach](#training-the-coach). Not committed to git.

### `data/thresholds.json` — per-class thresholds

Generated by `evaluate.py --calibrate` after training. Maps each of the 49 concept labels to a calibrated sigmoid threshold (0.0–1.0) that maximises per-class F1 on the validation set.

---

## Configuration System

The merge runs at startup:

```
default.yaml  --------+
dev.yaml  ------------+-- deep_merge() -> live config dict
%APPDATA%\CHESSPLAYER\config.yaml  --+
```

User overrides write back automatically when the active source changes. See [config/README_config.md](config/README_config.md) for the full schema.

---

## Development Scripts

| Script | Purpose |
|---|---|
| `retrain.ps1` | Full ML pipeline: annotated PGNs -> Lichess CSV -> game databases -> train -> evaluate -> inspect |
| `scripts/run_dev.ps1` | Launch the app from the correct working directory (Windows PowerShell) |
| `scripts/run_dev.sh` | Same, for bash |
| `scripts/build_windows.ps1` | Package the app for distribution |

---

## Contributing

- Module boundaries and the no-Qt-outside-ui rule are load-bearing — keep them.
- **Concept vocab (`concept_vocab.py`) is append-only.** Adding a concept in the middle shifts every output neuron index and invalidates all saved checkpoints. New concepts go at the end only.
- New algorithmic detectors go in `tools/label_positions.py` and must be added to both `DETECTABLE_CONCEPTS` and the `label_position()` call loop.
- New coaching phrases go in the phrase DB seed scripts keyed by `strategy/metric/severity/fragment_type`.
- New extractors go in `src/chess_coach/extractors/` and must be added to the strategy engine's extractor list.
- Run the coach test suite before pushing: `pytest src/chess_coach/tests/`.

---

## Code Review — Known Flaws & Technical Debt

> This section is a senior-developer audit of the current state of the project. It is written in the spirit of a hard but fair internship review: the goal is not to catalogue shame but to create a clear improvement backlog. Every item below is a real problem with a real fix. Work through them.

---

### 🔴 Critical — Bugs or Silently Wrong Behaviour

**1. ~~`--quick` mode references a nonexistent attribute~~ ✅ Fixed**
`train_ds._raw` → `train_ds._offsets`. Verified with a `--quick --epochs 1` run.

**2. ~~`LABEL_SMOOTHING` is dead config~~ ✅ Fixed**
`y_smooth = y * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING` applied to targets before loss call in `train.py`. Softens labels from (0,1) → (0.05, 0.95).

**3. ~~GRU is permanently disabled in live inference~~ ✅ Fixed**
`_build_history_rich()` added to `coach.py`. Replays `history_uci` moves to extract per-move (piece, capture, check, color) dicts and passes them to `predict_concepts()`. GRU now receives actual game history at inference time.

**4. SF cache zero-fill status unknown**
`build_sf_cache.py` now validates after build: it checks the fraction of all-zero rows and warns loudly if >95% are empty. Run `python tools/build_sf_cache.py --force` after verifying Stockfish is accessible and the classical eval table format is supported. If SF classical eval is unavailable, remove the 14 SF dims from the architecture (they waste capacity if always zero).

**5. ~~Training F1 checkpoint selection uses fixed 0.5 threshold~~ ✅ Fixed**
`train.py` now loads `data/thresholds.json` at startup (from the previous calibration run) and uses those per-class thresholds for validation F1 computation. Falls back to 0.5 per class on first run. Checkpoint selection is now consistent with post-training evaluation.

---

### 🟡 Significant — Architecture or Design Problems

**6. ~~Documentation says 53 concepts; code has 49~~ ✅ Fixed**
README, concept_vocab.py, and classifier.py all updated. Concept count: **49**. Architecture block corrected: 3,013-dim raw input, 1,714-dim combined, 3.08M parameters. Concept vocabulary table replaced with the actual 49-entry list from `concept_vocab.py`.

**7. NNUE task misalignment — documented lesson**
Three full training runs (Phase 5, 5C, 5D) confirmed that NNUE Feature Transformer activations do not improve concept classification (F1 stuck at 0.33 vs 0.56 for Phase 4B). The root cause is *task misalignment*: NNUE FT representations are organised around centipawn evaluation ("how good is this position"), not concept identity ("what pattern is present"). These are related but distinct prediction targets, and a representation optimised for one does not transfer cleanly to the other.

**Lesson:** Before adopting any pre-trained representation as a feature, verify that the pre-training task's labels correlate with your target labels. For NNUE → concept labels, the correlation is weak and indirect. The 1,811-dim `algo_v4` features are purpose-built for concept detection and remain the primary signal source.

**NNUE's correct role:** Post-classification gating in the coach layer — validate that concepts make sense given the position's evaluation — not as a training feature.

**8. Hysteresis thresholds need empirical validation**
`ACTIVATE_THRESHOLD = 0.65` and `HOLD_THRESHOLD = 0.40` are good intuitions but have not been validated against the model's actual probability distribution. Run `python tools/survey_hysteresis.py` (added this session) to sample 1,000 diverse positions and see where Phase 4B's concept probabilities actually cluster before committing to these values.

**9. ~~No ML tests exist~~ ✅ Fixed (smoke tests added)**
`src/chess_coach/tests/test_ml_smoke.py` added: deterministic `fen_to_tensor` output, valid logits shape and range from `ChessConceptClassifier`, checkpoint-backed `predict_concepts` call on a known position. Run with `pytest src/chess_coach/tests/test_ml_smoke.py`.

**10. ~~History-aware training, history-blind inference~~ ✅ Fixed**
`_build_history_rich()` in `coach.py` bridges `history_uci` strings to the rich-move dict format that `history_rich_to_tensor()` expects. `coach.analyze()` now passes real game history to `predict_concepts()` instead of `None`.

---

### 🟠 Technical Debt — Won't Break Anything Today But Will Hurt Tomorrow

**11. ~~Cache file paths are hardcoded strings in six different files~~ ✅ Fixed**
`src/chess_coach/ml/paths.py` created. All canonical paths (`CLASSIFIER_BEST`, `TRAINING_JSONL`, `THRESHOLDS`, `ALGO_CACHE`, `V3_CACHE`, `SF_CACHE`, `NNUE_CACHE`, `BOARD_CACHE`, `NNUE_WEIGHTS`) live there. `train.py`, `evaluate.py`, `coach.py`, `dataset.py`, `classifier.py`, and `nimzo_net_engine.py` all import from it.

**12. ~~Algo cache rebuild has no ground-truth validation~~ ✅ Fixed**
`build_algo_cache.py --verify N` added. Loads the existing cache, samples N positions from JSONL, recomputes `algo_feature_vector_v4(fen)` for each, and reports max/mean absolute difference. Use after any rebuild to confirm the cache matches freshly computed values.

**13. ~~`build_board_cache.py` fallback row count uses JSONL line count~~ ✅ Fixed**
Fallback now scans JSONL for `max(_ac) + 1` instead of counting lines. Line count was wrong whenever the JSONL was reordered or appended without rebuilding — `max(_ac) + 1` is always the correct cache dimension.

**14. ~~Early stopping patience is 10 epochs — too short~~ ✅ Fixed**
Default patience increased to 20. With cosine annealing over 100 epochs, the low-LR tail (epochs 80–100) can deliver meaningful F1 gains that patience=10 would prematurely cut off.

**15. ~~`num_workers=2` is undershooting~~ ✅ Fixed**
`num_workers` increased to 4 in `train.py`. With all features pre-cached as numpy mmaps, `__getitem__` is array lookups + `torch.cat` — CPU-cheap enough that 4 workers better saturates the GPU on a typical development machine.

**16. ~~Two unconnected integration layers~~ Documented and cleaned up**
`concept_signal_adapter.py` and `nimzo_net_engine.py` are WIP integration code, not dead files. `nimzo_net_engine.py` now uses paths.py for the checkpoint path and `predict_concepts()` with calibrated thresholds (fixed stale `threshold=0.45`). To connect them to the live app: replace `StrategyEngine` with `NimzoNetEngine` in the application's InitWorker import.

**17. ~~`docs/phase5_nnue_integration_plan.md` describes an abandoned approach~~ ✅ Fixed**
ABANDONED header added at the top of the document explaining why Phase 5 failed (task misalignment), what was learned, and the current direction. The document is preserved for reference — the failure analysis is as valuable as the original plan.

**18. ~~Hysteresis state is direction-unaware~~ ✅ Fixed**
`coach.py` now stores a `_ply_states` dict mapping ply number → concept snapshot. When `analyze()` is called with a ply ≤ the last seen ply (backward navigation), it restores the concept state from that ply's snapshot instead of carrying forward stale activations. `reset()` clears both `_active_concepts` and `_ply_states`.

---

### 🔵 Process & Habits

**19. ~~Experiments were not tracked~~ ✅ Fixed**
`docs/experiments.md` created. Contains a summary table of all 8 training runs with params, dataset size, best Macro F1, best epoch, result file pointer, and outcome. Each run has a detailed section with the hypothesis stated before training, key changes, result, and lesson. New runs should add a row to the table before the run starts and fill in results when done. The file also includes an Upcoming Experiments section for tracking what to try next and why.

**20. ~~The model was tested on the wrong failure mode~~ Documented**
`docs/experiments.md` §Evaluation Philosophy documents the gap between macro F1 (what we measure) and coaching quality (what we care about). Macro F1 confirms the model can identify patterns — it does not confirm that the pattern is the most salient thing in the position, that the retrieved annotation is appropriate, or that the coach is consistent across quiet moves. Practical next steps are listed: the hysteresis survey tool (already built), position spot-checks by a chess-knowledgeable reviewer, a consistency replay check, and a false-positive audit on the 5 highest-recall concepts. This does not require a large annotated corpus — 20 spot-checked positions per major architecture change would catch gross failures.

---

**Summary for the intern:** The engineering fundamentals here are solid — the cache architecture, the lazy mmap pattern, the multi-label setup, the separation between deterministic and neural coaches. The dataset pipeline is genuinely well-structured. The biggest lessons are: (1) validate pre-trained representations against your actual task before building on them; (2) dead config is worse than no config; (3) test your ML pipeline like you test your application code; (4) write down what you expected before you run the experiment.
