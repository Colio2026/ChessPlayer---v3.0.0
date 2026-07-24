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
    - [Theme Performance & Spatial Coverage](#theme-performance--spatial-coverage)
    - [Roadmap](#roadmap)
12. [Training the Coach](#training-the-coach)
13. [The Deterministic Coach (Legacy)](#the-deterministic-coach-legacy)
14. [Database Schema](#database-schema)
15. [Configuration System](#configuration-system)
16. [Development Scripts](#development-scripts)
17. [Contributing](#contributing)
18. [Experiment Log](#experiment-log)

---

## Features

- **PGN Game Browser** — load any `.pgn` file or directory; filter by player, event, opening, date, and ECO code; paginated with lazy loading for databases of any size
- **Interactive Board** — drag-and-drop piece moves, full variation tree support, promote/demote variations, inline move comments
- **Engine Analysis** — Stockfish UCI integration with multi-PV evaluation, animated eval bar, and best-move arrows; runs on a background thread so the UI stays responsive
- **Continuation Statistics** — see how often a position arises in your loaded library and what the top continuations are, powered by an O(1) position-tree lookup
- **Chess Coach (Deterministic)** — Nimzowitsch-style strategic coaching: classifies each position into one of four strategies (Blitz, Flank, Fortress, Feint) and generates natural-language guidance assembled from a curated phrase database
- **Coach Nimzowitsch (Neural Network)** — a 49-class multi-label classifier (Phase 4B champion, 3.08M parameters) trained to identify chess concepts by name from board position, full move history via GRU, and 2,877-dim spatial heuristics; trained on 1.59M examples from master games, Lichess puzzles, and algorithmically-labelled positions; best validated Macro F1: **0.5614** (Phase 4B champion, calibrated); Phase 4C architecture (B8+B9 tactical/pawn maps, 68-dim v3) coded and pending training; direction-aware Schmitt-trigger hysteresis gates coach output during live game navigation
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
|   +-- eco_db.json             # ECO opening database for RAG coach
|   +-- Carlsen.pgn             # Default game library
|   +-- Caissabase.pgn          # Master game database for ML training
|   +-- lichess_elite_2020-10.pgn
|   +-- lichess_db_puzzle.csv   # Lichess 6M puzzle database
|   +-- nn.nnue                 # SF16.1 NNUE weights (kept, not used in training)
|   +-- annotated_pgns/         # Annotated PGN training material
|   |   +-- Raw_pgn/            # Manually curated annotated games
|   |   +-- lichess_studies/    # Concept-organised Lichess studies
|   +-- training_raw.jsonl      # Assembled training dataset with _ac indices (generated)
|   +-- algo_cache.npy          # 3779-dim algo_v4 features per example (Phase 4C, generated)
|   +-- v3_cache.npy            # 82-dim v3 concept bits per example (Phase 4C, generated)
|   +-- sf_cache.npy            # 14-dim SF classical eval per example (83 MB, generated)
|   +-- board_cache.npy         # 1001-dim board tensors per example (7.38 GB, generated)
|   +-- nnue_cache.npy          # 2048-dim NNUE FT activations (15 GB, generated, Phase 5)
|   +-- classifier_best.pt      # Best model checkpoint by val Macro F1 (generated)
|   +-- classifier_last.pt      # Latest epoch checkpoint (generated)
|   +-- thresholds.json         # Per-class calibrated thresholds (generated)
|   +-- trees/                  # MoveTree cache files (.pkl.gz)
|
+-- assets/
|   +-- pieces/                 # PNG piece images
|   +-- engines/                # Place your Stockfish binary here
|   +-- scraping/               # Lichess data collection scripts
|   +-- ui.qss                  # Qt stylesheet
|
+-- docs/
|   +-- experiments.md          # Full experiment log: all training runs, hypotheses, results
|   +-- phase5_nnue_integration_plan.md   # ABANDONED — NNUE task misalignment analysis
|
+-- results/                    # Training/eval console output (gitignored)
|   +-- results####_YYYY-MM-DD_HHMM_train.txt
|   +-- results####_YYYY-MM-DD_HHMM_eval.txt
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
|       +-- coach/              # WIP: NimzoNetEngine (ML → live app bridge)
|       |   +-- nimzo_net_engine.py
|       +-- rag/                # RAG coaching layer
|       |   +-- coach.py        # ChessCoach: classifier → hysteresis → RAG
|       |   +-- retriever.py    # RAGRetriever: concept → annotation lookup
|       +-- ml/                 # Neural network coach
|       |   +-- paths.py        # Single source of truth for all data file paths
|       |   +-- concept_vocab.py    # 49 concept labels (stable, order matters)
|       |   +-- board_encoder.py    # FEN → 1001-dim tensor, move → 128-dim, GRU helpers
|       |   +-- dataset.py          # JSONL loader, mmap cache integration, train/val/test split
|       |   +-- classifier.py       # Phase 4B: spatial bottleneck + GRU(256) + MLP head
|       |   +-- train.py            # Training loop with calibrated early stopping
|       |   +-- evaluate.py         # Threshold calibration, per-class metrics, spot checks
|       +-- tests/
|           +-- test_ml_smoke.py    # Board encoder, classifier, checkpoint spot-checks
|
+-- tools/                      # ML data pipeline scripts
|   +-- parse_annotated_pgn.py      # Extracts examples from annotated PGNs
|   +-- ingest_lichess_csv.py       # Processes Lichess 6M puzzle CSV
|   +-- ingest_game_database.py     # Scans master game PGNs algorithmically
|   +-- label_positions.py          # Algorithmic concept detectors (algo_v4 + v3)
|   +-- build_algo_cache.py         # Builds algo_cache.npy + v3_cache.npy; --verify N
|   +-- build_sf_cache.py           # Builds sf_cache.npy with post-build validation
|   +-- build_board_cache.py        # Builds board_cache.npy
|   +-- build_nnue_cache.py         # Builds nnue_cache.npy (Phase 5, not used in training)
|   +-- build_rag_index.py          # Builds RAG annotation index + eco_db.json
|   +-- nnue_reader.py              # SF NNUE binary parser (HalfKAv2)
|   +-- inspect_weights.py          # Post-training weight norm report per concept
|   +-- survey_hysteresis.py        # Validates ACTIVATE/HOLD thresholds vs. real distribution
|   +-- audit_concepts.py           # FP/FN audit: samples misfiring positions with Lichess links
|   +-- check_consistency.py        # Game replay: flags concept transitions on quiet moves
|
+-- retrain_and_reparse.ps1     # Full pipeline: cache checks → train → eval → audit
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
| `chess_coach/ml/paths.py` | Single source of truth for all data file paths (`CLASSIFIER_BEST`, `TRAINING_JSONL`, all caches) |
| `chess_coach/ml/concept_vocab.py` | Canonical ordered list of all 49 concept labels |
| `chess_coach/ml/board_encoder.py` | `fen_to_tensor()` (1001-dim), `move_to_tensor()` (128-dim), `history_rich_to_tensor()`, `COMBINED_SIZE_V4B` (1737) |
| `chess_coach/ml/dataset.py` | JSONL loader with lazy mmap cache integration; 80/10/10 split; `pos_weight` for class imbalance |
| `chess_coach/ml/classifier.py` | `ChessConceptClassifier` — Phase 4B: spatial bottleneck + GRU(256) + 1024/512 MLP head |
| `chess_coach/ml/train.py` | Training loop; calibrated early stopping; label smoothing; cosine LR |
| `chess_coach/ml/evaluate.py` | Per-class threshold calibration, test-split metrics, spot checks on 50 named positions |
| `chess_coach/rag/coach.py` | `ChessCoach` — concept classifier → Schmitt-trigger hysteresis → RAG retrieval; direction-aware ply snapshots |
| `chess_coach/rag/retriever.py` | `RAGRetriever` — ECO opening lookup + concept annotation retrieval |
| `chess_coach/coach/nimzo_net_engine.py` | WIP: `NimzoNetEngine` — bridge from classifier to live application |
| `tools/label_positions.py` | `algo_feature_vector_v4()` (3779-dim) + `algo_feature_vector()` (82-dim) + 46 concept detectors (36 per-color × 2 + 10 global) |
| `tools/build_algo_cache.py` | Strips algo_features from JSONL → `algo_cache.npy` + `v3_cache.npy`; `--verify N` spot-check |
| `tools/build_sf_cache.py` | Runs SF classical eval → `sf_cache.npy`; post-build zero-row validation |
| `tools/build_board_cache.py` | Pre-computes `fen_to_tensor()` → `board_cache.npy` for all positions |
| `tools/survey_hysteresis.py` | Samples N positions; reports p50–p99 per concept; validates ACTIVATE/HOLD thresholds |
| `tools/audit_concepts.py` | FP/FN audit on test split; samples misfiring positions; outputs Lichess analysis links |
| `tools/check_consistency.py` | Replays a game through `coach.analyze()`; flags concept transitions on quiet moves |
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

**Phase 4C (current codebase — training pending)**

```
Raw static input: 4,088-dim
  - 1,001  board encoding  (12×64 piece channels, attack maps, pawn structure, king shelter, mobility)
  -   128  move one-hot    (64 from-square + 64 to-square — the key move being played)
  - 2,877  algo_v4 features  (explicit geometric chess heuristics — B1-B9 blocks; see Theme Coverage)
  -    68  v3 concept bits   (binary algorithmic concept flags — bypass the spatial bottleneck)
  -    14  SF classical eval  (Mobility, King safety, Threats, Passed, Space, Pawns, Imbalance × 2 sides)

Spatial bottleneck:  Linear(3779, 256) → ReLU → Dropout(0.3)   # compresses algo_v4

GRU:  144-dim per-step history  →  256-dim context
      (encodes piece type, capture, check, and side-to-move for each prior half-move)

Combined (post-projection):  1,723-dim
  = board(1001) + move(128) + spatial_proj(256) + v3(68) + sf(14) + gru(256)

MLP head:
  Hidden 1:  Linear(1737, 1024) → BatchNorm → ReLU → Dropout(0.4)
  Hidden 2:  Linear(1024,  512) → BatchNorm → ReLU → Dropout(0.2)
  Output:    Linear( 512,   49) → per-class sigmoid  (BCEWithLogitsLoss)

Parameters: ~3.08M  (head size unchanged; spatial_proj input grew 1811→3779)
```

*Phase 4B champion checkpoint (last trained model): algo_v4 1811-dim, v3 59-dim, combined 1714-dim, best Macro F1 0.5614 on 1.23M examples (epoch 68). Phase 4C caches must be rebuilt before training (`python tools/build_algo_cache.py --force`).*

**Training details:**
- Loss: `BCEWithLogitsLoss` with per-class `pos_weight` clamped to (1.0, 20.0); label smoothing ε=0.05
- Optimiser: AdamW, weight decay 6e-3, initial LR 1e-3 with cosine annealing over 100 epochs
- Early stopping: macro F1 on validation set, patience **20 epochs** (increased from 10 to capture the low-LR cosine tail)
- Checkpoint selection: uses calibrated thresholds from `data/thresholds.json` for F1 computation; falls back to 0.5 per class on first run before calibration
- Thresholds: calibrated per class on the validation set after training, saved to `data/thresholds.json`; sweep 0.05–0.95 in 0.05 steps, maximising per-class F1

**Why multi-label:** Most positions have multiple concepts simultaneously present. A position can be a `pin` + `battery` + `pawn_storm` + `king_safety` threat all at once. Single-label classification would lose this richness.

**Why the move feature:** The key move often determines the concept. A rook move to e7 could be `rook_seventh`, `battery`, or `clearance` depending on context. The 128-dim move one-hot gives the model the move being played alongside the static board, dramatically reducing ambiguity for tactical concepts.

### Training Pipeline

The full pipeline is orchestrated by `retrain_and_reparse.ps1`. Three data sources are assembled into `data/training_raw.jsonl`, pre-computed into mmap caches, and then trained on:

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

#### Scraping Scripts (`assets/scraping/`)

Run `.\scraping.ps1 -Token lip_xxxx` to collect new data. Each scraper targets a different hole in the dataset:

| Script | Method | What it fills |
|---|---|---|
| `scrape_lichess_studies.py` | Lichess Study Search API (`requests`) | **Concept-targeted.** Downloads up to 200 annotated studies per concept into `data/annotated_pgns/lichess_studies/<concept>/`. The folder name guarantees the label via `_inject_folder_concept` — no keyword match needed. Primary fix for data-starved concepts: `initiative` (~17K), `interference` (~18K), `x_ray` (~19K), `shouldering` (~1.3K). Requires a free Lichess OAuth token (`study:read` scope). |
| `scrape_chessgames.py` | Playwright (chessgames.com) | **Expert prose.** Crawls human-annotated master games. Not concept-targeted — labels flow from keyword matching in `{ }` comment blocks. Best coverage for strategic/positional prose concepts: `outpost`, `bad_bishop`, `pawn_chain`, `space_advantage`, `initiative`, `prophylaxis`, `battery`, `weak_square`. |
| `scrape_gameknot.py` | Playwright (gameknot.com) | **Expert prose (different pool).** Same keyword-matching pipeline as chessgames, different author base and annotation style. Complementary coverage for: `king_activity`, `king_safety`, `pawn_majority`, `development_lead`, `mating_attack`, `sacrifice`, `clearance`, `deflection`. |

After scraping: run `retrain_and_reparse.ps1` steps 1–3 to ingest the new PGNs, then step 4 to rebuild the 3779-dim algo cache.

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

**All 49 concepts now have algorithmic detectors.** Phase 4C adds 13 new detectors: `interference`, `initiative`, `prophylaxis`, `sacrifice`, `clearance`, `deflection`, `zwischenzug` (‡), plus 6 added in Phase 4C initial (`x_ray`, `discovery`, `mating_attack`, `double_check`, `zugzwang`, `shouldering`). Labels from Lichess tags and annotated PGNs remain the primary source for move-sequence concepts (`zwischenzug`, `clearance`) but detectors now provide geometric signal.

### Theme Performance & Spatial Coverage

Per-concept breakdown of spatial signals, algorithmic detection, and evaluation metrics. Use this table to identify where to invest — whether in better labels, more scraped data, or new spatial blocks.

**Eval base:** Phase 4B retrain, epoch 37, 1.59M training examples (159,145 test examples). Result file: `results/results0052_2026-07-21_2119_eval.txt`. **Phase 4C code is ready** (3779/82-dim, B8+B9 blocks, all 49 concepts detected) but has not yet been trained — B8/B9 and new v3 entries marked **‡** are not in the current checkpoint.

All concepts share a common base of **board(1001) + move(128) + B1-B5 general(512) + SF-eval(14) + GRU(256)**. The tables below document concept-specific features on top of that base. DB examples are estimated as `test_support × 10` (10% test split of 1.59M dataset).

**Spatial signal key:**
- **B1-B5 name(Xd)** — spatial map in the 512-dim B1-B5 block (weak_square, outpost, backward_pawn, passed_pawn — 128d each × 2 colors)
- **B6 name(Xd)** — dedicated sub-block within the 1151-dim B6 map; `Xd` = dimension count, both sides unless noted
- **B7 king_safety(148d)** — attack density in king zone + pawn shelter maps, both sides
- **B8 name(Xd) ‡** — Phase 4C tactical maps, not yet in trained model
- **B9 name(Xd) ‡** — Phase 4C pawn-structure + mating maps (new B9 block), not yet in trained model
- **v3 ✓** — binary algorithmic detection bit in 82-dim v3 bypass channel (skips the 256-dim bottleneck)
- **v3 ✓‡** — v3 bit added in Phase 4C; not in current trained checkpoint
- **—** — no dedicated signal beyond B1-B5 general maps

---

#### Tactical (15 concepts)

| Concept | Dedicated spatial | v3 | Thresh | Prec | Rec | F1 | ~DB | Notes |
|---|---|---|---|---|---|---|---|---|
| pin | B8 pin_vec(256d) ‡ | ✓ | 0.85 | 0.541 | 0.687 | 0.605 | 109K | Low precision: fires too broadly. Phase 4C adds per-square pinned/pinner maps [w_pinned(64), w_pinner(64), b_pinned(64), b_pinner(64)] — gives exact square-level signal. v3 bit anchors true pins in current model. |
| fork | B8 fork_vec(256d) ‡ | ✓ | 0.75 | 0.468 | 0.521 | 0.493 | 101K | Weakest tactical concept with full data volume. No B6/B7 in Phase 4B. Phase 4C fork_vec encodes exact forker and forked squares per side [w_forking(64), w_forked(64), b_forking(64), b_forked(64)]. v3 detector fires on ≥2 valuable pieces attacked. |
| skewer | B6 x_ray(384d) | ✓ | 0.85 | 0.577 | 0.594 | 0.585 | 100K | B6 x_ray covers sliding-piece ray alignment — shared geometry with x_ray concept. Balanced prec/rec. v3 detector labels it. No further spatial gap; label quality is the likely bottleneck. |
| discovery | — | ✓‡ | 0.80 | 0.329 | 0.395 | 0.359 | 49K | No dedicated spatial in Phase 4B. Phase 4C v3 detector: friendly piece screens a slider from a valuable enemy target. ~49K examples vs 100K+ for common concepts — data volume is the primary bottleneck alongside missing geometry. |
| x_ray | B6 x_ray(384d) | ✓‡ | 0.85 | 0.713 | 0.484 | 0.577 | 19K | Very high precision (0.713) but low recall. **CRITICALLY DATA-STARVED: only ~19K examples in DB.** Scrapers not yet rerun for x_ray synonyms. Phase 4C v3 detector: sliding piece with a second friendly piece behind a blocker on the same ray. Scraper rerun is the single highest-leverage action. |
| double_check | B6 double_check(130d) | ✓‡ | 0.85 | 0.645 | 0.531 | 0.582 | 29K | B6 block encodes checker count + piece type. Phase 4C adds v3 global bit (len(checkers) ≥ 2). Low recall despite dedicated geometry (~29K examples and complex trigger conditions). |
| clearance | B9 clearance_vec(128d) ‡ | ✓‡ | 0.80 | 0.309 | 0.378 | 0.340 | 52K | Phase 4C adds B9 clearance_vec (128d): marks friendly non-slider pieces that block a friendly slider's ray to a valuable enemy piece — moving this piece reveals the attack. v3 bit via `_has_clearance`. |
| deflection | B9 deflection_vec(128d) ‡ | ✓‡ | 0.75 | 0.256 | 0.416 | 0.317 | 55K | Phase 4C adds B9 deflection_vec (128d): marks enemy pieces that are the sole defender of an enemy queen/rook AND are attacked by us. v3 bit via `_has_deflection`. |
| overloading | — | — | 0.75 | 0.333 | 0.525 | 0.408 | 102K | No dedicated spatial. Conceptually related to fork but the overloaded piece is *defending* two targets, not attacking. Requires encoding defensive assignments, not just attack counts. Good data volume; geometry is the gap. |
| zwischenzug | B9 zwischenzug_vec(128d) ‡ | ✓‡ | 0.80 | 0.365 | 0.437 | 0.398 | 51K | Phase 4C adds B9 zwischenzug_vec (128d): marks under-attacked pieces when a forcing check is simultaneously available (the intermezzo pattern). v3 bit via `_has_zwischenzug`. Fires only on the side to move. |
| interference | B9 interference_vec(128d) ‡ | ✓‡ | 0.70 | 0.226 | 0.214 | 0.220 | 18K | **Lowest F1 of all 49 concepts.** ~18K examples. Phase 4C adds B9 interference_vec (128d): gap squares between enemy slider and defended piece, marked where we can interpose. v3 bit via `_has_interference`. Dataset volume remains the primary bottleneck. |
| back_rank | B7 king_safety(148d) | — | 0.90 | 0.683 | 0.818 | 0.745 | 102K | Best tactical F1. B7 king_safety encodes back-rank pawn cover (luft presence/absence) and open files near the king. Move tensor gives the attacking rook/queen square. High recall (0.818). |
| sacrifice | B9 sacrifice_vec(130d) ‡ | ✓‡ | 0.80 | 0.341 | 0.413 | 0.374 | 57K | Phase 4C adds B9 sacrifice_vec (130d): marks pieces attacked by less-valuable enemy pieces and under-defended (offered sacrifice squares) + material deficit norm. v3 bit via `_has_sacrifice`. Material-down + king-attack proxy for positional compensation. |
| mating_attack | B9 mating_pressure_vec(128d) ‡ | ✓‡ | 0.75 | 0.225 | 0.439 | 0.297 | 56K | Phase 4C adds: (1) v3 per-color detector (≥2 non-pawn pieces attacking king zone, outnumbering defenders); (2) B9 mating_pressure_vec [w_pieces_on_bk_zone(64), b_pieces_on_wk_zone(64)] — marks WHICH squares the attacking pieces sit on, not just whether an attack exists. Complements B7 which marks which king zone squares are under fire. Very low precision (0.225) — model overfires on active positions; per-square attacker location should help specificity. |
| trapped_piece | — | — | 0.75 | 0.285 | 0.424 | 0.341 | 101K | No dedicated spatial. Requires knowing a specific piece has no safe escape squares — closely related to mobility (B1-B5) but not explicitly encoded per-piece. Good data volume; per-piece mobility maps would be the fix. |

---

#### Piece Concepts (8 concepts)

| Concept | Dedicated spatial | v3 | Thresh | Prec | Rec | F1 | ~DB | Notes |
|---|---|---|---|---|---|---|---|---|
| outpost | — | ✓ | 0.80 | 0.391 | 0.658 | 0.491 | 104K | No dedicated B6 block. v3 detector flags outpost piece. High recall but poor precision — model fires on any well-placed piece. True outpost requires confirming the pawn structure prevents enemy piece challenge — a relationship only partial in B1-B5 pawn maps. |
| blockade | — | ✓ | 0.70 | 0.306 | 0.500 | 0.379 | 102K | No dedicated spatial. Piece-in-front-of-pawn geometry is simple but distinguishing a blockade (strategic) from any pawn obstruction requires pawn advance potential — not directly encoded. |
| bad_bishop | — | ✓ | 0.75 | 0.305 | 0.562 | 0.396 | 100K | v3 per-color detector. Pawn-color vs bishop-color relationship needs explicit pawn-complex maps. B6 bishop_pair block partially helps (tracks pawn color complex for bishop pair) but bad_bishop needs the same map for single-bishop positions. |
| good_bishop | — | ✓ | 0.75 | 0.313 | 0.580 | 0.406 | 102K | Mirror of bad_bishop — same spatial gap. High recall implies the model fires whenever a bishop is present with open diagonals, not specifically when the pawns are on the opposite color. |
| bishop_pair | B6 bishop_pair(130d) | ✓ | 0.80 | 0.434 | 0.786 | 0.559 | 104K | B6 dedicated block tracks both bishops + pawn color complex. High recall (0.786) but precision suffers — model fires even when bishop-pair advantage is minimal or position is closed. |
| piece_activity | — | ✓ | 0.85 | 0.527 | 0.736 | 0.615 | 104K | No dedicated B6 block; B1-B5 mobility maps carry the signal. v3 detector. Best piece-concept F1 — mobility in B1-B5 is genuinely informative for this concept. |
| battery | B6 battery(144d) | ✓ | 0.75 | 0.312 | 0.523 | 0.391 | 99K | B6 battery captures same-direction piece stacking (rooks on a file, bishop+queen on a diagonal). Precision poor — many positions have two rooks or Q+B without a genuine battery threat. |
| rook_seventh | B6 rook_seventh(34d) | ✓ | 0.90 | 0.669 | 0.772 | 0.717 | 100K | Dedicated B6 block (rook rank proximity to opponent's 2nd rank, both sides). Second-best piece concept F1. The geometry is explicit and the concept is well-defined — a model success story. |

---

#### Pawn Structure (9 concepts)

| Concept | Dedicated spatial | v3 | Thresh | Prec | Rec | F1 | ~DB | Notes |
|---|---|---|---|---|---|---|---|---|
| passed_pawn | B5 passed_pawn_map(128d) | ✓ | 0.75 | 0.338 | 0.553 | 0.420 | 104K | v3 detector. B5 _passed_pawn_map: [w_passed(64), b_passed(64)] — per-square passed pawn locations for both sides. Uses bitboard fill: a pawn is passed if no enemy pawn on same or adjacent files ahead. Already in Phase 4C code (B1-B5 block, slot 4). |
| promotion | B6 promotion(18d) | ✓ | 0.85 | 0.646 | 0.493 | 0.559 | 101K | B6 block captures pawn-on-7th geometry. Decent precision but low recall (0.493) — threshold is conservative. Move tensor captures the promoting move. The concept is well-specified; threshold tuning and more examples may improve recall. |
| isolated_pawn | B8 isolated_pawn_vec(128d) ‡ | ✓ | 0.70 | 0.273 | 0.409 | 0.328 | 105K | v3 detector. Phase 4C adds [w_isolated(64), b_isolated(64)] per-square maps. **Largest test support (105K) with lowest pawn-structure F1** — unambiguous evidence that the missing geometry, not data volume, is the bottleneck. Phase 4C should be decisive here. |
| backward_pawn | B3 backward_pawn_map(128d) | ✓ | 0.75 | 0.351 | 0.609 | 0.446 | 103K | v3 detector. B3 _backward_pawn_map: [w_backward(64), b_backward(64)] — per-square backward pawn locations. A backward pawn is one whose stop square is attacked by an enemy pawn AND has no friendly pawn on an adjacent file behind it. Already in Phase 4C code (B1-B5 block, slot 3). |
| doubled_pawn | — | ✓ | 0.85 | 0.537 | 0.758 | 0.629 | 103K | No dedicated B6 block. v3 detector. Higher F1 than most structure concepts — the same-file pawn pattern is distinctive enough in B1-B5 pawn maps. |
| pawn_majority | — | ✓ | 0.70 | 0.252 | 0.421 | 0.316 | 102K | v3 detector. No dedicated spatial. Requires comparing pawn counts per wing — a relationship the model can partially derive from B1-B5 pawn maps but without explicit wing-majority encoding. |
| pawn_chain | B9 pawn_chain_vec(128d) ‡ | ✓ | 0.70 | 0.232 | 0.404 | 0.295 | 102K | v3 detector exists (`_has_pawn_chain`). B9 _pawn_chain_vec now added: [w_chain(64), b_chain(64)] — marks all pawns that are part of a diagonal chain (both the base pawn doing the defending AND the member pawns being defended). Chain members = `wp & _wp_attacks(wp)`; base pawns found by stepping back from members. Previously the only chain signal was the binary v3 bit; now the net sees exact chain geometry per square. |
| pawn_storm | — | ✓ | 0.85 | 0.563 | 0.757 | 0.646 | 102K | No dedicated B6 block despite being a rich spatial concept. v3 detector. High recall (0.757) — advancing pawn wedge patterns in B1-B5 are informative. Precision (0.563) suffers from false-positive storms in closed positions. |
| pawn_island | B9 pawn_island_vec(130d) ‡ | ✓ | 0.70 | 0.225 | 0.386 | 0.285 | 100K | v3 detector exists (`_pawn_island_count ≥ 2`). B9 _pawn_island_vec now added: [w_connected(64), b_connected(64), w_island_count(1), b_island_count(1)] = 130d. connected[sq]=1 if pawn has ≥1 neighbor on adjacent file (in a group). island_count normalized to [0,1]. Complements B8 isolated_pawn_vec (which marks pawns WITH NO neighbors); together they encode full island structure. Island count scalar gives the model the exact group count signal needed. |

---

#### King & Endgame (11 concepts)

| Concept | Dedicated spatial | v3 | Thresh | Prec | Rec | F1 | ~DB | Notes |
|---|---|---|---|---|---|---|---|---|
| king_safety | B7 king_safety(148d) | ✓ | 0.80 | 0.491 | 0.527 | 0.509 | 103K | Dedicated 148-dim block: attack density in king zone, pawn shelter, open files near king, both sides encoded. Moderate F1 despite rich features — concept spans a wide spectrum (minimal risk to immediate mate), hard to threshold cleanly. Most improvable via better threshold calibration. |
| king_activity | B7 king_safety(148d) partial | ✓ | 0.85 | 0.573 | 0.726 | 0.641 | 100K | B7 was built for king *safety* (attacks, shelter). King *activity* (centralization, endgame penetration) is distinct. General B1-B5 mobility captures some of it. A dedicated endgame king-position map (king square relative to board center + pawn mass) would improve precision. |
| shouldering | B6 shouldering(21d) | ✓‡ | 0.85 | 0.570 | 0.714 | 0.634 | ~1.3K | B6 block: king file adjacency + rank offset in pawn endgame (21d). Phase 4C adds v3 global detector. **CRITICALLY DATA-STARVED: only ~1,260 examples in entire DB.** F1 0.634 is remarkable given the volume — the geometry is clearly learned. Scraper rerun for "shouldering" synonyms is the single most impactful data action. |
| opposition | B6 opposition(15d) | ✓ | 0.80 | 0.666 | 0.600 | 0.631 | 99K | B6 block: king relative distance in endgame (15d). Decent F1. Low recall may reflect the endgame context filter in the v3 detector being too strict (opposition in middlegame positions missed). |
| zugzwang | B6 zugzwang_tier1(4d) | ✓‡ | 0.90 | 0.623 | 0.785 | 0.695 | 48K | B6 tier-1 block (4d: quiet-endgame flags). Phase 4C v3 global detector: ≤5 legal moves + no pawn advances available in endgame. High recall (0.785) — model fires on many quiet endgames. Threshold 0.90 is appropriately conservative. |
| rook_endgame | — | ✓ | 0.85 | 0.859 | 0.715 | 0.780 | 101K | Best endgame F1. No dedicated B6 block — v3 detector identifies rook-only material, B1-B5 rook mobility + board tensor carry the pattern. Material composition is highly distinctive; the model reads it well. |
| pawn_endgame | — | ✓ | 0.90 | 0.662 | 0.683 | 0.672 | 78K | v3 detector identifies pawn-only endgame. No dedicated B6 block. B1-B5 pawn maps + material state provide sufficient signal. |
| bishop_endgame | B6 bishop_endgame(133d) | ✓ | 0.85 | 0.935 | 0.572 | 0.710 | 100K | Dedicated B6 block (133d: bishop + remaining material). **Very high precision (0.935)** — when it fires, it's almost always right. Low recall is a threshold artifact (0.85); reducing threshold slightly would improve F1. |
| knight_endgame | — | ✓ | 0.70 | 0.998 | 0.599 | 0.749 | 73K | Near-perfect precision (0.998). v3 detector flags knight-only material. Low recall: the 0.70 calibrated threshold indicates val-set optimisation was still conservative. Reducing threshold to ~0.60 would improve recall with minimal precision cost. |
| queen_endgame | — | ✓ | 0.95 | 0.997 | 0.624 | 0.768 | 88K | Near-perfect precision (0.997). Threshold 0.95 is the joint-highest. Queen endgame material state is unambiguous. Recall suffers from the conservative threshold — reducing to 0.90 should recover recall with no precision cost. |
| drawn_position | B6 drawn_position(2d) | ✓ | 0.80 | 0.956 | 0.465 | 0.626 | 34K | B6 block (2d: material insufficiency bits — K+B vs K, K+N vs K etc.). Very high precision (0.956). Low recall: positions with *potential* draws (perpetual threat, fortress) that don't hit the binary material threshold are missed. ~34K examples; more drawn-endgame annotated examples would expand coverage. |

---

#### Positional / Strategic (6 concepts)

| Concept | Dedicated spatial | v3 | Thresh | Prec | Rec | F1 | ~DB | Notes |
|---|---|---|---|---|---|---|---|---|
| weak_square | B1 weak_square_map(128d) | ✓ | 0.70 | 0.211 | 0.368 | 0.268 | 102K | **Lowest strategic F1.** v3 detector + B1 _weak_square_map: [w_weak(64), b_weak(64)] — a square is "weak" for color C if no pawn of C on an adjacent file can ever advance to defend it (pawn-controlled vs. pawn-hole). Already in Phase 4C code (B1-B5 block, slot 1). The B1 map is the correct geometry; low F1 may reflect label noise (keyword "weak square" is vague in game annotations) more than missing features. |
| open_file | B8 open_file_vec(40d) ‡ | ✓ | 0.70 | 0.295 | 0.610 | 0.398 | 104K | Phase 4C adds 40-dim maps: rook-file presence (8d per side) + fully open (8d) + semi-open for white (8d) + semi-open for black (8d). v3 detector. Current model has no dedicated open-file block; B1-B5 captures this weakly. High recall but very low precision — fires too readily. Phase 4C geometry should tighten this significantly. |
| space_advantage | — | ✓ | 0.90 | 0.943 | 0.573 | 0.713 | 98K | v3 detector (pawn advancement count heuristic). Excellent precision (0.943). Low recall is a calibration artifact — threshold 0.90 is conservative. The space heuristic in v3 is unambiguous; reducing threshold to 0.85 would recover recall. |
| development_lead | B6 development(136d) | ✓ | 0.95 | 0.996 | 0.572 | 0.727 | 101K | Near-perfect precision (0.996). B6 development block: piece rank advancement maps (136d, both sides — measures how far each side's pieces have moved from their starting squares). Threshold 0.95 is joint-highest; still achieves F1 0.727. Reduce to 0.90 to improve recall. |
| initiative | B9 initiative_vec(130d) ‡ | ✓‡ | 0.85 | 0.335 | 0.788 | 0.470 | 17K | Phase 4C adds B9 initiative_vec (130d): marks pieces generating winning threats (more attackers than defenders on enemy pieces) + normalized threat count. v3 bit via `_has_initiative` (proxy for SF Threats term imbalance). Small dataset (~17K). |
| prophylaxis | B9 prophylaxis_vec(130d) ‡ | ✓‡ | 0.75 | 0.217 | 0.369 | 0.273 | 52K | Phase 4C adds B9 prophylaxis_vec (130d): marks overprotected pieces (≥2 more defenders than attackers) and key outpost squares dominated against enemy pieces eyeing them. v3 bit via `_has_prophylaxis_pos`. Labels from annotated PGNs + new detector. |

---

**Common patterns across weak concepts:**
- **No dedicated spatial + no v3 bit** = worst outcomes in Phase 4B (prophylaxis 0.273, interference 0.220). Phase 4C adds B9 spatial maps and v3 bits for all three — interference_vec (128d gap-square map), initiative_vec (130d threat-piece map), prophylaxis_vec (130d overprotection + key-square map). Dataset volume (18K / 17K) remains the primary bottleneck for interference and initiative.
- **Data starvation despite spatial coverage** = x_ray (~19K), shouldering (~1.3K), drawn_position (~34K), double_check (~29K). Scraper rerun for synonym terms is the immediate fix.
- **High precision / low recall** = bishop_endgame (0.935/0.572), knight_endgame (0.998/0.599), queen_endgame (0.997/0.624), space_advantage (0.943/0.573), development_lead (0.996/0.572). Conservative thresholds only; reduce by 0.05 to recover recall with negligible precision cost.
- **Low precision / high recall** = initiative (0.335/0.788), bishop_pair (0.434/0.786), king_activity (0.573/0.726). Model fires too readily; spatial specificity or stricter v3 signal needed.

---

### Roadmap

The coach has been through four architectural phases. Phase 4B is the current production model. Full experiment history with per-run numbers, hypotheses, and lessons is in [`docs/experiments.md`](docs/experiments.md).

**Phase 1 (superseded)** — 1,188-dim static features, MLP baseline. Macro F1: ~0.30. Proof of concept only.

**Phase 2 (superseded)** — Scaled MLP, ~1M examples. Macro F1: ~0.51. Good recall, poor precision — overfires.

**Phase 4B (last trained champion)** — 3,013-dim raw input → 1,714-dim combined after spatial bottleneck + GRU(256). Spatial bottleneck (Linear 1811→256) compresses 1,811 explicit geometric heuristics into a 256-dim representation. GRU reads per-move (piece, capture, check, color) over up to 60 prior half-moves. Macro F1: **0.5614** (calibrated, 49 concepts, epoch 68). Champion run on 1.23M examples; retrain on 1.59M in progress at time of Phase 4C code upgrade.

**Phase 4C (code ready, training pending)** — 5,004-dim raw input → 1,737-dim combined. algo_v4 expanded 1,811 → 3,779 (+680 B8 tactical maps; +1,288 B9 pawn/mating/strategic/tactical maps: all six B9-wave-1/2 maps plus sacrifice_vec/clearance_vec/deflection_vec/zwischenzug_vec). v3 expanded 59 → 82 (+13 new detectors bringing coverage to all 49 concepts). Requires cache rebuild (`build_algo_cache.py --force`) before training.

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
.\retrain_and_reparse.ps1
```

Steps 1–8 (parse, ingest, cache builds) are commented out by default — they only run when new source data is added or caches need a forced rebuild. The active steps are training through the full evaluation suite:

| Step | Script | Input | Output | Notes |
|---|---|---|---|---|
| 1 | `parse_annotated_pgn.py` | `data/annotated_pgns/` | `training_raw.jsonl` | Commented out unless re-parsing |
| 2 | `ingest_lichess_csv.py` | `data/lichess_db_puzzle.csv` | appended to JSONL | Commented out unless re-parsing |
| 3 | `ingest_game_database.py` | `data/Caissabase.pgn` etc. | appended to JSONL | Commented out unless re-parsing |
| 4 | `build_algo_cache.py` | JSONL | `algo_cache.npy` (13.35 GB) + `v3_cache.npy` (435 MB) | Strips algo_features, stamps `_ac` indices |
| 5 | `build_sf_cache.py` | JSONL | `sf_cache.npy` (83 MB) | Validates non-zero row fraction after build |
| 6 | `build_nnue_cache.py` | JSONL | `nnue_cache.npy` (15 GB) | Kept but unused in Phase 4B training |
| 7 | `build_board_cache.py` | JSONL | `board_cache.npy` (7.38 GB) | Eliminates per-example FEN parsing |
| 8 | `build_rag_index.py` | annotation PGNs | `eco_db.json` + RAG index | — |
| 9 | `train.py --phase4` | all caches + JSONL | `classifier_best.pt` | Active |
| 10 | `evaluate.py --calibrate` | `classifier_best.pt` | `thresholds.json` + eval report | Active |
| 11 | `inspect_weights.py` | `classifier_best.pt` | Weight norm report | Active |
| 12 | `survey_hysteresis.py --n 2000` | JSONL + checkpoint | Console report | Active — validates ACTIVATE/HOLD thresholds |
| 13 | `audit_concepts.py --n 8` | test split + checkpoint | Console report + Lichess links | Active — FP/FN audit, open links manually |
| 14 | `check_consistency.py` | UCI moves + checkpoint | Console report | Commented — supply a game to replay |

**Running individual steps:**

```powershell
# Training only (caches already built)
python -m src.chess_coach.ml.train --phase4

# Calibrate thresholds + full evaluation
python -m src.chess_coach.ml.evaluate --calibrate

# Survey hysteresis threshold placement
python tools/survey_hysteresis.py --n 2000

# Audit false positives for the 8 worst concepts
python tools/audit_concepts.py --n 8 --samples 10

# Consistency check on a game (supply UCI moves)
python tools/check_consistency.py --uci "e2e4 e7e5 g1f3 b8c6 f1b5"

# Verify algo cache integrity after rebuild
python tools/build_algo_cache.py --verify 500

# Dry run: label distribution from Lichess CSV without writing
python tools/ingest_lichess_csv.py --input data/lichess_db_puzzle.csv --count-only --limit 50000
```

**JSONL format** — each training example (after caches are built):
```json
{
  "fen":          "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
  "move_uci":     "f3g5",
  "themes":       ["pin", "fork", "battery"],
  "history_rich": [{"uci": "e2e4", "piece": 1, "captured": null, "is_check": false, "color": 1}],
  "comment":      "The knight move creates a fork threat while exploiting the pin...",
  "phase":        "opening",
  "_ac":          12345
}
```
`_ac` is the cache row index assigned by `build_algo_cache.py`. `history_rich` is the per-move context list consumed by the GRU.

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
| `scraping.ps1` | Collect annotated PGN training data from web sources (Lichess studies API, chessgames.com, gameknot.com). Run before `retrain_and_reparse.ps1` when adding new data. |
| `retrain_and_reparse.ps1` | Full ML pipeline: parse → ingest → cache rebuild → train (Phase 4C) → calibrate → inspect weights → hysteresis survey → concept audit |
| `scripts/run_dev.ps1` | Launch the app from the correct working directory (Windows PowerShell) |
| `scripts/run_dev.sh` | Same, for bash |
| `scripts/build_windows.ps1` | Package the app for distribution |

---

## Contributing

- Module boundaries and the no-Qt-outside-ui rule are load-bearing — keep them.
- **Concept vocab (`concept_vocab.py`) is append-only.** Adding a concept in the middle shifts every output neuron index and invalidates all saved checkpoints. New concepts go at the end only.
- **All data paths go through `src/chess_coach/ml/paths.py`.** Never hardcode `data/...` strings anywhere else.
- New algorithmic detectors go in `tools/label_positions.py` and must be added to both `DETECTABLE_CONCEPTS` and the `label_position()` call loop.
- New coaching phrases go in the phrase DB seed scripts keyed by `strategy/metric/severity/fragment_type`.
- New extractors go in `src/chess_coach/extractors/` and must be added to the strategy engine's extractor list.
- **Document every training run in `docs/experiments.md`** before it starts (hypothesis) and after it finishes (result + lesson). Undocumented runs are invisible to the next person, including future you.
- Run the coach test suite before pushing: `pytest src/chess_coach/tests/`.
- After every retrain run the full evaluation suite: `evaluate.py --calibrate` → `survey_hysteresis.py` → `audit_concepts.py` → spot-check Lichess URLs manually.

---

## Experiment Log

See [`docs/experiments.md`](docs/experiments.md) for the full training history including per-run hypotheses, results, lessons learned, and upcoming experiments.

**Quick reference:**

| Phase | Params | Dataset | Best Macro F1 | Notes |
|---|---|---|---|---|
| Phase 1 | ~1.2M | ~500K | 0.30 | MLP baseline, proof of concept |
| Phase 2 | ~1.2M | ~1M | 0.51 | Scaled MLP, overfires |
| Phase 4B | 3.07M | 1.23M | **0.5614** | Champion. Calibrated, 49 concepts. |
| Phase 5 | 1.74M | 1.07M | 0.47 | NNUE experiment — ambiguous signal |
| Phase 5C/5D | 1.35–1.75M | 1.59M | 0.33–0.35 | Pure NNUE stalled. Task misalignment confirmed. |
| Phase 4B retrain | 3.08M | 1.59M | in progress | More data, patience 20, full eval suite |

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
README, concept_vocab.py, and classifier.py all updated. Concept count: **49**. Architecture block corrected to Phase 4C: 3,702-dim raw input, 1,723-dim combined, ~3.08M parameters. Concept vocabulary table replaced with the actual 49-entry list from `concept_vocab.py`.

**7. NNUE task misalignment — documented lesson**
Three full training runs (Phase 5, 5C, 5D) confirmed that NNUE Feature Transformer activations do not improve concept classification (F1 stuck at 0.33 vs 0.56 for Phase 4B). The root cause is *task misalignment*: NNUE FT representations are organised around centipawn evaluation ("how good is this position"), not concept identity ("what pattern is present"). These are related but distinct prediction targets, and a representation optimised for one does not transfer cleanly to the other.

**Lesson:** Before adopting any pre-trained representation as a feature, verify that the pre-training task's labels correlate with your target labels. For NNUE → concept labels, the correlation is weak and indirect. The 2,491-dim `algo_v4` features are purpose-built for concept detection and remain the primary signal source.

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
