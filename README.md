# ChessPlayer v3.0.0

A desktop chess analysis and coaching application for Windows. Browse large PGN game databases, analyze positions with a UCI engine, and receive Nimzowitsch-style strategic coaching — all running locally with no cloud dependency.

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
11. [Chess Coach Pipeline](#chess-coach-pipeline)
12. [Database Schema](#database-schema)
13. [Configuration System](#configuration-system)
14. [Development Scripts](#development-scripts)
15. [Contributing](#contributing)

---

## Features

- **PGN Game Browser** — load any `.pgn` file or directory, filter by player, event, opening, date, and ECO code; paginated with lazy loading for databases of any size
- **Interactive Board** — drag-and-drop piece moves, full variation tree support, promote/demote variations, inline move comments
- **Engine Analysis** — Stockfish UCI integration with multi-PV evaluation, animated eval bar, and best-move arrows; runs on a background thread so the UI stays responsive
- **Continuation Statistics** — see how often a position arises in your loaded library and what the top continuations are, powered by an O(1) position-tree lookup
- **Chess Coach** — Nimzowitsch-style strategic coaching: what's wrong with the position, why, what to do, and why now; weak squares highlighted on a secondary board with historical GM precedents
- **Offline-first** — all analysis, coaching, and database queries run locally

---

## Screenshots

**PGN Editor — game browser, interactive board, and engine analysis**

![PGN Editor](docs/screenshots/ChessPNGEditorAlpha.png)

**Chess Coach — strategic coaching panel with weak-square overlay**

![Chess Coach](docs/screenshots/ChessCoachAlpha.png)

---

## Requirements

- **Python** 3.11 or 3.12
- **Windows** 10 / 11 (64-bit)
- **Stockfish** binary — see [Stockfish Setup](#stockfish-setup)

Python packages (installed via `pip`):

| Package | Version |
|---|---|
| PySide6 | `>=6.8, <6.9` |
| python-chess | `>=1.11, <2.0` |
| PyYAML | `>=6.0.2, <7.0` |

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

The app uses flat imports and **must be launched from the `src/chessplayer` directory**:

```powershell
cd src\chessplayer
python main.py
```

Or use the provided dev script from the repo root:

```powershell
.\scripts\run_dev.ps1
```

**CLI flags:**

| Flag | Effect |
|---|---|
| *(none)* | Launch the GUI |
| `--index` | Rebuild the index for all configured sources, then exit (no GUI) |
| `--index-source <path>` | Index a single `.pgn` file or directory, then exit |

---

## Configuration

Configuration is a three-layer merge system. You should rarely need to touch anything beyond the engine path.

| Layer | File | Purpose |
|---|---|---|
| 1 — Defaults | `config/default.yaml` | Canonical defaults, never edited at runtime |
| 2 — Dev overrides | `config/dev.yaml` | Local developer overrides (gitignored) |
| 3 — User overrides | `%APPDATA%\CHESSPLAYER\config.yaml` | Auto-saved user preferences (last source, UI state) |

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
  pgn_source: "data/Carlsen.pgn"   # game library for GM precedent matching
  phrase_db: "data/chess_coach.db"
  movetime_ms: 500                  # Stockfish analysis time per position

browsing:
  page_size: 200                    # rows per page in the game table
```

See [config/README_config.md](config/README_config.md) for the full schema and merge rules.

---

## Using the App

### Game Browser

The left panel is the game archive. Use the filter bar at the top to narrow results by:

- **White / Black** — player name (partial match)
- **Event / Site** — tournament name or venue
- **ECO** — opening code (e.g. `E60`)
- **Date range** — from/to year

Click any row to load that game onto the board. The table is paged and lazy-loads from SQLite, so multi-gigabyte databases open instantly.

### Board & PGN Panel

- **Drag pieces** to make moves. Promotion prompts appear automatically.
- **Click moves in the PGN panel** to navigate to that ply.
- **Arrow keys** step forward/back through the mainline.
- Right-click a variation move to promote it to the mainline.
- Click the comment icon on a move to add or edit a PGN comment.

### Engine Panel

Toggle the engine on/off with the toolbar button. When active, Stockfish analyses the current position continuously and shows:

- Centipawn score and the animated eval bar
- Best move (highlighted on the board)
- Principal variation line(s)

Click any move in the PV line to replay it on the secondary coach board.

### Variations Panel

Shows how often the current position appears in your loaded game library and what moves were played from it, along with win/draw/loss percentages. This uses a pre-built move tree (not live SQL) so lookups are instant.

### Chess Coach

Click **Request Coach Analysis** to get a structured coaching message for the current position:

1. **Diagnosis** — what structural or tactical problem exists
2. **Evidence** — the signals that support the diagnosis (weak squares, pawn fixedness, king exposure, etc.)
3. **Plan** — a concrete recommendation (which piece to reroute, which pawn to advance, etc.)
4. **Urgency** — why this needs to happen now

Weak squares identified by the coach are overlaid as corner indicators on the secondary board.

---

## Indexing PGN Files

On first run, ChessPlayer indexes whatever PGN source is set in config. Indexing reads game headers and records a byte offset into the source file — it does **not** copy game text into the database. This means a 4 GB `.pgn` file indexes in a few minutes and opens games in milliseconds.

To add a new source:

1. Drop your `.pgn` file into `data/`.
2. Update `pgn_sources.active_source.path` in `config/default.yaml` (or via the UI source switcher).
3. Run with `--index` or let auto-indexing run on next launch.

The move tree for continuation statistics is built separately and cached in `data/trees/` as a gzip-pickled file keyed by the source's SHA-1 hash. It is rebuilt automatically when the source changes.

---

## Project Architecture

### Directory Layout

```
ChessPlayer - v3.0.0/
├── config/                  # YAML configuration (3-layer merge)
│   ├── default.yaml
│   ├── dev.yaml             # gitignored, local overrides
│   └── README_config.md
│
├── data/                    # Runtime data (gitignored)
│   ├── index.sqlite         # Game metadata + byte offsets
│   ├── chess_coach.db       # Coaching phrase database
│   ├── Carlsen.pgn          # Default game library
│   └── trees/              # MoveTree cache files (.pkl.gz)
│
├── assets/
│   ├── pieces/              # PNG piece images (WP.png, BK.png, …)
│   ├── engines/             # Place your Stockfish binary here
│   └── ui.qss               # Qt stylesheet
│
├── src/
│   ├── main.py              # CLI entry point
│   ├── chessplayer/         # Main application package
│   │   ├── app.py           # QApplication factory
│   │   ├── core/            # Chess state machine (no Qt)
│   │   ├── pgn/             # Database, indexer, move tree (no Qt)
│   │   ├── engine/          # UCI engine wrapper (no Qt)
│   │   ├── config/          # Config loader
│   │   ├── utils/           # Path utilities
│   │   └── ui/              # All PySide6 widgets + QML
│   │       ├── qml/BoardView.qml
│   │       └── main_window/window.py
│   └── chess_coach/         # Coaching backend (independent package)
│       ├── core/            # Data types + strategy engine
│       ├── extractors/      # Six metric extractor modules
│       ├── strategies/      # Four strategy detectors
│       ├── database/        # Phrase DB, pattern matcher, PGN indexer
│       └── tests/
│
└── scripts/                 # run_dev.ps1, run_dev.sh, build_windows.ps1
```

### Module Map

| Module | Role |
|---|---|
| `core/pgn_edit.py` | Central chess controller — `PgnEditor` owns all move/navigation/annotation logic |
| `core/game_session.py` | `GameSession` — undo/redo wrapper around `chess.Board` |
| `pgn/store.py` | `PgnStore` — SQLite query layer for game metadata |
| `pgn/indexer.py` | `build_or_rebuild_index_for_source()` — PGN scanner, writes offsets to SQLite |
| `pgn/move_tree.py` | `MoveTree` — position-frequency tree for O(1) continuation lookup |
| `pgn/continuations.py` | Queries MoveTree (full library) or store (filtered subset) |
| `pgn/query.py` | `Query` + `Clause` — filter spec compiler to parameterised SQL |
| `engine/uci_engine.py` | `UciEngine` — Stockfish subprocess controller, runs on QThread |
| `config/loader.py` | `load_config()` three-layer merge + `save_user_config_patch()` |
| `ui/main_window/window.py` | `MainWindow` — central coordinator, owns all signals |
| `ui/board_model.py` | `BoardListModel` + `BoardBridge` — Python ↔ QML bridge |
| `ui/pgn_panel.py` | HTML-rendered PGN tree viewer |
| `ui/variations_panel.py` | Continuation stats display |
| `ui/engine_panel.py` | Stockfish eval display, multi-PV lines |
| `ui/eval_bar.py` | Animated centipawn eval bar widget |
| `ui/game_table_model.py` | Virtual table model with 200-row lazy paging |
| `ui/query_builder.py` | Filter bar widget, emits `query_changed(Query)` |
| `chess_coach/core/strategy_engine.py` | Top-level coaching orchestrator |
| `chess_coach/core/data_types.py` | `MetricSignal`, `CoachOutput`, `GMPrecedent` dataclasses |
| `chess_coach/database/phrase_db.py` | `PhraseDB` — phrase lookup with slot-filling |
| `chess_coach/database/pattern_matcher.py` | `PatternMatcher` — GM precedent matching |

### Dependency Rules

These rules are enforced to prevent circular imports. Qt is never imported outside of `ui/`.

```
ui/          → core/, pgn/, engine/, config/, utils/
core/        → (none — pure chess logic)
pgn/         → utils/
engine/      → (none — subprocess wrapper)
config/      → utils/
chess_coach/ → core/ only
```

### Signal Flow

```
User drags piece on QML board
  └─ BoardBridge.attemptMove(from_sq, to_sq)
       └─ PgnEditor.try_user_move()
            └─ PgnEditor updates board + PGN node
                 └─ bridge.moveMade.emit(san)
                      └─ MainWindow._on_position_changed()
                           ├─ pgn_panel.refresh()
                           ├─ variations_panel.refresh(prefix_uci)
                           ├─ engine_panel.trigger_analysis()
                           └─ coachRequested.emit(fen, prefix_uci)
                                └─ MainWindow._on_coach_help_requested()
                                     └─ StrategyEngine.analyse(board, side)
                                          └─ CoachOutput → CoachPanel
```

---

## Chess Coach Pipeline

The coaching backend in `src/chess_coach/` runs a six-stage analysis pipeline on every position:

| Stage | Module | What it does |
|---|---|---|
| 1. Extract | `extractors/` (6 modules) | Measure board features → list of `MetricSignal` objects |
| 2. Phase filter | `phase_filter.py` | Classify opening / middlegame / endgame, re-weight signals |
| 3. Score strategies | `strategies/` (4 detectors) | Score each strategy 0.0–1.0 (blitz, flank, fortress, feint) |
| 4. Resolve conflict | `conflict_resolver.py` | Cascade rules → pick primary strategy + confidence |
| 5. Recommend plan | `plan_recommender.py` | Select weak squares, set move flags |
| 6. Narrate | `narrator.py` | Slot-fill phrase templates → `CoachOutput` |

**Extractors** and the signals they produce:

| Extractor | Signals |
|---|---|
| `king_safety.py` | `king_exposure`, `sacrifice_delta` |
| `space_control.py` | `space_delta_queenside`, `space_delta_kingside` |
| `piece_mobility.py` | `piece_mobility_ratio` |
| `pawn_structure.py` | `pawn_fixedness`, `weak_pawns`, `passed_pawn` |
| `material_balance.py` | `material_imbalance` |
| `tactic_scanner.py` | `tactic_pin`, `tactic_fork`, `tactic_skewer`, `tactic_discovery` |

Coaching phrases are stored in `data/chess_coach.db` and are sourced from Nimzowitsch's *My System*. Phrases are keyed by `strategy / metric / severity / fragment_type / cause_tag` and support slot substitution (`{square}`, `{file}`, `{piece}`, `{side}`, `{target}`).

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
    pgn_path     TEXT,           -- absolute path to the .pgn file
    offset_bytes INTEGER,        -- byte offset of this game within the file
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

Full game text is never duplicated into the database. `PgnStore.open_game_pgn_text(game_id)` seeks to `offset_bytes` and parses one game on demand.

### `data/trees/<sha1>.pkl.gz` — move tree cache

A gzip-pickled `MoveTree` object keyed by the source file's SHA-1. Position keys are the first four FEN fields (piece placement, turn, castling rights, en-passant), so transpositions from different move orders correctly hash to the same node.

---

## Configuration System

Full schema reference is in [config/README_config.md](config/README_config.md). The merge runs at startup:

```
default.yaml  ──┐
dev.yaml  ──────┤  deep_merge()  →  live config dict
%APPDATA%\CHESSPLAYER\config.yaml  ──┘
```

User overrides are written back automatically when the active source changes (e.g. switching game libraries in the UI). You can also create `config/dev.yaml` to set `app.debug: true` or override any key without touching defaults.

---

## Development Scripts

Located in `scripts/`:

| Script | Purpose |
|---|---|
| `run_dev.ps1` | Launch the app from the correct working directory (Windows PowerShell) |
| `run_dev.sh` | Same, for bash |
| `build_windows.ps1` | Package the app for distribution (Windows) |

---

## Contributing

- Module boundaries and the no-Qt-outside-ui rule are load-bearing — keep them.
- New extractors go in `src/chess_coach/extractors/`; add the module to the strategy engine's extractor list.
- New coaching phrases go in the phrase DB seed scripts keyed by `strategy/metric/severity/fragment_type`.
- Run the coach test suite before pushing: `pytest src/chess_coach/tests/`.
