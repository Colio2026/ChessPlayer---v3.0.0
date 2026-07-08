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

**Chess Coach — strategic coaching panel with Stockfish16 Evaluation Terms**

![Chess Coach](docs/screenshots/ChessCoachAlpha.png)

**Stockfish Integration — live evaluation bar and multi-PV analysis**

![Stockfish Integration](docs/screenshots/StockFishIntegration.png)

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

The Coach tab analyses the current position and produces a fully structured strategic recommendation in natural language — no bullet points, no engine gibberish. Every word is assembled from a curated phrase database sourced from Aaron Nimzowitsch's *My System* (1925), the foundational text of prophylactic positional chess. Nimzowitsch was the first theorist to describe chess in terms of strategic principles — pawn chains, blockades, overprotection, outposts, the prophylactic move — and his language maps cleanly onto the four strategies the coach detects. The result is coaching that reads like a chess teacher, not a search engine.

#### What the coach identifies

The coach classifies every position into one of four strategic categories:

| Strategy | What it means |
|---|---|
| **Blitz** | Direct assault on an exposed king — concentrate attackers, open lines, strike before the defence consolidates |
| **Flank** | Positional squeeze — restrict mobility, occupy outposts, cramp the opponent until the position collapses of its own weight |
| **Fortress** | Blockade defence — when objectively worse, seal open files, fix the pawn structure, and hold the wall |
| **Feint** | Wing misdirection — hold tension deliberately, bait a commitment to one wing, then switch to the other |

The strategy is shown as a badge alongside a confidence score and the detected game phase (opening / middlegame / endgame). When two strategies score within 8% of each other the coach surfaces both and notes the tension between plans.

#### What the output looks like

Each coaching response has four ordered parts:

1. **Headline** — one sentence naming the strategy, phase, and confidence
2. **Plan sentences** — up to four sentences assembled in sequence: *what is wrong → why we know → what to do → why right now*
3. **Tactics** — separate tactical observations (pins, forks, skewers, discoveries) shown in a distinct section
4. **Weak squares** — every square identified by the extractors, listed and overlaid as corner indicators on the coach board

**Example — Fortress detected in an opening position:**

```
FORTRESS                                                       29% · opening

Fortress Defence is suggested — opening (29% confidence).

  Our pieces are restricted — this is not a failure. In the fortress
  strategy, restriction is the design.

  The move count tells the story — our pieces command more squares.
  This is the first measure of positional advantage.

  Development demands activity. Every piece must have a purpose;
  every tempo must be spent wisely.

  The isolated pawn will not defend itself. Attack it now while
  every piece can participate in the assault.

⚡ Tactics
  The fork on c5 wins material by force — one of the attacked
  pieces must be surrendered.

Weak squares: a7 · b5 · b6 · b8 · c6 · c7 · e6 · e7
```

Clicking **Coach ON / OFF** in the toolbar enables or disables analysis. The side badge (White / Black) controls which player is being coached. Analysis runs automatically on each move.

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

The coaching backend (`src/chess_coach/`) is a fully deterministic, auditable analysis pipeline. There are no AI calls at runtime. Every coaching sentence is assembled from a curated SQLite phrase database using slot-filling — the same phrase template can produce different text depending on the specific squares, files, and pieces the extractors identify.

### The seven-stage pipeline

```
chess.Board
    │
    ▼
1. Extractors ──────── six modules measure board features
    │                  → list[MetricSignal]
    ▼
2. Phase filter ─────── classify opening / middlegame / endgame
    │                   re-weight signals for the current phase
    ▼
3. Strategy scoring ─── four detectors score 0.0 – 1.0 each
    │                   blitz · flank · fortress · feint
    ▼
4. Conflict resolver ── 5-rule priority cascade picks primary + confidence
    │                   outputs ResolverResult (primary, secondary, tie_band)
    ▼
5. Plan recommender ─── selects weakness_squares and move_flags
    │                   optionally consults Stockfish for move scoring
    ▼
6. GM precedents ─────── PatternMatcher queries coach_positions table
    │                    returns 0–3 matching GM games
    ▼
7. Narrator ─────────── slot-fills phrase templates → CoachOutput
                         headline + plan_sentences + tactic_hints
```

### Stage 1 — Extractors

Six independent modules each measure one aspect of the position and emit `MetricSignal` objects. A `MetricSignal` carries a normalised score (0.0–1.0), the side it applies to, a machine-readable cause tag, the key squares and pieces involved, a severity tier, and an `action_hint` — a plain-English fallback written by the extractor for use if no phrase DB entry matches.

| Extractor | Signals produced |
|---|---|
| `king_safety.py` | `king_exposure`, `sacrifice_delta` |
| `space_control.py` | `space_delta_queenside`, `space_delta_kingside` |
| `piece_mobility.py` | `piece_mobility_ratio` |
| `pawn_structure.py` | `pawn_fixedness`, `weak_pawns`, `passed_pawn` |
| `material_balance.py` | `material_imbalance`, `eval_deficit` (from Stockfish) |
| `tactic_scanner.py` | `tactic_pin`, `tactic_fork`, `tactic_skewer`, `tactic_discovery` |

No extractor ever produces English text — that is the narrator's job. No narrator ever produces scores — that is the extractor's job. This separation is a hard architectural rule.

### Stage 3 — Strategy scoring

Each of the four strategy detectors reads the full `MetricSignal` list and returns a score between 0.0 and 1.0. The detectors also accept optional history (recent board states and prior signal lists) to detect trends like a kingside space expansion building toward a blitz, or a feint that has been prepared over several moves.

### Stage 4 — Conflict resolver

The resolver applies a five-rule priority cascade to the raw strategy scores:

| Rule | Condition | Effect |
|---|---|---|
| 1 — Eval check | Stockfish eval deficit > 1.5 pawns | Fortress score gets +0.25 bonus |
| 2 — King emergency | Opponent king exposure > 0.80 | Blitz overrides Flank regardless of score |
| 3 — Phase override | Endgame + both Blitz and Flank > 0.65 | Flank promoted over Blitz |
| 4 — Tie band | Primary and secondary within 0.08 | Both strategies surfaced, `tie_band = True` |
| 5 — Feint gate | Feint is top scorer but no GM DB confirmation | Feint demoted to secondary |

A strategy must score above **0.65** to be considered "fired". Below that threshold the highest scorer still wins but with lower confidence. The feint gate (Rule 5) is intentional — misdirection is only named as the primary recommendation when the GM pattern database confirms it has been used by strong players from this position.

### Stage 6 — Phrase database

`data/chess_coach.db` contains two tables:

- **`phrases`** — keyed by `strategy / phase / metric / severity / fragment_type / cause_tag`. Each row is one sentence template attributed to a chapter of *My System*.
- **`tactics`** — keyed by `tactic_type / phase / severity`. One row per tactical pattern type.

The narrator queries the database for each of the four output slots in order — *diagnosis → evidence → plan → urgency* — picking the highest-priority phrase that matches the leading signal. If a specific phrase for `(strategy, metric, severity)` is not found, the query falls back first to `strategy=general`, then relaxes severity to `any`. If the database produces no match at all, the signal's own `action_hint` is used instead.

Phrase templates support five placeholders filled from `MetricSignal` data at query time:

| Placeholder | Filled with |
|---|---|
| `{square}` | First key square (e.g. `g7`) |
| `{file}` | File letter of the first key square (e.g. `g`) |
| `{piece}` | First key piece descriptor (e.g. `Ng5`) |
| `{side}` | `White` or `Black` |
| `{target}` | Second key square, if present |

Example phrase template and result:

```
Template:  "The pawn cover before the {side} king has been shattered —
            the {square} square gapes like an open wound."

Filled:    "The pawn cover before the Black king has been shattered —
            the g7 square gapes like an open wound."
```

All phrases carry a `source` field citing the chapter of *My System* they were drawn from, a `voice` field (`nimzowitsch` or `neutral`), and a `priority` integer (1–10) used to rank competing matches.

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
