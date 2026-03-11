# src/chessplayer/  —  Application Root

Entry point and top-level wiring for ChessPlayer v3.0.0.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point. Parses `--index` / `--index-source` flags for headless indexing. Calls `run_app(config)` for normal GUI launch. |
| `app.py` | Creates the `QApplication`, instantiates `MainWindow`, and starts the Qt event loop. |

---

## Launch Sequence

```
python -m chessplayer          (or: python src/chessplayer/main.py)
  │
  ├─ load_config()             Merge default.yaml + dev.yaml + user override
  │
  ├─ run_app(config)
  │   ├─ QApplication()
  │   ├─ MainWindow(config)
  │   │   ├─ Build all widgets
  │   │   ├─ Wire all signals
  │   │   ├─ Initialise PgnStore from SQLite index
  │   │   └─ _restore_active_source_from_config()
  │   └─ win.show()
  │
  └─ sys.exit(app.exec())
```

---

## Headless Indexing (CLI)

```bash
# Index a single PGN file
python main.py --index-source data/lichess_elite.pgn

# Index a directory of PGN files
python main.py --index-source data/my_pgns/

# Rebuild the default configured source
python main.py --index
```

Indexing writes to `data/index.sqlite` and exits without opening the GUI.

---

## Package Layout

```
src/chessplayer/
├── main.py             Entry point
├── app.py              QApplication launcher
│
├── core/               Chess logic — no Qt (see README_core.md)
│   ├── pgn_edit.py
│   ├── game_session.py
│   └── log.py
│
├── pgn/                PGN storage and analysis (see README_pgn.md)
│   ├── store.py
│   ├── indexer.py
│   ├── move_tree.py
│   ├── continuations.py
│   ├── query.py
│   └── models.py
│
├── engine/             UCI engine wrapper (see README_engine.md)
│   └── uci_engine.py
│
├── ui/                 PySide6 widgets (see README_ui.md)
│   ├── main_window/
│   │   └── window.py   ← MainWindow (sole active file in this package)
│   ├── pgn_panel.py
│   ├── variations_panel.py
│   ├── coach_board.py
│   ├── engine_panel.py
│   ├── eval_bar.py
│   ├── board_model.py
│   ├── game_table_model.py
│   ├── query_builder.py
│   ├── continuation_stats_model.py
│   ├── variation_model.py
│   └── comment_dialog.py
│
├── config/             YAML config + loader (see README_config.md)
│   ├── default.yaml
│   ├── dev.yaml
│   └── loader.py
│
├── utils/
│   └── paths.py        resolve_path() — repo-relative path resolution
│
└── assets/
    ├── pieces/         PNG piece images (WP.png, BK.png, etc.)
    ├── engines/        Stockfish binary
    └── Board.qml       QML chess board component
```

---

## Dependency Rules

```
ui/  →  can import from:  core/, pgn/, engine/, config/, utils/
core/  →  can import from:  (nothing in this project — stdlib + python-chess only)
pgn/   →  can import from:  utils/  (and stdlib + python-chess + sqlite3)
engine/ → can import from:  (nothing in this project — stdlib only)
config/ → can import from:  utils/
```

No circular imports. No Qt in `core/`, `pgn/`, `engine/`, or `config/`.

---

## Data Directory (`data/`)

```
data/
├── index.sqlite          Game metadata index (written by indexer)
├── trees/                MoveTree gzip-pickle files (one per library source)
│   └── <sha1>.pkl.gz
├── coach.db              Coach phrase database (Phase H — not yet built)
└── *.pgn                 PGN library files
```

`data/` is gitignored. Users populate it by loading a PGN library through the GUI or CLI.
