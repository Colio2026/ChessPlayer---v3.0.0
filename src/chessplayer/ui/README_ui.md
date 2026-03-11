# ui/  —  PySide6 GUI Widgets

All user-facing components. No widget accesses python-chess directly.
All chess logic goes through `PgnEditor` (core/) or `BoardBridge`.

---

## Files

| File | Class | Purpose |
|------|-------|---------|
| `board_model.py` | `BoardListModel`, `BoardBridge` | Exposes piece positions to QML. `BoardBridge` is the QML↔PgnEditor translation layer. |
| `coach_board.py` | `CoachBoardWidget` | Secondary chess board for replaying comment lines and engine PV lines without affecting the main game. |
| `comment_dialog.py` | `CommentDialog` | Modal dialog for entering/editing PGN move comments. Hint shows parenthesis syntax for coach links. |
| `continuation_stats_model.py` | `ContinuationStatsModel` | `QAbstractTableModel` backing the Variations/Lines tab table. Holds `ContinuationStat` rows. |
| `engine_panel.py` | `EnginePanel` | Owns the Stockfish engine lifecycle. Runs analysis on a QThread. Displays multi-PV evaluation table. |
| `eval_bar.py` | `EvalBar` | Vertical evaluation bar widget. Animates between centipawn scores. |
| `game_table_model.py` | `GameTableModel` | Lazy-loading virtual table model for the game archive browser. Fetches rows from SQLite in 200-row chunks as the user scrolls. |
| `pgn_panel.py` | `PgnPanel` | HTML-rendered PGN tree viewer. Clickable moves, collapsible variations, comment coach links, native header table. |
| `query_builder.py` | `QueryBuilder` | Filter bar above the game archive. Emits `query_changed(Query)` signal. |
| `variation_model.py` | `VariationTreeModel` | `QAbstractItemModel` for a tree view of the current game's variation branches. |
| `variations_panel.py` | `VariationsPanel` | Position-relative continuation statistics tab. Uses `MoveTree` for O(1) queries. |

## Sub-packages

### `main_window/`
Contains `window.py` — the `MainWindow` class. Central coordinator for all widget wiring.
All other files in this directory (`browser_ops.py`, `indexing_ops.py`, `source_state.py`, `worker.py`) are superseded by `window.py` and should be deleted.

---

## Signal Flow Summary

```
QML Board drag
  └─> BoardBridge.attemptMove()
        └─> PgnEditor.try_user_move()
        └─> bridge.moveMade.emit(san)
              └─> MainWindow._on_position_changed()
                    ├─> pgn_panel.refresh()
                    ├─> variations_panel.refresh(prefix)
                    ├─> engine_panel.trigger_analysis()
                    └─> coachRequested.emit(fen, prefix)
```

---

## Coach Interface

The `Coach` tab in the right panel is currently a placeholder `QLabel`.
When Phase H/I builds the coach widget, it connects to:
```python
MainWindow.coachRequested  # Signal(str, list) — (fen, prefix_uci)
```
No other changes to `window.py` are required.
