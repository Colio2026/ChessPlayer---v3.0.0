# core/  —  Chess Logic (No Qt)

Pure Python chess logic. Zero Qt dependencies.
Every component here can be unit-tested without a display.
The GUI layer accesses chess state exclusively through these classes.

---

## Files

| File | Contents | Purpose |
|------|----------|---------|
| `pgn_edit.py` | `PgnEditor`, `LoadedGame` | Central chess controller. Owns the live board and the PGN tree. All move input, navigation, annotation, variation management, and persistence goes through here. |
| `game_session.py` | `GameSession`, `MoveResult` | Wraps a python-chess `Board` with undo/redo stacks. Handles raw move validation, promotion detection, and board state tracking. |
| `log.py` | `log` | Module-level `logging.Logger` instance named `'chessplayer'`. Import and use directly — `from core.log import log`. |

---

## pgn_edit.py — PgnEditor

The single source of truth for game state. Holds two views of the same game:

- **`session`** (`GameSession`) — the live `chess.Board`, tracking the exact current position.
- **`loaded.game`** (`chess.pgn.Game`) — the full PGN tree, including all variations and comments.

`current_node` is the `GameNode` in the tree that corresponds to the last move played. Navigation updates both `session.board` and `current_node` atomically.

### Key Method Groups

**Loading**
```python
editor.new_freeplay()             # Start a blank game
editor.load_pgn_text(pgn_str)     # Load from PGN string
```

**Navigation**
```python
editor.step_back()                # One move back
editor.step_forward_mainline()    # One move forward (mainline)
editor.navigate_to_ply(ply)       # Jump to any ply from start
editor.navigate_to_node(node)     # Jump to any GameNode (incl. variations)
```

**Move Input**
```python
editor.try_user_move(from_sq, to_sq)   # Board drag (validates, records in tree)
editor.resolve_promotion(promo)         # Complete a pending promotion ('q','r','b','n')
editor.apply_uci_move(uci)             # Programmatic move (engine, variation click)
```

**Annotation**
```python
editor.insert_comment(text)            # Attach comment to current_node
editor.insert_comment_at_ply(ply, text)
editor.set_header(key, value)          # Edit PGN tag
editor.add_header(key, value)
```

**Variation Management**
```python
editor.promote_variation(node)    # Make variation the mainline
editor.demote_variation(node)     # Move variation down
editor.delete_variation(node)     # Remove entire branch
editor.delete_from_node(node)     # Truncate tree from node
```

**SAN ↔ UCI Translation**
```python
# Used by PgnPanel to resolve comment coach links
uci_list, err = editor.san_to_uci(base_ply, ['Nf3', 'Nc6', 'Bb5'])
san_list       = editor.uci_to_san(base_ply, ['g1f3', 'b8c6', 'f1b5'])

# Used by engine / variations tab
prefix = editor.played_prefix_uci()   # All moves from start to current position
```

**Persistence**
```python
editor.export_pgn()              # Full PGN string with all annotations
editor.export_pgn_to_file(path)  # Write to a new file (Save As)
editor.replace_in_library_file() # Overwrite original file at stored byte offset (Save to Library)
```

---

## game_session.py — GameSession

Used internally by `PgnEditor`. You rarely need to access it directly.
Available as `editor.session` and `editor.session.board`.

```python
session.board          # The live chess.Board
session.fen()          # Current FEN string
session.can_undo()     # True if there are moves to step back through
session.try_move(from_sq, to_sq)       # Returns MoveResult
session.try_promotion(uci_prefix, 'q') # Complete pending promotion
```

### MoveResult fields

| Field | Type | Set when |
|-------|------|----------|
| `ok` | `bool` | Always |
| `uci` | `str \| None` | Move was legal |
| `san` | `str \| None` | Move was legal |
| `reason` | `str \| None` | Move was illegal — e.g. `"illegal"` |
| `promotion_required` | `bool` | Pawn reached back rank without specifying piece |
| `promotion_uci_prefix` | `str \| None` | e.g. `"e7e8"` — use to complete promotion |

---

## log.py

```python
from core.log import log

log.debug("Detailed trace: %s", value)
log.info("Event: %s", message)
log.warning("Something unexpected: %s", detail)
log.error("Failed: %s", exc)
```

Configure the log level in `main.py` or via the `logging` module.
The logger name is `'chessplayer'` — configure it in any standard logging setup.

---

## Design Rules

- **No Qt imports anywhere in `core/`.** If you find yourself importing from PySide6 here, the logic belongs in `ui/` instead.
- **No file I/O in `GameSession`.** Persistence is `PgnEditor`'s responsibility.
- **`PgnEditor` is not thread-safe.** All calls must come from the main Qt thread. The engine and indexer run on QThreads and communicate back via signals, never calling `PgnEditor` directly.
