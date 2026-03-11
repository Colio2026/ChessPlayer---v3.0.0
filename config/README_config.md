# config/  —  Configuration Files and Loader

Three-layer configuration system: defaults → dev overrides → user overrides.
All layers are deep-merged at startup. User overrides persist between sessions.

---

## Files

| File | Purpose |
|------|---------|
| `default.yaml` | Canonical defaults. Checked into source control. Never edited at runtime. |
| `dev.yaml` | Developer overrides (e.g. `app.debug: true`). Gitignored. Applied after default. |
| `loader.py` | `load_config()` and `save_user_config_patch()`. |

---

## Merge Order

```
default.yaml
    ↓ deep_merge
dev.yaml  (if present)
    ↓ deep_merge
%APPDATA%\CHESSPLAYER\config.yaml  (user overrides, if present)
    ↓
final config dict used by the application
```

---

## Key Sections

### `paths`
```yaml
paths:
  data_dir: "data"          # SQLite index, MoveTree cache, PGN files
  assets_dir: "assets"      # Piece images, engine binary
  pieces_dir: "assets/pieces"
  engines_dir: "assets/engines"
```

### `engine`
```yaml
engine:
  enabled_on_start: false
  path: "assets/engines/stockfish-.../stockfish.exe"
```

### `browsing`
```yaml
browsing:
  page_size: 200            # Rows fetched per page in the game browser
```

### `coach`  *(Phase H+)*
```yaml
coach:
  db_path: "data/coach.db"  # SQLite phrase/pattern database
  enabled: true             # Master switch
  auto_annotate: false      # Auto-insert coach output as PGN comments
  movetime_ms: 2000         # Analysis time per position (ms)
```

### `ui`  *(persisted by the app, not set manually)*
```yaml
ui:
  last_source_id: 1
  last_source_type: "archive_file"
  last_source_path: "/path/to/library.pgn"
```
Written automatically by `_save_active_source_to_user_config()` whenever the user changes the active library. Restored at startup by `_restore_active_source_from_config()`.

---

## loader.py

### `load_config() → dict`
Loads and merges all three layers. Called once in `main.py`. The returned dict is passed to `MainWindow` and threaded through to all components that need config access.

### `save_user_config_patch(patch: dict) → Path`
Deep-merges `patch` into the user config file. Only the keys in `patch` are affected — all other user settings are preserved.

```python
# Example: persist last source
save_user_config_patch({
    "ui": {
        "last_source_id": 3,
        "last_source_path": "/data/lichess_elite.pgn"
    }
})
```

---

## Adding a New Config Key

1. Add the key with a sensible default to `default.yaml`.
2. Document it in this README.
3. Read it from `config["section"]["key"]` in the relevant module.
4. Never hardcode a fallback in application code — the default belongs in `default.yaml`.
