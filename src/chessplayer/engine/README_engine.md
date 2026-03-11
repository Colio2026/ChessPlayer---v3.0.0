# engine/  —  UCI Engine Wrapper

Thin synchronous wrapper around the Stockfish subprocess.
Zero Qt dependencies. Called exclusively from `ui/engine_panel.py` on a QThread.

---

## Files

| File | Class | Purpose |
|------|-------|---------|
| `uci_engine.py` | `UciEngine`, `BestMove` | Launches and communicates with a UCI-compatible chess engine (Stockfish) via stdin/stdout. |

---

## UciEngine

```python
engine = UciEngine(engine_exe=Path("assets/engines/stockfish.exe"))
engine.start()
result = engine.analyze_movetime(prefix_uci, movetime_ms=2000)
engine.stop()
```

### BestMove fields

| Field | Type | Description |
|-------|------|-------------|
| `uci` | `str` | Best move in UCI format (e.g. `"e2e4"`). |
| `pv_uci_list` | `list[str]` | Principal variation — sequence of best moves. |
| `score_cp` | `int \| None` | Evaluation in centipawns. Positive = White advantage. |
| `score_mate` | `int \| None` | Mate in N. Positive = White wins, negative = Black wins. |
| `depth` | `int` | Search depth reached. |

---

## Engine Binary Location

Configured in `default.yaml`:
```yaml
engine:
  path: "assets/engines/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
  enabled_on_start: false
```

To swap engines: replace the binary and update the path. Any UCI-compatible
engine will work (Stockfish, Leela, Komodo, etc.).

---

## Adding a New Engine Feature

1. Add a method to `UciEngine` that sends the appropriate UCI command and reads the response.
2. Call it from `EnginePanel` on the analysis QThread.
3. Emit a new signal from `EnginePanel` to deliver the result to `MainWindow`.

Do not call `UciEngine` methods from the main Qt thread — all engine I/O blocks until the engine responds.
