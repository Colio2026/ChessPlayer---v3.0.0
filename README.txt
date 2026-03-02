CHESSPLAYER 3.0.0

PySide6 (Qt6) chess database viewer/editor with Stockfish analysis.

Goals:
- Scale to single large PGN archive or millions of PGN files (indexed/normalized).
- Interactive board editing before/after loading a game.
- All edits stored as PGN variations.
- On close: prompt to overwrite / save-as / discard (no automatic backups).
- Branching UI (variation tree).
- Stockfish toggle with user knobs (time/depth/nodes, MultiPV, update rate, PV length).
- Filters: player/event/opening/ECO/result/date with sorting.
- Modern grayscale theme.
