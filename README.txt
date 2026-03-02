CHESSPLAYER v3.0.0 - Phase 2 (Flat Imports)

This scaffold is intentionally "flat-import" style:
- main.py imports `config.*`, `ui.*`, `pgn.*`, `utils.*` directly (no `chessplayer.` prefix).
- Run from the `src/chessplayer/` directory OR configure VSCode to run that file with cwd=src/chessplayer.

Typical run:
  cd src/chessplayer
  python main.py

Index-only:
  python main.py --index
