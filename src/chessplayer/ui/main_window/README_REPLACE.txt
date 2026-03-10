Replace the old single-file module with this package.

Steps:
1. Delete: src/chessplayer/ui/main_window.py
2. Copy this folder to: src/chessplayer/ui/main_window/
3. Keep app.py importing:
      from ui.main_window import MainWindow

The package __init__.py re-exports MainWindow, so app.py does not need any other change.
