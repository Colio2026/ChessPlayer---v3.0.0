# Fix Left/Right Arrow Keyboard Navigation

## Steps:
- [x] Step 1: Edit src/chessplayer/ui/main_window/window.py - Uncomment self._build_shortcuts() call in __init__, add self._active_board = "main" initialization.
- [x] Step 2: Test by running python src/main.py, load a game, click main board, press left/right arrows.
- [x] Step 3: Verify coach board switching if applicable.
- [x] Step 4: Mark complete and cleanup TODO.md.

**Arrows fixed ✅. Now fixing "Request Coach Analysis" button (greyed out).**

Steps:
- [x] Port coachRequested Signal + emit + enable_save_to_library.
- [ ] Test.

Current: Editing window.py.
