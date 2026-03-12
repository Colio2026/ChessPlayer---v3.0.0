"""
conftest.py
===========
Pytest configuration for the chess_coach package.

This file is placed at src/chess_coach/ (the package root) and does one job:
adds the package root to sys.path so that `from core.X import Y` resolves
correctly regardless of where pytest is invoked from.

Run from src/chess_coach/:
    pytest tests/test_foundation.py -v

Run from repo root:
    pytest src/chess_coach/tests/test_foundation.py -v
"""

import sys
from pathlib import Path

# Insert the chess_coach package root (this file's directory) at the front
# of sys.path so `import core`, `import extractors`, etc. all resolve.
_package_root = Path(__file__).resolve().parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))
