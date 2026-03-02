import os
import sys
from pathlib import Path

def is_frozen() -> bool:
    return hasattr(sys, "_MEIPASS")

def base_path() -> Path:
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[3]

def resolve_path(relative: str) -> Path:
    return base_path() / relative
