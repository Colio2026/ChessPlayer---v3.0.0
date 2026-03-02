from __future__ import annotations
from pathlib import Path

def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[3]

def resolve_path(relative: str) -> Path:
    return repo_root_from_this_file() / relative
