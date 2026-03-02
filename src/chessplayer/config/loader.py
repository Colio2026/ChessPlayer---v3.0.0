from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from utils.paths import resolve_path


def _default_user_data_dir() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "CHESSPLAYER"
        return Path.home() / "AppData" / "Roaming" / "CHESSPLAYER"
    return Path.home() / ".chessplayer"


def _user_override_path() -> Path:
    # Optional user override (same locations as before)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = (Path(appdata) / "CHESSPLAYER") if appdata else _default_user_data_dir()
        return base / "config.yaml"
    return _default_user_data_dir() / "config.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> Dict[str, Any]:
    # Default from repo root /config
    default_path = resolve_path("config/default.yaml")
    with open(default_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    dev_path = resolve_path("config/dev.yaml")
    if dev_path.exists():
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, dev_cfg)

    user_path = _user_override_path()
    if user_path.exists():
        with open(user_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)

    return cfg
