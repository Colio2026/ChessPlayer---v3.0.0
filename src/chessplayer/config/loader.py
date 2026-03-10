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


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load_config() -> Dict[str, Any]:
    default_path = resolve_path("config/default.yaml")
    cfg: Dict[str, Any] = _load_yaml_file(default_path)

    dev_path = resolve_path("config/dev.yaml")
    if dev_path.exists():
        dev_cfg = _load_yaml_file(dev_path)
        cfg = _deep_merge(cfg, dev_cfg)

    user_path = _user_override_path()
    if user_path.exists():
        user_cfg = _load_yaml_file(user_path)
        cfg = _deep_merge(cfg, user_cfg)

    return cfg


def save_user_config_patch(patch: Dict[str, Any]) -> Path:
    user_path = _user_override_path()
    user_path.parent.mkdir(parents=True, exist_ok=True)

    current = _load_yaml_file(user_path)
    merged = _deep_merge(current, patch)

    with open(user_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    return user_path