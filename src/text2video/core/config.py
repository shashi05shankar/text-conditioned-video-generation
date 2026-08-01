"""Minimal YAML config loading.

Configs are plain YAML dicts. `load_config` returns a nested `Config` object that
supports both attribute access (`cfg.model.latent_dim`) and dict access
(`cfg["model"]["latent_dim"]`), and can be converted back to a plain dict so the
full config can be snapshotted into every experiment log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Nested attribute-accessible config wrapper around a dict."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return self[key]

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view, used to snapshot the config into run logs."""
        return _deep_copy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


def load_config(path: str | Path) -> Config:
    """Load a YAML config file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping, got {type(data).__name__}")
    return Config(data)


def merge_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Apply dotted-key overrides (e.g. {"train.batch_size": 8}) to a config.

    Used by CLI scripts so a single YAML can be reused for a quick smoke run
    without editing the file.
    """
    data = cfg.to_dict()
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        node = data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return Config(data)
