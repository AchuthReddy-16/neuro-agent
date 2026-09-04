"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from neuro_agent.paths import CONFIGS_DIR, load_config_path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_config(name: str = "base") -> dict[str, Any]:
    return load_yaml(load_config_path(name))


def load_benchmark_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "benchmark.yaml")


def load_eval_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "eval.yaml")
