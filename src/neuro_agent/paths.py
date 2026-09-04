"""Centralized path resolution for project artifacts and model caches."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (this repo)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Cache / volume root. Override with NEURO_WORKSPACE_ROOT for shared persistent storage.
WORKSPACE_ROOT = Path(os.environ.get("NEURO_WORKSPACE_ROOT", str(PROJECT_ROOT)))

# Hugging Face cache directories under the workspace root
HF_HOME = WORKSPACE_ROOT / ".cache" / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"
HF_DATASETS_CACHE = HF_HOME / "datasets"
HF_TRANSFORMERS_CACHE = HF_HOME / "transformers"

# Project artifact directories
CONFIGS_DIR = PROJECT_ROOT / "configs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"


def ensure_dirs() -> None:
    """Create required directories if they do not exist."""
    for d in (
        HF_HOME,
        HF_HUB_CACHE,
        HF_DATASETS_CACHE,
        HF_TRANSFORMERS_CACHE,
        CHECKPOINTS_DIR,
        RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def configure_hf_cache() -> None:
    """Set environment variables so HF libraries use the configured cache root."""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_TRANSFORMERS_CACHE))
    os.environ.setdefault("TORCH_HOME", str(WORKSPACE_ROOT / ".cache" / "torch"))


def load_config_path(name: str = "base") -> Path:
    """Return path to a named config file."""
    return CONFIGS_DIR / f"{name}.yaml"
