"""Inference configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Runtime inference configuration."""

    model_name: str
    dtype: str = "bfloat16"
    seed: int = 42
    do_sample: bool = False
    max_new_tokens: int = 64
    use_cache: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False
