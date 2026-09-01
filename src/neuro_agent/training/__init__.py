"""Training pipelines (stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QLoRAConfig:
    """QLoRA training configuration for 24GB VRAM."""

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    output_dir: str | Path = "/workspace/neuro-agent/checkpoints"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_seq_length: int = 8192
    gradient_checkpointing: bool = True


class QLoRATrainer:
    """Placeholder QLoRA trainer; will use PEFT + bitsandbytes."""

    def __init__(self, config: QLoRAConfig) -> None:
        self.config = config

    def train(self, dataset_path: str | Path, **kwargs: Any) -> Path:
        """Run QLoRA fine-tuning. Not implemented in scaffold."""
        raise NotImplementedError("QLoRA training will be implemented in the SFT stage.")
