#!/usr/bin/env python3
"""Run multimodal QLoRA SFT training (Stage F)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.multimodal.trainer import MultimodalSFTTrainer
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT, configure_hf_cache, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal QLoRA SFT training")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "multimodal_sft.yaml",
        help="Path to multimodal SFT YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()

    config = load_yaml(args.config)
    trainer = MultimodalSFTTrainer(config, PROJECT_ROOT)
    summary = trainer.train()

    print("\nTraining complete.")
    print(
        json.dumps(
            {
                "model": summary.model_name,
                "train_examples": summary.train_examples,
                "validation_examples": summary.validation_examples,
                "total_steps": summary.total_steps,
                "final_train_loss": summary.final_train_loss,
                "final_eval_loss": summary.final_eval_loss,
                "runtime_s": summary.runtime_s,
                "peak_vram_mb": summary.peak_vram_mb,
                "adapter_size_mb": summary.adapter_size_mb,
                "checkpoint_path": summary.checkpoint_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
