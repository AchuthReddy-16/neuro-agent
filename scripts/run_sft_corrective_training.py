#!/usr/bin/env python3
"""Run corrective QLoRA continuation from existing SFT adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT, configure_hf_cache, ensure_dirs
from neuro_agent.training.trainer import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrective SFT continuation from existing adapter"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "sft_corrective.yaml",
        help="Path to corrective SFT YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()

    config = load_yaml(args.config)
    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(config, PROJECT_ROOT)
    summary = trainer.train()

    print("\nCorrective training complete.")
    print(
        json.dumps(
            {
                "model": summary.model_name,
                "base_adapter_path": summary.base_adapter_path,
                "train_examples": summary.train_examples,
                "validation_examples": summary.validation_examples,
                "num_train_epochs": summary.num_train_epochs,
                "learning_rate": summary.learning_rate,
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
