#!/usr/bin/env python3
"""Run conservative GRPO multimodal RLVR post-training (Stage F.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.multimodal.rlvr_trainer import MultimodalRLVRTrainer
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT, configure_hf_cache, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal GRPO RLVR training")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "multimodal_rlvr.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()

    config = load_yaml(args.config)
    trainer = MultimodalRLVRTrainer(config, PROJECT_ROOT)
    summary = trainer.train()

    print("\nMultimodal RLVR training complete.")
    print(
        json.dumps(
            {
                "method": summary.method,
                "model": summary.model_name,
                "base_adapter_path": summary.base_adapter_path,
                "train_examples": summary.train_examples,
                "max_steps": summary.max_steps,
                "total_steps": summary.total_steps,
                "num_generations": summary.num_generations,
                "learning_rate": summary.learning_rate,
                "beta": summary.beta,
                "max_completion_length": summary.max_completion_length,
                "reward_mean": summary.reward_mean,
                "reward_std": summary.reward_std,
                "kl_mean": summary.kl_mean,
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
