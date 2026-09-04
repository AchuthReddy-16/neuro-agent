"""Conservative GRPO RLVR trainer for Qwen2.5-VL multimodal models."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from PIL import Image
from trl import GRPOConfig, GRPOTrainer

from neuro_agent.inference.engine import set_seed
from neuro_agent.multimodal.dataset import (
    format_sft_user_text,
    load_multimodal_examples,
    resolve_image_path,
)
from neuro_agent.multimodal.model import load_vlm_for_training, print_architecture_summary
from neuro_agent.training.rewards import (
    MULTIMODAL_REWARD_TRACKER,
    REWARD_SPEC,
    multimodal_verifiable_reward,
)
from neuro_agent.training.trainer import _adapter_size_mb


@dataclass
class MultimodalRLVRTrainingSummary:
    method: str
    model_name: str
    base_adapter_path: str
    train_examples: int
    max_steps: int
    num_generations: int
    learning_rate: float
    beta: float
    max_completion_length: int
    total_steps: int
    runtime_s: float
    peak_vram_mb: float
    adapter_size_mb: float
    checkpoint_path: str
    seed: int
    reward_mean: float | None
    reward_std: float | None
    reward_min: float | None
    reward_max: float | None
    kl_mean: float | None = None
    per_task_reward: dict[str, dict[str, float]] | None = None
    reward_spec: dict[str, str] | None = None


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every message uses structured content blocks for Arrow serialization."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        normalized.append({"role": message["role"], "content": content})
    return normalized


def _build_rlvr_dataset(
    examples: list[dict[str, Any]],
    system_prompt: str,
    project_root: Path,
) -> Dataset:
    rows: list[dict[str, Any]] = []
    for example in examples:
        image_path = resolve_image_path(project_root, example["image_path"])
        rows.append(
            {
                "image_path": str(image_path),
                "user_text": format_sft_user_text(example),
                "verification_type": example["verification_type"],
                "ground_truth": json.dumps(example["ground_truth"]),
                "context": json.dumps(example.get("context", example.get("relevant_context", {}))),
                "tolerance": json.dumps(example.get("tolerance")),
                "task_family": example.get("task_family", ""),
                "example_id": example.get("id", ""),
            }
        )

    dataset = Dataset.from_list(rows)

    def _attach_image_batch(batch: dict[str, Any]) -> dict[str, Any]:
        prompts: list[list[dict[str, Any]]] = []
        for image_path, user_text in zip(batch["image_path"], batch["user_text"], strict=True):
            image = Image.open(image_path).convert("RGB")
            prompts.append(
                _normalize_messages(
                    [
                        {"role": "system", "content": system_prompt.strip()},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": user_text},
                            ],
                        },
                    ]
                )
            )
        batch["prompt"] = prompts
        return batch

    dataset.set_transform(_attach_image_batch)
    return dataset


class MultimodalRLVRTrainer:
    """Run verifiable-reward GRPO continuation from a multimodal SFT adapter."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def train(self) -> MultimodalRLVRTrainingSummary:
        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        qlora_cfg = self.config["qlora"]
        train_cfg = self.config["training"]
        prompt_cfg = self.config["prompt"]
        output_cfg = self.config["output"]

        seed = int(train_cfg["seed"])
        set_seed(seed)
        MULTIMODAL_REWARD_TRACKER.totals.clear()

        dataset_path = self.project_root / data_cfg["train_path"]
        examples = load_multimodal_examples(dataset_path)

        forbidden = set(data_cfg.get("forbidden_subjects", []))
        if forbidden:
            import re

            def _subject(ex: dict[str, Any]) -> str | None:
                for field in ("image_path", "id", "image_id"):
                    m = re.search(r"(S\d{3})", str(ex.get(field, "")))
                    if m:
                        return m.group(1)
                return None

            before = len(examples)
            examples = [ex for ex in examples if _subject(ex) not in forbidden]
            leaked = before - len(examples)
            if leaked:
                raise ValueError(f"Refusing to train: {leaked} examples match forbidden subjects")

        max_examples = data_cfg.get("max_examples")
        if max_examples is not None:
            examples = examples[: int(max_examples)]

        checkpoint_dir = self.project_root / output_cfg["checkpoint_dir"]
        results_dir = self.project_root / output_cfg["results_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        freeze_vision = bool(model_cfg.get("freeze_vision_tower", True))
        model, processor, arch_info = load_vlm_for_training(
            model_cfg,
            qlora_cfg,
            freeze_vision_tower=freeze_vision,
        )
        print_architecture_summary(arch_info, qlora_cfg)

        if train_cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        adapter_path = self.project_root / model_cfg["adapter_path"]
        system_prompt = prompt_cfg["system"]
        train_dataset = _build_rlvr_dataset(examples, system_prompt, self.project_root)

        per_device_batch = int(train_cfg["per_device_train_batch_size"])
        grad_accum = int(train_cfg["gradient_accumulation_steps"])
        max_steps = int(train_cfg["max_steps"])
        num_generations = int(train_cfg["num_generations"])
        max_completion_length = int(train_cfg["max_completion_length"])

        grpo_args = GRPOConfig(
            output_dir=str(checkpoint_dir / "runs"),
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            steps_per_generation=train_cfg.get("steps_per_generation"),
            learning_rate=float(train_cfg["learning_rate"]),
            max_steps=max_steps,
            num_generations=num_generations,
            max_completion_length=max_completion_length,
            beta=float(train_cfg.get("beta", 0.04)),
            sync_ref_model=bool(train_cfg.get("sync_ref_model", False)),
            ref_model_sync_steps=int(train_cfg.get("ref_model_sync_steps", 64)),
            logging_steps=int(train_cfg.get("logging_steps", 5)),
            save_strategy=train_cfg.get("save_strategy", "steps"),
            save_steps=int(train_cfg.get("save_steps", 40)),
            save_total_limit=int(train_cfg.get("save_total_limit", 2)),
            bf16=bool(train_cfg.get("bf16", True)),
            gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
            seed=seed,
            report_to=[],
            temperature=float(train_cfg.get("temperature", 1.0)),
            top_p=float(train_cfg.get("top_p", 1.0)),
            remove_unused_columns=False,
            log_completions=bool(train_cfg.get("log_completions", False)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=multimodal_verifiable_reward,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=processor,
        )

        t0 = time.perf_counter()
        train_result = trainer.train()
        runtime_s = time.perf_counter() - t0

        final_dir = checkpoint_dir / "final"
        if final_dir.exists():
            for child in final_dir.iterdir():
                if child.is_file():
                    child.unlink()
        else:
            final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        history = trainer.state.log_history
        reward_rows = [row for row in history if "reward" in row]
        reward_values = [row.get("reward") for row in reward_rows if row.get("reward") is not None]
        kl_values = [row.get("kl") for row in history if row.get("kl") is not None]

        reward_mean = reward_std = reward_min = reward_max = None
        if reward_values:
            reward_mean = float(statistics.mean(reward_values))
            reward_std = float(statistics.pstdev(reward_values)) if len(reward_values) > 1 else 0.0
            reward_min = float(min(reward_values))
            reward_max = float(max(reward_values))

        metrics_path = results_dir / "training_metrics.csv"
        with metrics_path.open("w", newline="") as handle:
            fieldnames = sorted({key for row in history for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        per_task_reward = MULTIMODAL_REWARD_TRACKER.summary()
        with (results_dir / "per_task_reward.json").open("w") as handle:
            json.dump(per_task_reward, handle, indent=2)

        with (results_dir / "reward_spec.json").open("w") as handle:
            json.dump(REWARD_SPEC, handle, indent=2)

        summary = MultimodalRLVRTrainingSummary(
            method="GRPO",
            model_name=model_cfg["name"],
            base_adapter_path=str(adapter_path),
            train_examples=len(examples),
            max_steps=max_steps,
            num_generations=num_generations,
            learning_rate=float(train_cfg["learning_rate"]),
            beta=float(train_cfg.get("beta", 0.04)),
            max_completion_length=max_completion_length,
            total_steps=int(trainer.state.global_step),
            runtime_s=runtime_s,
            peak_vram_mb=peak_vram_mb,
            adapter_size_mb=_adapter_size_mb(final_dir),
            checkpoint_path=str(final_dir),
            seed=seed,
            reward_mean=reward_mean,
            reward_std=reward_std,
            reward_min=reward_min,
            reward_max=reward_max,
            kl_mean=float(statistics.mean(kl_values)) if kl_values else None,
            per_task_reward=per_task_reward,
            reward_spec=REWARD_SPEC,
        )

        summary_path = results_dir / "training_summary.json"
        with summary_path.open("w") as handle:
            json.dump(asdict(summary), handle, indent=2)

        train_metrics_path = results_dir / "train_result.json"
        with train_metrics_path.open("w") as handle:
            json.dump(train_result.metrics, handle, indent=2)

        return summary
