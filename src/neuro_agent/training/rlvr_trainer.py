"""Conservative GRPO RLVR trainer on top of an existing QLoRA adapter."""

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
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from neuro_agent.inference.engine import set_seed
from neuro_agent.training.dataset import load_sft_examples
from neuro_agent.training.rewards import verifiable_reward


@dataclass
class RLVRTrainingSummary:
    method: str
    model_name: str
    base_adapter_path: str
    train_examples: int
    max_steps: int
    num_generations: int
    learning_rate: float
    beta: float
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


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    return mapping[dtype_name.lower()]


def _adapter_size_mb(adapter_dir: Path) -> float:
    total = 0
    for path in adapter_dir.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total / (1024 * 1024)


def _build_prompt_messages(example: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    user_content = (
        f"Context:\n{json.dumps(example.get('context', {}), indent=2, sort_keys=True)}\n\n"
        f"Question: {example['question'].strip()}"
    )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_content},
    ]


def _build_rlvr_dataset(
    examples: list[dict[str, Any]],
    system_prompt: str,
) -> Dataset:
    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                "prompt": _build_prompt_messages(example, system_prompt),
                "verification_type": example["verification_type"],
                "ground_truth": json.dumps(example["ground_truth"]),
                "context": json.dumps(example.get("context", {})),
                "tolerance": json.dumps(example.get("tolerance")),
                "task_type": example.get("task_type", ""),
                "example_id": example.get("id", ""),
            }
        )
    return Dataset.from_list(rows)


class RLVRTrainer:
    """Run verifiable-reward GRPO continuation from an existing adapter."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def train(self) -> RLVRTrainingSummary:
        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        qlora_cfg = self.config["qlora"]
        train_cfg = self.config["training"]
        prompt_cfg = self.config["prompt"]
        output_cfg = self.config["output"]

        seed = int(train_cfg["seed"])
        set_seed(seed)

        dataset_path = self.project_root / data_cfg["train_path"]
        examples = load_sft_examples(dataset_path)
        max_examples = data_cfg.get("max_examples")
        if max_examples is not None:
            examples = examples[: int(max_examples)]

        checkpoint_dir = self.project_root / output_cfg["checkpoint_dir"]
        results_dir = self.project_root / output_cfg["results_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        compute_dtype = _resolve_dtype(qlora_cfg["bnb_4bit_compute_dtype"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        model = prepare_model_for_kbit_training(model)
        if train_cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        adapter_path = self.project_root / model_cfg["adapter_path"]
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)

        system_prompt = prompt_cfg["system"]
        train_dataset = _build_rlvr_dataset(examples, system_prompt)

        per_device_batch = int(train_cfg["per_device_train_batch_size"])
        grad_accum = int(train_cfg["gradient_accumulation_steps"])
        max_steps = int(train_cfg["max_steps"])
        num_generations = int(train_cfg["num_generations"])

        grpo_args = GRPOConfig(
            output_dir=str(checkpoint_dir / "runs"),
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=float(train_cfg["learning_rate"]),
            max_steps=max_steps,
            num_generations=num_generations,
            max_completion_length=int(train_cfg["max_completion_length"]),
            beta=float(train_cfg.get("beta", 0.04)),
            sync_ref_model=bool(train_cfg.get("sync_ref_model", True)),
            ref_model_sync_steps=int(train_cfg.get("ref_model_sync_steps", 64)),
            logging_steps=int(train_cfg.get("logging_steps", 5)),
            save_strategy=train_cfg.get("save_strategy", "steps"),
            save_steps=int(train_cfg.get("save_steps", 50)),
            save_total_limit=int(train_cfg.get("save_total_limit", 2)),
            bf16=bool(train_cfg.get("bf16", True)),
            gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
            seed=seed,
            report_to=[],
            temperature=float(train_cfg.get("temperature", 1.0)),
            top_p=float(train_cfg.get("top_p", 1.0)),
            remove_unused_columns=False,
            log_completions=bool(train_cfg.get("log_completions", False)),
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=verifiable_reward,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
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
        tokenizer.save_pretrained(final_dir)

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

        summary = RLVRTrainingSummary(
            method="GRPO",
            model_name=model_cfg["name"],
            base_adapter_path=str(adapter_path),
            train_examples=len(examples),
            max_steps=max_steps,
            num_generations=num_generations,
            learning_rate=float(train_cfg["learning_rate"]),
            beta=float(train_cfg.get("beta", 0.04)),
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
        )

        summary_path = results_dir / "training_summary.json"
        with summary_path.open("w") as handle:
            json.dump(asdict(summary), handle, indent=2)

        train_metrics_path = results_dir / "train_result.json"
        with train_metrics_path.open("w") as handle:
            json.dump(train_result.metrics, handle, indent=2)

        return summary
