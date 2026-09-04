"""QLoRA SFT trainer for Qwen3 models."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from neuro_agent.inference.engine import set_seed
from neuro_agent.training.dataset import (
    load_sft_examples,
    split_by_subjects,
    tokenize_sft_example,
)


@dataclass
class TrainingSummary:
    model_name: str
    qlora_config: dict[str, Any]
    lora_target_modules: list[str]
    train_examples: int
    validation_examples: int
    train_subjects: list[str]
    validation_subjects: list[str]
    max_seq_length: int
    num_train_epochs: int
    total_steps: int
    effective_batch_size: int
    final_train_loss: float | None
    final_eval_loss: float | None
    runtime_s: float
    peak_vram_mb: float
    adapter_size_mb: float
    checkpoint_path: str
    total_tokens_seen: int
    seed: int
    base_adapter_path: str | None = None
    learning_rate: float | None = None


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


def _build_dataset(
    examples: list[dict[str, Any]],
    tokenizer,
    system_prompt: str,
    max_seq_length: int,
) -> Dataset:
    rows = [
        tokenize_sft_example(example, tokenizer, system_prompt, max_seq_length)
        for example in examples
    ]
    return Dataset.from_list(rows)


class SFTTrainer:
    """Run QLoRA supervised fine-tuning."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = project_root

    def train(self) -> TrainingSummary:
        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        qlora_cfg = self.config["qlora"]
        train_cfg = self.config["training"]
        prompt_cfg = self.config["prompt"]
        output_cfg = self.config["output"]

        seed = int(train_cfg["seed"])
        set_seed(seed)

        train_subjects = set(data_cfg["train_subjects"])
        validation_subjects = set(data_cfg["validation_subjects"])
        forbidden_subjects = set(data_cfg["forbidden_subjects"])

        dataset_path = self.project_root / data_cfg["train_path"]
        examples = load_sft_examples(dataset_path)
        train_examples, val_examples = split_by_subjects(
            examples,
            train_subjects=train_subjects,
            validation_subjects=validation_subjects,
            forbidden_subjects=forbidden_subjects,
        )

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

        base_adapter_path = model_cfg.get("adapter_path")
        if base_adapter_path:
            adapter_dir = self.project_root / base_adapter_path
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
            lora_target_modules = list(qlora_cfg["target_modules"])
        else:
            lora_config = LoraConfig(
                r=int(qlora_cfg["lora_r"]),
                lora_alpha=int(qlora_cfg["lora_alpha"]),
                lora_dropout=float(qlora_cfg["lora_dropout"]),
                target_modules=list(qlora_cfg["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            lora_target_modules = list(qlora_cfg["target_modules"])

        max_seq_length = int(train_cfg["max_seq_length"])
        system_prompt = prompt_cfg["system"]
        train_dataset = _build_dataset(train_examples, tokenizer, system_prompt, max_seq_length)
        eval_dataset = _build_dataset(val_examples, tokenizer, system_prompt, max_seq_length)

        per_device_batch = int(train_cfg["per_device_train_batch_size"])
        grad_accum = int(train_cfg["gradient_accumulation_steps"])
        effective_batch = per_device_batch * grad_accum
        steps_per_epoch = max(1, (len(train_examples) + effective_batch - 1) // effective_batch)
        total_steps = steps_per_epoch * int(train_cfg["num_train_epochs"])
        warmup_steps = max(1, int(total_steps * float(train_cfg.get("warmup_ratio", 0.03))))

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir / "runs"),
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=float(train_cfg["num_train_epochs"]),
            learning_rate=float(train_cfg["learning_rate"]),
            warmup_steps=warmup_steps,
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            logging_steps=int(train_cfg["logging_steps"]),
            eval_strategy=train_cfg["eval_strategy"],
            save_strategy=train_cfg["save_strategy"],
            bf16=bool(train_cfg.get("bf16", True)),
            optim=str(train_cfg.get("optim", "paged_adamw_8bit")),
            lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
            seed=seed,
            report_to=[],
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            dataloader_pin_memory=True,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
        train_losses = [row["loss"] for row in history if "loss" in row]
        eval_losses = [row["eval_loss"] for row in history if "eval_loss" in row]

        loss_csv = results_dir / "loss_history.csv"
        with loss_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["step", "epoch", "loss", "eval_loss", "learning_rate"],
            )
            writer.writeheader()
            for row in history:
                writer.writerow(
                    {
                        "step": row.get("step"),
                        "epoch": row.get("epoch"),
                        "loss": row.get("loss"),
                        "eval_loss": row.get("eval_loss"),
                        "learning_rate": row.get("learning_rate"),
                    }
                )

        total_tokens = int(train_result.metrics.get("train_samples", len(train_examples))) * max_seq_length

        summary = TrainingSummary(
            model_name=model_cfg["name"],
            qlora_config={
                "load_in_4bit": qlora_cfg["load_in_4bit"],
                "bnb_4bit_compute_dtype": qlora_cfg["bnb_4bit_compute_dtype"],
                "bnb_4bit_quant_type": qlora_cfg["bnb_4bit_quant_type"],
                "bnb_4bit_use_double_quant": qlora_cfg["bnb_4bit_use_double_quant"],
                "lora_r": qlora_cfg["lora_r"],
                "lora_alpha": qlora_cfg["lora_alpha"],
                "lora_dropout": qlora_cfg["lora_dropout"],
            },
            lora_target_modules=lora_target_modules,
            train_examples=len(train_examples),
            validation_examples=len(val_examples),
            train_subjects=sorted(train_subjects - validation_subjects),
            validation_subjects=sorted(validation_subjects),
            max_seq_length=max_seq_length,
            num_train_epochs=int(train_cfg["num_train_epochs"]),
            total_steps=int(trainer.state.global_step),
            effective_batch_size=effective_batch,
            final_train_loss=train_losses[-1] if train_losses else None,
            final_eval_loss=eval_losses[-1] if eval_losses else None,
            runtime_s=runtime_s,
            peak_vram_mb=peak_vram_mb,
            adapter_size_mb=_adapter_size_mb(final_dir),
            checkpoint_path=str(final_dir),
            total_tokens_seen=total_tokens,
            seed=seed,
            base_adapter_path=str(self.project_root / base_adapter_path)
            if base_adapter_path
            else None,
            learning_rate=float(train_cfg["learning_rate"]),
        )

        summary_path = results_dir / "training_summary.json"
        with summary_path.open("w") as handle:
            json.dump(asdict(summary), handle, indent=2)

        return summary
