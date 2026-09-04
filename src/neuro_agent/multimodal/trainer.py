"""QLoRA SFT trainer for Qwen2.5-VL multimodal models."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from neuro_agent.inference.engine import set_seed
from neuro_agent.multimodal.dataset import (
    build_multimodal_messages,
    format_sft_user_text,
    load_multimodal_examples,
    resolve_image_path,
    split_multimodal_by_subjects,
)
from neuro_agent.multimodal.model import load_vlm_for_training, print_architecture_summary
from neuro_agent.training.trainer import TrainingSummary, _adapter_size_mb


class MultimodalSFTDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        processor,
        system_prompt: str,
        project_root: Path,
        max_seq_length: int,
    ) -> None:
        self.examples = examples
        self.processor = processor
        self.system_prompt = system_prompt
        self.project_root = project_root
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        image_path = resolve_image_path(self.project_root, example["image_path"])
        user_text = format_sft_user_text(example)
        answer = (example.get("grounded_answer") or example.get("answer", "")).strip()

        full_messages = build_multimodal_messages(
            system_prompt=self.system_prompt,
            user_text=user_text,
            image_uri=f"file://{image_path}",
            assistant_text=answer,
        )
        prompt_messages = build_multimodal_messages(
            system_prompt=self.system_prompt,
            user_text=user_text,
            image_uri=f"file://{image_path}",
        )

        full_text = self.processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_image_inputs, full_video_inputs = process_vision_info(full_messages)
        prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_messages)

        full_inputs = self.processor(
            text=[full_text],
            images=full_image_inputs,
            videos=full_video_inputs,
            padding=False,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            padding=False,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        prompt_len = prompt_inputs["input_ids"].shape[-1]

        if input_ids.shape[0] > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            attention_mask = attention_mask[: self.max_seq_length]
            prompt_len = min(prompt_len, self.max_seq_length)

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in full_inputs:
            item["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs:
            item["image_grid_thw"] = full_inputs["image_grid_thw"][0]
        return item


class MultimodalDataCollator:
    def __init__(self, processor) -> None:
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)
        batch_input_ids = []
        batch_attention = []
        batch_labels = []

        for feature in features:
            pad_len = max_len - feature["input_ids"].shape[0]
            input_ids = torch.nn.functional.pad(feature["input_ids"], (0, pad_len), value=self.pad_token_id)
            attention = torch.nn.functional.pad(feature["attention_mask"], (0, pad_len), value=0)
            labels = torch.nn.functional.pad(feature["labels"], (0, pad_len), value=-100)
            batch_input_ids.append(input_ids)
            batch_attention.append(attention)
            batch_labels.append(labels)

        batch: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention),
            "labels": torch.stack(batch_labels),
        }

        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        if "image_grid_thw" in features[0]:
            grids = [
                f["image_grid_thw"].unsqueeze(0)
                if f["image_grid_thw"].ndim == 1
                else f["image_grid_thw"]
                for f in features
            ]
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
        return batch


class MultimodalSFTTrainer:
    """Run QLoRA supervised fine-tuning on image+text pairs."""

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
        examples = load_multimodal_examples(dataset_path)
        train_examples, val_examples = split_multimodal_by_subjects(
            examples,
            train_subjects=train_subjects,
            validation_subjects=validation_subjects,
            forbidden_subjects=forbidden_subjects,
        )

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

        max_seq_length = int(train_cfg["max_seq_length"])
        system_prompt = prompt_cfg["system"]
        train_dataset = MultimodalSFTDataset(
            train_examples,
            processor,
            system_prompt,
            self.project_root,
            max_seq_length,
        )
        eval_dataset = MultimodalSFTDataset(
            val_examples,
            processor,
            system_prompt,
            self.project_root,
            max_seq_length,
        )
        data_collator = MultimodalDataCollator(processor)

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
            data_collator=data_collator,
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
                "freeze_vision_tower": freeze_vision,
            },
            lora_target_modules=list(qlora_cfg["target_modules"]),
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
            base_adapter_path=str(self.project_root / model_cfg["adapter_path"])
            if model_cfg.get("adapter_path")
            else None,
            learning_rate=float(train_cfg["learning_rate"]),
        )

        summary_path = results_dir / "training_summary.json"
        with summary_path.open("w") as handle:
            payload = asdict(summary)
            payload["architecture"] = {
                "total_parameters": arch_info.total_parameters,
                "vision_parameters": arch_info.vision_parameters,
                "merger_parameters": arch_info.merger_parameters,
                "text_checkpoint_reused": arch_info.text_checkpoint_reused,
            }
            json.dump(payload, handle, indent=2)

        return summary
