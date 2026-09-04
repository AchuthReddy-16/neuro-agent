"""SFT dataset loading and chat-template formatting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

SUBJECT_RE = re.compile(r"^(S\d{3})")


def extract_subjects(example: dict[str, Any]) -> list[str]:
    subjects: list[str] = []
    for sample in example.get("source_samples", []):
        match = SUBJECT_RE.match(str(sample))
        if match:
            subjects.append(match.group(1))
    return subjects


def load_sft_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def split_by_subjects(
    examples: list[dict[str, Any]],
    train_subjects: set[str],
    validation_subjects: set[str],
    forbidden_subjects: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for example in examples:
        subjects = set(extract_subjects(example))
        if subjects & forbidden_subjects:
            raise ValueError(
                f"Forbidden subject in training data: example={example.get('id')} subjects={subjects}"
            )
        if not subjects:
            raise ValueError(f"Example missing subject metadata: {example.get('id')}")

        if subjects <= validation_subjects:
            val_rows.append(example)
        elif subjects <= train_subjects:
            train_rows.append(example)
        else:
            raise ValueError(
                f"Example spans multiple split subjects: id={example.get('id')} subjects={subjects}"
            )

    return train_rows, val_rows


def format_sft_messages(example: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    user_content = (
        f"Context:\n{json.dumps(example['tool_context'], indent=2, sort_keys=True)}\n\n"
        f"Question: {example['question'].strip()}"
    )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["answer"].strip()},
    ]


def tokenize_sft_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    system_prompt: str,
    max_seq_length: int,
) -> dict[str, list[int]]:
    messages = format_sft_messages(example, system_prompt)
    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
        prompt_len = min(len(prompt_ids), max_seq_length)
    else:
        prompt_len = len(prompt_ids)

    labels = [-100] * len(full_ids)
    for idx in range(prompt_len, len(full_ids)):
        labels[idx] = full_ids[idx]

    attention_mask = [1] * len(full_ids)
    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
