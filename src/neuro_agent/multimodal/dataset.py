"""Multimodal vision dataset loading and formatting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from neuro_agent.training.dataset import extract_subjects, split_by_subjects

SUBJECT_RE = re.compile(r"^(S\d{3})")

TASK_CLASS_TO_VERIFIER = {
    "numeric": "numeric",
    "categorical": "categorical",
    "comparison": "categorical",
    "ranking": "ranking",
    "set_membership": "set",
}


def load_multimodal_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def _merge_source_values_into_context(
    context: dict[str, Any],
    source_values: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(context)
    if not source_values:
        return merged
    if "values" in source_values and "values" not in merged:
        merged["values"] = source_values["values"]
    for key in ("operation", "value", "units", "k"):
        if key in source_values and key not in merged:
            merged[key] = source_values[key]
    return merged


def normalize_eval_example(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize vision eval rows to the verifier/eval harness schema."""
    normalized = dict(example)
    normalized["category"] = example.get("task_family", example.get("category", "unknown"))
    normalized["question"] = (example.get("question") or example.get("researcher_question", "")).strip()

    context = dict(example.get("context") or example.get("relevant_context") or {})
    source_values = example.get("source_values") or example.get("supporting_numeric_evidence")
    normalized["context"] = _merge_source_values_into_context(context, source_values)

    if "verification_type" not in normalized:
        task_class = example.get("task_class", "")
        normalized["verification_type"] = TASK_CLASS_TO_VERIFIER.get(task_class, "categorical")

    if "tolerance" not in normalized and normalized["verification_type"] == "numeric":
        normalized["tolerance"] = {"absolute": 1e-6, "relative": 1e-6}

    return normalized


def split_multimodal_by_subjects(
    examples: list[dict[str, Any]],
    train_subjects: set[str],
    validation_subjects: set[str],
    forbidden_subjects: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return split_by_subjects(
        examples,
        train_subjects=train_subjects,
        validation_subjects=validation_subjects,
        forbidden_subjects=forbidden_subjects,
    )


def resolve_image_path(project_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def format_sft_user_text(example: dict[str, Any]) -> str:
    context = example.get("relevant_context") or example.get("context") or {}
    question = (example.get("researcher_question") or example.get("question", "")).strip()
    return (
        f"Context:\n{json.dumps(context, indent=2, sort_keys=True)}\n\n"
        f"Question: {question}"
    )


def build_multimodal_messages(
    *,
    system_prompt: str,
    user_text: str,
    image_uri: str,
    assistant_text: str | None = None,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": image_uri},
        {"type": "text", "text": user_text},
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_content},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text.strip()})
    return messages
