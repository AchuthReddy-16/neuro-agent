"""Research request and evidence bundle schemas."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

QuestionType = Literal[
    "band_power",
    "rms",
    "psd_peak",
    "channel_ranking",
    "threshold_set",
    "condition_comparison",
]

SortDirection = Literal["ascending", "descending"]

ProvenanceSource = Literal[
    "stored_feature",
    "raw_eeg",
    "welch_psd",
    "metadata",
    "comparison",
    "vision_metadata",
]


@dataclass
class ResearchToolRequest:
    """Structured research request — no natural-language parsing."""

    question_type: QuestionType
    sample_id: str | None = None
    subject_id: str | None = None
    run_id: str | None = None
    epoch: int | None = None
    channels: list[str] | str | None = None
    frequency_band: str | None = None
    frequency_range: tuple[float, float] | None = None
    metric: str | None = None
    condition_a: str | None = None
    condition_b: str | None = None
    threshold: float | None = None
    threshold_mode: Literal["absolute", "median", "upper_quartile"] = "absolute"
    comparator: Literal["gt", "ge", "lt", "le"] | None = None
    top_k: int | None = None
    sort_direction: SortDirection = "descending"
    requested_visual_type: str | None = None
    image_id: str | None = None
    include_vision_evidence: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "epoch": self.epoch,
            "channels": self.channels,
            "frequency_band": self.frequency_band,
            "frequency_range": self.frequency_range,
            "metric": self.metric,
            "condition_a": self.condition_a,
            "condition_b": self.condition_b,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "comparator": self.comparator,
            "top_k": self.top_k,
            "sort_direction": self.sort_direction,
            "requested_visual_type": self.requested_visual_type,
            "image_id": self.image_id,
            "include_vision_evidence": self.include_vision_evidence,
            "extra": dict(self.extra),
        }


@dataclass
class ProvenanceRecord:
    """Provenance for a single numeric or derived result."""

    field: str
    source: ProvenanceSource
    detail: str
    sample_id: str | None = None
    artifact_path: str | None = None
    method: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "source": self.source,
            "detail": self.detail,
            "sample_id": self.sample_id,
            "artifact_path": self.artifact_path,
            "method": self.method,
            "note": self.note,
        }


@dataclass
class ToolInvocation:
    """Record of a single tool call within a routed request."""

    name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    runtime_ms: float
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "runtime_ms": self.runtime_ms,
            "provenance": [p.to_dict() for p in self.provenance],
            "success": self.success,
            "error": self.error,
        }


@dataclass
class EvidenceBundle:
    """Assembled evidence from a routed research request."""

    request_id: str
    question_type: QuestionType
    metadata: dict[str, Any] | None = None
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    numeric_evidence: dict[str, Any] = field(default_factory=dict)
    ranked_evidence: dict[str, Any] | None = None
    set_evidence: dict[str, Any] | None = None
    condition_evidence: dict[str, Any] | None = None
    vision_evidence: list[dict[str, Any]] = field(default_factory=list)
    units: str | None = None
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "question_type": self.question_type,
            "metadata": self.metadata,
            "tool_invocations": [inv.to_dict() for inv in self.tool_invocations],
            "numeric_evidence": self.numeric_evidence,
            "ranked_evidence": self.ranked_evidence,
            "set_evidence": self.set_evidence,
            "condition_evidence": self.condition_evidence,
            "vision_evidence": self.vision_evidence,
            "units": self.units,
            "provenance": [p.to_dict() for p in self.provenance],
            "warnings": list(self.warnings),
            "uncertainty_notes": list(self.uncertainty_notes),
            "success": self.success,
            "error": self.error,
        }


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


BAND_POWER_RECOMPUTE_NOTE = (
    "Band power production uses source='features' (features.parquet). "
    "Welch recompute from raw EEG may differ for low-frequency bands "
    "(delta/theta) due to integration window and filter edge effects."
)
