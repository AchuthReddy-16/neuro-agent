"""Vision evidence resolver — structured refs without VLM inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from neuro_agent.tools.metadata import VisionAssetIndex, default_vision_asset_index, lookup_sample_metadata
from neuro_agent.tools.schemas import SampleNotFoundError

VISUALIZATION_ALIASES: dict[str, str] = {
    "topomap": "topomap_multi_band",
    "topomap_multi_band": "topomap_multi_band",
    "psd": "power_spectral_density",
    "power_spectral_density": "power_spectral_density",
    "spectrogram": "spectrogram",
    "band_power": "channel_band_power",
    "channel_band_power": "channel_band_power",
    "waveform": "waveform",
    "condition_comparison": "condition_comparison",
    "comparison": "condition_comparison",
}


class VisionFamily(str, Enum):
    TOPOMAP = "topomap_multi_band"
    PSD = "power_spectral_density"
    SPECTROGRAM = "spectrogram"
    BAND_POWER = "channel_band_power"
    WAVEFORM = "waveform"
    CONDITION_COMPARISON = "condition_comparison"


@dataclass(frozen=True)
class VisionEvidenceRef:
    """Structured vision artifact reference with numeric sidecar values."""

    image_id: str
    image_path: str
    family: str
    source_sample_id: str
    subject_id: str
    run_id: str
    source_numeric_values: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "family": self.family,
            "source_sample_id": self.source_sample_id,
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "source_numeric_values": self.source_numeric_values,
            "metadata": dict(self.metadata),
        }


def _normalize_visual_type(visual_type: str | None) -> str | None:
    if visual_type is None:
        return None
    key = visual_type.strip().lower().replace("-", "_").replace(" ", "_")
    return VISUALIZATION_ALIASES.get(key, key)


def _record_to_ref(record: dict[str, Any]) -> VisionEvidenceRef:
    return VisionEvidenceRef(
        image_id=str(record["image_id"]),
        image_path=str(record["image_path"]),
        family=str(record["visualization_type"]),
        source_sample_id=str(record.get("epoch_sample_id", "")),
        subject_id=str(record.get("subject_id", "")),
        run_id=str(record.get("run_id", "")),
        source_numeric_values=dict(record.get("source_numeric_values") or {}),
        metadata={
            "condition": record.get("condition"),
            "movement_condition": record.get("movement_condition"),
            "task_state": record.get("task_state"),
            "frequency_band": record.get("frequency_band"),
            "split": record.get("split"),
            "image_sha256": record.get("image_sha256"),
            "eeg_channels_used": record.get("eeg_channels_used"),
            "plotting_parameters": record.get("plotting_parameters"),
            "source_processed_data_reference": record.get("source_processed_data_reference"),
        },
    )


def resolve_vision_evidence(
    *,
    sample_id: str | None = None,
    image_id: str | None = None,
    visual_type: str | None = None,
    subject_id: str | None = None,
    run_id: str | None = None,
    epoch: int | None = None,
    vision_index: VisionAssetIndex | None = None,
) -> list[VisionEvidenceRef]:
    """Connect a sample or image to vision artifacts and numeric sidecar values.

    Returns structured references only — no VLM or pixel inference.
    """
    vision_index = vision_index or default_vision_asset_index()

    if image_id is not None:
        record = vision_index.get_image_record(image_id)
        if record is None:
            raise SampleNotFoundError(f"Vision image not found: {image_id}")
        ref = _record_to_ref(record)
        if visual_type is not None:
            requested = _normalize_visual_type(visual_type)
            if requested and ref.family != requested:
                raise ValueError(
                    f"Image {image_id} is {ref.family!r}, not {requested!r}"
                )
        return [ref]

    if sample_id is None and subject_id is not None:
        meta = lookup_sample_metadata(
            subject_id=subject_id,
            run_id=run_id,
            epoch=epoch,
            vision_index=vision_index,
        )
        sample_id = meta.sample_id
    elif sample_id is not None:
        lookup_sample_metadata(sample_id=sample_id, vision_index=vision_index)

    if sample_id is None:
        raise ValueError("Provide image_id or sample_id (or subject/run/epoch)")

    records: list[dict[str, Any]] = []
    vision_index._load()
    assert vision_index._by_sample is not None
    for record in vision_index._by_sample.get(sample_id, []):
        records.append(record)

    requested = _normalize_visual_type(visual_type)
    if requested:
        records = [r for r in records if r.get("visualization_type") == requested]

    if not records:
        if requested:
            raise SampleNotFoundError(
                f"No vision evidence for sample {sample_id} with type {requested!r}"
            )
        return []

    return [_record_to_ref(record) for record in records]
