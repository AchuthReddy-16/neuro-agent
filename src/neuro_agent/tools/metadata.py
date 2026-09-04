"""Sample metadata lookup for deterministic research routing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from neuro_agent.paths import PROJECT_ROOT
from neuro_agent.tools._stores import SampleStore, default_sample_store, official_channels
from neuro_agent.tools.normalization import sample_labels_from_record
from neuro_agent.tools.schemas import SAMPLE_ID_RE, SampleNotFoundError

VISION_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "vision" / "metadata" / "images.jsonl"
EPOCH_FROM_PARTS_RE = re.compile(r"^(S\d{3})_(R\d{2})_(E\d{3})$")


@dataclass(frozen=True)
class FeatureRefs:
    """Pointers to precomputed per-channel features."""

    feature_path: str
    available_columns: tuple[str, ...] = (
        "mean",
        "variance",
        "std",
        "rms",
        "peak_to_peak",
        "delta_power",
        "theta_power",
        "alpha_mu_power",
        "beta_power",
    )


@dataclass(frozen=True)
class ArrayRef:
    """Pointer to raw epoch array in HDF5."""

    array_path: str
    array_index: int
    n_channels: int
    n_samples: int


@dataclass(frozen=True)
class VisionAssetRef:
    """Lightweight vision artifact reference tied to a sample."""

    image_id: str
    image_path: str
    visualization_type: str
    frequency_band: str | None = None


@dataclass(frozen=True)
class SampleMetadata:
    """Normalized sample metadata with explicit, non-merged label fields."""

    sample_id: str
    subject_id: str
    run_id: str
    epoch_index: int
    task_type: str
    movement: str
    condition: str
    protocol: str | None
    event_code: str | None
    split: str
    sampling_rate_hz: float
    channels: tuple[str, ...]
    array_ref: ArrayRef
    feature_refs: FeatureRefs
    vision_assets: tuple[VisionAssetRef, ...] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "epoch_index": self.epoch_index,
            "task_type": self.task_type,
            "movement": self.movement,
            "condition": self.condition,
            "protocol": self.protocol,
            "event_code": self.event_code,
            "split": self.split,
            "sampling_rate_hz": self.sampling_rate_hz,
            "channels": list(self.channels),
            "array_ref": {
                "array_path": self.array_ref.array_path,
                "array_index": self.array_ref.array_index,
                "n_channels": self.array_ref.n_channels,
                "n_samples": self.array_ref.n_samples,
            },
            "feature_refs": {
                "feature_path": self.feature_refs.feature_path,
                "available_columns": list(self.feature_refs.available_columns),
            },
            "vision_assets": [
                {
                    "image_id": asset.image_id,
                    "image_path": asset.image_path,
                    "visualization_type": asset.visualization_type,
                    "frequency_band": asset.frequency_band,
                }
                for asset in self.vision_assets
            ],
            "extra": dict(self.extra),
        }


class VisionAssetIndex:
    """Read-only index of vision image metadata keyed by sample and image id."""

    def __init__(self, metadata_path: Path = VISION_METADATA_PATH) -> None:
        self.metadata_path = metadata_path
        self._by_sample: dict[str, list[dict[str, Any]]] | None = None
        self._by_image: dict[str, dict[str, Any]] | None = None

    def _load(self) -> None:
        if self._by_sample is not None:
            return
        by_sample: dict[str, list[dict[str, Any]]] = {}
        by_image: dict[str, dict[str, Any]] = {}
        with self.metadata_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample_id = str(record.get("epoch_sample_id", ""))
                by_sample.setdefault(sample_id, []).append(record)
                image_id = str(record.get("image_id", ""))
                if image_id:
                    by_image[image_id] = record
        self._by_sample = by_sample
        self._by_image = by_image

    def assets_for_sample(self, sample_id: str) -> list[VisionAssetRef]:
        self._load()
        assert self._by_sample is not None
        refs: list[VisionAssetRef] = []
        for record in self._by_sample.get(sample_id, []):
            refs.append(
                VisionAssetRef(
                    image_id=str(record["image_id"]),
                    image_path=str(record["image_path"]),
                    visualization_type=str(record["visualization_type"]),
                    frequency_band=record.get("frequency_band"),
                )
            )
        return refs

    def get_image_record(self, image_id: str) -> dict[str, Any] | None:
        self._load()
        assert self._by_image is not None
        return self._by_image.get(image_id)


@lru_cache(maxsize=1)
def default_vision_asset_index() -> VisionAssetIndex:
    return VisionAssetIndex()


def _epoch_index_from_sample_id(sample_id: str) -> int:
    match = SAMPLE_ID_RE.match(sample_id)
    if not match:
        raise ValueError(f"Invalid sample_id format: {sample_id!r}")
    return int(sample_id.rsplit("_", 1)[-1][1:])


def _resolve_sample_id(
    *,
    sample_id: str | None = None,
    subject_id: str | None = None,
    run_id: str | None = None,
    epoch: int | None = None,
    sample_store: SampleStore,
) -> str:
    if sample_id is not None:
        if not SAMPLE_ID_RE.match(sample_id):
            raise ValueError(f"Invalid sample_id format: {sample_id!r}")
        return sample_id

    if subject_id is None or run_id is None or epoch is None:
        raise ValueError(
            "Provide sample_id or all of subject_id, run_id, and epoch"
        )

    subject = subject_id.strip().upper()
    run = run_id.strip().upper()
    if not run.startswith("R"):
        run = f"R{run}"
    epoch_str = f"E{int(epoch):03d}"
    candidate = f"{subject}_{run}_{epoch_str}"
    if not SAMPLE_ID_RE.match(candidate):
        raise ValueError(f"Could not construct valid sample_id from parts: {candidate!r}")
    sample_store.get(candidate)
    return candidate


def lookup_sample_metadata(
    *,
    sample_id: str | None = None,
    subject_id: str | None = None,
    run_id: str | None = None,
    epoch: int | None = None,
    sample_store: SampleStore | None = None,
    vision_index: VisionAssetIndex | None = None,
) -> SampleMetadata:
    """Resolve sample_id or subject/run/epoch into normalized metadata."""
    sample_store = sample_store or default_sample_store()
    vision_index = vision_index or default_vision_asset_index()

    resolved_id = _resolve_sample_id(
        sample_id=sample_id,
        subject_id=subject_id,
        run_id=run_id,
        epoch=epoch,
        sample_store=sample_store,
    )

    try:
        record = sample_store.get(resolved_id)
    except SampleNotFoundError:
        raise

    labels = sample_labels_from_record(record)
    channels = tuple(official_channels())
    array_ref = ArrayRef(
        array_path=str(record["array_path"]),
        array_index=int(record["array_index"]),
        n_channels=int(record.get("n_channels", len(channels))),
        n_samples=int(record.get("n_samples", 640)),
    )
    feature_refs = FeatureRefs(feature_path=str(record.get("feature_path", "data/processed/features.parquet")))
    vision_assets = tuple(vision_index.assets_for_sample(resolved_id))

    extra: dict[str, Any] = {}
    if "metadata" in record:
        extra["preprocessing"] = record["metadata"]
    if record.get("start_time") is not None:
        extra["start_time_s"] = float(record["start_time"])
    if record.get("end_time") is not None:
        extra["end_time_s"] = float(record["end_time"])

    return SampleMetadata(
        sample_id=labels.sample_id,
        subject_id=labels.subject_id,
        run_id=labels.run_id,
        epoch_index=_epoch_index_from_sample_id(labels.sample_id),
        task_type=labels.task_type,
        movement=labels.movement,
        condition=labels.condition,
        protocol=labels.protocol,
        event_code=labels.event_code,
        split=str(labels.split or record.get("split", "")),
        sampling_rate_hz=float(record.get("sampling_rate", 160.0)),
        channels=channels,
        array_ref=array_ref,
        feature_refs=feature_refs,
        vision_assets=vision_assets,
        extra=extra,
    )
