"""Read-only data access for samples, features, and epoch arrays."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from neuro_agent.tools.schemas import (
    ARRAYS_DIR,
    FEATURES_PATH,
    SAMPLE_ID_RE,
    SAMPLES_PATH,
    SampleNotFoundError,
    load_channel_names,
)


class SampleStore:
    """Lazy loader for samples.jsonl and HDF5 epoch arrays."""

    def __init__(self, samples_path: Path = SAMPLES_PATH, arrays_dir: Path = ARRAYS_DIR) -> None:
        self.samples_path = samples_path
        self.arrays_dir = arrays_dir
        self._index: dict[str, dict[str, Any]] | None = None

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if self._index is None:
            index: dict[str, dict[str, Any]] = {}
            with self.samples_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    index[str(record["sample_id"])] = record
            self._index = index
        return self._index

    def get(self, sample_id: str) -> dict[str, Any]:
        if not SAMPLE_ID_RE.match(sample_id):
            raise ValueError(f"Invalid sample_id format: {sample_id!r}")
        record = self._load_index().get(sample_id)
        if record is None:
            raise SampleNotFoundError(f"Sample not found: {sample_id}")
        return record

    def load_epoch(self, sample_id: str) -> tuple[np.ndarray, list[str], float]:
        record = self.get(sample_id)
        array_name = Path(str(record["array_path"])).name
        array_path = self.arrays_dir / array_name
        if not array_path.exists():
            array_path = Path(str(record["array_path"]))
        with h5py.File(array_path, "r") as handle:
            epoch = handle["eeg"][int(record["array_index"])].astype(np.float32)
            channels = [
                item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
                for item in handle["channel_names"][:]
            ]
            sampling_rate = float(handle.attrs.get("sampling_rate", record["sampling_rate"]))
        channels = [ch.upper() for ch in channels]
        self._validate_epoch_shape(epoch, channels)
        return epoch, channels, sampling_rate

    @staticmethod
    def _validate_epoch_shape(epoch: np.ndarray, channels: list[str]) -> None:
        if epoch.ndim != 2:
            raise ValueError(f"Expected epoch shape (n_channels, n_samples); got {epoch.shape}")
        if epoch.shape[0] != len(channels):
            raise ValueError(
                f"Channel count mismatch: epoch has {epoch.shape[0]} rows, "
                f"channel_names has {len(channels)}"
            )


class FeatureStore:
    """Lazy loader for features.parquet."""

    def __init__(self, features_path: Path = FEATURES_PATH) -> None:
        self.features_path = features_path
        self._frame: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = pd.read_parquet(self.features_path)
        return self._frame

    def get_sample_features(self, sample_id: str) -> pd.DataFrame:
        frame = self._load()
        rows = frame[frame["sample_id"] == sample_id]
        if rows.empty:
            raise SampleNotFoundError(f"No feature rows for sample: {sample_id}")
        return rows

    def get_channel_value(self, sample_id: str, channel: str, column: str) -> float:
        rows = self.get_sample_features(sample_id)
        match = rows[rows["channel"].str.upper() == channel.upper()]
        if match.empty:
            raise SampleNotFoundError(f"Channel {channel!r} missing for sample {sample_id}")
        if column not in match.columns:
            raise KeyError(f"Feature column missing: {column}")
        return float(match.iloc[0][column])


@lru_cache(maxsize=1)
def default_sample_store() -> SampleStore:
    return SampleStore()


@lru_cache(maxsize=1)
def default_feature_store() -> FeatureStore:
    return FeatureStore()


@lru_cache(maxsize=1)
def official_channels() -> tuple[str, ...]:
    return tuple(load_channel_names())
