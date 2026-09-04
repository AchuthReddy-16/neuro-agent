"""Lightweight file-backed experiment store for the demo API."""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuro_agent.paths import PROJECT_ROOT, RESULTS_DIR


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ExperimentRecord:
    id: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    eeg: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    visualizations: list[dict[str, Any]] = field(default_factory=list)
    modalities: dict[str, bool] = field(default_factory=dict)
    status: str = "empty"
    is_demo: bool = False
    error_message: str | None = None
    analysis_history: list[dict[str, Any]] = field(default_factory=list)
    linked_sample_id: str | None = None
    linked_image_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentStore:
    """JSON + files under results/api_experiments/ (or configurable root)."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (RESULTS_DIR / "api_experiments")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _exp_dir(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def _meta_path(self, experiment_id: str) -> Path:
        return self._exp_dir(experiment_id) / "experiment.json"

    def create(self, *, is_demo: bool = False) -> ExperimentRecord:
        with self._lock:
            eid = _new_id("exp")
            now = time.time()
            rec = ExperimentRecord(
                id=eid,
                created_at=now,
                updated_at=now,
                status="empty",
                is_demo=is_demo,
                modalities={"eeg": False, "metadata": False, "vision": False, "text": True},
            )
            d = self._exp_dir(eid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "uploads").mkdir(exist_ok=True)
            self._write(rec)
            return rec

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        path = self._meta_path(experiment_id)
        if not path.exists():
            return None
        with self._lock:
            data = json.loads(path.read_text())
            return ExperimentRecord(**data)

    def save(self, rec: ExperimentRecord) -> None:
        with self._lock:
            rec.updated_at = time.time()
            self._write(rec)

    def _write(self, rec: ExperimentRecord) -> None:
        path = self._meta_path(rec.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec.to_dict(), indent=2, default=str))

    def store_upload(
        self,
        experiment_id: str,
        *,
        filename: str,
        content: bytes,
        kind: str,
        content_type: str | None,
    ) -> dict[str, Any]:
        rec = self.get(experiment_id)
        if rec is None:
            raise KeyError(experiment_id)
        asset_id = _new_id("asset")
        safe_name = Path(filename).name
        dest = self._exp_dir(experiment_id) / "uploads" / f"{asset_id}_{safe_name}"
        dest.write_bytes(content)
        artifact = {
            "id": asset_id,
            "name": safe_name,
            "kind": kind,
            "size_bytes": len(content),
            "content_type": content_type,
            "stored_path": str(dest.relative_to(PROJECT_ROOT))
            if dest.is_relative_to(PROJECT_ROOT)
            else str(dest),
            "image_id": None,
        }
        rec.artifacts.append(artifact)
        rec.status = "ready"
        self.save(rec)
        return artifact

    def append_analysis(self, experiment_id: str, summary: dict[str, Any]) -> None:
        rec = self.get(experiment_id)
        if rec is None:
            raise KeyError(experiment_id)
        rec.analysis_history.append(summary)
        self.save(rec)

    def delete(self, experiment_id: str) -> None:
        with self._lock:
            d = self._exp_dir(experiment_id)
            if d.exists():
                shutil.rmtree(d)
