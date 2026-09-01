"""Data schemas and loaders (stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingExample:
    """Single SFT/RLVR training example."""

    instruction: str
    input: str = ""
    output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NeuroDatasetSchema:
    """Expected schema for neuroscience training data."""

    version: str = "0.1"
    fields: list[str] = field(
        default_factory=lambda: ["instruction", "input", "output", "modality", "source"]
    )
    modalities: list[str] = field(
        default_factory=lambda: ["text", "eeg", "fmri", "behavioral"]
    )

    def validate(self, example: dict[str, Any]) -> bool:
        """Check that an example conforms to the schema."""
        return all(f in example for f in self.fields if f not in ("input",))
