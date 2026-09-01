"""Research agent orchestration (stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Agent runtime configuration."""

    model_path: str
    tools: list[str] = field(default_factory=list)
    max_turns: int = 10
    system_prompt: str = ""


class NeuroResearchAgent:
    """Placeholder multimodal neuroscience research agent."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute an agent research loop. Not implemented in scaffold."""
        raise NotImplementedError("Agent orchestration will be implemented after RLVR stage.")
