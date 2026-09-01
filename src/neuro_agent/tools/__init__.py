"""Agent tool interfaces (stub)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for agent-callable tools."""

    name: str = "base_tool"
    description: str = ""

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Run the tool and return structured output."""


class LiteratureSearchTool(Tool):
    """Placeholder literature search tool."""

    name = "literature_search"
    description = "Search neuroscience literature databases."

    def execute(self, query: str, max_results: int = 10, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Literature search will be implemented with the agent.")


class DataAnalysisTool(Tool):
    """Placeholder data analysis tool."""

    name = "data_analysis"
    description = "Analyze neuroscience datasets and produce summaries."

    def execute(self, dataset_id: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Data analysis will be implemented with the agent.")
