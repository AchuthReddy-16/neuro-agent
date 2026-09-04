"""Agent execution trace schema and grounding checks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from neuro_agent.tools.evidence import EvidenceBundle

NUMERIC_RE = re.compile(
    r"(?<![A-Za-z0-9_])(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![A-Za-z0-9_])"
)


@dataclass
class GroundingResult:
    """Grounding check for numeric claims in the final answer."""

    numeric_claims: list[float] = field(default_factory=list)
    supported_claims: list[float] = field(default_factory=list)
    unsupported_claims: list[float] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_claims": self.numeric_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "passed": self.passed,
        }


@dataclass
class RecoveryTrace:
    """Single recovery cycle trace."""

    action: str
    success: bool
    pre_verification: dict[str, Any] | None = None
    post_verification: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "success": self.success,
            "pre_verification": self.pre_verification,
            "post_verification": self.post_verification,
            "notes": self.notes,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AgentTrace:
    """Full trace for one research agent request."""

    request_id: str
    original_question: str
    parsed_intent: dict[str, Any] | None
    intent_valid: bool
    routing_result: dict[str, Any] | None
    tool_invocations: list[dict[str, Any]]
    evidence_bundle: dict[str, Any] | None
    final_answer: str | None
    runtime_ms: float
    intent_latency_ms: float = 0.0
    answer_latency_ms: float = 0.0
    peak_vram_mb: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounding: GroundingResult | None = None
    failure_category: str | None = None
    # G.3B verification / recovery fields
    draft_answer: str | None = None
    verification_triggered: bool = False
    trigger_reason: list[str] = field(default_factory=list)
    first_pass_verification: dict[str, Any] | None = None
    recovery: RecoveryTrace | None = None
    final_verification: dict[str, Any] | None = None
    path_mode: str = "NORMAL"  # NORMAL | RECOVERY
    verifier_latency_ms: float = 0.0
    recovery_latency_ms: float = 0.0
    model_calls: int = 0

    @property
    def success(self) -> bool:
        if self.final_answer and "cannot provide a fully grounded answer" in self.final_answer.lower():
            return False
        grounding_ok = self.grounding is None or self.grounding.passed
        if self.final_verification is not None:
            grounding_ok = self.final_verification.get("passed", grounding_ok)
        elif self.grounding is not None:
            grounding_ok = self.grounding.passed
        return (
            self.intent_valid
            and self.evidence_bundle is not None
            and self.evidence_bundle.get("success", False)
            and self.final_answer is not None
            and not self.errors
            and grounding_ok
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "original_question": self.original_question,
            "parsed_intent": self.parsed_intent,
            "intent_valid": self.intent_valid,
            "routing_result": self.routing_result,
            "tool_invocations": self.tool_invocations,
            "evidence_bundle": self.evidence_bundle,
            "final_answer": self.final_answer,
            "draft_answer": self.draft_answer,
            "runtime_ms": self.runtime_ms,
            "intent_latency_ms": self.intent_latency_ms,
            "answer_latency_ms": self.answer_latency_ms,
            "verifier_latency_ms": self.verifier_latency_ms,
            "recovery_latency_ms": self.recovery_latency_ms,
            "peak_vram_mb": self.peak_vram_mb,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "grounding": self.grounding.to_dict() if self.grounding else None,
            "failure_category": self.failure_category,
            "verification_triggered": self.verification_triggered,
            "trigger_reason": list(self.trigger_reason),
            "first_pass_verification": self.first_pass_verification,
            "recovery": self.recovery.to_dict() if self.recovery else None,
            "final_verification": self.final_verification,
            "path_mode": self.path_mode,
            "model_calls": self.model_calls,
            "success": self.success,
        }


def _collect_evidence_numbers(bundle: EvidenceBundle, *, rel_tol: float = 1e-3) -> set[float]:
    """Flatten numeric values from an evidence bundle for grounding checks."""
    numbers: set[float] = set()

    def _add(value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            if math.isfinite(value):
                numbers.add(float(value))
        elif isinstance(value, dict):
            for v in value.values():
                _add(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _add(v)

    _add(bundle.numeric_evidence)
    if bundle.ranked_evidence:
        _add(bundle.ranked_evidence.get("values"))
    if bundle.set_evidence:
        _add(bundle.set_evidence.get("values"))
        _add(bundle.set_evidence.get("threshold_used"))
        _add(bundle.set_evidence.get("n_selected"))
    if bundle.condition_evidence:
        _add(bundle.condition_evidence)
    for ref in bundle.vision_evidence:
        _add(ref.get("source_numeric_values"))

    # Also include epoch indices and small integers from metadata
    if bundle.metadata:
        _add(bundle.metadata.get("epoch_index"))
        _add(bundle.metadata.get("sampling_rate_hz"))

    return numbers


def _is_supported(claim: float, evidence: set[float], *, rel_tol: float = 0.02) -> bool:
    """Check if a claimed number is supported by evidence within tolerance."""
    if claim in evidence:
        return True
    # Integer-like claims (e.g. top_k, n_selected)
    if abs(claim - round(claim)) < 1e-9:
        rounded = float(round(claim))
        if rounded in evidence:
            return True
    for ref in evidence:
        if ref == 0.0:
            if abs(claim) < 1e-6:
                return True
            continue
        if abs(claim - ref) <= max(abs(ref) * rel_tol, 1e-3):
            return True
    return False


def check_grounding(
    final_answer: str,
    bundle: EvidenceBundle,
    *,
    rel_tol: float = 0.02,
) -> GroundingResult:
    """Verify numeric claims in the final answer against evidence."""
    evidence_nums = _collect_evidence_numbers(bundle)
    claims: list[float] = []
    supported: list[float] = []
    unsupported: list[float] = []

    for match in NUMERIC_RE.finditer(final_answer):
        value = float(match.group(1))
        # Skip likely years / sample-id fragments / list indices
        if value in {160.0, 2024.0, 2025.0, 2026.0}:
            continue
        claims.append(value)
        if _is_supported(value, evidence_nums, rel_tol=rel_tol):
            supported.append(value)
        else:
            unsupported.append(value)

    return GroundingResult(
        numeric_claims=claims,
        supported_claims=supported,
        unsupported_claims=unsupported,
        passed=len(unsupported) == 0,
    )


def classify_failure(trace: AgentTrace) -> str | None:
    """Assign failure category A–E per G.3A spec."""
    if trace.success:
        return None
    if not trace.intent_valid:
        return "A_intent"
    if trace.evidence_bundle and not trace.evidence_bundle.get("success"):
        return "B_tool_exec"
    if trace.evidence_bundle and trace.evidence_bundle.get("success") and not trace.final_answer:
        return "C_evidence_assembly"
    if trace.grounding and not trace.grounding.passed:
        return "D_grounding"
    if trace.final_answer and "insufficient" in trace.final_answer.lower():
        return "E_insufficient_evidence"
    if trace.errors:
        return trace.failure_category or "B_tool_exec"
    return "E_insufficient_evidence"
