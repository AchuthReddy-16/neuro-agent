"""Trigger and recommendation policies for conditional verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuro_agent.agent.intent import IntentParseResult
    from neuro_agent.agent.verifier import DeterministicCheckResult, VerificationResult

from neuro_agent.tools.evidence import EvidenceBundle

INFO_WARNING_PREFIXES = (
    "Band power production uses source=",
)


def _actionable_warnings(warnings: list[str]) -> list[str]:
    return [w for w in warnings if not any(w.startswith(p) for p in INFO_WARNING_PREFIXES)]

Recommendation = str  # ACCEPT | RETRY_TOOL | REPLAN | REWRITE | INSUFFICIENT_EVIDENCE


def should_trigger_verifier(
    deterministic: DeterministicCheckResult,
    bundle: EvidenceBundle,
    intent_result: IntentParseResult,
    draft_answer: str | None,
    *,
    tool_count: int,
    intent_confidence: float | None = None,
) -> tuple[bool, list[str]]:
    """Decide whether to invoke the verifier model (deterministic checks always run)."""
    reasons: list[str] = []

    if not deterministic.passed:
        reasons.extend(deterministic.failure_codes)

    if _actionable_warnings(bundle.warnings):
        reasons.append("tool_warnings")

    if intent_result.warnings:
        reasons.append("intent_warnings")

    confidence = intent_confidence
    if confidence is None and intent_result.raw_json:
        raw_conf = intent_result.raw_json.get("confidence")
        if isinstance(raw_conf, (int, float)):
            confidence = float(raw_conf)
    if confidence is not None and confidence < 0.7:
        reasons.append("low_intent_confidence")

    if tool_count > 1:
        reasons.append("multi_tool")

    if deterministic.ambiguous:
        reasons.append("ambiguity")

    if _vision_ref_missing(bundle, intent_result):
        reasons.append("visual_ref_missing")

    if deterministic.unsupported_candidates:
        reasons.append("unsupported_value_candidate")

    # Fast-path: clean single-tool requests with deterministic pass skip model verifier.
    if (
        deterministic.passed
        and tool_count <= 1
        and not _actionable_warnings(bundle.warnings)
        and not intent_result.warnings
        and not deterministic.ambiguous
        and not _vision_ref_missing(bundle, intent_result)
    ):
        return False, []

    return (len(reasons) > 0, sorted(set(reasons)))


def _vision_ref_missing(bundle: EvidenceBundle, intent_result: IntentParseResult) -> bool:
    if intent_result.request is None:
        return False
    if not intent_result.request.include_vision_evidence:
        return False
    return len(bundle.vision_evidence) == 0


def select_recovery_action(
    verification: VerificationResult,
    deterministic: DeterministicCheckResult,
    *,
    tool_count: int,
    max_tool_calls: int,
    recovery_attempted: bool,
) -> Recommendation:
    """Map verification failure to a single recovery action (no recursion)."""
    if recovery_attempted:
        return "INSUFFICIENT_EVIDENCE"

    rec = verification.recommendation
    if rec == "ACCEPT":
        if not verification.passed and (
            "missing_answer_sections" in deterministic.failure_codes
            or verification.unsupported_claims
        ):
            return "REWRITE"
        return "INSUFFICIENT_EVIDENCE"

    if rec == "INSUFFICIENT_EVIDENCE":
        return "INSUFFICIENT_EVIDENCE"

    if rec == "RETRY_TOOL":
        if tool_count >= max_tool_calls:
            return "REWRITE" if verification.unsupported_claims else "INSUFFICIENT_EVIDENCE"
        if "tool_execution_failed" in deterministic.failure_codes:
            return "RETRY_TOOL"
        if "missing_evidence" in deterministic.failure_codes:
            return "RETRY_TOOL"

    if rec == "REPLAN":
        if "condition_mismatch" in deterministic.failure_codes or "tool_param_mismatch" in deterministic.failure_codes:
            return "REPLAN"
        return "REWRITE"

    if rec == "REWRITE":
        return "REWRITE"

    # Fallback from failure codes
    if verification.unsupported_claims or "unsupported_numeric" in deterministic.failure_codes:
        return "REWRITE"
    if "tool_execution_failed" in deterministic.failure_codes and tool_count < max_tool_calls:
        return "RETRY_TOOL"
    if "condition_mismatch" in deterministic.failure_codes:
        return "REPLAN"
    if "missing_evidence" in deterministic.failure_codes:
        return "INSUFFICIENT_EVIDENCE"

    return "INSUFFICIENT_EVIDENCE"


def is_format_only_failure(det: "DeterministicCheckResult") -> bool:
    """True when only answer formatting failed but numeric grounding is sound."""
    if det.passed:
        return False
    codes = set(det.failure_codes)
    if codes == {"missing_answer_sections"}:
        return bool(det.grounding and det.grounding.passed)
    if codes <= {"missing_answer_sections", "unit_mismatch"} and det.grounding and det.grounding.passed:
        return True
    return False


def build_insufficient_evidence_answer(
    question: str,
    *,
    failure_codes: list[str] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Safe final response when evidence is insufficient after recovery."""
    parts = [
        "Answer: I cannot provide a fully grounded answer for this question.",
        "Evidence: Available evidence was insufficient or could not be verified.",
        "Tools used: See trace for attempted tool invocations.",
        "Uncertainty: "
        + "; ".join(notes or failure_codes or ["insufficient verified evidence"]),
    ]
    return "\n".join(parts)
