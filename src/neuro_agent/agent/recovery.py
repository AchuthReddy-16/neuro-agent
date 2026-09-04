"""Single-cycle recovery actions after verification failure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from neuro_agent.agent.intent import IntentParseResult, parse_and_validate_intent, validate_intent
from neuro_agent.agent.answer_format import format_grounded_answer
from neuro_agent.agent.policies import (
    Recommendation,
    build_insufficient_evidence_answer,
    select_recovery_action,
)
from neuro_agent.agent.verifier import (
    DeterministicCheckResult,
    VerificationResult,
    run_deterministic_checks,
)
from neuro_agent.tools.evidence import EvidenceBundle, ResearchToolRequest
from neuro_agent.tools.router import route_research_request


@dataclass
class RecoveryResult:
    """Outcome of one recovery cycle."""

    action: Recommendation
    success: bool
    final_answer: str | None = None
    bundle: EvidenceBundle | None = None
    intent_result: IntentParseResult | None = None
    tool_invocations: list[dict[str, Any]] = field(default_factory=list)
    verification: VerificationResult | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "success": self.success,
            "final_answer": self.final_answer,
            "evidence_bundle": self.bundle.to_dict() if self.bundle else None,
            "parsed_intent": self.intent_result.raw_json if self.intent_result else None,
            "tool_invocations": self.tool_invocations,
            "verification": self.verification.to_dict() if self.verification else None,
            "notes": self.notes,
        }


GenerateFn = Callable[[str, EvidenceBundle], tuple[str, float, float]]
ParseIntentFn = Callable[[str], tuple[IntentParseResult, float, float]]


def execute_recovery(
    action: Recommendation,
    *,
    question: str,
    draft_answer: str | None,
    bundle: EvidenceBundle,
    intent_result: IntentParseResult,
    verification: VerificationResult,
    deterministic: DeterministicCheckResult,
    generate_answer: GenerateFn,
    parse_intent: ParseIntentFn,
    request_id: str,
    tool_count: int,
    max_tool_calls: int = 6,
    prior_tool_keys: set[tuple[str, str]] | None = None,
) -> RecoveryResult:
    """Execute exactly one recovery action."""
    prior_keys = set(prior_tool_keys or [])
    for inv in bundle.tool_invocations:
        prior_keys.add((inv.name, _inputs_key(inv.inputs)))

    if action == "INSUFFICIENT_EVIDENCE":
        answer = build_insufficient_evidence_answer(
            question,
            failure_codes=verification.failure_codes,
            notes=verification.missing_evidence or verification.unsupported_claims,
        )
        return RecoveryResult(
            action=action,
            success=False,
            final_answer=answer,
            bundle=bundle,
            intent_result=intent_result,
            tool_invocations=[inv.to_dict() for inv in bundle.tool_invocations],
            notes=["insufficient_evidence_response"],
        )

    if action == "REWRITE":
        answer, _, _ = generate_answer(question, bundle)
        post_det = run_deterministic_checks(
            answer,
            bundle,
            request=intent_result.request,
        )
        if not post_det.passed and post_det.grounding and post_det.grounding.passed:
            answer = format_grounded_answer(question, bundle)
            post_det = run_deterministic_checks(
                answer,
                bundle,
                request=intent_result.request,
            )
            notes = ["rewrite_from_same_evidence", "deterministic_format_fallback"]
        else:
            notes = ["rewrite_from_same_evidence"]
        post_ver = VerificationResult(
            passed=post_det.passed,
            confidence_score=0.9 if post_det.passed else 0.4,
            failure_codes=post_det.failure_codes,
            unsupported_claims=post_det.unsupported_candidates,
            missing_evidence=post_det.missing_evidence,
            unit_issues=post_det.unit_issues,
            condition_mismatch=post_det.condition_mismatches,
            recommendation="ACCEPT" if post_det.passed else "INSUFFICIENT_EVIDENCE",
            deterministic=post_det,
        )
        return RecoveryResult(
            action=action,
            success=post_det.passed,
            final_answer=answer,
            bundle=bundle,
            intent_result=intent_result,
            tool_invocations=[inv.to_dict() for inv in bundle.tool_invocations],
            verification=post_ver,
            notes=notes,
        )

    if action == "RETRY_TOOL":
        if tool_count >= max_tool_calls:
            return execute_recovery(
                "INSUFFICIENT_EVIDENCE",
                question=question,
                draft_answer=draft_answer,
                bundle=bundle,
                intent_result=intent_result,
                verification=verification,
                deterministic=deterministic,
                generate_answer=generate_answer,
                parse_intent=parse_intent,
                request_id=request_id,
                tool_count=tool_count,
                max_tool_calls=max_tool_calls,
                prior_tool_keys=prior_keys,
            )
        if intent_result.request is None:
            return execute_recovery(
                "INSUFFICIENT_EVIDENCE",
                question=question,
                draft_answer=draft_answer,
                bundle=bundle,
                intent_result=intent_result,
                verification=verification,
                deterministic=deterministic,
                generate_answer=generate_answer,
                parse_intent=parse_intent,
                request_id=request_id,
                tool_count=tool_count,
                max_tool_calls=max_tool_calls,
                prior_tool_keys=prior_keys,
            )
        new_bundle = route_research_request(intent_result.request, request_id=request_id)
        new_count = tool_count + len(new_bundle.tool_invocations)
        dup = _has_duplicate_tools(new_bundle, prior_keys)
        if dup or new_count > max_tool_calls:
            return execute_recovery(
                "REWRITE" if not dup else "INSUFFICIENT_EVIDENCE",
                question=question,
                draft_answer=draft_answer,
                bundle=bundle,
                intent_result=intent_result,
                verification=verification,
                deterministic=deterministic,
                generate_answer=generate_answer,
                parse_intent=parse_intent,
                request_id=request_id,
                tool_count=tool_count,
                max_tool_calls=max_tool_calls,
                prior_tool_keys=prior_keys,
            )
        answer, _, _ = generate_answer(question, new_bundle)
        post_det = run_deterministic_checks(answer, new_bundle, request=intent_result.request)
        post_ver = VerificationResult(
            passed=post_det.passed,
            confidence_score=0.85 if post_det.passed else 0.35,
            failure_codes=post_det.failure_codes,
            unsupported_claims=post_det.unsupported_candidates,
            recommendation="ACCEPT" if post_det.passed else "INSUFFICIENT_EVIDENCE",
            deterministic=post_det,
        )
        return RecoveryResult(
            action=action,
            success=post_det.passed and new_bundle.success,
            final_answer=answer,
            bundle=new_bundle,
            intent_result=intent_result,
            tool_invocations=[inv.to_dict() for inv in new_bundle.tool_invocations],
            verification=post_ver,
            notes=["retry_tool_routing"],
        )

    if action == "REPLAN":
        new_intent, _, _ = parse_intent(question)
        if not new_intent.success or new_intent.request is None:
            return execute_recovery(
                "INSUFFICIENT_EVIDENCE",
                question=question,
                draft_answer=draft_answer,
                bundle=bundle,
                intent_result=intent_result,
                verification=verification,
                deterministic=deterministic,
                generate_answer=generate_answer,
                parse_intent=parse_intent,
                request_id=request_id,
                tool_count=tool_count,
                max_tool_calls=max_tool_calls,
                prior_tool_keys=prior_keys,
            )
        new_bundle = route_research_request(new_intent.request, request_id=request_id)
        new_count = tool_count + len(new_bundle.tool_invocations)
        if _has_duplicate_tools(new_bundle, prior_keys) or new_count > max_tool_calls:
            return execute_recovery(
                "REWRITE",
                question=question,
                draft_answer=draft_answer,
                bundle=bundle,
                intent_result=intent_result,
                verification=verification,
                deterministic=deterministic,
                generate_answer=generate_answer,
                parse_intent=parse_intent,
                request_id=request_id,
                tool_count=tool_count,
                max_tool_calls=max_tool_calls,
                prior_tool_keys=prior_keys,
            )
        answer, _, _ = generate_answer(question, new_bundle)
        post_det = run_deterministic_checks(answer, new_bundle, request=new_intent.request)
        post_ver = VerificationResult(
            passed=post_det.passed,
            confidence_score=0.8 if post_det.passed else 0.3,
            failure_codes=post_det.failure_codes,
            unsupported_claims=post_det.unsupported_candidates,
            recommendation="ACCEPT" if post_det.passed else "INSUFFICIENT_EVIDENCE",
            deterministic=post_det,
        )
        return RecoveryResult(
            action=action,
            success=post_det.passed and new_bundle.success,
            final_answer=answer,
            bundle=new_bundle,
            intent_result=new_intent,
            tool_invocations=[inv.to_dict() for inv in new_bundle.tool_invocations],
            verification=post_ver,
            notes=["replan_intent_and_route"],
        )

    return RecoveryResult(action=action, success=False, notes=["unknown_action"])


def plan_recovery(
    verification: VerificationResult,
    deterministic: DeterministicCheckResult,
    *,
    tool_count: int,
    max_tool_calls: int = 6,
    recovery_attempted: bool = False,
) -> Recommendation:
    return select_recovery_action(
        verification,
        deterministic,
        tool_count=tool_count,
        max_tool_calls=max_tool_calls,
        recovery_attempted=recovery_attempted,
    )


def _inputs_key(inputs: dict[str, Any]) -> str:
    import json

    return json.dumps(inputs, sort_keys=True)


def _has_duplicate_tools(bundle: EvidenceBundle, prior_keys: set[tuple[str, str]]) -> bool:
    for inv in bundle.tool_invocations:
        key = (inv.name, _inputs_key(inv.inputs))
        if key in prior_keys:
            return True
    return False
