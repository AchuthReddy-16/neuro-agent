"""Deterministic and model-based verification for grounded answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from neuro_agent.agent.intent import IntentParseResult, extract_json_object
from neuro_agent.agent.traces import GroundingResult, check_grounding
from neuro_agent.tools.evidence import EvidenceBundle, ResearchToolRequest

Recommendation = Literal[
    "ACCEPT",
    "RETRY_TOOL",
    "REPLAN",
    "REWRITE",
    "INSUFFICIENT_EVIDENCE",
]

VERIFIER_SYSTEM_PROMPT = """You are a grounding verifier for neuroscience research answers.
You do NOT solve the problem. You only check whether the draft answer is supported by the evidence bundle.

Output ONLY a single JSON object:
{
  "passed": true|false,
  "confidence_score": 0.0-1.0,
  "failure_codes": ["..."],
  "unsupported_claims": ["..."],
  "evidence_conflicts": ["..."],
  "missing_evidence": ["..."],
  "unit_issues": ["..."],
  "condition_mismatch": ["..."],
  "recommendation": "ACCEPT|RETRY_TOOL|REPLAN|REWRITE|INSUFFICIENT_EVIDENCE"
}

Rules:
- Every numeric value in the draft answer must appear in the evidence (within tolerance).
- Channel names cited must exist in evidence or metadata.
- Units in the answer must match evidence units.
- For condition comparisons, cited conditions must match the request.
- If evidence is missing for the question, recommend INSUFFICIENT_EVIDENCE.
- If only wording is wrong but numbers are grounded, recommend REWRITE.
- If tools failed or params look wrong, recommend RETRY_TOOL or REPLAN.
- If fully grounded, passed=true and recommendation=ACCEPT."""

ANSWER_SECTIONS = ("answer:", "evidence:", "tools used:", "uncertainty:")
CHANNEL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,3})\b")
NON_CHANNEL_TOKENS = {
    "THE", "AND", "FOR", "RMS", "PSD", "BETA", "DELTA", "THETA", "NONE",
    "RAW", "EEG", "SEE", "ALL", "NOT", "USE", "HZ", "UV", "BAND", "TOOL",
    "USED", "REST", "WELCH", "FROM", "LOW", "DUE", "MAY", "VIA",
}
UNIT_PATTERNS = {
    "power": ("µv²", "uv2", "μv²", "power", "µv^2"),
    "amplitude": ("µv", "uv", "μv", "rms", "amplitude"),
    "frequency": ("hz", "hertz", "frequency"),
}


@dataclass
class DeterministicCheckResult:
    """Outcome of cheap pre-model verification checks."""

    passed: bool
    failure_codes: list[str] = field(default_factory=list)
    grounding: GroundingResult | None = None
    missing_channels: list[str] = field(default_factory=list)
    unit_issues: list[str] = field(default_factory=list)
    condition_mismatches: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    evidence_conflicts: list[str] = field(default_factory=list)
    unsupported_candidates: list[str] = field(default_factory=list)
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failure_codes": self.failure_codes,
            "grounding": self.grounding.to_dict() if self.grounding else None,
            "missing_channels": self.missing_channels,
            "unit_issues": self.unit_issues,
            "condition_mismatches": self.condition_mismatches,
            "missing_evidence": self.missing_evidence,
            "evidence_conflicts": self.evidence_conflicts,
            "unsupported_candidates": self.unsupported_candidates,
            "ambiguous": self.ambiguous,
        }


@dataclass
class VerificationResult:
    """Full verification outcome (deterministic + optional model)."""

    passed: bool
    confidence_score: float
    failure_codes: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    evidence_conflicts: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    unit_issues: list[str] = field(default_factory=list)
    condition_mismatch: list[str] = field(default_factory=list)
    recommendation: Recommendation = "ACCEPT"
    deterministic: DeterministicCheckResult | None = None
    model_verified: bool = False
    raw_model_json: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "confidence_score": self.confidence_score,
            "failure_codes": self.failure_codes,
            "unsupported_claims": self.unsupported_claims,
            "evidence_conflicts": self.evidence_conflicts,
            "missing_evidence": self.missing_evidence,
            "unit_issues": self.unit_issues,
            "condition_mismatch": self.condition_mismatch,
            "recommendation": self.recommendation,
            "deterministic": self.deterministic.to_dict() if self.deterministic else None,
            "model_verified": self.model_verified,
            "raw_model_json": self.raw_model_json,
        }


def _bundle_from_dict(data: dict[str, Any] | None) -> EvidenceBundle | None:
    if not data:
        return None
    from neuro_agent.tools.evidence import ToolInvocation, ProvenanceRecord

    invocations = []
    for inv in data.get("tool_invocations", []):
        prov = [ProvenanceRecord(**p) for p in inv.get("provenance", [])]
        invocations.append(
            ToolInvocation(
                name=inv["name"],
                inputs=inv.get("inputs", {}),
                outputs=inv.get("outputs", {}),
                runtime_ms=inv.get("runtime_ms", 0.0),
                provenance=prov,
                success=inv.get("success", True),
                error=inv.get("error"),
            )
        )
    prov_records = [ProvenanceRecord(**p) for p in data.get("provenance", [])]
    return EvidenceBundle(
        request_id=data.get("request_id", ""),
        question_type=data.get("question_type", "band_power"),
        metadata=data.get("metadata"),
        tool_invocations=invocations,
        numeric_evidence=data.get("numeric_evidence", {}),
        ranked_evidence=data.get("ranked_evidence"),
        set_evidence=data.get("set_evidence"),
        condition_evidence=data.get("condition_evidence"),
        vision_evidence=data.get("vision_evidence", []),
        units=data.get("units"),
        provenance=prov_records,
        warnings=list(data.get("warnings", [])),
        uncertainty_notes=list(data.get("uncertainty_notes", [])),
        success=data.get("success", False),
        error=data.get("error"),
    )


def _collect_evidence_channels(bundle: EvidenceBundle) -> set[str]:
    channels: set[str] = set()

    def _scan(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in {"channel", "channels", "top_channel", "highest_rms_channel"}:
                    if isinstance(v, str):
                        channels.add(v.upper())
                    elif isinstance(v, list):
                        channels.update(str(c).upper() for c in v)
                _scan(v)
        elif isinstance(obj, list):
            for item in obj:
                _scan(item)

    _scan(bundle.numeric_evidence)
    _scan(bundle.ranked_evidence)
    _scan(bundle.set_evidence)
    _scan(bundle.metadata)
    for inv in bundle.tool_invocations:
        _scan(inv.outputs)
    return channels


def _has_required_evidence(bundle: EvidenceBundle, request: ResearchToolRequest | None) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not bundle.success:
        missing.append("bundle_not_successful")
        return False, missing

    qtype = bundle.question_type
    if qtype in {"band_power", "rms", "psd_peak"} and not bundle.numeric_evidence:
        missing.append("numeric_evidence")
    if qtype == "channel_ranking" and not bundle.ranked_evidence:
        missing.append("ranked_evidence")
    if qtype == "threshold_set" and not bundle.set_evidence:
        missing.append("set_evidence")
    if qtype == "condition_comparison" and not bundle.condition_evidence:
        missing.append("condition_evidence")
    if request and request.include_vision_evidence and not bundle.vision_evidence:
        missing.append("vision_evidence")
    return len(missing) == 0, missing


def _check_answer_sections(answer: str) -> bool:
    lower = answer.lower()
    return all(section in lower for section in ANSWER_SECTIONS)


def _check_units(answer: str, bundle: EvidenceBundle) -> list[str]:
    issues: list[str] = []
    if not bundle.units:
        return issues
    lower = answer.lower()
    unit = bundle.units.lower()
    if "power" in unit or "µv" in unit:
        if "hz" in lower and "power" not in lower and bundle.question_type != "psd_peak":
            issues.append(f"frequency_unit_in_power_answer expected={bundle.units}")
    if bundle.question_type == "psd_peak" and bundle.units and "hz" not in lower:
        issues.append("missing_hz_for_frequency_answer")
    return issues


def _check_conditions(answer: str, request: ResearchToolRequest | None) -> list[str]:
    if request is None or request.question_type != "condition_comparison":
        return []
    mismatches: list[str] = []
    lower = answer.lower()
    for cond in (request.condition_a, request.condition_b):
        if cond and cond.replace("_", " ") not in lower and cond not in lower:
            mismatches.append(f"missing_condition:{cond}")
    return mismatches


def _detect_tool_loops(invocations: list[dict[str, Any]]) -> bool:
    seen: set[tuple[str, str]] = set()
    for inv in invocations:
        key = (inv.get("name", ""), json.dumps(inv.get("inputs", {}), sort_keys=True))
        if key in seen:
            return True
        seen.add(key)
    return False


def _detect_conflicting_numerics(bundle: EvidenceBundle) -> list[str]:
    conflicts: list[str] = []
    values = bundle.numeric_evidence.get("values")
    if isinstance(values, dict) and len(values) != len(set(values.values())):
        # duplicate values are fine; look for same channel different values in outputs
        pass
    for inv in bundle.tool_invocations:
        if not inv.success and inv.error:
            conflicts.append(f"tool_error:{inv.name}")
    return conflicts


def run_deterministic_checks(
    draft_answer: str | None,
    bundle: EvidenceBundle,
    *,
    request: ResearchToolRequest | None = None,
    tool_invocations: list[dict[str, Any]] | None = None,
) -> DeterministicCheckResult:
    """Run cheap verification checks before optional model verifier."""
    failure_codes: list[str] = []
    invocations = tool_invocations or [inv.to_dict() for inv in bundle.tool_invocations]

    if draft_answer is None:
        return DeterministicCheckResult(passed=False, failure_codes=["no_draft_answer"])

    if not _check_answer_sections(draft_answer):
        failure_codes.append("missing_answer_sections")

    if not bundle.success:
        failure_codes.append("tool_execution_failed")

    if any(not inv.get("success", True) for inv in invocations):
        failure_codes.append("tool_execution_failed")

    if _detect_tool_loops(invocations):
        failure_codes.append("tool_loop")

    has_required, missing = _has_required_evidence(bundle, request)
    if not has_required:
        failure_codes.append("missing_evidence")

    grounding = check_grounding(draft_answer, bundle)
    if not grounding.passed:
        failure_codes.append("unsupported_numeric")
    unsupported_candidates = [str(v) for v in grounding.unsupported_claims]

    evidence_channels = _collect_evidence_channels(bundle)
    if bundle.metadata and bundle.metadata.get("channels"):
        evidence_channels.update(str(c).upper() for c in bundle.metadata["channels"])

    cited_channels: set[str] = set()
    for section_name in ("answer:", "evidence:"):
        start = draft_answer.lower().find(section_name)
        if start < 0:
            continue
        next_starts = [
            draft_answer.lower().find(s, start + len(section_name))
            for s in ("evidence:", "tools used:", "uncertainty:")
            if s != section_name
        ]
        next_starts = [i for i in next_starts if i >= 0]
        end = min(next_starts) if next_starts else len(draft_answer)
        section_text = draft_answer[start:end]
        for match in CHANNEL_RE.finditer(section_text):
            ch = match.group(1)
            if ch in NON_CHANNEL_TOKENS:
                continue
            cited_channels.add(ch)

    missing_channels = sorted(
        ch for ch in cited_channels if ch not in evidence_channels and len(ch) <= 4
    )
    if missing_channels and bundle.question_type in {"band_power", "rms", "channel_ranking"}:
        failure_codes.append("channel_not_in_evidence")

    unit_issues = _check_units(draft_answer, bundle)
    if unit_issues:
        failure_codes.append("unit_mismatch")

    condition_mismatches = _check_conditions(draft_answer, request)
    if condition_mismatches:
        failure_codes.append("condition_mismatch")

    evidence_conflicts = _detect_conflicting_numerics(bundle)
    if evidence_conflicts:
        failure_codes.append("evidence_conflict")

    ambiguous = (
        len(grounding.numeric_claims) > 3
        and bundle.question_type == "channel_ranking"
        and bool(bundle.warnings)
    )

    passed = len(failure_codes) == 0
    return DeterministicCheckResult(
        passed=passed,
        failure_codes=sorted(set(failure_codes)),
        grounding=grounding,
        missing_channels=missing_channels,
        unit_issues=unit_issues,
        condition_mismatches=condition_mismatches,
        missing_evidence=missing,
        evidence_conflicts=evidence_conflicts,
        unsupported_candidates=unsupported_candidates,
        ambiguous=ambiguous,
    )


def deterministic_to_verification(det: DeterministicCheckResult) -> VerificationResult:
    """Convert deterministic-only result to VerificationResult."""
    if det.passed:
        return VerificationResult(
            passed=True,
            confidence_score=1.0,
            recommendation="ACCEPT",
            deterministic=det,
            model_verified=False,
        )

    recommendation: Recommendation = "REWRITE"
    if "tool_execution_failed" in det.failure_codes or "missing_evidence" in det.failure_codes:
        recommendation = "RETRY_TOOL" if "tool_execution_failed" in det.failure_codes else "INSUFFICIENT_EVIDENCE"
    elif "condition_mismatch" in det.failure_codes:
        recommendation = "REPLAN"
    elif "unsupported_numeric" in det.failure_codes:
        recommendation = "REWRITE"
    elif "unit_mismatch" in det.failure_codes:
        recommendation = "REWRITE"

    return VerificationResult(
        passed=False,
        confidence_score=0.3,
        failure_codes=det.failure_codes,
        unsupported_claims=det.unsupported_candidates,
        evidence_conflicts=det.evidence_conflicts,
        missing_evidence=det.missing_evidence,
        unit_issues=det.unit_issues,
        condition_mismatch=det.condition_mismatches,
        recommendation=recommendation,
        deterministic=det,
        model_verified=False,
    )


def merge_verification(
    deterministic: DeterministicCheckResult,
    model_json: dict[str, Any],
) -> VerificationResult:
    """Merge model verifier JSON with deterministic results (deterministic failures win)."""
    rec = model_json.get("recommendation", "REWRITE")
    if rec not in {"ACCEPT", "RETRY_TOOL", "REPLAN", "REWRITE", "INSUFFICIENT_EVIDENCE"}:
        rec = "REWRITE"

    model_passed = bool(model_json.get("passed", False))
    passed = deterministic.passed and model_passed

    failure_codes = sorted(set(deterministic.failure_codes + list(model_json.get("failure_codes", []))))
    if not deterministic.passed:
        passed = False

    return VerificationResult(
        passed=passed,
        confidence_score=float(model_json.get("confidence_score", 0.5)),
        failure_codes=failure_codes,
        unsupported_claims=[str(x) for x in model_json.get("unsupported_claims", [])]
        or deterministic.unsupported_candidates,
        evidence_conflicts=[str(x) for x in model_json.get("evidence_conflicts", [])]
        or deterministic.evidence_conflicts,
        missing_evidence=[str(x) for x in model_json.get("missing_evidence", [])]
        or deterministic.missing_evidence,
        unit_issues=[str(x) for x in model_json.get("unit_issues", [])] or deterministic.unit_issues,
        condition_mismatch=[str(x) for x in model_json.get("condition_mismatch", [])]
        or deterministic.condition_mismatches,
        recommendation=rec if not passed else "ACCEPT",
        deterministic=deterministic,
        model_verified=True,
        raw_model_json=model_json,
    )


def parse_verification_json(text: str) -> dict[str, Any]:
    return extract_json_object(text)


def build_verifier_user_prompt(
    question: str,
    intent: dict[str, Any] | None,
    bundle: EvidenceBundle,
    draft_answer: str,
    deterministic: DeterministicCheckResult,
) -> str:
    payload = {
        "question": question,
        "intent": intent,
        "evidence_bundle": bundle.to_dict(),
        "draft_answer": draft_answer,
        "deterministic_checks": deterministic.to_dict(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)
