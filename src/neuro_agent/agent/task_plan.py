"""Conversation-aware task / component planner for production routing.

Determines which components to run (TEXT, TOOLS, VISION, VERIFY, VISUALIZE)
from the current question, relevant history, and available artifacts.

Conservative fail-safes: prefer explanation over tools when ambiguous;
never invent missing inputs; never inherit prior tool runs by default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

Component = Literal["TEXT", "TOOLS", "VISION", "VERIFY", "VISUALIZE", "NEEDS_INPUT"]
NeedKind = Literal["image", "dataset", "document", "clarification", "none"]


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str
    tools_used: list[str] = field(default_factory=list)
    route: str | None = None
    evidence_summary: str | None = None
    question_type: str | None = None


@dataclass
class ArtifactContext:
    has_sample: bool = False
    has_image: bool = False
    sample_id: str | None = None
    image_id: str | None = None


@dataclass
class TaskPlan:
    components: list[str]
    text_only: bool
    use_tools: bool
    use_vision: bool
    use_verify: bool
    use_visualize: bool
    needs_input: bool
    need_kind: NeedKind
    missing_input_message: str | None
    resolved_question: str
    is_follow_up: bool
    prior_context_for_answer: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": list(self.components),
            "text_only": self.text_only,
            "use_tools": self.use_tools,
            "use_vision": self.use_vision,
            "use_verify": self.use_verify,
            "use_visualize": self.use_visualize,
            "needs_input": self.needs_input,
            "need_kind": self.need_kind,
            "missing_input_message": self.missing_input_message,
            "resolved_question": self.resolved_question,
            "is_follow_up": self.is_follow_up,
            "reason": self.reason,
        }


# --- pattern helpers (behavioral guidance, used with conversation context) ---

_GREETING = re.compile(
    r"^\s*(hey|hi|hello|yo|sup|good\s+(morning|afternoon|evening)|thanks|thank you|ok|okay|cool)\s*[!.?]?\s*$",
    re.I,
)
_CAPABILITY = re.compile(
    r"\b(what (can|do) you (do|know)|who are you|help me|your capabilities|what are you)\b",
    re.I,
)
_CONCEPT = re.compile(
    r"\b(what is|what'?s|define|explain|meaning of|generally|in general|tell me about)\b",
    re.I,
)
_VISION = re.compile(
    r"\b(topomap|spectrogram|figure|plot|image|heatmap|visual(?:ly|s|ization)?|"
    r"look(?:ing)?\s+at|what\s+does\s+this\s+(?:figure|plot|image|topomap)|"
    r"interpret(?:ing)?\s+(?:the\s+)?(?:map|plot|figure|image)|"
    r"show(?:s|ing)?\s+(?:this\s+)?(?:figure|plot|image|topomap))\b",
    re.I,
)
_COMPUTE = re.compile(
    r"\b(which\s+channels?|rank|highest|lowest|band[\s-]?power|rms|psd\s*peak|"
    r"compare|threshold|discriminative|outlier|effect\s+size|compute|calculate|"
    r"for\s+this\s+sample|this\s+(?:sample|recording|eeg|dataset|csv))\b",
    re.I,
)
_EXPLICIT_DATA_REF = re.compile(
    r"\b(this\s+(?:sample|recording|eeg|dataset|csv|experiment|run)|"
    r"these\s+data|current\s+(?:sample|result|figure)|selected\s+figure|"
    r"uploaded|for\s+sample\s+S\d+)\b",
    re.I,
)
_DOCUMENT = re.compile(
    r"\b(pdf|document|paper|manuscript|docx?)\b",
    re.I,
)
_FOLLOW_UP = re.compile(
    r"^\s*(why\??|why\s+is\b|what about\b|and\b|how come\b|is that\b|"
    r"what does that mean|show me the difference|compare that\b|"
    r"the (second|third|first|previous|last) (one|result|channel)|"
    r"what about the previous|is that significant)\b",
    re.I,
)
_FOLLOW_UP_COMPUTE = re.compile(
    r"\b(compare (that|it|this) with|vs\.?|versus|difference between|"
    r"what about (channel\s+)?[A-Z]\d|also (compute|calculate|rank))\b",
    re.I,
)
_CHANNEL_TOKEN = re.compile(r"\b([A-Z]{1,3}\d{1,2}|Iz|IZ|Cz|CZ|Fz|FZ|Pz|PZ|Oz|OZ)\b")


def _last_user_assistant(
    history: list[ConversationTurn],
) -> tuple[ConversationTurn | None, ConversationTurn | None]:
    last_user = None
    last_asst = None
    for turn in reversed(history):
        if last_asst is None and turn.role == "assistant":
            last_asst = turn
        elif last_user is None and turn.role == "user":
            last_user = turn
        if last_user and last_asst:
            break
    return last_user, last_asst


def _prior_tools(history: list[ConversationTurn]) -> list[str]:
    tools: list[str] = []
    for turn in reversed(history):
        if turn.role == "assistant" and turn.tools_used:
            tools.extend(turn.tools_used)
            break
    return tools


def _looks_like_concept_only(question: str) -> bool:
    q = question.strip()
    if _GREETING.match(q) or _CAPABILITY.search(q):
        return True
    # Channel/metric lookups are computations, not definitions.
    channel_metric = bool(_CHANNEL_TOKEN.search(q) and _COMPUTE.search(q))
    # Definitions / general explanations — even if the topic mentions band power, etc.
    if re.match(r"(?i)^\s*what is\b", q) and not _EXPLICIT_DATA_REF.search(q):
        if channel_metric:
            return False
        return True
    if re.search(r"(?i)\bgenerally\b|\bin general\b|\bmeaning of\b", q) and not _EXPLICIT_DATA_REF.search(
        q
    ):
        return True
    if re.search(r"(?i)^explain\b", q) and not _EXPLICIT_DATA_REF.search(q) and not _VISION.search(q):
        return True
    if _CONCEPT.search(q) and not _EXPLICIT_DATA_REF.search(q) and not _COMPUTE.search(q):
        return True
    return False


def _needs_vision(question: str) -> bool:
    if _COMPUTE.search(question) and not _VISION.search(question):
        # ranking / band power without visual language
        return False
    return bool(_VISION.search(question))


def _needs_compute(question: str, *, is_follow_up_compute: bool) -> bool:
    if is_follow_up_compute:
        return True
    if _looks_like_concept_only(question):
        return False
    if _needs_vision(question) and not _COMPUTE.search(question):
        return False
    if _EXPLICIT_DATA_REF.search(question) and _COMPUTE.search(question):
        return True
    if _COMPUTE.search(question):
        # Conservative: computation language without explicit data ref still
        # needs tools IF a sample is available; otherwise NEEDS_INPUT.
        return True
    return False


def _resolve_follow_up(
    question: str,
    history: list[ConversationTurn],
) -> tuple[str, bool, bool, str | None]:
    """Return (resolved_question, is_follow_up, wants_new_compute, prior_context)."""
    last_user, last_asst = _last_user_assistant(history)
    if not last_user and not last_asst:
        return question, False, False, None

    is_fu = bool(_FOLLOW_UP.search(question)) or bool(_FOLLOW_UP_COMPUTE.search(question))
    # Short anaphoric questions after an analytical turn
    if not is_fu and len(question.split()) <= 8:
        if re.search(r"\b(that|it|this|those|previous|above)\b", question, re.I):
            if last_asst and (last_asst.tools_used or last_asst.evidence_summary):
                is_fu = True

    if not is_fu:
        return question, False, False, None

    prior_bits: list[str] = []
    if last_user:
        prior_bits.append(f"Previous question: {last_user.content}")
    if last_asst:
        prior_bits.append(f"Previous answer: {last_asst.content[:800]}")
        if last_asst.evidence_summary:
            prior_bits.append(f"Previous evidence: {last_asst.evidence_summary[:600]}")
    prior = "\n".join(prior_bits) if prior_bits else None

    wants_compute = bool(_FOLLOW_UP_COMPUTE.search(question))
    # "why is T8 highest?" — explanatory, use prior result, not new ranking
    if re.search(r"(?i)^\s*why\b", question) and not wants_compute:
        resolved = (
            f"{question.strip()} (in the context of the previous result: "
            f"{(last_asst.content if last_asst else '')[:400]})"
        )
        return resolved, True, False, prior

    # "compare that with C4" — resolve "that"
    if wants_compute and last_user:
        channels = _CHANNEL_TOKEN.findall(question)
        prior_channels = _CHANNEL_TOKEN.findall(last_user.content + " " + (last_asst.content if last_asst else ""))
        resolved = question
        if re.search(r"\b(that|it|this)\b", question, re.I) and prior_channels:
            # Prefer channel from prior user question (e.g. C3)
            ref = prior_channels[0]
            resolved = re.sub(
                r"\b(that|it|this)\b",
                f"{ref} beta-band power from the previous result",
                question,
                count=1,
                flags=re.I,
            )
            if channels:
                resolved += f" Compare with {channels[0]}."
        elif last_user:
            resolved = f"{question} (referring to: {last_user.content})"
        return resolved, True, True, prior

    # Generic follow-up explanation
    resolved = question
    if last_user:
        resolved = f"{question} (follow-up to: {last_user.content})"
    return resolved, True, False, prior


def plan_task(
    question: str,
    *,
    history: list[ConversationTurn] | None = None,
    artifacts: ArtifactContext | None = None,
) -> TaskPlan:
    """Plan which components the current turn requires."""
    q = (question or "").strip()
    history = history or []
    artifacts = artifacts or ArtifactContext()

    if not q:
        return TaskPlan(
            components=["TEXT", "NEEDS_INPUT"],
            text_only=True,
            use_tools=False,
            use_vision=False,
            use_verify=False,
            use_visualize=False,
            needs_input=True,
            need_kind="clarification",
            missing_input_message="Enter a research question before analyzing.",
            resolved_question=q,
            is_follow_up=False,
            prior_context_for_answer=None,
            reason="empty_question",
        )

    if _DOCUMENT.search(q):
        return TaskPlan(
            components=["TEXT", "NEEDS_INPUT"],
            text_only=True,
            use_tools=False,
            use_vision=False,
            use_verify=False,
            use_visualize=False,
            needs_input=True,
            need_kind="document",
            missing_input_message=(
                "PDF/document questions are not supported yet. "
                "Ask a text question, attach an EEG sample, or select a figure."
            ),
            resolved_question=q,
            is_follow_up=False,
            prior_context_for_answer=None,
            reason="unsupported_document",
        )

    resolved, is_follow_up, wants_fu_compute, prior = _resolve_follow_up(q, history)

    # Pure conversation / concept — never inherit prior tools
    if _GREETING.match(q) or _CAPABILITY.search(q):
        return TaskPlan(
            components=["TEXT"],
            text_only=True,
            use_tools=False,
            use_vision=False,
            use_verify=False,
            use_visualize=False,
            needs_input=False,
            need_kind="none",
            missing_input_message=None,
            resolved_question=q,
            is_follow_up=False,
            prior_context_for_answer=None,
            reason="conversational",
        )

    if _looks_like_concept_only(q) and not wants_fu_compute:
        return TaskPlan(
            components=["TEXT"],
            text_only=True,
            use_tools=False,
            use_vision=False,
            use_verify=False,
            use_visualize=False,
            needs_input=False,
            need_kind="none",
            missing_input_message=None,
            resolved_question=q,
            is_follow_up=is_follow_up,
            prior_context_for_answer=prior if is_follow_up else None,
            reason="concept_explanation",
        )

    # Explanatory follow-up after tools — TEXT with prior context, no new tools
    if is_follow_up and not wants_fu_compute:
        prior_tools = _prior_tools(history)
        if prior_tools or (prior and "Previous answer" in prior):
            # "why is T8 highest?" / "what does that mean?"
            if not _needs_vision(q) and not wants_fu_compute:
                return TaskPlan(
                    components=["TEXT"],
                    text_only=True,
                    use_tools=False,
                    use_vision=False,
                    use_verify=False,
                    use_visualize=False,
                    needs_input=False,
                    need_kind="none",
                    missing_input_message=None,
                    resolved_question=resolved,
                    is_follow_up=True,
                    prior_context_for_answer=prior,
                    reason="follow_up_explanation",
                )

    needs_vision = _needs_vision(q)
    needs_compute = _needs_compute(q, is_follow_up_compute=wants_fu_compute)

    if needs_vision:
        if not artifacts.has_image:
            return TaskPlan(
                components=["TEXT", "NEEDS_INPUT"],
                text_only=True,
                use_tools=False,
                use_vision=False,
                use_verify=False,
                use_visualize=False,
                needs_input=True,
                need_kind="image",
                missing_input_message=(
                    "This question needs an image. Attach or select a figure first."
                ),
                resolved_question=resolved,
                is_follow_up=is_follow_up,
                prior_context_for_answer=prior,
                reason="vision_missing_image",
            )
        return TaskPlan(
            components=["TEXT", "VISION"],
            text_only=False,
            use_tools=False,
            use_vision=True,
            use_verify=False,
            use_visualize=True,
            needs_input=False,
            need_kind="none",
            missing_input_message=None,
            resolved_question=resolved,
            is_follow_up=is_follow_up,
            prior_context_for_answer=prior,
            reason="vision_analysis",
        )

    if needs_compute:
        if not artifacts.has_sample:
            return TaskPlan(
                components=["TEXT", "NEEDS_INPUT"],
                text_only=True,
                use_tools=False,
                use_vision=False,
                use_verify=False,
                use_visualize=False,
                needs_input=True,
                need_kind="dataset",
                missing_input_message=(
                    "This question needs an EEG sample or dataset. "
                    "Upload a sample JSON (with sample_id) or select an experiment sample first."
                ),
                resolved_question=resolved,
                is_follow_up=is_follow_up,
                prior_context_for_answer=prior,
                reason="tools_missing_sample",
            )
        # VERIFY only for tool-backed research results
        return TaskPlan(
            components=["TEXT", "TOOLS", "VERIFY"],
            text_only=False,
            use_tools=True,
            use_vision=False,
            use_verify=True,
            use_visualize=False,
            needs_input=False,
            need_kind="none",
            missing_input_message=None,
            resolved_question=resolved,
            is_follow_up=is_follow_up,
            prior_context_for_answer=prior,
            reason="deterministic_tools",
        )

    # Ambiguous: prefer explanation
    return TaskPlan(
        components=["TEXT"],
        text_only=True,
        use_tools=False,
        use_vision=False,
        use_verify=False,
        use_visualize=False,
        needs_input=False,
        need_kind="none",
        missing_input_message=None,
        resolved_question=resolved,
        is_follow_up=is_follow_up,
        prior_context_for_answer=prior,
        reason="ambiguous_prefer_text",
    )


def history_from_payload(raw: list[dict[str, Any]] | None) -> list[ConversationTurn]:
    """Parse API conversation_history into ConversationTurn list."""
    out: list[ConversationTurn] = []
    if not raw:
        return out
    for item in raw[-12:]:  # bound context window
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").lower()
        content = str(item.get("content") or item.get("text") or "").strip()
        if not content:
            continue
        tools = item.get("tools_used") or item.get("toolsUsed") or []
        if isinstance(tools, str):
            tools = [tools]
        out.append(
            ConversationTurn(
                role="assistant" if role in {"assistant", "agent", "system"} else "user",
                content=content,
                tools_used=[str(t) for t in tools],
                route=item.get("route"),
                evidence_summary=item.get("evidence_summary") or item.get("evidenceSummary"),
                question_type=item.get("question_type") or item.get("questionType"),
            )
        )
    return out
