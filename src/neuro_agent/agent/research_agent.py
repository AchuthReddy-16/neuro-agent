"""Primary tool-using neuroscience research agent."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from neuro_agent.agent.intent import IntentParseResult, parse_and_validate_intent
from neuro_agent.agent.answer_format import format_grounded_answer
from neuro_agent.agent.policies import is_format_only_failure, should_trigger_verifier
from neuro_agent.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    RECOVERY_ANSWER_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_intent_user_prompt,
)
from neuro_agent.agent.recovery import RecoveryResult, execute_recovery, plan_recovery
from neuro_agent.agent.traces import AgentTrace, RecoveryTrace, check_grounding, classify_failure
from neuro_agent.agent.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    build_verifier_user_prompt,
    deterministic_to_verification,
    merge_verification,
    parse_verification_json,
    run_deterministic_checks,
)
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.model_loader import load_model_and_tokenizer
from neuro_agent.tools.evidence import EvidenceBundle, new_request_id
from neuro_agent.tools.router import route_research_request

MAX_TOOL_CALLS = 6


@dataclass
class ResearchAgentConfig:
    """Runtime configuration for the primary research agent."""

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    adapter_path: str = "checkpoints/sft_corrected_v2/final"
    dtype: str = "bfloat16"
    seed: int = 42
    intent_max_new_tokens: int = 256
    answer_max_new_tokens: int = 512
    verifier_max_new_tokens: int = 256
    device: str = "cuda:0"
    enable_verification: bool = True
    max_recovery_cycles: int = 1


class PrimaryResearchAgent:
    """NL question → intent → router → grounded answer."""

    def __init__(
        self,
        config: ResearchAgentConfig | None = None,
        *,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config or ResearchAgentConfig()
        self._model = model
        self._tokenizer = tokenizer
        self._loaded = model is not None and tokenizer is not None

    def load(self) -> None:
        if self._loaded:
            return
        inference_cfg = InferenceConfig(
            model_name=self.config.model_name,
            dtype=self.config.dtype,
            seed=self.config.seed,
            adapter_path=self.config.adapter_path,
            do_sample=False,
            max_new_tokens=self.config.intent_max_new_tokens,
        )
        self._model, self._tokenizer, self._load_info = load_model_and_tokenizer(
            inference_cfg,
            device=self.config.device,
        )
        self._loaded = True

    def unload(self) -> None:
        """Release GPU-resident text weights so a VLM can load safely."""
        self._model = None
        self._tokenizer = None
        self._load_info = None  # type: ignore[assignment]
        self._loaded = False
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def is_loaded(self) -> bool:
        return bool(self._loaded and self._model is not None)

    def _chat_prompt(self, system: str, user: str) -> str:
        assert self._tokenizer is not None
        messages = [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float, float]:
        assert self._model is not None and self._tokenizer is not None
        device = self._model.device
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]

        if device.type == "cuda":
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(dev_idx)

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        new_ids = output_ids[0, prompt_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        peak_vram = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )
        return text, elapsed_ms, peak_vram

    def parse_intent(self, question: str) -> tuple[IntentParseResult, float, float]:
        """Run model intent selection and validation."""
        self.load()
        prompt = self._chat_prompt(INTENT_SYSTEM_PROMPT, build_intent_user_prompt(question))
        raw, latency_ms, peak_vram = self._generate(prompt, self.config.intent_max_new_tokens)
        result = parse_and_validate_intent(raw)
        if not result.success:
            result.warnings.append(f"raw_model_output={raw[:500]}")
        return result, latency_ms, peak_vram

    def generate_answer(
        self,
        question: str,
        bundle: EvidenceBundle,
        *,
        recovery: bool = False,
    ) -> tuple[str, float, float]:
        """Generate grounded final answer from evidence bundle."""
        self.load()
        system = RECOVERY_ANSWER_SYSTEM_PROMPT if recovery else ANSWER_SYSTEM_PROMPT
        prompt = self._chat_prompt(
            system,
            build_answer_user_prompt(question, bundle.to_dict()),
        )
        text, latency_ms, peak_vram = self._generate(prompt, self.config.answer_max_new_tokens)
        return text, latency_ms, peak_vram

    def verify_answer(
        self,
        question: str,
        bundle: EvidenceBundle,
        draft_answer: str,
        intent_result: IntentParseResult,
        *,
        run_model: bool,
    ) -> tuple[Any, float, float]:
        """Run deterministic checks and optional verifier model."""
        from neuro_agent.agent.verifier import VerificationResult

        det = run_deterministic_checks(
            draft_answer,
            bundle,
            request=intent_result.request,
        )
        if not run_model:
            return deterministic_to_verification(det), 0.0, 0.0

        self.load()
        prompt = self._chat_prompt(
            VERIFIER_SYSTEM_PROMPT,
            build_verifier_user_prompt(
                question,
                intent_result.raw_json,
                bundle,
                draft_answer,
                det,
            ),
        )
        raw, latency_ms, peak_vram = self._generate(prompt, self.config.verifier_max_new_tokens)
        try:
            model_json = parse_verification_json(raw)
            result = merge_verification(det, model_json)
        except Exception:
            result = deterministic_to_verification(det)
            result.failure_codes.append("verifier_parse_error")
        return result, latency_ms, peak_vram

    def ask(
        self,
        question: str,
        *,
        request_id: str | None = None,
        draft_corruption: Any | None = None,
    ) -> AgentTrace:
        """Full pipeline: intent → route → grounded answer."""
        t_start = time.perf_counter()
        req_id = request_id or new_request_id()
        errors: list[str] = []
        warnings: list[str] = []
        peak_vram = 0.0

        intent_result, intent_ms, intent_vram = self.parse_intent(question)
        peak_vram = max(peak_vram, intent_vram)

        if not intent_result.success or intent_result.request is None:
            trace = AgentTrace(
                request_id=req_id,
                original_question=question,
                parsed_intent=intent_result.raw_json,
                intent_valid=False,
                routing_result=None,
                tool_invocations=[],
                evidence_bundle=None,
                final_answer=None,
                runtime_ms=(time.perf_counter() - t_start) * 1000.0,
                intent_latency_ms=intent_ms,
                errors=[intent_result.error or "intent validation failed"],
                warnings=intent_result.warnings,
            )
            trace.failure_category = classify_failure(trace)
            return trace

        parsed_intent = intent_result.request.to_dict()
        bundle = route_research_request(intent_result.request, request_id=req_id)

        tool_count = len(bundle.tool_invocations)
        if tool_count > MAX_TOOL_CALLS:
            errors.append(f"tool call limit exceeded: {tool_count} > {MAX_TOOL_CALLS}")

        if not bundle.success:
            errors.append(bundle.error or "routing failed")

        final_answer: str | None = None
        answer_ms = 0.0
        grounding = None

        model_calls = 1  # intent
        draft_answer: str | None = None
        recovery_trace: RecoveryTrace | None = None
        first_pass_verification = None
        final_verification = None
        verification_triggered = False
        trigger_reason: list[str] = []
        verifier_ms = 0.0
        recovery_ms = 0.0
        path_mode = "NORMAL"

        if bundle.success and not errors:
            try:
                draft_answer, answer_ms, answer_vram = self.generate_answer(question, bundle)
                model_calls += 1
                peak_vram = max(peak_vram, answer_vram)
                if draft_corruption and draft_answer:
                    draft_answer = draft_corruption(draft_answer)
                final_answer = draft_answer
                grounding = check_grounding(final_answer, bundle)

                if self.config.enable_verification and draft_answer:
                    det = run_deterministic_checks(
                        draft_answer,
                        bundle,
                        request=intent_result.request,
                    )
                    trigger, trigger_reason = should_trigger_verifier(
                        det,
                        bundle,
                        intent_result,
                        draft_answer,
                        tool_count=tool_count,
                    )
                    verification_triggered = trigger
                    run_model = trigger

                    if not trigger and det.passed:
                        first_pass_verification = deterministic_to_verification(det).to_dict()
                        final_verification = first_pass_verification
                    elif is_format_only_failure(det):
                        verification_triggered = False
                        path_mode = "NORMAL"
                        t_rec = time.perf_counter()
                        ver_result = deterministic_to_verification(det)
                        first_pass_verification = ver_result.to_dict()
                        final_answer = format_grounded_answer(question, bundle)
                        model_calls += 0
                        post_det = run_deterministic_checks(
                            final_answer,
                            bundle,
                            request=intent_result.request,
                        )
                        from neuro_agent.agent.verifier import VerificationResult

                        post_ver = VerificationResult(
                            passed=post_det.passed,
                            confidence_score=0.95 if post_det.passed else 0.4,
                            failure_codes=post_det.failure_codes,
                            unsupported_claims=post_det.unsupported_candidates,
                            recommendation="ACCEPT" if post_det.passed else "REWRITE",
                            deterministic=post_det,
                        )
                        final_verification = post_ver.to_dict()
                        recovery_ms = (time.perf_counter() - t_rec) * 1000.0
                        grounding = check_grounding(final_answer, bundle)
                        recovery_trace = None
                    else:
                        verification_triggered = True
                        ver_result, ver_ms, ver_vram = self.verify_answer(
                            question,
                            bundle,
                            draft_answer,
                            intent_result,
                            run_model=run_model,
                        )
                        if run_model:
                            model_calls += 1
                        verifier_ms += ver_ms
                        peak_vram = max(peak_vram, ver_vram)
                        first_pass_verification = ver_result.to_dict()
                        final_verification = first_pass_verification

                        if not ver_result.passed and self.config.max_recovery_cycles > 0:
                            path_mode = "RECOVERY"
                            t_rec = time.perf_counter()
                            action = plan_recovery(
                                ver_result,
                                det,
                                tool_count=tool_count,
                                max_tool_calls=MAX_TOOL_CALLS,
                            )

                            def _gen_rec(q: str, b: EvidenceBundle) -> tuple[str, float, float]:
                                return self.generate_answer(q, b, recovery=True)

                            rec_result = execute_recovery(
                                action,
                                question=question,
                                draft_answer=draft_answer,
                                bundle=bundle,
                                intent_result=intent_result,
                                verification=ver_result,
                                deterministic=det,
                                generate_answer=_gen_rec,
                                parse_intent=self.parse_intent,
                                request_id=req_id,
                                tool_count=tool_count,
                                max_tool_calls=MAX_TOOL_CALLS,
                            )
                            recovery_ms = (time.perf_counter() - t_rec) * 1000.0
                            if rec_result.action in {"REWRITE", "REPLAN"}:
                                model_calls += 1
                            if rec_result.action == "REPLAN":
                                model_calls += 1

                            final_answer = rec_result.final_answer
                            if rec_result.bundle is not None:
                                bundle = rec_result.bundle
                                tool_count = len(bundle.tool_invocations)
                            if rec_result.intent_result and rec_result.intent_result.success:
                                intent_result = rec_result.intent_result
                                parsed_intent = (
                                    intent_result.request.to_dict()
                                    if intent_result.request
                                    else parsed_intent
                                )

                            post_ver = rec_result.verification
                            if post_ver is None and final_answer:
                                post_det = run_deterministic_checks(
                                    final_answer,
                                    bundle,
                                    request=intent_result.request,
                                )
                                from neuro_agent.agent.verifier import VerificationResult

                                post_ver = VerificationResult(
                                    passed=post_det.passed,
                                    confidence_score=0.5,
                                    failure_codes=post_det.failure_codes,
                                    unsupported_claims=post_det.unsupported_candidates,
                                    recommendation=(
                                        "ACCEPT" if post_det.passed else "INSUFFICIENT_EVIDENCE"
                                    ),
                                    deterministic=post_det,
                                )
                            if post_ver is not None:
                                final_verification = post_ver.to_dict()

                            grounding = (
                                check_grounding(final_answer or "", bundle)
                                if final_answer
                                else grounding
                            )
                            recovery_trace = RecoveryTrace(
                                action=rec_result.action,
                                success=rec_result.success,
                                pre_verification=first_pass_verification,
                                post_verification=final_verification,
                                notes=rec_result.notes,
                                latency_ms=recovery_ms,
                            )
                        elif not ver_result.passed:
                            grounding = (
                                check_grounding(final_answer or "", bundle)
                                if final_answer
                                else grounding
                            )

                if grounding and not grounding.passed:
                    warnings.append(
                        f"unsupported numeric claims: {grounding.unsupported_claims}"
                    )
            except Exception as exc:
                errors.append(f"answer generation failed: {exc}")

        warnings.extend(bundle.warnings)
        warnings.extend(intent_result.warnings)

        trace = AgentTrace(
            request_id=req_id,
            original_question=question,
            parsed_intent=parsed_intent,
            intent_valid=True,
            routing_result={"success": bundle.success, "error": bundle.error},
            tool_invocations=[inv.to_dict() for inv in bundle.tool_invocations],
            evidence_bundle=bundle.to_dict(),
            final_answer=final_answer,
            draft_answer=draft_answer,
            runtime_ms=(time.perf_counter() - t_start) * 1000.0,
            intent_latency_ms=intent_ms,
            answer_latency_ms=answer_ms,
            verifier_latency_ms=verifier_ms,
            recovery_latency_ms=recovery_ms,
            peak_vram_mb=peak_vram,
            errors=errors,
            warnings=warnings,
            grounding=grounding,
            verification_triggered=verification_triggered,
            trigger_reason=trigger_reason,
            first_pass_verification=first_pass_verification,
            recovery=recovery_trace,
            final_verification=final_verification,
            path_mode=path_mode,
            model_calls=model_calls,
        )
        trace.failure_category = classify_failure(trace)
        return trace

    @property
    def model(self) -> PreTrainedModel | None:
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        return self._tokenizer
