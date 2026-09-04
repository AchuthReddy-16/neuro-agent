"""Unit tests for primary research agent intent parsing (Stage G.3A)."""

from __future__ import annotations

import json

import pytest

from neuro_agent.agent.intent import (
    IntentValidationError,
    intent_matches_expected,
    parse_and_validate_intent,
    validate_intent,
)


class TestIntentValidation:
    def test_band_power_valid(self) -> None:
        data = {
            "question_type": "band_power",
            "sample_id": "S001_R01_E000",
            "frequency_band": "beta",
            "channels": ["C3"],
        }
        req = validate_intent(data)
        assert req.question_type == "band_power"
        assert req.sample_id == "S001_R01_E000"

    def test_condition_comparison_valid(self) -> None:
        data = {
            "question_type": "condition_comparison",
            "subject_id": "S013",
            "condition_a": "left_fist",
            "condition_b": "right_fist",
            "metric": "alpha_mu_power",
        }
        req = validate_intent(data)
        assert req.subject_id == "S013"

    def test_invalid_question_type(self) -> None:
        with pytest.raises(IntentValidationError):
            validate_intent({"question_type": "unknown"})

    def test_missing_sample_locator(self) -> None:
        with pytest.raises(IntentValidationError):
            validate_intent({"question_type": "rms"})

    def test_parse_json_from_fenced_block(self) -> None:
        raw = 'Here is the intent:\n```json\n{"question_type": "rms", "sample_id": "S001_R01_E000", "channels": ["C3"]}\n```'
        result = parse_and_validate_intent(raw)
        assert result.success
        assert result.request is not None
        assert result.request.question_type == "rms"

    def test_intent_matches_expected(self) -> None:
        data = {
            "question_type": "band_power",
            "sample_id": "S001_R01_E000",
            "frequency_band": "beta",
            "channels": ["C3"],
        }
        result = parse_and_validate_intent(json.dumps(data))
        expected = {
            "question_type": "band_power",
            "sample_id": "S001_R01_E000",
            "frequency_band": "beta",
            "channels": ["C3"],
        }
        assert intent_matches_expected(result, expected)

    def test_normalize_truncated_sample_id(self) -> None:
        data = {
            "question_type": "band_power",
            "sample_id": "S001_R01_E0",
            "frequency_band": "delta",
            "channels": ["FZ"],
        }
        req = validate_intent(data)
        assert req.sample_id == "S001_R01_E000"
