from __future__ import annotations

import pytest

from deepresearch.domain.exceptions import (
    DeepResearchError,
    FetchError,
    InvalidReportError,
    LLMError,
    LLMRefusalError,
    MaxIterationsExceededError,
    SearchProviderError,
)


def test_subclasses_are_caught_by_base_exception() -> None:
    cases = [
        SearchProviderError("x"),
        FetchError("x"),
        LLMRefusalError(reason="refused"),
        MaxIterationsExceededError(iterations=3),
        InvalidReportError("x"),
    ]
    for exc in cases:
        with pytest.raises(DeepResearchError):
            raise exc


def test_llm_refusal_exposes_reason_attribute() -> None:
    exc = LLMRefusalError(reason="policy violation")
    assert exc.reason == "policy violation"
    assert "policy violation" in str(exc)


def test_max_iterations_exposes_iterations_attribute() -> None:
    exc = MaxIterationsExceededError(iterations=4)
    assert exc.iterations == 4
    assert "4" in str(exc)


def test_llm_refusal_is_caught_as_llm_error() -> None:
    with pytest.raises(LLMError):
        raise LLMRefusalError(reason="x")
