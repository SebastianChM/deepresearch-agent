from __future__ import annotations


class DeepResearchError(Exception):
    pass


class SearchProviderError(DeepResearchError):
    pass


class FetchError(DeepResearchError):
    pass


class LLMError(DeepResearchError):
    pass


class LLMRefusalError(LLMError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"LLM refused to produce output: {reason}")
        self.reason = reason


class MaxIterationsExceededError(DeepResearchError):
    def __init__(self, iterations: int) -> None:
        super().__init__(f"Agent exceeded the maximum allowed iterations ({iterations})")
        self.iterations = iterations


class InvalidReportError(DeepResearchError):
    pass
