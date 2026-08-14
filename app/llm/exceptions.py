"""Custom exception hierarchy for the LLM abstraction module."""

from __future__ import annotations


from typing import Optional
from app.llm.schemas import LLMUsage

class LLMError(Exception):
    """Base exception class for all LLM errors."""

    def __init__(self, message: str, usage: Optional[LLMUsage] = None, latency_seconds: Optional[float] = None):
        super().__init__(message)
        self.usage = usage
        self.latency_seconds = latency_seconds


class LLMConfigurationError(LLMError):
    """Raised when LLM provider configuration or API keys are invalid or missing."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds the configured timeout."""

    pass


from typing import Optional
from app.llm.schemas import LLMUsage

class LLMValidationError(LLMError):
    """Raised when an LLM response fails JSON parsing or Pydantic validation after retries."""

    pass


class LLMProviderError(LLMError):
    """Raised when an underlying LLM provider API returns an HTTP or API error."""

    pass


class LLMGuardrailError(LLMError):
    """Raised when a prompt or response violates safety, cost, or prompt-injection guardrails."""

    pass
