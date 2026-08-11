"""Custom exception hierarchy for the LLM abstraction module."""

from __future__ import annotations


class LLMError(Exception):
    """Base exception class for all LLM errors."""

    pass


class LLMConfigurationError(LLMError):
    """Raised when LLM provider configuration or API keys are invalid or missing."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds the configured timeout."""

    pass


class LLMValidationError(LLMError):
    """Raised when an LLM response fails JSON parsing or Pydantic validation after retries."""

    pass


class LLMProviderError(LLMError):
    """Raised when an underlying LLM provider API returns an HTTP or API error."""

    pass


class LLMGuardrailError(LLMError):
    """Raised when a prompt or response violates safety, cost, or prompt-injection guardrails."""

    pass
