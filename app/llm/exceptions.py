"""Custom exception hierarchy for the LLM abstraction module."""


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
