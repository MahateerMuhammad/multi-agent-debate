"""LLM Provider Factory module."""

from __future__ import annotations

from app.core.config import Settings
from app.core.config import settings as default_settings
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import LLMConfigurationError
from app.llm.providers.mock import MockLLMProvider
from app.llm.providers.qwen_provider import QwenProvider


def get_llm_provider(
    config: Settings | None = None, provider_name: str | None = None
) -> BaseLLMProvider:
    """Factory function instantiating the configured LLM provider instance."""
    cfg = config or default_settings
    provider_type = (provider_name or cfg.LLM_PROVIDER).lower()


    if provider_type in ("openrouter", "qwen", "ollama", "vllm", "openai-compatible"):
        if not cfg.LLM_BASE_URL:
            raise LLMConfigurationError(
                f"LLM_BASE_URL must be configured for provider '{provider_type}'."
            )
        return QwenProvider(
            model_name=cfg.LLM_MODEL,
            base_url=cfg.LLM_BASE_URL,
            api_key=cfg.LLM_API_KEY,
            timeout=cfg.LLM_TIMEOUT,
            max_retries=cfg.LLM_MAX_RETRIES,
            temperature=cfg.LLM_TEMPERATURE,
            max_tokens=cfg.LLM_MAX_TOKENS,
            enable_guardrails=cfg.ENABLE_GUARDRAILS,
        )

    if provider_type == "mock":
        return MockLLMProvider(
            model_name=cfg.LLM_MODEL,
            timeout=cfg.LLM_TIMEOUT,
            max_retries=cfg.LLM_MAX_RETRIES,
            temperature=cfg.LLM_TEMPERATURE,
            max_tokens=cfg.LLM_MAX_TOKENS,
            enable_guardrails=cfg.ENABLE_GUARDRAILS,
        )

    raise LLMConfigurationError(
        f"Unsupported LLM provider: '{cfg.LLM_PROVIDER}'. "
        f"Supported providers are: 'openrouter', 'qwen', 'ollama', 'vllm', 'mock'."
    )
