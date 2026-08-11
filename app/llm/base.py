"""Abstract base class contract for LLM providers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from app.llm.schemas import LLMResponse

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    """Abstract base class defining the standard interface for all LLM providers."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "",
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'qwen', 'ollama', 'mock')."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> LLMResponse[str]:
        """Generate unstructured text from the LLM."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> LLMResponse[T]:
        """Generate structured data conforming to a Pydantic response model."""
        pass

    def _sanitize_log(self, text: str) -> str:
        """Sanitize sensitive credentials, API keys, and authorization headers from logs."""
        if not text:
            return text
        sanitized = text
        if self.api_key and len(self.api_key) > 4:
            sanitized = sanitized.replace(self.api_key, f"{self.api_key[:3]}...***")
        sanitized = re.sub(
            r"(Bearer\s+)[A-Za-z0-9_\-\.]+",
            r"\1***MASKED***",
            sanitized,
            flags=re.IGNORECASE,
        )
        return sanitized
