"""OpenAI-compatible LLM Provider implementation for Qwen and open-source models."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider
from app.llm.exceptions import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMTimeoutError,
    LLMValidationError,
)
from app.llm.schemas import LLMResponse, LLMUsage

T = TypeVar("T", bound=BaseModel)
logger = get_logger("app.llm.providers.qwen")


class QwenProvider(BaseLLMProvider):
    """LLM Provider for Qwen models using an OpenAI-compatible REST endpoint."""

    def __init__(
        self,
        model_name: str = "qwen2.5-72b-instruct",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.7,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
        )
        if not self.base_url:
            raise LLMConfigurationError("LLM base_url must be provided for QwenProvider.")

    @property
    def provider_name(self) -> str:
        return "qwen"

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> LLMResponse[str]:
        """Generate text completion from Qwen model."""
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        start_time = time.perf_counter()
        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                log_msg = (
                    f"Attempt {attempt}/{self.max_retries} posting to {endpoint} "
                    f"with model {self.model_name}"
                )
                logger.debug(self._sanitize_log(log_msg))

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        endpoint,
                        json=payload,
                        headers=self._get_headers(),
                    )

                if response.status_code in (401, 403):
                    raise LLMConfigurationError(
                        f"Authentication failed for provider {self.provider_name}: {response.text}"
                    )
                if response.status_code != 200:
                    raise LLMProviderError(
                        f"Provider {self.provider_name} returned status {response.status_code}: "
                        f"{response.text}"
                    )

                res_json = response.json()
                content = res_json["choices"][0]["message"]["content"]
                usage_data = res_json.get("usage", {})

                usage = LLMUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
                latency = time.perf_counter() - start_time

                return LLMResponse[str](
                    data=content,
                    raw_response=content,
                    usage=usage,
                    latency_seconds=latency,
                    model_name=self.model_name,
                    provider=self.provider_name,
                )

            except httpx.TimeoutException:
                last_exception = LLMTimeoutError(
                    f"Request to {self.provider_name} timed out after {self.timeout}s"
                )
            except (httpx.RequestError, LLMProviderError) as e:
                last_exception = e
            except LLMConfigurationError:
                raise

            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        if isinstance(last_exception, LLMTimeoutError):
            raise last_exception
        raise LLMProviderError(
            f"Failed to generate response after {self.max_retries} attempts: {last_exception}"
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> LLMResponse[T]:
        """Generate structured response matching Pydantic response_model schema."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        format_instruction = (
            "\n\nYou MUST respond ONLY with a valid JSON object matching the following schema:\n"
            f"```json\n{schema_json}\n```\nDo not include commentary outside the JSON block."
        )

        full_prompt = f"{prompt}{format_instruction}"
        start_time = time.perf_counter()
        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = await self.generate(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    **kwargs,
                )

                clean_json_str = raw_response.data.strip()
                if "```json" in clean_json_str:
                    match = re.search(r"```json\s*(.*?)\s*```", clean_json_str, re.DOTALL)
                    if match:
                        clean_json_str = match.group(1)
                elif "```" in clean_json_str:
                    match = re.search(r"```\s*(.*?)\s*```", clean_json_str, re.DOTALL)
                    if match:
                        clean_json_str = match.group(1)

                validated_data = response_model.model_validate_json(clean_json_str)
                latency = time.perf_counter() - start_time

                return LLMResponse[T](
                    data=validated_data,
                    raw_response=raw_response.data,
                    usage=raw_response.usage,
                    latency_seconds=latency,
                    model_name=self.model_name,
                    provider=self.provider_name,
                )

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    f"Structured validation failed attempt {attempt}/{self.max_retries} "
                    f"for {response_model.__name__}: {e}"
                )
                last_exception = LLMValidationError(
                    f"Failed to validate response against {response_model.__name__}: {e}"
                )
            except LLMError as e:
                last_exception = e

            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        if isinstance(last_exception, LLMValidationError):
            raise last_exception
        raise LLMValidationError(
            f"Failed to produce valid output for {response_model.__name__} "
            f"after {self.max_retries} attempts: {last_exception}"
        )
