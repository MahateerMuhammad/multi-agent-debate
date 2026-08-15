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
        model_name: str = "qwen/qwen-2.5-72b-instruct:free",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_guardrails: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_guardrails=enable_guardrails,
        )
        if not self.base_url:
            raise LLMConfigurationError("LLM base_url must be provided for QwenProvider.")
            
        import httpx
        # Strict timeout on connection pool
        strict_timeout = httpx.Timeout(connect=5.0, read=timeout, write=5.0, pool=5.0)
        self.client = httpx.AsyncClient(timeout=strict_timeout)

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
        """Generate text completion from Qwen model with guardrail validation."""
        self.validate_prompt(prompt)

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Hard clamp max_tokens to prevent DoW regardless of caller
        requested_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        safe_max_tokens = min(requested_tokens, self.max_tokens)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": safe_max_tokens,
        }
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        start_time = time.perf_counter()
        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                log_msg = (
                    f"Attempt {attempt}/{self.max_retries} posting to {endpoint} "
                    f"with model {self.model_name}"
                )
                logger.debug(self._sanitize_log(log_msg))

                response = await self.client.post(
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
        self.validate_prompt(prompt)

        # Generate XML format instruction dynamically from the Pydantic schema
        schema_props = response_model.model_json_schema().get("properties", {})
        xml_template = "\n".join(f"<{key}>...</{key}>" for key in schema_props.keys())
        
        format_instruction = (
            "\n\nYou MUST format your final answers using XML tags corresponding to the required fields.\n"
            "You can think step-by-step or write your analysis outside the tags, but the final extracted values MUST be wrapped exactly like this:\n"
            f"```xml\n{xml_template}\n```\n"
            "For arrays/lists, separate items with a semicolon (;). For nested objects, just output a simple string and we will parse it."
        )

        full_prompt = f"{prompt}{format_instruction}"
        start_time = time.perf_counter()
        last_exception: Exception | None = None
        
        cumulative_usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        import re

        for attempt in range(1, self.max_retries + 1):
            try:
                # Ask the model to generate plain text (no JSON schema constraints)
                raw_response = await self.generate(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    **kwargs,
                )
                
                # Accumulate tokens across all attempts
                if raw_response.usage.prompt_tokens is not None:
                    cumulative_usage.prompt_tokens = (cumulative_usage.prompt_tokens or 0) + raw_response.usage.prompt_tokens
                if raw_response.usage.completion_tokens is not None:
                    cumulative_usage.completion_tokens = (cumulative_usage.completion_tokens or 0) + raw_response.usage.completion_tokens
                cumulative_usage.total_tokens = (cumulative_usage.prompt_tokens or 0) + (cumulative_usage.completion_tokens or 0)

                # Extract XML tags using Regex
                text = raw_response.data
                extracted_dict = {}
                for key, prop_info in schema_props.items():
                    match = re.search(f"<{key}>(.*?)</{key}>", text, re.DOTALL | re.IGNORECASE)
                    if match:
                        val = match.group(1).strip()
                        # Simple type casting based on Pydantic schema types
                        prop_str = str(prop_info).lower()
                        if 'array' in prop_str:
                            extracted_dict[key] = [v.strip() for v in val.split(';') if v.strip()]
                        elif 'number' in prop_str or 'integer' in prop_str or 'float' in prop_str:
                            try:
                                import re as inner_re
                                float_match = inner_re.search(r"[-+]?\d*\.\d+|\d+", val)
                                if float_match:
                                    extracted_dict[key] = float(float_match.group())
                                else:
                                    extracted_dict[key] = 0.0
                            except Exception:
                                extracted_dict[key] = 0.0
                        elif 'boolean' in prop_str:
                            extracted_dict[key] = val.lower() in ('true', '1', 'yes')
                        else:
                            extracted_dict[key] = val

                # The shock-absorber validator in schemas.py will handle any missing keys!
                parsed_obj = response_model(**extracted_dict)
                
                return LLMResponse[T](
                    data=parsed_obj,
                    raw_response=text,
                    usage=cumulative_usage,
                    latency_seconds=time.perf_counter() - start_time,
                    model_name=self.model_name,
                    provider=self.provider_name,
                )
                
            except Exception as e:
                # Catch Pydantic validation errors or missing tag errors
                last_exception = LLMValidationError(
                    f"Failed to parse XML into {response_model.__name__}: {str(e)}"
                )
                last_exception.usage = cumulative_usage
                last_exception.latency_seconds = time.perf_counter() - start_time

            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

        if last_exception:
            raise last_exception
        raise LLMProviderError("Exhausted retries.")

    async def aclose(self) -> None:
        """Gracefully close the underlying HTTPX client connection pool."""
        await self.client.aclose()
