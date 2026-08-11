"""Mock LLM Provider implementation for offline testing and deterministic simulation."""

from __future__ import annotations

import json
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.base import BaseLLMProvider
from app.llm.exceptions import LLMTimeoutError, LLMValidationError
from app.llm.schemas import LLMResponse, LLMUsage

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(BaseLLMProvider):
    """Deterministic Mock LLM Provider for unit testing and offline development."""

    def __init__(
        self,
        model_name: str = "mock-qwen2.5",
        mock_response: str = "Mock generated text response.",
        mock_json_response: dict[str, Any] | None = None,
        simulate_timeout: bool = False,
        simulate_invalid_json: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.mock_response = mock_response
        self.mock_json_response = mock_json_response
        self.simulate_timeout = simulate_timeout
        self.simulate_invalid_json = simulate_invalid_json

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> LLMResponse[str]:
        if self.simulate_timeout:
            raise LLMTimeoutError(f"Mock request timed out after {self.timeout}s")

        start_time = time.perf_counter()
        usage = LLMUsage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(self.mock_response.split()),
            total_tokens=len(prompt.split()) + len(self.mock_response.split()),
        )
        latency = time.perf_counter() - start_time

        return LLMResponse[str](
            data=self.mock_response,
            raw_response=self.mock_response,
            usage=usage,
            latency_seconds=latency,
            model_name=self.model_name,
            provider=self.provider_name,
        )

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> LLMResponse[T]:
        if self.simulate_timeout:
            raise LLMTimeoutError(f"Mock structured request timed out after {self.timeout}s")

        if self.simulate_invalid_json:
            raise LLMValidationError(
                f"Mock generated malformed output for {response_model.__name__}"
            )

        start_time = time.perf_counter()

        if self.mock_json_response is not None:
            raw_str = json.dumps(self.mock_json_response)
        else:
            raw_str = self._generate_default_json(response_model)

        try:
            validated_data = response_model.model_validate_json(raw_str)
        except (ValidationError, json.JSONDecodeError) as e:
            raise LLMValidationError(
                f"Mock data failed schema validation for {response_model.__name__}: {e}"
            ) from e

        usage = LLMUsage(prompt_tokens=20, completion_tokens=30, total_tokens=50)
        latency = time.perf_counter() - start_time

        return LLMResponse[T](
            data=validated_data,
            raw_response=raw_str,
            usage=usage,
            latency_seconds=latency,
            model_name=self.model_name,
            provider=self.provider_name,
        )

    def _generate_default_json(self, response_model: type[BaseModel]) -> str:
        """Construct fallback mock JSON matching required Pydantic model fields."""
        schema = response_model.model_json_schema()
        properties = schema.get("properties", {})
        dummy_data: dict[str, Any] = {}

        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            if prop_type == "string":
                dummy_data[prop_name] = f"Mock {prop_name}"
            elif prop_type == "integer":
                dummy_data[prop_name] = 1
            elif prop_type == "number":
                dummy_data[prop_name] = 0.95
            elif prop_type == "boolean":
                dummy_data[prop_name] = True
            elif prop_type == "array":
                dummy_data[prop_name] = [f"Mock {prop_name} item"]
            else:
                dummy_data[prop_name] = "Mock value"

        return json.dumps(dummy_data)
