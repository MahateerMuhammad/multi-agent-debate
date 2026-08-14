"""Mock LLM Provider implementation for offline testing and deterministic simulation."""

from __future__ import annotations

import json
import re
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
        self.validate_prompt(prompt)

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
        self.validate_prompt(prompt)

        if self.simulate_timeout:
            raise LLMTimeoutError(f"Mock structured request timed out after {self.timeout}s")

        if self.simulate_invalid_json:
            raise LLMValidationError(
                f"Mock generated malformed output for {response_model.__name__}",
                usage=LLMUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
                latency_seconds=0.1
            )

        start_time = time.perf_counter()

        if self.mock_json_response is not None:
            raw_str = json.dumps(self.mock_json_response)
        else:
            raw_str = self._generate_default_json(response_model, prompt=prompt)

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

    def _generate_default_json(self, response_model: type[BaseModel], prompt: str = "") -> str:
        """Construct query-sensitive, role-aware, and round-dependent mock JSON."""
        schema = response_model.model_json_schema()
        properties = schema.get("properties", {})
        defs = schema.get("$defs", {})
        dummy_data: dict[str, Any] = {}

        prompt_hash = abs(hash(prompt))
        doc_ids = re.findall(r"\b[A-Z0-9]{3,}(?:-[A-Z0-9]+)+\b", prompt)
        doc_ids_unique = list(dict.fromkeys(doc_ids))

        # Extract active debate round from prompt if present
        round_match = re.search(r"Round:?\s*(\d+)", prompt, re.IGNORECASE)
        round_num = int(round_match.group(1)) if round_match else 1

        # Query topic classification
        prompt_lower = prompt.lower()
        if "open-source" in prompt_lower or "cset" in prompt_lower:
            topic_key = "opensource"
        elif "traffic" in prompt_lower or "judge" in prompt_lower or "court" in prompt_lower:
            topic_key = "legal"
        else:
            topic_key = "ubi"

        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type")

            if prop_name in ("claim", "target_claim"):
                if topic_key == "opensource":
                    dummy_data[prop_name] = (
                        f"Round {round_num}: Open-source AI models democratize security "
                        "research and mitigate single-vendor systemic dependency risk."
                    )
                elif topic_key == "legal":
                    dummy_data[prop_name] = (
                        f"Round {round_num}: Automated judicial algorithms risk violating "
                        "procedural due process standards in municipal traffic court."
                    )
                else:
                    dummy_data[prop_name] = (
                        f"Round {round_num}: Universal basic income stabilizes aggregate "
                        "consumer demand during technology-driven labor market transitions."
                    )

            elif prop_name == "winner":
                dummy_data[prop_name] = (
                    "Position A" if (prompt_hash + round_num) % 2 == 0 else "Position B"
                )

            elif prop_name in ("verdict_summary", "verdict_rationale", "summary"):
                if round_num == 1:
                    dummy_data[prop_name] = (
                        f"Round 1 preliminary evaluation on {topic_key}: Position A "
                        "demonstrates initial structural coherence, but key counter-evidence "
                        "remains unaddressed."
                    )
                elif round_num == 2:
                    dummy_data[prop_name] = (
                        f"Round 2 intermediate synthesis on {topic_key}: Position B "
                        "introduced compelling rebuttals regarding empirical risk parameters "
                        "and procedural flaws."
                    )
                else:
                    dummy_data[prop_name] = (
                        f"Round 3 final adjudication on {topic_key}: Comprehensive "
                        "multi-round synthesis confirms preponderance of grounded evidence "
                        "favors the affirmative case."
                    )

            elif prop_type == "string":
                dummy_data[prop_name] = (
                    f"Mock {prop_name} statement for {topic_key} round {round_num}"
                )

            elif prop_name == "confidence":
                conf_val = 0.75 + (round_num * 0.06) + ((prompt_hash % 7) / 100.0)
                dummy_data[prop_name] = round(min(conf_val, 0.94), 2)

            elif prop_type in ("number", "integer"):
                if prop_name in ("total_score_a", "total_score_b"):
                    # Gradually increase score across rounds to test adaptive stopping thresholds
                    base_score = 0.72 + (round_num * 0.08) + ((prompt_hash % 5) / 100.0)
                    dummy_data[prop_name] = round(min(base_score, 0.96), 2)
                else:
                    dummy_data[prop_name] = round(0.70 + (prompt_hash % 25) / 100.0, 2)

            elif prop_type == "boolean":
                dummy_data[prop_name] = True

            elif prop_type == "array":
                items_schema = prop_schema.get("items", {})
                ref = items_schema.get("$ref")
                if ref:
                    def_name = ref.split("/")[-1]
                    def_props = defs.get(def_name, {}).get("properties", {})
                    nested: dict[str, Any] = {}
                    for p, s in def_props.items():
                        stype = s.get("type", "string")
                        if stype == "string":
                            nested[p] = f"Rubric evaluation for {p} in round {round_num}"
                        elif stype in ("number", "integer"):
                            nested[p] = round(0.70 + (round_num * 0.05), 2)
                        elif stype == "boolean":
                            nested[p] = True
                        else:
                            nested[p] = f"Mock {p}"
                    dummy_data[prop_name] = [nested]
                else:
                    is_cite = prop_name in ("supporting_evidence", "sources_cited", "citations")
                    if is_cite and doc_ids_unique:
                        dummy_data[prop_name] = doc_ids_unique
                    elif is_cite:
                        if topic_key == "opensource":
                            dummy_data[prop_name] = (
                                ["CSET-2026-01"]
                                if round_num == 1
                                else ["CSET-2026-01", "NIST-AI-6001"]
                            )
                        elif topic_key == "legal":
                            dummy_data[prop_name] = ["NJAT-2025-04"]
                        else:
                            dummy_data[prop_name] = []

                    elif prop_name == "reasoning":
                        if topic_key == "opensource":
                            dummy_data[prop_name] = (
                                [
                                    "Public code visibility enables global vulnerability auditing.",
                                    "Monolithic API lock-in creates single point vulnerability.",
                                ]
                                if round_num == 1
                                else [
                                    "Public code visibility enables global vulnerability auditing.",
                                    "Monolithic API lock-in creates single point vulnerability.",
                                    f"Round {round_num} safety benchmarks show open resilience.",
                                ]
                            )
                        elif topic_key == "legal":
                            dummy_data[prop_name] = [
                                "Algorithmic decision systems lack human discretionary nuance.",
                                "Evidentiary validation standards require cross-examination.",
                            ]
                        else:
                            dummy_data[prop_name] = [
                                "Direct cash transfers preserve consumer purchasing power."
                            ]

                    elif prop_name == "counter_arguments":
                        if topic_key == "opensource":
                            dummy_data[prop_name] = [
                                f"Round {round_num} rebuttal: Open weight distribution risk."
                            ]
                        else:
                            dummy_data[prop_name] = [
                                f"Round {round_num} rebuttal: Automated sensors reduce bias."
                            ]

                    elif prop_name == "logical_fallacies":
                        has_fallacy = (prompt_hash + round_num) % 3 == 0
                        dummy_data[prop_name] = ["hasty_generalization"] if has_fallacy else []

                    else:
                        dummy_data[prop_name] = [f"Item for {topic_key} round {round_num}"]
            else:
                dummy_data[prop_name] = "Mock value"

        return json.dumps(dummy_data)
