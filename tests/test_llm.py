from __future__ import annotations

import pytest

from app.core.config import Settings
from app.llm.exceptions import (
    LLMConfigurationError,
    LLMTimeoutError,
    LLMValidationError,
)
from app.llm.factory import get_llm_provider
from app.llm.providers.mock import MockLLMProvider
from app.llm.providers.qwen_provider import QwenProvider
from app.llm.schemas import (
    ArgumentOutput,
    ResearcherOutput,
    VerdictOutput,
)


@pytest.mark.asyncio
async def test_mock_provider_text_generation() -> None:
    """Test text generation with MockLLMProvider."""
    provider = MockLLMProvider(mock_response="Qwen test response")
    response = await provider.generate("What is AI ethics?")

    assert response.data == "Qwen test response"
    assert response.provider == "mock"
    assert response.usage.total_tokens > 0
    assert response.latency_seconds >= 0.0


@pytest.mark.asyncio
async def test_mock_provider_structured_output() -> None:
    """Test structured output generation using Pydantic schema."""
    custom_json = {
        "topic": "AI Governance",
        "summary": "Key regulation frameworks required",
        "key_facts": ["Act 1 passed", "Fact 2 verified"],
        "sources": ["https://example.org/report"],
    }
    provider = MockLLMProvider(mock_json_response=custom_json)
    response = await provider.generate_structured("Research AI governance", ResearcherOutput)

    assert isinstance(response.data, ResearcherOutput)
    assert response.data.topic == "AI Governance"
    assert len(response.data.key_facts) == 2
    assert response.data.sources[0] == "https://example.org/report"


@pytest.mark.asyncio
async def test_malformed_structured_output_rejection() -> None:
    """Test that malformed/invalid JSON outputs are strictly rejected with LLMValidationError."""
    provider = MockLLMProvider(simulate_invalid_json=True)

    with pytest.raises(LLMValidationError) as exc_info:
        await provider.generate_structured("Generate argument", ArgumentOutput)

    assert (
        "malformed output" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()
    )


@pytest.mark.asyncio
async def test_mock_provider_timeout_simulation() -> None:
    """Test timeout exception handling."""
    provider = MockLLMProvider(simulate_timeout=True, timeout=5.0)

    with pytest.raises(LLMTimeoutError) as exc_info:
        await provider.generate("Test prompt")

    assert "timed out" in str(exc_info.value).lower()


def test_factory_provider_instantiation() -> None:
    """Test get_llm_provider factory for Qwen and Mock providers."""
    qwen_cfg = Settings(
        LLM_PROVIDER="qwen",
        LLM_MODEL="qwen2.5-72b-instruct",
        LLM_BASE_URL="http://localhost:11434/v1",
        LLM_API_KEY="test_key_12345",
    )
    provider = get_llm_provider(qwen_cfg)
    assert isinstance(provider, QwenProvider)
    assert provider.model_name == "qwen2.5-72b-instruct"

    mock_cfg = Settings(LLM_PROVIDER="mock")
    mock_provider = get_llm_provider(mock_cfg)
    assert isinstance(mock_provider, MockLLMProvider)


def test_factory_invalid_provider_configuration() -> None:
    """Test that unsupported provider raises LLMConfigurationError."""
    invalid_cfg = Settings(LLM_PROVIDER="invalid_vendor_xyz")

    with pytest.raises(LLMConfigurationError) as exc_info:
        get_llm_provider(invalid_cfg)

    assert "unsupported llm provider" in str(exc_info.value).lower()


def test_factory_missing_base_url() -> None:
    """Test missing base_url for Qwen provider raises LLMConfigurationError."""
    bad_cfg = Settings(LLM_PROVIDER="qwen", LLM_BASE_URL="")

    with pytest.raises(LLMConfigurationError) as exc_info:
        get_llm_provider(bad_cfg)

    assert "base_url must be configured" in str(exc_info.value).lower()


def test_secret_log_masking() -> None:
    """Test that API keys and Authorization headers are masked in logs."""
    provider = QwenProvider(
        model_name="qwen2.5-72b",
        base_url="http://localhost:8000/v1",
        api_key="secret_qwen_token_xyz999",
    )
    raw_log = "Sending request with Bearer secret_qwen_token_xyz999 to endpoint."
    sanitized = provider._sanitize_log(raw_log)

    assert "secret_qwen_token_xyz999" not in sanitized
    assert "***MASKED***" in sanitized or "...***" in sanitized


@pytest.mark.asyncio
async def test_structured_schemas_instantiation() -> None:
    """Test instantiation and validation of all debate agent structured schemas."""
    arg = ArgumentOutput(
        claim="Open Source models ensure transparency",
        reasoning=["Auditable weights", "Local hosting capability"],
        supporting_evidence=["Qwen 2.5 benchmark scores"],
        confidence=0.95,
    )
    assert arg.claim.startswith("Open Source")
    assert arg.confidence == 0.95

    verdict = VerdictOutput(
        winner="Proponent",
        summary="Proponent presented verifiable evidence",
        key_deciding_factors=["Factuality", "Logical consistency"],
        proponent_score=0.9,
        opponent_score=0.7,
    )
    assert verdict.winner == "Proponent"
    assert verdict.proponent_score > verdict.opponent_score
