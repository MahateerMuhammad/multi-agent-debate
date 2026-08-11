from __future__ import annotations

import json

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings managed via Pydantic BaseSettings and environment variables."""

    PROJECT_NAME: str = "Multi-Agent Debate"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "console"  # "console" or "json"
    API_V1_STR: str = "/api/v1"

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Open-Source LLM Configuration (Qwen 2.5 default)
    LLM_PROVIDER: str = "openrouter"  # Options: "openrouter", "qwen", "ollama", "vllm", "mock"
    LLM_MODEL: str = "qwen/qwen-2.5-72b-instruct:free"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_API_KEY: str = ""

    # Guardrails & Cost Controls
    LLM_TIMEOUT: float = 30.0
    LLM_MAX_RETRIES: int = 3
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048  # Hard cap on output generation per request
    LLM_MAX_DEBATE_ROUNDS: int = 5  # Hard cap on max debate rounds
    ENABLE_GUARDRAILS: bool = True  # Enforce prompt sanitization & injection defense

    VECTORSTORE_DIR: str = "./data/vectorstore"
    PROCESSED_DATA_DIR: str = "./data/processed"
    RAW_DATA_DIR: str = "./data/raw"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse string-based CORS origins into a list of strings."""
        if isinstance(v, str):
            v_trimmed = v.strip()
            if v_trimmed.startswith("[") and v_trimmed.endswith("]"):
                try:
                    res = json.loads(v_trimmed)
                    if isinstance(res, list):
                        return [str(item) for item in res]
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in v_trimmed.split(",") if item.strip()]
        return list(v) if isinstance(v, list) else [str(v)]


settings = Settings()
