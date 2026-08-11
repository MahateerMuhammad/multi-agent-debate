from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.config import Settings


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Pytest fixture providing a FastAPI TestClient instance."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def custom_settings() -> Settings:
    """Pytest fixture providing isolated settings for configuration tests."""
    return Settings(
        PROJECT_NAME="Test Debate Project",
        VERSION="0.0.1",
        ENVIRONMENT="testing",
        DEBUG=True,
        LOG_LEVEL="DEBUG",
        CORS_ORIGINS=["http://testserver"],
    )
