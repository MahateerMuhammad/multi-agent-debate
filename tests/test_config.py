from app.core.config import Settings


def test_default_settings() -> None:
    """Verify application default settings values."""
    s = Settings()
    assert s.PROJECT_NAME == "Multi-Agent Debate"
    assert s.VERSION == "0.1.0"
    assert s.ENVIRONMENT == "development"
    assert isinstance(s.CORS_ORIGINS, list)


def test_cors_origins_parsing() -> None:
    """Test parsing string list and comma-separated string for CORS_ORIGINS."""
    s1 = Settings(CORS_ORIGINS='["http://localhost:3000"]')  # type: ignore[arg-type]
    assert s1.CORS_ORIGINS == ["http://localhost:3000"]

    s2 = Settings(CORS_ORIGINS="http://localhost:3000, http://example.com")  # type: ignore[arg-type]
    assert s2.CORS_ORIGINS == ["http://localhost:3000", "http://example.com"]
