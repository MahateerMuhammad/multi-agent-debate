from fastapi.testclient import TestClient


def test_health_check_endpoint(test_client: TestClient) -> None:
    """Test the /health endpoint response structure and status code."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "project" in data
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data


def test_cors_headers(test_client: TestClient) -> None:
    """Test that CORS response headers are properly set."""
    response = test_client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
