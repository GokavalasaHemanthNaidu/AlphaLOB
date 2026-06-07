import pytest
from fastapi.testclient import TestClient
from src.api.main import app

# We need a clean instance of _request_log or to mock the IP so we don't interfere with other tests
# Since TestClient usually uses "testclient" as host, we can just run 31 requests in a row

def test_rate_limit():
    import src.api.main
    src.api.main._request_log.clear()
    
    client = TestClient(app)
    
    # 30 requests should succeed
    for _ in range(30):
        # We use GET / to be fast
        response = client.get("/")
        assert response.status_code == 200
        
    # The 31st request should fail with 429 Too Many Requests
    response = client.get("/")
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]
