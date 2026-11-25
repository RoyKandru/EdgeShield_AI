from fastapi.testclient import TestClient

from app.main import app


def test_system_metrics_endpoint_returns_resource_sections():
    response = TestClient(app).get("/api/system/metrics")
    assert response.status_code == 200
    payload = response.json()
    for section in ("cpu", "memory", "storage", "temperature", "frequency", "npu", "gpu", "network", "system"):
        assert section in payload
    assert "usage_percent" in payload["cpu"]
    assert "total_bytes" in payload["memory"]
    assert "usage_percent" in payload["storage"]
    assert isinstance(payload["network"], list)
    assert isinstance(payload["limitations"], list)


def test_system_metrics_does_not_require_accelerator_counters():
    payload = TestClient(app).get("/api/system/metrics").json()
    # Accelerator counters may legitimately be unavailable on some kernels.
    assert "busy_percent" in payload["npu"]
    assert "busy_percent" in payload["gpu"]
