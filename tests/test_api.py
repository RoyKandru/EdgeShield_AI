from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_url_api():
    response = client.post("/api/url/analyze", json={"url": "http://192.0.2.10/verify"})
    assert response.status_code == 200
    assert "event_id" in response.json()


def test_sms_api():
    response = client.post(
        "/api/sms/analyze",
        json={"message": "Urgent: update KYC and share OTP now"}
    )
    assert response.status_code == 200
    assert response.json()["source"] == "sms_detector"


def test_sci_scope_is_three_inputs():
    capabilities = client.get("/api/sci/capabilities").json()
    assert capabilities["input_pipelines"] == ["network", "url", "sms"]
    assert "sensor" not in capabilities["components_monitored"]
    assert capabilities["external_sensors"] is False
