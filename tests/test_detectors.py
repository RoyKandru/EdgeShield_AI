from app.detectors.url import analyse_url
from app.detectors.sms import analyse_sms
from app.detectors.network import analyse_flows


def test_url_suspicious_indicators():
    event = analyse_url("http://192.0.2.10/urgent/bank/verify-account")
    assert event.source == "url_detector"
    assert event.risk_score > 0
    assert event.evidence


def test_sms_otp_scam_indicators():
    event = analyse_sms("Urgent! Your bank KYC will expire. Share OTP and verify account at http://bad.test")
    assert event.source == "sms_detector"
    assert event.risk_score > 0
    assert len(event.evidence) >= 2


def test_network_syn_burst():
    events = analyse_flows([{
        "src_ip": "10.0.0.2", "dst_ip": "10.0.0.1",
        "src_port": 50000, "dst_port": 22, "protocol": "TCP",
        "packets": 40, "bytes": 5000, "duration_seconds": 1.0, "syn_count": 25
    }])
    assert len(events) == 1
    assert events[0].risk_score > 0


def test_url_malformed_does_not_crash():
    # urlparse raises ValueError("Invalid IPv6 URL") on an unbalanced '['
    event = analyse_url("http://[bad-host/login")
    assert event.source == "url_detector"
    assert any(e.code == "INVALID_HOST" for e in event.evidence)
