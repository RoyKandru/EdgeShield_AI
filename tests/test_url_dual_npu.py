"""Tests for Prasad (2026) URL Dual-Branch MLP + Char 1D-CNN on RK3588J NPU_CORE_1."""
from __future__ import annotations

from app.detectors.url import analyse_url
from app.edge_models.url_dual_npu import get_url_dual_npu_client


def test_live_rk3588_npu_core1_url_inference_and_fusion():
    client = get_url_dual_npu_client()
    assert client.ready, f"Expected RK3588 NPU_CORE_1 URL model ready, got: {client.status()}"

    phish_ip = client.infer("http://192.168.0.1/login.php")
    phish_brand = client.infer("https://secure-facebook-com-login-alert.ga/verify")
    benign_sbi = client.infer("https://onlinesbi.sbi/")
    assert phish_ip["backend"] == "rknn-npu"
    assert phish_ip["core_id"] == 1
    assert phish_ip["core"] == "NPU_CORE_1"
    assert phish_ip["npu_latency_ms"] < 25.0
    assert phish_ip["score"] > 0.95
    assert phish_brand["score"] > 0.95
    assert benign_sbi["score"] < 0.30

    # End-to-end hybrid URL detector fusion (incl. Prasad Fig. 16 failure cases fixed!)
    ev_google = analyse_url("https://www.google.com")
    assert ev_google.prediction == "no_obvious_indicators"
    assert ev_google.metadata.get("npu_url_core") == "NPU_CORE_1"
    assert ev_google.metadata.get("npu_url_backend") == "rknn-npu"

    ev_ip = analyse_url("http://192.168.0.1/login.php")
    assert ev_ip.prediction == "suspicious"
    assert ev_ip.metadata.get("npu_url_model") == "url-dualbranch-mlp-charcnn-rknn-v1"

    ev_brand = analyse_url("http://accounts.google.com.mailverify.cc/signin")
    assert ev_brand.prediction == "suspicious"
