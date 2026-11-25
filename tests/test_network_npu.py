"""Tests for 26-feature Network IDS on RK3588J NPU_CORE_0 + 3-Core Concurrent NPU execution."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.detectors.network import analyse_flows
from app.edge_models.network_npu import get_network_npu_client
from app.edge_models.service import get_edge_inference_service


def test_live_rk3588_npu_core0_network_and_3core_concurrency():
    client = get_network_npu_client()
    assert client.ready, f"Expected RK3588 NPU_CORE_0 Network model ready, got: {client.status()}"

    benign = client.infer_flow_dict({
        "src_ip": "192.168.1.50", "dst_ip": "192.168.1.6", "dst_port": 443,
        "packets": 8, "bytes": 4200, "duration_seconds": 1.2, "syn_count": 1,
    })
    port_scan = client.infer_flow_dict({
        "src_ip": "192.168.1.99", "dst_ip": "192.168.1.6", "dst_port": 4444,
        "packets": 2, "bytes": 120, "duration_seconds": 0.001, "syn_count": 1,
        "sw_flows": 250, "sw_ndport": 240, "sw_small_frac": 1.0,
    })
    syn_flood = client.infer_flow_dict({
        "src_ip": "192.168.1.99", "dst_ip": "192.168.1.6", "dst_port": 80,
        "packets": 400, "bytes": 24000, "duration_seconds": 0.5, "syn_count": 380,
        "sw_flows": 200, "dw_flows": 200,
    })
    assert benign["backend"] == "rknn-npu" and benign["core_id"] == 0 and benign["core"] == "NPU_CORE_0"
    assert benign["npu_latency_ms"] < 25.0
    assert benign["score"] < 0.40
    assert port_scan["score"] > 0.85
    assert syn_flood["score"] > 0.85

    # Verify analyse_flows() wires NPU_CORE_0 cleanly
    evs = analyse_flows([
        {"src_ip": "192.168.1.50", "dst_ip": "192.168.1.6", "dst_port": 443, "packets": 8, "bytes": 4200, "duration_seconds": 1.2, "syn_count": 1},
        {"src_ip": "192.168.1.99", "dst_ip": "192.168.1.6", "dst_port": 80, "packets": 400, "bytes": 24000, "duration_seconds": 0.5, "syn_count": 380, "sw_flows": 200},
    ])
    assert evs[0].prediction == "no_configured_anomaly"
    assert evs[0].metadata.get("npu_net_core") == "NPU_CORE_0"
    assert evs[1].prediction == "anomalous"

    # Verify ALL 3 RK3588 NPU cores concurrently (Core 0 Network + Core 1 URL + Core 2 SMS)!
    svc = get_edge_inference_service()
    st = svc.status()
    assert st["all_3_cores_active"] is True
    assert st["active_npu_cores"] == ["NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2"]

    with ThreadPoolExecutor(max_workers=3) as ex:
        f0 = ex.submit(svc.infer_network, {"dst_port": 80, "packets": 400, "duration_seconds": 0.5, "syn_count": 380, "sw_flows": 200})
        f1 = ex.submit(svc.infer_url, "http://192.168.0.1/login.php")
        f2 = ex.submit(svc.infer_sms, "URGENT! Update KYC at http://sbi-kyc.xyz/login or account blocked!")
        r0, r1, r2 = f0.result(), f1.result(), f2.result()
    assert (r0.core_id, r1.core_id, r2.core_id) == (0, 1, 2)
    assert r0.backend == r1.backend == r2.backend == "rknn-npu"
    assert r0.flagged and r1.flagged and r2.flagged
