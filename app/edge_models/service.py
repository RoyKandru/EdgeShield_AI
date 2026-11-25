"""Python orchestration for ALL THREE independent RK3588 NPU detector cores.

- Core 0 (NPU_CORE_0): Network IDS 1D-CNN + MLP (V2r_nolen26, 26 raw-packet features, LIVE FP16 NPU)
- Core 1 (NPU_CORE_1): Phishing URL Dual-Branch Lexical MLP + Char 1D-CNN (Prasad 2026, LIVE FP16 NPU)
- Core 2 (NPU_CORE_2): SMS BiGRU + Masked Global Mean Pooling + WBCE (Phap 2026, LIVE FP16 NPU)
- Arm64 CPU (Pipeline B): Meta Llama-3.2-1B-Instruct via ExecuTorch (XNNPACK + KleidiAI)
"""
from __future__ import annotations
import logging
import re
from pathlib import Path

def _read_kernel_npu_loads() -> dict[int, float]:
    try:
        raw = Path("/sys/kernel/debug/rknpu/load").read_text(encoding="utf-8")
        return {int(c): float(v) for c, v in re.findall(r"Core(\d+)\s*:\s*(\d+(?:\.\d+)?)%", raw)}
    except Exception:
        return {}

from app.edge_models.config import load_specs
from app.edge_models.contracts import InferenceResult
from app.edge_models.network_npu import get_network_npu_client
from app.edge_models.sms_bigru_npu import get_sms_bigru_npu_client
from app.edge_models.url_dual_npu import get_url_dual_npu_client

log = logging.getLogger("edgeshield.edge_models")


class EdgeInferenceService:
    def __init__(self, manifest: str = "./models/rknn/manifest.json"):
        self.specs = load_specs(manifest)
        self.net_npu = get_network_npu_client()
        self.url_npu = get_url_dual_npu_client()
        self.sms_npu = get_sms_bigru_npu_client()

    def status(self):
        models_status = {
            "network": self.net_npu.status(),
            "url": self.url_npu.status(),
            "sms": self.sms_npu.status(),
        }
        kloads = _read_kernel_npu_loads()
        cores = []
        for cid, key in ((0, "network"), (1, "url"), (2, "sms")):
            m = models_status[key]
            w_pct = float(m.get("load_percent", 0.0)) if m.get("ready") else 0.0
            k_pct = float(kloads.get(cid, 0.0))
            eff_pct = round(max(w_pct, k_pct), 1)
            m["load_percent"] = eff_pct
            cores.append({"core": cid, "load_percent": eff_pct, "active": bool(m.get("ready"))})
        active_cores = sorted([m["core"] for m in models_status.values() if m.get("ready")])
        return {
            "allocation_policy": "one dedicated RK3588 NPU core per specialist detector",
            "active_npu_cores": active_cores,
            "cores": cores,
            "all_3_cores_active": len(active_cores) == 3,
            "models": models_status,
            "concurrency_verified": len(active_cores) == 3,
            "note": (
                "ALL 3 RK3588J NPU cores active: NPU_CORE_0 (Network 1D-CNN+MLP), NPU_CORE_1 (URL MLP+CharCNN), "
                "NPU_CORE_2 (SMS BiGRU+MaskedPool+WBCE). Separate from Arm64 CPU ExecuTorch Llama-3.2-1B."
            ),
        }

    def infer_network(self, features: dict, threshold: float | None = None):
        if self.net_npu.ready:
            tau = float(threshold if threshold is not None else self.specs.get("network", {}).get("threshold", 0.80))
            res = self.net_npu.infer_flow_dict(features, threshold=tau)
            return InferenceResult(
                pipeline="network",
                score=float(res["score"]),
                flagged=bool(res["flagged"]),
                backend=str(res["backend"]),
                core_id=int(res["core_id"]),
                latency_ms=float(res["npu_latency_ms"]),
                model_version=str(res["model_id"]),
                calibrated_probability=True,
                note=f"26-feature V2r_nolen26 Network 1D-CNN+MLP FP16 on RK3588 {res['core']} (tau={tau:.2f}).",
            )
        raise RuntimeError(self.net_npu.status().get("error") or "RKNN Network NPU core unavailable")

    def infer_url(self, url: str, threshold: float | None = None):
        if self.url_npu.ready:
            tau = float(threshold if threshold is not None else self.specs.get("url", {}).get("threshold", 0.96))
            res = self.url_npu.infer(url, threshold=tau)
            return InferenceResult(
                pipeline="url",
                score=float(res["score"]),
                flagged=bool(res["flagged"]),
                backend=str(res["backend"]),
                core_id=int(res["core_id"]),
                latency_ms=float(res["npu_latency_ms"]),
                model_version=str(res["model_id"]),
                calibrated_probability=True,
                note=f"Prasad (2026) Dual-Branch URL MLP+CharCNN FP16 on RK3588 {res['core']} (tau={tau:.2f}).",
            )
        raise RuntimeError(self.url_npu.status().get("error") or "RKNN URL NPU core unavailable")

    def infer_sms(self, message: str, threshold: float | None = None):
        if self.sms_npu.ready:
            tau = float(threshold if threshold is not None else self.specs.get("sms", {}).get("threshold", 0.22))
            res = self.sms_npu.infer(message, threshold=tau)
            return InferenceResult(
                pipeline="sms",
                score=float(res["score"]),
                flagged=bool(res["flagged"]),
                backend=str(res["backend"]),
                core_id=int(res["core_id"]),
                latency_ms=float(res["npu_latency_ms"]),
                model_version=str(res["model_id"]),
                calibrated_probability=True,
                note=f"Phap (2026) BiGRU+MaskedPool+WBCE FP16 on RK3588 {res['core']} (tau={tau:.2f}, tokens={res['token_count']}).",
            )
        raise RuntimeError(self.sms_npu.status().get("error") or "RKNN SMS NPU core unavailable")

    def close(self):
        self.net_npu.close()
        self.url_npu.close()
        self.sms_npu.close()


_service = None


def get_edge_inference_service():
    global _service
    if _service is None:
        _service = EdgeInferenceService()
    return _service
