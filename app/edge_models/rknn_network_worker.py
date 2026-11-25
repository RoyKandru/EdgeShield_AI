#!/usr/bin/env python3
"""Persistent RK3588 NPU_CORE_0 inference worker for 26-feature Network 1D-CNN+MLP (V2r_nolen26)."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

FRAME_PREFIX = "__RKNN_JSON__:"


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def emit(payload: dict) -> None:
    sys.stdout.write(FRAME_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./models/rknn")
    core_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    rknn_path = model_dir / "network_ids_fp16.rknn"
    meta_path = model_dir / "network_ids_meta.json"

    try:
        from rknnlite.api import RKNNLite
        for p in (rknn_path, meta_path):
            if not p.is_file():
                raise FileNotFoundError(str(p))

        t_init = time.perf_counter()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prep = meta["preprocessing"]
        feat_names = list(prep["feature_names"])
        mean = np.asarray(prep["mean"], dtype=np.float32)
        std = np.asarray(prep["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        default_tau = float(meta["calibration"]["default_threshold"])

        masks = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1, 2: RKNNLite.NPU_CORE_2}
        core_names = {0: "NPU_CORE_0", 1: "NPU_CORE_1", 2: "NPU_CORE_2"}
        if core_id not in masks:
            raise ValueError(f"Invalid RK3588 NPU core_id: {core_id}")

        rknn = RKNNLite(verbose=False)
        rc = rknn.load_rknn(str(rknn_path))
        if rc != 0:
            raise RuntimeError(f"load_rknn returned {rc}")
        rc = rknn.init_runtime(core_mask=masks[core_id])
        if rc != 0:
            raise RuntimeError(f"init_runtime returned {rc}")

        # Warm-up on NPU_CORE_0
        z = np.zeros((1, len(feat_names)), dtype=np.float32)
        _ = rknn.inference(inputs=[z])
        init_ms = round((time.perf_counter() - t_init) * 1000.0, 2)

        emit({
            "ready": True,
            "backend": "rknn-npu",
            "core_id": core_id,
            "core": core_names[core_id],
            "model_id": meta.get("model_id", "network-ids-1dcnn-mlp-v2r26-rknn-v1"),
            "architecture": meta["architecture"]["name"],
            "precision": meta["architecture"].get("npu_precision", "fp16"),
            "n_features": len(feat_names),
            "default_threshold": default_tau,
            "init_ms": init_ms,
        })
    except Exception as exc:
        emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    def infer_one(vec_raw: list[float], tau: float) -> dict:
        x = np.asarray(vec_raw, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip((signed_log1p(x) - mean) / std, -8.0, 8.0).astype(np.float32)[None, :]
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[x])
        npu_ms = (time.perf_counter() - t0) * 1000.0
        logit = float(np.clip(np.asarray(outputs[0], dtype=np.float32).reshape(-1)[0], -40.0, 40.0))
        score = 1.0 / (1.0 + math.exp(-logit))
        return {
            "score": round(score, 6),
            "logit": round(logit, 5),
            "flagged": bool(score >= tau),
            "threshold": tau,
            "npu_latency_ms": round(npu_ms, 3),
            "backend": "rknn-npu",
            "core_id": core_id,
            "core": core_names[core_id],
            "model_id": meta.get("model_id", "network-ids-1dcnn-mlp-v2r26-rknn-v1"),
        }

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if req.get("cmd") == "shutdown":
                    break
                tau = float(req.get("threshold", default_tau))
                if "vectors" in req:
                    results = [infer_one(v, tau) for v in req["vectors"]]
                    emit({"ok": True, "results": results})
                else:
                    vec = req.get("vector")
                    if vec is None:
                        feats = req.get("features") or {}
                        vec = [float(feats.get(k, 0.0)) for k in feat_names]
                    res = infer_one(vec, tau)
                    res["ok"] = True
                    emit(res)
            except Exception as exc:
                emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
