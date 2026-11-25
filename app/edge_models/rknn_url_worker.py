#!/usr/bin/env python3
"""Persistent RK3588 NPU_CORE_1 inference worker for Prasad (2026) URL Dual-Branch MLP + CharCNN."""
from __future__ import annotations

import html
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

FRAME_PREFIX = "__RKNN_JSON__:"

_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "ow.ly", "rb.gy"}
SUSPICIOUS_TERMS = (
    "verify", "urgent", "login", "signin", "update", "secure", "account", "bank", "kyc",
    "wallet", "password", "bonus", "prize", "confirm", "suspend", "billing", "paypal",
)


def normalise_url(raw: str) -> str:
    u = html.unescape((raw or "").strip()).lower()
    u = _SCHEME.sub("", u)
    if u.startswith("www."):
        u = u[4:]
    return u


def url_host(norm: str) -> str:
    authority = re.split(r"[/?#]", norm, maxsplit=1)[0]
    authority = authority.rsplit("@", 1)[-1]
    return authority.split(":", 1)[0]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def extract_url_features(raw: str) -> np.ndarray:
    n = normalise_url(raw)
    host = url_host(n)
    rest = n[len(host):] if n.startswith(host) else ""
    path, _, query = rest.partition("?")
    labels = host.split(".") if host else [""]
    digits = sum(c.isdigit() for c in n)
    host_digits = sum(c.isdigit() for c in host)
    return np.asarray([
        float(len(n)), float(len(host)), float(len(path)), float(len(query)),
        float(n.count(".")), float(host.count(".")), float(host.count("-")),
        digits / max(1, len(n)), host_digits / max(1, len(host)),
        float(n.count("/")), float(n.count("@")), float(n.count("%")),
        float(n.count("=")), float(n.count("&")),
        1.0 if _IPV4.match(host) else 0.0,
        1.0 if "xn--" in host else 0.0,
        1.0 if host in SHORTENERS else 0.0,
        float(sum(t in n for t in SUSPICIOUS_TERMS)),
        _entropy(host), _entropy(n),
        1.0 if len(path) > 1 else 0.0,
        1.0 if query else 0.0,
        float(len(labels[-1])),
        float(max(len(x) for x in labels)),
        float(sum(x.isdigit() for x in labels)),
    ], dtype=np.float32)


def emit(payload: dict) -> None:
    sys.stdout.write(FRAME_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./models/rknn")
    core_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    rknn_path = model_dir / "url_hybrid_fp16.rknn"
    meta_path = model_dir / "url_hybrid_meta.json"

    try:
        from rknnlite.api import RKNNLite
        for p in (rknn_path, meta_path):
            if not p.is_file():
                raise FileNotFoundError(str(p))

        t_init = time.perf_counter()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        prep = meta["preprocessing"]
        mean = np.asarray(prep["mean"], dtype=np.float32)
        std = np.asarray(prep["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        alphabet = str(prep["alphabet"])
        char_lookup = {c: i + 2 for i, c in enumerate(alphabet)}
        char_vocab = len(alphabet) + 2
        max_len = int(prep["char_max_len"])
        default_tau = float(meta["calibration"]["high_precision_threshold"])

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

        # Warm-up on NPU_CORE_1
        z_lex = np.zeros((1, len(mean)), dtype=np.float32)
        z_ch = np.zeros((1, char_vocab, max_len), dtype=np.float32)
        z_ch[0, 0, :] = 1.0
        _ = rknn.inference(inputs=[z_lex, z_ch])
        init_ms = round((time.perf_counter() - t_init) * 1000.0, 2)

        emit({
            "ready": True,
            "backend": "rknn-npu",
            "core_id": core_id,
            "core": core_names[core_id],
            "model_id": meta.get("model_id", "url-dualbranch-mlp-charcnn-rknn-v1"),
            "architecture": meta["architecture"]["name"],
            "precision": meta["architecture"].get("npu_precision", "fp16"),
            "default_threshold": default_tau,
            "init_ms": init_ms,
        })
    except Exception as exc:
        emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if req.get("cmd") == "shutdown":
                    break
                url = str(req.get("url", ""))
                tau = float(req.get("threshold", default_tau))

                t_total = time.perf_counter()
                norm = normalise_url(url)
                lx = np.clip((extract_url_features(url) - mean) / std, -8.0, 8.0).astype(np.float32)[None, :]
                ch = np.zeros((1, char_vocab, max_len), dtype=np.float32)
                s = norm[:max_len]
                for i, c in enumerate(s):
                    ch[0, char_lookup.get(c, 1), i] = 1.0
                if len(s) < max_len:
                    ch[0, 0, len(s):] = 1.0

                t_npu = time.perf_counter()
                outputs = rknn.inference(inputs=[lx, ch])
                npu_ms = (time.perf_counter() - t_npu) * 1000.0
                total_ms = (time.perf_counter() - t_total) * 1000.0

                logit = float(np.clip(np.asarray(outputs[0], dtype=np.float32).reshape(-1)[0], -40.0, 40.0))
                score = 1.0 / (1.0 + math.exp(-logit))

                emit({
                    "ok": True,
                    "score": round(score, 6),
                    "logit": round(logit, 5),
                    "flagged": bool(score >= tau),
                    "threshold": tau,
                    "npu_latency_ms": round(npu_ms, 3),
                    "total_latency_ms": round(total_ms, 3),
                    "backend": "rknn-npu",
                    "core_id": core_id,
                    "core": core_names[core_id],
                    "model_id": meta.get("model_id", "url-dualbranch-mlp-charcnn-rknn-v1"),
                })
            except Exception as exc:
                emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
