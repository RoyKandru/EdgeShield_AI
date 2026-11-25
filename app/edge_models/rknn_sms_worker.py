#!/usr/bin/env python3
"""Persistent RK3588 NPU_CORE_2 inference worker for Phap (2026) BiGRU+MaskedPool+WBCE.

Runs inside the board's rknn-python environment (where rknnlite is installed).
Frames all JSON responses with `__RKNN_JSON__:` so Rockchip librknnrt C logs on stdout/stderr
can never corrupt IPC parsing.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

FRAME_PREFIX = "__RKNN_JSON__:"

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
_SPACED_WORD_RE = re.compile(r"(?<![A-Za-z0-9])((?:[A-Za-z0-9]\s+){2,}[A-Za-z0-9])(?![A-Za-z0-9])")
_DOTTED_ACRO_RE = re.compile(r"\b([A-Za-z])\.([A-Za-z])\.([A-Za-z])(?:\.([A-Za-z]))?\b")
_USSD_RE = re.compile(r"\*+(?:401|21|61|62|67)\*[0-9]{4,15}#?")
_URL_RE = re.compile(
    r"(?:hxxps?|https?)://[^\s<>\"'\)\]]+|www\.[^\s<>\"'\)\]]+|"
    r"\b(?:[a-z0-9-]+\.)+(?:com|in|co\.in|org|net|gov\.in|nic\.in|uk|co\.uk|top|xyz|zip|mov|click|info|biz|site|online|vip|club|tk|ml|ga|cf|gq|cfd|sbs|pages\.dev|workers\.dev|ngrok-free\.app|bit\.ly|rb\.gy|tinyurl\.com|t\.co|apk)(?:/[^\s<>\"'\)\]]*)?",
    re.I,
)
_EMAIL_RE = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
_CURRENCY_NUM_RE = re.compile(r"(?:rs\.?|inr|₹|\$|£|€)\s*\d[\d,]*(?:\.\d+)?", re.I)
_LONG_NUM_RE = re.compile(r"\b\d[\d\-\s]{3,}\d\b|\b\d{4,}\b")
_TOKEN_RE = re.compile(r"<url>|<num>|<ussd>|<email>|<cur>|[a-z0-9']+|[!?]")


def preprocess_sms(text: str) -> list[str]:
    s = unicodedata.normalize("NFKC", text or "")
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _SPACED_WORD_RE.sub(lambda m: re.sub(r"\s+", "", m.group(1)), s)
    s = _DOTTED_ACRO_RE.sub(lambda m: "".join(g for g in m.groups() if g), s)
    s = s.lower()
    s = _USSD_RE.sub(" <ussd> ", s)
    s = _EMAIL_RE.sub(" <email> ", s)
    s = _URL_RE.sub(" <url> ", s)
    s = _CURRENCY_NUM_RE.sub(" <cur> <num> ", s)
    s = _LONG_NUM_RE.sub(" <num> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _TOKEN_RE.findall(s)


def emit(payload: dict) -> None:
    sys.stdout.write(FRAME_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./models/rknn")
    core_id = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    rknn_path = model_dir / "sms_bigru_fp16.rknn"
    emb_path = model_dir / "sms_bigru_emb.npz"
    meta_path = model_dir / "sms_bigru_meta.json"

    try:
        from rknnlite.api import RKNNLite
        for p in (rknn_path, emb_path, meta_path):
            if not p.is_file():
                raise FileNotFoundError(str(p))

        t_init = time.perf_counter()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vocab: dict[str, int] = meta["vocab"]
        unk_id = int(vocab.get("<UNK>", 1))
        arch = meta["architecture"]
        t_max = int(arch["t_max"])
        emb_dim = int(arch["emb_dim"])
        default_tau = float(meta["calibration"]["default_threshold"])
        emb = np.load(emb_path)["embedding"].astype(np.float32)

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

        # Warm-up inference on real NPU core so first request has zero cold start
        z_f = np.zeros((1, t_max, emb_dim), dtype=np.float32)
        z_b = np.zeros((1, t_max, emb_dim), dtype=np.float32)
        z_w = np.zeros((1, 1, t_max), dtype=np.float32)
        z_w[0, 0, 0] = 1.0
        _ = rknn.inference(inputs=[z_f, z_b, z_w])
        init_ms = round((time.perf_counter() - t_init) * 1000.0, 2)

        emit({
            "ready": True,
            "backend": "rknn-npu",
            "core_id": core_id,
            "core": core_names[core_id],
            "model_id": meta.get("model_id", "sms-bigru-maskedpool-wbce-rknn-v1"),
            "architecture": arch.get("name"),
            "precision": arch.get("npu_precision", "fp16"),
            "t_max": t_max,
            "emb_dim": emb_dim,
            "vocab_size": len(vocab),
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
                text = str(req.get("text", ""))
                tau = float(req.get("threshold", default_tau))

                t_total = time.perf_counter()
                toks = preprocess_sms(text)
                ids = [vocab.get(t, unk_id) for t in toks[:t_max]]
                L = len(ids)

                sf = np.zeros((1, t_max, emb_dim), dtype=np.float32)
                sb = np.zeros((1, t_max, emb_dim), dtype=np.float32)
                pw = np.zeros((1, 1, t_max), dtype=np.float32)
                if L > 0:
                    sf[0, :L] = emb[ids]
                    sb[0, :L] = emb[ids[::-1]]
                    pw[0, 0, :L] = 1.0 / float(L)
                else:
                    sf[0, 0] = emb[unk_id]
                    sb[0, 0] = emb[unk_id]
                    pw[0, 0, 0] = 1.0

                t_npu = time.perf_counter()
                outputs = rknn.inference(inputs=[sf, sb, pw])
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
                    "token_count": L,
                    "tokens_preview": toks[:16],
                    "npu_latency_ms": round(npu_ms, 3),
                    "total_latency_ms": round(total_ms, 3),
                    "backend": "rknn-npu",
                    "core_id": core_id,
                    "core": core_names[core_id],
                    "model_id": meta.get("model_id", "sms-bigru-maskedpool-wbce-rknn-v1"),
                })
            except Exception as exc:
                emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
