"""Persistent RK3588 NPU_CORE_1 client for Prasad (2026) URL Dual-Branch MLP + Char 1D-CNN."""
from __future__ import annotations

import atexit
import json
import logging
import os
import select
import subprocess
import sys
from collections import deque
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("edgeshield.rknn.url_dual")

FRAME_PREFIX = "__RKNN_JSON__:"
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _ROOT / "models" / "rknn"
DEFAULT_RKNN_PYTHON = Path(os.environ.get("EDGESHIELD_RKNN_PYTHON", "/home/open/miniforge3/envs/rknn-python/bin/python"))


def _find_rknn_python() -> str | None:
    for cand in (str(DEFAULT_RKNN_PYTHON), "/home/open/miniforge3/envs/rknn-python/bin/python", sys.executable):
        p = Path(cand)
        if not p.is_file() or not os.access(p, os.X_OK):
            continue
        if cand == sys.executable:
            try:
                import rknnlite  # noqa: F401
                return cand
            except Exception:
                continue
        return cand
    return None


class URLDualNPUClient:
    """Manages a persistent RKNNLite NPU_CORE_1 worker for URLDualBranchNPU."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, core_id: int = 1) -> None:
        if core_id not in (0, 1, 2):
            raise ValueError("RK3588 NPU core_id must be 0, 1, or 2")
        self.model_dir = Path(model_dir)
        self.core_id = core_id
        self.worker_script = Path(__file__).resolve().parent / "rknn_url_worker.py"
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._ready = False
        self._error: str | None = None
        self._banner: dict[str, Any] = {}
        self._meta_summary: dict[str, Any] = {}
        self._infer_count = 0
        self._latency_sum_ms = 0.0
        self._last_latency_ms: float | None = None
        self._recent_runs: deque[tuple[float, float]] = deque()
        self._load_meta_summary()
        self._start_locked()
        atexit.register(self.close)

    def _load_meta_summary(self) -> None:
        meta_path = self.model_dir / "url_hybrid_meta.json"
        if meta_path.is_file():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                self._meta_summary = {
                    "model_id": m.get("model_id"),
                    "paper": m.get("paper"),
                    "architecture": m.get("architecture"),
                    "calibration": m.get("calibration"),
                    "test_metrics": m.get("test_metrics"),
                }
            except Exception as exc:
                log.warning("Failed reading url_hybrid_meta.json: %s", exc)

    def _read_framed_json(self, timeout: float = 15.0) -> dict[str, Any]:
        if not self._proc or not self._proc.stdout:
            raise RuntimeError("RKNN URL worker stdout unavailable")
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            rlist, _, _ = select.select([fd], [], [], min(0.5, remaining))
            if not rlist:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"RKNN URL worker exited with code {self._proc.returncode}")
                continue
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("RKNN URL worker closed stdout unexpectedly")
            idx = line.find(FRAME_PREFIX)
            if idx != -1:
                return json.loads(line[idx + len(FRAME_PREFIX):].strip())
        raise TimeoutError(f"RKNN URL worker timed out after {timeout}s")

    def _start_locked(self) -> None:
        self._ready = False
        self._error = None
        for fname in ("url_hybrid_fp16.rknn", "url_hybrid_meta.json"):
            p = self.model_dir / fname
            if not p.is_file():
                self._error = f"FileNotFoundError: {p}"
                return
        py_bin = _find_rknn_python()
        if not py_bin:
            self._error = "RuntimeError: No Python environment with rknnlite found"
            return
        try:
            self._proc = subprocess.Popen(
                [py_bin, "-u", str(self.worker_script), str(self.model_dir), str(self.core_id)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            banner = self._read_framed_json(timeout=20.0)
            if not banner.get("ready"):
                self._error = str(banner.get("error", "RKNN URL worker failed to initialize"))
                self._terminate_locked()
                return
            self._banner = banner
            self._ready = True
            log.info("RK3588 %s READY: %s (init=%.1fms)", banner.get("core"), banner.get("model_id"), float(banner.get("init_ms", 0.0)))
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            self._terminate_locked()
            log.warning("RKNN URL worker unavailable: %s", self._error)

    def _terminate_locked(self) -> None:
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    @property
    def ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    def infer(self, url: str, threshold: float | None = None) -> dict[str, Any]:
        with self._lock:
            if not self.ready:
                self._start_locked()
            if not self.ready or not self._proc or not self._proc.stdin:
                raise RuntimeError(self._error or "RK3588 NPU URL worker unavailable")
            req: dict[str, Any] = {"url": url}
            if threshold is not None:
                req["threshold"] = float(threshold)
            try:
                self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
                resp = self._read_framed_json(timeout=10.0)
            except Exception:
                self._terminate_locked()
                self._start_locked()
                if not self.ready or not self._proc or not self._proc.stdin:
                    raise RuntimeError(self._error or "RK3588 NPU URL worker restart failed")
                self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
                resp = self._read_framed_json(timeout=10.0)
            if not resp.get("ok"):
                raise RuntimeError(str(resp.get("error", "RKNN URL NPU inference failed")))
            ms = float(resp.get("npu_latency_ms", 0.0))
            self._infer_count += 1
            self._latency_sum_ms += ms
            self._last_latency_ms = ms
            now_m = time.monotonic()
            self._recent_runs.append((now_m, ms))
            while self._recent_runs and (now_m - self._recent_runs[0][0]) > 1.5:
                self._recent_runs.popleft()
            return resp


    def live_load_percent(self, window_s: float = 1.5) -> float:
        """Real-time NPU utilization % over the last window_s seconds (0.0% when idle)."""
        if not self.ready:
            return 0.0
        now_m = time.monotonic()
        with self._lock:
            while self._recent_runs and (now_m - self._recent_runs[0][0]) > window_s:
                self._recent_runs.popleft()
            busy_ms = sum(ms for _, ms in self._recent_runs)
        pct = (busy_ms / (window_s * 1000.0)) * 100.0
        return round(min(100.0, max(0.0, pct)), 1)

    def status(self) -> dict[str, Any]:
        core_names = {0: "NPU_CORE_0", 1: "NPU_CORE_1", 2: "NPU_CORE_2"}
        avg_ms = round(self._latency_sum_ms / self._infer_count, 2) if self._infer_count else None
        return {
            "ready": self.ready,
            "core_id": self.core_id,
            "core": core_names[self.core_id],
            "artifact": str(self.model_dir / "url_hybrid_fp16.rknn"),
            "error": None if self.ready else self._error,
            "backend": "rknn-npu" if self.ready else "unavailable",
            "model_id": self._banner.get("model_id") or self._meta_summary.get("model_id"),
            "architecture": self._banner.get("architecture") or "URLDualBranchNPU (Prasad 2026 Lexical MLP + Char 1D-CNN)",
            "precision": self._banner.get("precision", "fp16"),
            "init_ms": self._banner.get("init_ms"),
            "inferences_completed": self._infer_count,
            "avg_npu_latency_ms": avg_ms,
            "last_npu_latency_ms": self._last_latency_ms,
            "load_percent": self.live_load_percent(),
            "calibration": self._meta_summary.get("calibration", {}),
            "test_metrics": (self._meta_summary.get("test_metrics") or {}).get("at_high_precision"),
            "paper": self._meta_summary.get("paper"),
        }

    def close(self) -> None:
        with self._lock:
            self._terminate_locked()


_singleton: URLDualNPUClient | None = None
_singleton_lock = threading.Lock()


def get_url_dual_npu_client() -> URLDualNPUClient:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = URLDualNPUClient()
        return _singleton
