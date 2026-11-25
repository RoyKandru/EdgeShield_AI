"""One RKNNLite context per detector, pinned to a requested RK3588 NPU core.

This is an opt-in hardware adapter. Missing SDK/model/driver degrades to unavailable; it never
silently reports NPU inference when CPU fallback was used.
"""
from __future__ import annotations
import logging, threading, time
from pathlib import Path
from typing import Any
import numpy as np

log = logging.getLogger("edgeshield.rknn")
CORE_NAMES = {0: "NPU_CORE_0", 1: "NPU_CORE_1", 2: "NPU_CORE_2"}

class RKNNModel:
    def __init__(self, pipeline: str, model_path: str, core_id: int, input_shape: tuple[int, ...]):
        if core_id not in (0, 1, 2):
            raise ValueError("RK3588 NPU core_id must be 0, 1 or 2")
        self.pipeline, self.path, self.core_id = pipeline, Path(model_path), core_id
        self.input_shape = input_shape
        self.runtime = None
        self.lock = threading.Lock()
        self.error: str | None = None
        self._init()

    def _init(self):
        try:
            from rknnlite.api import RKNNLite
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self.runtime = RKNNLite()
            rc = self.runtime.load_rknn(str(self.path))
            if rc != 0:
                raise RuntimeError(f"load_rknn returned {rc}")
            masks = {0: RKNNLite.NPU_CORE_0, 1: RKNNLite.NPU_CORE_1, 2: RKNNLite.NPU_CORE_2}
            rc = self.runtime.init_runtime(core_mask=masks[self.core_id])
            if rc != 0:
                raise RuntimeError(f"init_runtime returned {rc}")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.runtime = None
            log.warning("RKNN %s unavailable: %s", self.pipeline, self.error)

    @property
    def ready(self) -> bool:
        return self.runtime is not None

    def infer(self, tensor: np.ndarray) -> tuple[float, float]:
        if not self.runtime:
            raise RuntimeError(self.error or "RKNN runtime unavailable")
        x = np.asarray(tensor, dtype=np.float32)
        # Model exports use fixed batch-1 NCHW tensors. RKNNLite expects a list of input arrays.
        if x.ndim == len(self.input_shape):
            x = np.expand_dims(x, 0)
        if tuple(x.shape) != (1, *self.input_shape):
            raise ValueError(f"{self.pipeline}: expected {(1, *self.input_shape)}, got {x.shape}")
        start = time.perf_counter()
        with self.lock:
            outputs = self.runtime.inference(inputs=[x])
        elapsed = (time.perf_counter() - start) * 1000.0
        if not outputs:
            raise RuntimeError("RKNN returned no outputs")
        arr = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if arr.size != 1:
            raise RuntimeError(f"Expected one logit, received output shape {np.asarray(outputs[0]).shape}")
        # Stable sigmoid for a single logit.
        logit = float(np.clip(arr[0], -40.0, 40.0))
        score = 1.0 / (1.0 + np.exp(-logit))
        return float(score), elapsed

    def close(self):
        if self.runtime:
            try: self.runtime.release()
            finally: self.runtime = None

class RKNNPool:
    def __init__(self, specs: dict[str, dict[str, Any]]):
        self.models: dict[str, RKNNModel] = {}
        for name, spec in specs.items():
            self.models[name] = RKNNModel(name, spec["artifact"], int(spec["core_id"]), tuple(spec["input_shape"]))

    def status(self):
        return {name: {"ready": m.ready, "core_id": m.core_id, "core": CORE_NAMES[m.core_id],
                       "artifact": str(m.path), "error": m.error,
                       "backend": "rknn-npu" if m.ready else "unavailable"}
                for name, m in self.models.items()}

    def infer(self, pipeline: str, tensor: np.ndarray, threshold: float = 0.5):
        from app.edge_models.contracts import InferenceResult
        model = self.models[pipeline]
        score, latency = model.infer(tensor)
        return InferenceResult(pipeline, score, score >= threshold, "rknn-npu", model.core_id,
                               latency, "candidate-int8-v1", False)

    def close(self):
        for model in self.models.values(): model.close()
