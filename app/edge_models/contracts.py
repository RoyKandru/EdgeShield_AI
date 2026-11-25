from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal

Pipeline = Literal["network", "url", "sms"]

@dataclass(frozen=True)
class ModelSpec:
    pipeline: Pipeline
    artifact: str
    input_name: str
    input_shape: tuple[int, ...]
    threshold: float = 0.5
    core_id: int = 0
    version: str = "candidate-int8-v1"

    def to_dict(self):
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        return d

@dataclass(frozen=True)
class InferenceResult:
    pipeline: str
    score: float
    flagged: bool
    backend: str
    core_id: int | None
    latency_ms: float
    model_version: str
    calibrated_probability: bool = False
    note: str = "Raw model score; calibrate threshold on held-out validation data."

