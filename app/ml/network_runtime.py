"""Lazy, offline loader/scorer for the network v2r model. Never touches the network.

Returns None (caller falls back to the rule-only detector) when ML is disabled, the artefact is
missing/corrupt, or its feature list does not match the reducer's (a stale artefact would otherwise
score garbage silently).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from app.config import settings
from app.ml.flow_reducer import MODEL_FEATURES, FlowRecord

log = logging.getLogger("edgeshield.network_ml")
BUNDLE_FILE = "network_model_v2r.joblib"

_lock = threading.Lock()
_cache: dict[str, dict | None] = {}


def load_bundle(model_dir: str | None = None) -> dict | None:
    if not settings.ml_detectors:
        return None
    path = str((Path(model_dir or settings.model_dir) / BUNDLE_FILE).resolve())
    with _lock:
        if path in _cache:
            return _cache[path]
        bundle = None
        try:
            if Path(path).exists():
                import joblib
                b = joblib.load(path)
                if list(b.get("features", [])) == list(MODEL_FEATURES):
                    bundle = b
                else:
                    log.warning("network model %s feature list != flow_reducer.MODEL_FEATURES; "
                                "falling back to rules", path)
        except Exception as exc:  # corrupt file, sklearn version mismatch, ...
            log.warning("could not load network model %s (%s); falling back to rules", path, exc)
        _cache[path] = bundle
        return bundle


def reset_cache() -> None:
    with _lock:
        _cache.clear()


def score_records(bundle: dict, records: list[FlowRecord]) -> np.ndarray:
    """P(attack) per flow, one batched predict_proba call."""
    if not records:
        return np.zeros(0)
    X = np.asarray([r.vector() for r in records], dtype=np.float32)
    return bundle["model"].predict_proba(X)[:, 1]
