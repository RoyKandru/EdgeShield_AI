"""
Runtime scorer for the URL ML model (models/url_model_v2.joblib).

Mirrors exactly what scripts/url_improve.py evaluated (reports/URL_IMPROVED.md):
    s    = mean(predict_proba) over the bundle's models, fed normalise_url(raw)
    flag = (s >= threshold_hybrid) AND registrable_domain(host) NOT in allowlist

Everything needed at inference ships inside the joblib bundle (models, threshold,
Tranco allowlist, Public Suffix List rules) - no network access, no data/external files.

Failure policy: if the artefact is missing, unreadable, malformed or scoring raises,
return None and let the caller fall back to the rule-based detector. Never raises.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from app.ml import features as F
from app.ml.url_features_v2 import registrable_domain

log = logging.getLogger("edgeshield.ml.url")

ARTEFACT = "url_model_v2.joblib"
MODEL_VERSION = "url-hybrid-v2"
_REQUIRED = ("models", "threshold_hybrid", "allowlist", "psl_rules")


@dataclass(frozen=True)
class UrlMl:
    score: float
    threshold: float
    allowlisted: bool
    flagged: bool          # score >= threshold and not allowlisted (the evaluated hybrid flag)
    domain: str


class _Scorer:
    def __init__(self, bundle: dict):
        self.models = list(bundle["models"])
        self.threshold = float(bundle["threshold_hybrid"])
        self.allow = frozenset(bundle["allowlist"])
        self.psl = frozenset(bundle["psl_rules"])
        self.trained = bundle.get("trained", "")

    def score(self, raw_url: str) -> UrlMl:
        norm = F.normalise_url(raw_url)
        s = sum(float(m.predict_proba([norm])[0, 1]) for m in self.models) / len(self.models)
        dom = registrable_domain(F.url_host(norm), self.psl)
        allowlisted = dom in self.allow
        return UrlMl(score=s, threshold=self.threshold, allowlisted=allowlisted,
                     flagged=(s >= self.threshold) and not allowlisted, domain=dom)


_lock = threading.Lock()
_cache: dict[str, _Scorer | None] = {}


def reset_cache() -> None:
    with _lock:
        _cache.clear()


def _load(path: Path) -> _Scorer | None:
    key = str(path)
    with _lock:
        if key in _cache:
            return _cache[key]
        scorer = None
        try:
            if not path.is_file():
                log.warning("URL ML model not found at %s; using rules only", path)
            else:
                import joblib  # lazy: keeps rules-only deployments light
                bundle = joblib.load(path)
                missing = [k for k in _REQUIRED if k not in bundle]
                if missing:
                    raise ValueError(f"bundle missing keys {missing}")
                scorer = _Scorer(bundle)
                log.info("URL ML model loaded from %s (trained %s)", path, scorer.trained)
        except Exception as exc:  # noqa: BLE001 - any failure must degrade, not crash
            log.warning("URL ML model unavailable (%s: %s); using rules only", type(exc).__name__, exc)
            scorer = None
        _cache[key] = scorer
        return scorer


def score_url(raw_url: str, model_dir: str | None = None, enabled: bool | None = None) -> UrlMl | None:
    """Return the ML result, or None when ML is disabled/unavailable/failed."""
    try:
        from app.config import settings
        if enabled is None:
            enabled = settings.ml_detectors
        if not enabled:
            return None
        scorer = _load(Path(model_dir or settings.model_dir) / ARTEFACT)
        if scorer is None:
            return None
        return scorer.score(raw_url)
    except Exception as exc:  # noqa: BLE001
        log.warning("URL ML scoring failed (%s: %s); using rules only", type(exc).__name__, exc)
        return None
