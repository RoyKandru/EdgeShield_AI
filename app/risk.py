import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.schemas import EvidenceItem


def calculate_risk(evidence: list[EvidenceItem], base_score: int = 0) -> int:
    """Deterministic, explainable score."""
    score = base_score + sum(max(0, min(item.weight, 45)) for item in evidence)
    return max(0, min(100, score))


def risk_level(score: int) -> str:
    if score >= 85:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


_URL_RE = re.compile(r'https?://[^\s<>"\']+|(?:[a-z0-9-]+\.)+(?:com|in|org|net|top|xyz|zip|dev|app|ly|gy|co|io|click|cfd|sbs)(?:/[^\s<>"\']*)?', re.I)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def extract_indicators(event: dict[str, Any]) -> list[str]:
    indicators: list[str] = []
    preview = str(event.get("raw_input_preview", ""))
    meta = event.get("metadata") or {}
    for key in ("hostname", "registrable_domain", "src_ip", "dst_ip"):
        v = str(meta.get(key, "") or "").strip().lower()
        if v and v not in {"127.0.0.1", "0.0.0.0", "localhost"}:
            indicators.append(v)
    for m in _URL_RE.findall(preview):
        u = m if m.lower().startswith(("http://", "https://")) else "http://" + m
        try:
            h = (urlparse(u).hostname or "").lower()
            if h:
                indicators.append(h)
        except Exception:
            pass
    for ip in _IP_RE.findall(preview):
        if not ip.startswith(("127.", "0.")):
            indicators.append(ip)
    # Deduplicate preserving order
    seen = set()
    out = []
    for ind in indicators:
        if ind not in seen:
            seen.add(ind)
            out.append(ind)
    return out


class CrossChannelCorrelator:
    """Correlates SMS -> embedded URL -> Network traffic events sharing an indicator within 10 min."""

    def __init__(self, window_s: float = 600.0) -> None:
        self.window_s = window_s
        # indicator -> list of (ts_epoch, source, event_id, risk_score)
        self._seen: dict[str, list[tuple[float, str, str, int]]] = defaultdict(list)

    def ingest(self, event: dict[str, Any]) -> dict[str, Any] | None:
        risk = int(event.get("risk_score", 0))
        if risk < 30:
            return None
        now = time.time()
        src = str(event.get("source", "unknown"))
        eid = str(event.get("event_id", ""))
        indicators = extract_indicators(event)

        best_corr: dict[str, Any] | None = None
        for ind in indicators:
            bucket = self._seen[ind]
            bucket[:] = [row for row in bucket if now - row[0] <= self.window_s]
            bucket.append((now, src, eid, risk))
            distinct_sources = sorted({row[1] for row in bucket})
            if len(distinct_sources) >= 2:
                fused_score = min(100, max(row[3] for row in bucket) + 15 * (len(distinct_sources) - 1))
                level = "critical" if fused_score >= 80 else "high" if fused_score >= 60 else "medium"
                event_ids = [row[2] for row in bucket[-8:]]
                best_corr = {
                    "correlation_id": f"corr-{uuid.uuid4().hex[:10]}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "indicator": ind,
                    "channels": distinct_sources,
                    "risk_level": level,
                    "risk_score": fused_score,
                    "summary": (
                        f"Cross-channel kill-chain detected across {' + '.join(distinct_sources)} "
                        f"sharing indicator '{ind}' ({len(event_ids)} linked events)."
                    ),
                    "event_ids": event_ids,
                }
        return best_corr
