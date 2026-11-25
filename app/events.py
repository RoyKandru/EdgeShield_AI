from datetime import datetime, timezone
from uuid import uuid4
from app.schemas import ThreatEvent, EvidenceItem
from app.risk import calculate_risk, risk_level


def create_event(
    source: str,
    category: str,
    prediction: str,
    confidence: float,
    evidence: list[EvidenceItem],
    actions: list[str],
    model_version: str,
    raw_input: str,
    metadata: dict | None = None,
    base_score: int = 0,
) -> ThreatEvent:
    score = calculate_risk(evidence, base_score)
    return ThreatEvent(
        event_id=f"ES-{uuid4().hex[:12].upper()}",
        timestamp=datetime.now(timezone.utc),
        source=source,
        threat_category=category,
        risk_level=risk_level(score),
        risk_score=score,
        confidence=max(0.0, min(1.0, confidence)),
        prediction=prediction,
        evidence=evidence,
        recommended_actions=actions,
        model_version=model_version,
        raw_input_preview=raw_input[:180],
        metadata=metadata or {},
    )
