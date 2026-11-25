from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high", "critical"]


class AnalyzeURLRequest(BaseModel):
    url: str = Field(min_length=3, max_length=4096)


class AnalyzeSMSRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)


class AnalyzeNetworkFlowRequest(BaseModel):
    src_ip: str = "0.0.0.0"
    dst_ip: str = "0.0.0.0"
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "TCP"
    packets: int = 0
    bytes: int = 0
    duration_seconds: float = 0.0
    syn_count: int = 0
    syn_packets: int = 0
    rst_packets: int = 0
    failed_connections: int = 0
    dns_query: str = ""
    payload_preview: str = ""


class EvidenceItem(BaseModel):
    code: str
    description: str
    weight: int = 0


class ThreatEvent(BaseModel):
    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    threat_category: str
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0.0, le=1.0)
    prediction: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    model_version: str
    raw_input_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdvisoryRequest(BaseModel):
    event_id: str
    language: str = "en"  # "en" | "hi" | "hinglish"


class SelectModelRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=255)


class AdvisoryResponse(BaseModel):
    event_id: str
    provider: str
    language: str = "en"
    explanation: str
    recommendations: list[str]
    generated_locally: bool = True
    runtime_note: str = ""
    validated: bool = True
    fallback_used: bool = False
    prompt_injection_guarded: bool = True
    validation_notes: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class SystemStatus(BaseModel):
    status: str
    application: str
    offline_mode: bool
    llama_provider: str
    llama_ready: bool
    database: str
    optional_network_capture: bool
    llama_model: str = ""
    llama_runtime: str = ""
    llama_load_ms: float | None = None
    live_monitor_running: bool = False
    live_monitor_interface: str = ""
    api_auth_enabled: bool = False
    detector_versions: dict[str, str] = Field(default_factory=dict)


class ContainmentRequest(BaseModel):
    event_id: str = ""
    action: str = ""  # sinkhole_domain | drop_ip | quarantine_mac (auto-inferred if empty + event_id set)
    target: str = ""
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    actor: str = "soc-operator"
    reason: str = "high_risk_threat_containment"
    force: bool = False


class LinkSandboxRequest(BaseModel):
    url: str = Field(min_length=3, max_length=4096)
    html_content: str | None = None
    allow_network: bool = False
