from typing import Any, Literal
from pydantic import BaseModel, Field


class SCIIncidentRequest(BaseModel):
    component: str
    symptom: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    correction_action: str
    verification: dict[str, Any] = Field(default_factory=dict)
    actor: str = "local-operator"


class SCIClosedLoopSubmitRequest(BaseModel):
    event_id: str
    claimed_verdict: Literal["false_positive", "false_negative", "correct"]
    claimed_label: Literal["benign", "malicious"]
    reviewer: str = Field(default="security-operator", min_length=1, max_length=80)
    independent_checks: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = Field(default="", max_length=2000)
    auto_deploy_if_validated: bool = False


class SCIProposalDeployRequest(BaseModel):
    actor: str = Field(default="security-operator", min_length=1, max_length=80)


class SCIRollbackRequest(BaseModel):
    target_version: str | None = None
    actor: str = Field(default="security-operator", min_length=1, max_length=80)
    reason: str = Field(default="operator_rollback", max_length=300)
