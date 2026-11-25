"""SCI Stage 5: Explicit Deterministic Root-Cause -> Correction Action Planner.

Implements the exact 7-row mapping table from Section 4 of the SCI Architecture spec.
Hard security rule: LLMs (Llama/ExecuTorch) are NEVER permitted to create, approve,
or deploy security policy or model updates.
"""
from __future__ import annotations

from typing import Any

FORBIDDEN_LLM_ACTORS = ("llama", "llm", "executorch", "advisory", "ai-advisor", "gpt", "gemini")

ROOT_CAUSE_ACTION_MAP: dict[str, dict[str, Any]] = {
    "missing_domain_reputation": {
        "action": "update_verified_domain_intelligence",
        "execution_target": "rk3588_on_device",
        "retrain_model": False,
        "description": "Update local verified domain intelligence (allow/deny list) without retraining.",
    },
    "incorrect_malicious_domain_classification": {
        "action": "update_verified_threat_indicators",
        "execution_target": "rk3588_on_device",
        "retrain_model": False,
        "description": "Correct verified threat/benign indicator classification in local intelligence.",
    },
    "excessive_feature_weighting": {
        "action": "adjust_risk_weights",
        "execution_target": "rk3588_on_device",
        "retrain_model": False,
        "description": "Adjust over-weighted deterministic rule weight within safe bounds and validate.",
    },
    "incorrect_url_feature_extraction": {
        "action": "fix_preprocessing_and_feature_extraction",
        "execution_target": "rk3588_on_device+engineering",
        "retrain_model": False,
        "description": "Enable/verify URL normalization preprocessing flags and log engineering patch record.",
    },
    "poor_probability_calibration": {
        "action": "recalibrate_thresholds",
        "execution_target": "rk3588_on_device",
        "retrain_model": False,
        "description": "Recalibrate decision probability threshold using verified validation set.",
    },
    "repeated_model_misclassification": {
        "action": "queue_workstation_retraining",
        "execution_target": "dev_workstation",
        "retrain_model": True,
        "description": "Queue verified labelled examples into workstation retrain pack (never train heavy models on RK3588).",
    },
    "new_phishing_technique": {
        "action": "extract_indicators_and_queue_retrain",
        "execution_target": "rk3588_on_device+dev_workstation",
        "retrain_model": True,
        "description": "Extract verified IOC/indicator into on-device detection logic AND add representative example to workstation retrain queue.",
    },
}


def enforce_non_llm_actor(actor: str) -> None:
    low = (actor or "").strip().lower()
    if any(tok in low for tok in FORBIDDEN_LLM_ACTORS):
        raise PermissionError(
            f"Security policy violation: LLM actor '{actor}' is strictly prohibited from modifying SCI policies or deploying updates."
        )


def plan_correction(diagnosis: dict[str, Any], actor: str) -> dict[str, Any]:
    enforce_non_llm_actor(actor)
    cause = str(diagnosis.get("root_cause", ""))
    if cause not in ROOT_CAUSE_ACTION_MAP:
        raise ValueError(f"Root cause '{cause}' has no mapped corrective action.")

    spec = ROOT_CAUSE_ACTION_MAP[cause]
    details = diagnosis.get("details") or {}
    channel = diagnosis.get("channel", "url")
    action = spec["action"]

    patch: dict[str, Any] = {"action": action, "channel": channel}
    if action == "update_verified_domain_intelligence":
        patch["domain"] = details.get("domain", "")
        patch["disposition"] = details.get("disposition", "deny")
    elif action == "update_verified_threat_indicators":
        patch["domain"] = details.get("domain", "")
        patch["indicator"] = details.get("indicator", "")
        patch["disposition"] = details.get("disposition", "deny")
    elif action == "adjust_risk_weights":
        patch["rule_code"] = details.get("rule_code", "")
        patch["old_weight"] = details.get("current_weight")
        patch["new_weight"] = details.get("suggested_weight", 15)
    elif action == "fix_preprocessing_and_feature_extraction":
        patch["unicode_normalization"] = True
        patch["embedded_url_handoff"] = True
    elif action == "recalibrate_thresholds":
        patch["threshold_key"] = details.get("threshold_key", "url_ml_threshold")
        patch["old_value"] = details.get("current_threshold")
        patch["new_value"] = details.get("suggested_threshold", 0.5)
    elif action == "extract_indicators_and_queue_retrain":
        patch["domain"] = details.get("domain", "")
        patch["indicator"] = details.get("indicator", "")
        patch["raw_input"] = details.get("raw_input", "")
    elif action == "queue_workstation_retraining":
        patch["raw_input"] = details.get("raw_input", "")

    return {
        "root_cause": cause,
        "action": action,
        "execution_target": spec["execution_target"],
        "retrain_model": spec["retrain_model"],
        "description": spec["description"],
        "patch": patch,
    }
