"""SCI: bounded self-correction workflow for EdgeShield's three input pipelines.

This module records detected faults and verification evidence. It does not
execute arbitrary commands or claim that an unobserved fault was detected.
"""
from __future__ import annotations
import hashlib, json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64

def canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

class SCIEngine:
    ALLOWED_ACTIONS = {
        "network": {"reparse_capture", "rebuild_flow_features", "use_validated_fallback", "mark_uncertain", "quarantine_detector", "reversible_containment_apply", "reversible_containment_undo"},
        "url": {"reparse_url", "recompute_url_features", "use_validated_fallback", "mark_uncertain", "quarantine_detector", "reversible_containment_apply", "reversible_containment_undo"},
        "sms": {"renormalise_message", "recompute_text_features", "use_validated_fallback", "mark_uncertain", "quarantine_detector"},
        "model": {"retry_validated_preprocessing", "use_validated_fallback", "mark_uncertain", "quarantine_model"},
        "soc": {"switch_to_cpu_fallback", "reduce_workload", "quarantine_accelerator", "request_hardware_inspection"},
        "runtime": {"restart_isolated_service", "rollback_software_version", "reject_invalid_input", "disable_faulty_backend"},
    }

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self.connect() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS sci_ledger (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT UNIQUE NOT NULL,
                record_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")

    def connect(self):
        return sqlite3.connect(self.db_path)

    def diagnose(self, component: str, symptom: str, evidence: dict[str, Any]):
        s = symptom.lower()
        if component == "network":
            cause = "pcap_or_flow_processing_issue"
            actions = ["reparse_capture", "rebuild_flow_features", "use_validated_fallback", "mark_uncertain"]
            if evidence.get("flow_schema_invalid"): cause = "flow_feature_schema_mismatch"
            if evidence.get("capture_truncated"): cause = "incomplete_or_truncated_capture"
        elif component == "url":
            cause = "url_parsing_or_feature_pipeline_issue"
            actions = ["reparse_url", "recompute_url_features", "use_validated_fallback", "mark_uncertain"]
            if evidence.get("normalisation_mismatch"): cause = "url_normalisation_mismatch"
        elif component == "sms":
            cause = "message_normalisation_or_feature_pipeline_issue"
            actions = ["renormalise_message", "recompute_text_features", "use_validated_fallback", "mark_uncertain"]
            if evidence.get("encoding_error"): cause = "message_encoding_or_decoding_error"
        elif component == "model":
            cause = "model_output_or_input_validation_failure"
            actions = ["retry_validated_preprocessing", "use_validated_fallback", "mark_uncertain", "quarantine_model"]
            if evidence.get("model_disagreement"): cause = "model_disagreement_or_uncertain_prediction"
            if evidence.get("model_artifact_integrity_failed"): cause = "model_artifact_integrity_failure"
        elif component == "soc":
            cause = "compute_resource_or_accelerator_fault"
            actions = ["switch_to_cpu_fallback", "reduce_workload", "quarantine_accelerator", "request_hardware_inspection"]
            if evidence.get("thermal_limit_exceeded"): cause = "thermal_limit_exceeded"
            if evidence.get("memory_pressure"): cause = "memory_pressure"
        else:
            cause = "software_pipeline_or_runtime_failure"
            actions = ["restart_isolated_service", "rollback_software_version", "reject_invalid_input", "disable_faulty_backend"]
            if evidence.get("integrity_check_failed") or "checksum" in s:
                cause = "integrity_or_data_corruption_suspected"
        return {
            "suspected_root_cause": cause,
            "candidate_actions": actions,
            "diagnosis_method": "deterministic_rules",
            "diagnosis_confidence": "not_calibrated",
            "requires_human_review": component in {"soc", "runtime"} or bool(evidence.get("safety_critical")),
        }

    def record_incident(self, component: str, symptom: str, evidence: dict,
                        correction_action: str, verification: dict,
                        actor: str = "local-system"):
        if component not in self.ALLOWED_ACTIONS:
            raise ValueError(f"component must be one of: {', '.join(self.ALLOWED_ACTIONS)}")
        if correction_action not in self.ALLOWED_ACTIONS[component]:
            raise ValueError("correction_action is not allow-listed for this component")
        diagnosis = self.diagnose(component, symptom, evidence)
        checks = verification.get("checks", [])
        all_passed = bool(checks) and all(isinstance(x, dict) and x.get("passed") is True for x in checks)
        any_failed = any(isinstance(x, dict) and x.get("passed") is False for x in checks)
        independent = verification.get("independent") is True
        status = "verified" if all_passed and independent else ("failed" if any_failed else "unverified")
        now = datetime.now(timezone.utc).isoformat()
        incident_id = "SCI-" + uuid.uuid4().hex[:12].upper()
        record = {
            "incident_id": incident_id, "timestamp": now, "component": component,
            "symptom": symptom, "evidence": evidence, "diagnosis": diagnosis,
            "correction": {"action": correction_action,
                           "execution_status": "recommendation_recorded_not_executed"},
            "verification": {"status": status, "checks": checks,
                             "independent_verification": independent},
            "actor": actor,
            "limitations": [
                "Correction action is recorded, not executed by this generic prototype.",
                "Verification depends on the quality and independence of supplied checks.",
                "Hash-chain integrity does not prove factual correctness.",
            ],
        }
        with self.connect() as c:
            previous = c.execute("SELECT record_hash FROM sci_ledger ORDER BY sequence DESC LIMIT 1").fetchone()
            previous_hash = previous[0] if previous else GENESIS
            digest = hashlib.sha256(previous_hash.encode() + canonical(record)).hexdigest()
            c.execute("INSERT INTO sci_ledger (incident_id,record_json,previous_hash,record_hash,created_at) VALUES (?,?,?,?,?)",
                      (incident_id, json.dumps(record), previous_hash, digest, now))
        return {**record, "previous_hash": previous_hash, "record_hash": digest}

    def list_incidents(self, limit=50):
        with self.connect() as c:
            rows = c.execute("SELECT record_json,previous_hash,record_hash FROM sci_ledger ORDER BY sequence DESC LIMIT ?",
                             (max(1, min(int(limit), 200)),)).fetchall()
        return [{**json.loads(r[0]), "previous_hash": r[1], "record_hash": r[2]} for r in rows]

    def verify_ledger(self):
        with self.connect() as c:
            rows = c.execute("SELECT incident_id,record_json,previous_hash,record_hash FROM sci_ledger ORDER BY sequence").fetchall()
        prev = GENESIS
        for incident_id, raw, stored_prev, stored_hash in rows:
            record = json.loads(raw)
            calculated = hashlib.sha256(prev.encode() + canonical(record)).hexdigest()
            if stored_prev != prev or stored_hash != calculated or record.get("incident_id") != incident_id:
                return {"valid": False, "failed_incident": incident_id}
            prev = stored_hash
        return {"valid": True, "records_checked": len(rows), "head_hash": prev}
