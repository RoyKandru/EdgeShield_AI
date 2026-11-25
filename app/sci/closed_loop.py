"""SCI Stages 6, 8 & 9: Closed-Loop Self-Learning, Controlled Deployment, Rollback & Continuous Monitoring."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.sci.diagnosis import diagnose_prediction_error
from app.sci.engine import SCIEngine
from app.sci.planner import ROOT_CAUSE_ACTION_MAP, enforce_non_llm_actor, plan_correction
from app.sci.policy import PolicyManager, apply_patch_to_policy, get_policy_manager
from app.sci.validator import validate_candidate_policy
from app.sci.verifier import OutcomeVerifier


class SCIClosedLoopController:
    def __init__(
        self,
        db_path: str,
        sci_engine: SCIEngine,
        policy_manager: PolicyManager | None = None,
    ) -> None:
        self.db_path = db_path
        self.sci = sci_engine
        self.policy_mgr = policy_manager or get_policy_manager()
        self.verifier = OutcomeVerifier(db_path)
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self) -> None:
        with self._connect() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS sci_proposals (
                proposal_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                outcome TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                action TEXT NOT NULL,
                execution_target TEXT NOT NULL,
                status TEXT NOT NULL,
                actor TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                validation_json TEXT NOT NULL,
                deployed_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS sci_retrain_queue (
                item_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                verified_label TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                raw_input TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS sci_version_monitor (
                policy_version TEXT PRIMARY KEY,
                decisions_count INTEGER NOT NULL DEFAULT 0,
                verified_fp INTEGER NOT NULL DEFAULT 0,
                verified_fn INTEGER NOT NULL DEFAULT 0,
                uncertain_count INTEGER NOT NULL DEFAULT 0,
                regression_alert INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )""")

    def _bump_monitor(self, version: str, field: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO sci_version_monitor (policy_version, updated_at) VALUES (?, ?)",
                (version, now),
            )
            if field in {"decisions_count", "verified_fp", "verified_fn", "uncertain_count"}:
                c.execute(
                    f"UPDATE sci_version_monitor SET {field} = {field} + 1, updated_at = ? WHERE policy_version = ?",
                    (now, version),
                )
            row = c.execute(
                "SELECT verified_fp, verified_fn FROM sci_version_monitor WHERE policy_version = ?",
                (version,),
            ).fetchone()
            if row and (row["verified_fp"] + row["verified_fn"]) >= 3:
                c.execute(
                    "UPDATE sci_version_monitor SET regression_alert = 1, updated_at = ? WHERE policy_version = ?",
                    (now, version),
                )

    def record_decision(self, policy_version: str | None = None) -> None:
        ver = policy_version or self.policy_mgr.get_active().get("version", "v1")
        self._bump_monitor(ver, "decisions_count")

    def submit_error_cycle(
        self,
        event: dict[str, Any],
        claimed_verdict: str,
        claimed_label: str,
        reviewer: str,
        independent_checks: list[dict[str, Any]] | None = None,
        notes: str = "",
        auto_deploy_if_validated: bool = False,
    ) -> dict[str, Any]:
        """Run Stages 3 -> 4 -> 5 -> 7 (and optionally controlled Stage 8) for a suspected error."""
        enforce_non_llm_actor(reviewer)
        active_ver = self.policy_mgr.get_active().get("version", "v1")
        rec_id = "FB-" + uuid.uuid4().hex[:10].upper()

        # Stage 3: Independent Offline Verification
        ver = self.verifier.verify(
            event=event,
            claimed_verdict=claimed_verdict,
            claimed_label=claimed_label,
            reviewer=reviewer,
            independent_checks=independent_checks,
            notes=notes,
        )

        if not ver["verified"]:
            self.verifier.store_unverified(
                rec_id, event, claimed_verdict, claimed_label, reviewer, notes,
                ver.get("poisoning_reason") or ver.get("reason", "unverified"),
            )
            self._bump_monitor(active_ver, "uncertain_count")
            # Audit poisoning attempts cleanly in immutable SHA-256 sci_ledger
            if ver.get("poisoning_blocked"):
                self.sci.record_incident(
                    component="runtime",
                    symptom="Blocked potential feedback poisoning attempt",
                    evidence={"event_id": event.get("event_id"), "reviewer": reviewer, "reason": ver["poisoning_reason"]},
                    correction_action="reject_invalid_input",
                    verification={"independent": True, "checks": [{"name": "anti_poisoning_guard", "passed": True}]},
                    actor=reviewer,
                )
            return {
                "stage_reached": "stage_3_outcome_verification",
                "verification": ver,
                "stored_in": "sci_unverified_feedback",
                "proposal": None,
                "deployed": False,
            }

        # Store independently verified outcome separately
        self.verifier.store_verified(
            rec_id, event, ver["outcome"], claimed_label, reviewer, ver["independent_signals"], ver["indicator"]
        )
        if ver["outcome"] == "VERIFIED_FP":
            self._bump_monitor(active_ver, "verified_fp")
        elif ver["outcome"] == "VERIFIED_FN":
            self._bump_monitor(active_ver, "verified_fn")
        else:
            return {
                "stage_reached": "stage_3_verified_no_error",
                "verification": ver,
                "stored_in": "sci_verified_feedback",
                "proposal": None,
                "deployed": False,
            }

        # Stage 4: Root-Cause Error Diagnosis
        src = str(event.get("source", ""))
        channel = "url" if "url" in src else "sms" if "sms" in src else "network"
        recent_errs = self.verifier.count_recent_verified_errors(channel)
        diag = diagnose_prediction_error(event, ver, recent_verified_errors_in_channel=recent_errs)

        # Stage 5: Explicit Deterministic Correction Planner
        plan = plan_correction(diag, actor=reviewer)

        # Queue for workstation retraining if planned
        if plan["retrain_model"]:
            with self._connect() as c:
                c.execute(
                    "INSERT INTO sci_retrain_queue VALUES (?,?,?,?,?,?,?)",
                    ("RT-" + uuid.uuid4().hex[:10].upper(), event.get("event_id", ""), channel,
                     claimed_label, diag["root_cause"], str(event.get("raw_input_preview", "")),
                     datetime.now(timezone.utc).isoformat()),
                )

        proposal_id = "PROP-" + uuid.uuid4().hex[:10].upper()
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as c:
            c.execute(
                "INSERT INTO sci_proposals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (proposal_id, event.get("event_id", ""), channel, ver["outcome"], diag["root_cause"],
                 plan["action"], plan["execution_target"], "planned", reviewer,
                 json.dumps(plan), json.dumps({}), "", now, now),
            )

        # Stage 7: Automated Validation Engine
        target_case = {
            "channel": channel,
            "input": event.get("metadata") if channel == "network" else event.get("raw_input_preview", ""),
            "expected_flagged": claimed_label == "malicious",
        }
        val_res = self.validate_proposal(proposal_id, target_case=target_case)

        deployed_info = None
        if auto_deploy_if_validated and val_res["passed"] and plan["action"] != "queue_workstation_retraining":
            deployed_info = self.deploy_proposal(proposal_id, actor=reviewer)

        return {
            "stage_reached": "stage_8_deployed" if deployed_info else "stage_7_validated",
            "verification": ver,
            "diagnosis": diag,
            "plan": plan,
            "proposal_id": proposal_id,
            "validation": val_res,
            "deployed": deployed_info is not None,
            "deployment": deployed_info,
        }

    def _get_proposal(self, proposal_id: str) -> dict[str, Any]:
        with self._connect() as c:
            row = c.execute("SELECT * FROM sci_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
        if not row:
            raise KeyError(f"Proposal '{proposal_id}' not found")
        d = dict(row)
        d["plan"] = json.loads(d.pop("plan_json"))
        d["validation"] = json.loads(d.pop("validation_json") or "{}")
        return d

    def validate_proposal(self, proposal_id: str, target_case: dict[str, Any] | None = None) -> dict[str, Any]:
        prop = self._get_proposal(proposal_id)
        plan = prop["plan"]
        cur_policy = self.policy_mgr.get_active()

        if plan["action"] == "queue_workstation_retraining":
            val = {
                "passed": True,
                "reasons": [],
                "target_resolved": True,
                "note": "Queued for dev-workstation retraining; no live on-device policy mutation required.",
            }
        else:
            cand_policy = apply_patch_to_policy(cur_policy, plan["patch"])
            val = validate_candidate_policy(cur_policy, cand_policy, target_case=target_case)

        new_status = "validated_pass" if val["passed"] else "validation_failed"
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as c:
            c.execute(
                "UPDATE sci_proposals SET status=?, validation_json=?, updated_at=? WHERE proposal_id=?",
                (new_status, json.dumps(val), now, proposal_id),
            )
        return val

    def deploy_proposal(self, proposal_id: str, actor: str = "operator") -> dict[str, Any]:
        enforce_non_llm_actor(actor)
        prop = self._get_proposal(proposal_id)
        val = prop.get("validation") or {}
        if not val.get("passed"):
            raise RuntimeError(f"Proposal {proposal_id} cannot be deployed without passing Stage 7 validation.")
        plan = prop["plan"]
        if plan["action"] == "queue_workstation_retraining":
            raise ValueError("Workstation retraining proposals are exported via /api/sci/retrain-pack, not hot-deployed.")

        cur_policy = self.policy_mgr.get_active()
        cand_policy = apply_patch_to_policy(cur_policy, plan["patch"])
        deployed = self.policy_mgr.deploy(
            cand_policy,
            actor=actor,
            reason=f"{proposal_id}: {prop['root_cause']} -> {plan['action']}",
            validation_summary={
                "f1": val.get("candidate", {}).get("f1"),
                "fpr": val.get("candidate", {}).get("fpr"),
                "critical_threat_recall": val.get("candidate", {}).get("critical_threat_recall"),
            },
        )
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as c:
            c.execute(
                "UPDATE sci_proposals SET status='deployed', deployed_version=?, updated_at=? WHERE proposal_id=?",
                (deployed["version"], now, proposal_id),
            )

        # Immutable SHA-256 audit entry in sci_ledger
        comp = prop["channel"] if prop["channel"] in {"url", "sms", "network"} else "model"
        allowed_map = {"url": "recompute_url_features", "sms": "recompute_text_features", "network": "rebuild_flow_features"}
        ledger_entry = self.sci.record_incident(
            component=comp,
            symptom=f"SCI Closed-Loop deployed {deployed['version']} for {prop['root_cause']} ({plan['action']})",
            evidence={"proposal_id": proposal_id, "policy_version": deployed["version"], "sha256": deployed["sha256"], "patch": plan["patch"]},
            correction_action=allowed_map.get(comp, "use_validated_fallback"),
            verification={"independent": True, "checks": [{"name": "golden_regression_suite", "passed": True, "metrics": deployed["validation_summary"]}]},
            actor=actor,
        )
        return {
            "proposal_id": proposal_id,
            "deployed_version": deployed["version"],
            "parent_version": deployed["parent_version"],
            "sha256": deployed["sha256"],
            "ledger_incident_id": ledger_entry["incident_id"],
            "ledger_hash": ledger_entry["record_hash"],
        }

    def rollback(self, target_version: str | None = None, actor: str = "operator", reason: str = "operator_rollback") -> dict[str, Any]:
        enforce_non_llm_actor(actor)
        res = self.policy_mgr.rollback(target_version=target_version, actor=actor, reason=reason)
        ledger = self.sci.record_incident(
            component="runtime",
            symptom=f"SCI policy rolled back from {res['rolled_back_from']} to {res['active_version']}",
            evidence=res,
            correction_action="rollback_software_version",
            verification={"independent": True, "checks": [{"name": "policy_sha256_verified", "passed": True}]},
            actor=actor,
        )
        return {**res, "ledger_incident_id": ledger["incident_id"], "ledger_hash": ledger["record_hash"]}

    def list_proposals(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as c:
            rows = c.execute(
                "SELECT * FROM sci_proposals ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["plan"] = json.loads(d.pop("plan_json"))
            d["validation"] = json.loads(d.pop("validation_json") or "{}")
            out.append(d)
        return out

    def export_retrain_pack(self) -> dict[str, Any]:
        with self._connect() as c:
            qrows = [dict(r) for r in c.execute("SELECT * FROM sci_retrain_queue ORDER BY created_at DESC").fetchall()]
            vrows = [dict(r) for r in c.execute("SELECT event_id, channel, outcome, verified_label, raw_input, indicator, created_at FROM sci_verified_feedback ORDER BY created_at DESC").fetchall()]
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "active_policy_version": self.policy_mgr.get_active().get("version"),
            "workstation_instructions": "Train & evaluate on dev workstation; export validated artifact & SHA-256 back to RK3588.",
            "retrain_queue_count": len(qrows),
            "verified_feedback_count": len(vrows),
            "retrain_queue": qrows,
            "verified_examples": vrows,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        payload["pack_sha256"] = digest
        return payload

    def monitor_status(self) -> list[dict[str, Any]]:
        with self._connect() as c:
            rows = c.execute("SELECT * FROM sci_version_monitor ORDER BY updated_at DESC LIMIT 20").fetchall()
        return [dict(r) for r in rows]

    def pipeline_status(self) -> dict[str, Any]:
        active = self.policy_mgr.get_active()
        return {
            "architecture": "SCI Evidence-Grounded Closed-Loop Self-Learning & Self-Correction Pipeline",
            "stages_implemented": [
                "1. Threat Detection (URL/SMS/Network + Live NIC)",
                "2. Decision & Evidence Logging (SQLite WAL + Policy Version)",
                "3. Independent Offline Outcome Verification (with UNCERTAIN + Anti-Poisoning)",
                "4. Root-Cause Error Diagnosis (7 Explicit Causes)",
                "5. Deterministic Correction Planner (LLM Policy-Write Locked)",
                "6. Bounded On-Device Correction Engine + Workstation Retrain Pack",
                "7. Automated Golden Regression Validation Engine",
                "8. Versioned Atomic Deployment & Rollback + SHA-256 Audit Ledger",
                "9. Continuous Post-Deployment Version Monitoring",
            ],
            "llm_policy_modification_allowed": False,
            "active_policy": {
                "version": active.get("version"),
                "parent_version": active.get("parent_version"),
                "sha256": active.get("sha256"),
                "verified_allow_domains_count": len(active.get("verified_allow_domains", [])),
                "verified_deny_domains_count": len(active.get("verified_deny_domains", [])),
                "verified_deny_indicators_count": len(active.get("verified_deny_indicators", [])),
                "rule_weights": active.get("rule_weights"),
                "thresholds": active.get("thresholds"),
            },
            "datasets": self.verifier.dataset_counts(),
            "policy_versions": self.policy_mgr.list_versions()[:10],
            "root_cause_action_map": {k: v["action"] for k, v in ROOT_CAUSE_ACTION_MAP.items()},
            "version_monitoring": self.monitor_status(),
            "proposals_recent": len(self.list_proposals(limit=50)),
        }
