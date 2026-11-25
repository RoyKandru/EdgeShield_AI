"""Versioned Dynamic SCI Policy & Local Verified Threat Intelligence Store (with Rollback).

Supports:
  - Atomic versioned policy activation (v1, v1, ...) with SHA-256 checksums
  - 1-click rollback to any prior working version
  - Thread-local `policy_override_context(candidate)` so the Validation Engine can safely
    replay any proposed patch through the real detectors without affecting live traffic
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIR = _ROOT / "data" / "sci_policy"

WEIGHT_MIN = 5
WEIGHT_MAX = 65
THRESHOLD_CLAMPS = {
    "url_ml_threshold": (0.25, 0.90),
    "url_npu_high_threshold": (0.50, 0.99),
    "url_npu_cost_threshold": (0.20, 0.90),
    "sms_ml_threshold": (0.55, 0.96),
    "sms_bigru_threshold": (0.15, 0.90),
    "sms_bigru_high_threshold": (0.35, 0.95),
    "sms_flag_score": (20, 60),
    "network_ml_threshold": (0.35, 0.95),
    "network_npu_threshold": (0.35, 0.98),
}


def default_policy() -> dict[str, Any]:
    return {
        "version": "v1",
        "parent_version": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actor": "system-bootstrap",
        "reason": "Initial verified RK3588J EdgeShield baseline policy",
        "verified_allow_domains": [],
        "verified_deny_domains": [],
        "verified_deny_indicators": [],
        "verified_benign_phrases": [],
        "rule_weights": {
            "url": {},
            "sms": {},
            "network": {},
        },
        "thresholds": {
            "url_ml_threshold": None,
            "url_npu_high_threshold": 0.96,
            "url_npu_cost_threshold": 0.70,
            "sms_ml_threshold": 0.85,
            "sms_bigru_threshold": 0.22,
            "sms_bigru_high_threshold": 0.56,
            "sms_flag_score": 45,
            "network_ml_threshold": None,
            "network_npu_threshold": 0.80,
        },
        "preprocessing": {
            "unicode_normalization": True,
            "embedded_url_handoff": True,
        },
        "validation_summary": {"status": "baseline"},
    }


def policy_sha256(policy: dict[str, Any]) -> str:
    clean = {k: v for k, v in policy.items() if k != "sha256"}
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class PolicyManager:
    def __init__(self, policy_dir: str | Path = DEFAULT_POLICY_DIR) -> None:
        self.dir = Path(policy_dir)
        self.versions_dir = self.dir / "versions"
        self.active_path = self.dir / "active.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cached: dict[str, Any] | None = None
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        with self._lock:
            if not self.active_path.is_file():
                base = default_policy()
                base["sha256"] = policy_sha256(base)
                self._write_version_locked(base)
                self._write_active_locked(base)
            self._cached = json.loads(self.active_path.read_text(encoding="utf-8"))

    def _write_version_locked(self, pol: dict[str, Any]) -> None:
        vp = self.versions_dir / f"{pol['version']}.json"
        vp.write_text(json.dumps(pol, indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_active_locked(self, pol: dict[str, Any]) -> None:
        tmp = self.dir / "active.json.tmp"
        tmp.write_text(json.dumps(pol, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.active_path)
        self._cached = copy.deepcopy(pol)

    def get_active(self) -> dict[str, Any]:
        override = getattr(_TLS, "policy_override", None)
        if override is not None:
            return override
        with self._lock:
            if self._cached is None:
                self._ensure_initialized()
            assert self._cached is not None
            return copy.deepcopy(self._cached)

    def next_version(self) -> str:
        with self._lock:
            existing = sorted(p.stem for p in self.versions_dir.glob("v*.json"))
            max_patch = 0
            for v in existing:
                parts = v.lstrip("v").split(".")
                if len(parts) == 3 and parts[2].isdigit():
                    max_patch = max(max_patch, int(parts[2]))
            return f"v1.{max_patch + 1}"

    def list_versions(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for p in sorted(self.versions_dir.glob("v*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    out.append({
                        "version": d.get("version"),
                        "parent_version": d.get("parent_version"),
                        "created_at": d.get("created_at"),
                        "actor": d.get("actor"),
                        "reason": d.get("reason"),
                        "sha256": d.get("sha256"),
                        "active": self._cached is not None and d.get("version") == self._cached.get("version"),
                        "validation_summary": d.get("validation_summary"),
                    })
                except Exception:
                    pass
            return out

    def load_version(self, version: str) -> dict[str, Any]:
        vp = self.versions_dir / f"{version}.json"
        if not vp.is_file():
            raise FileNotFoundError(f"Policy version '{version}' not found")
        return json.loads(vp.read_text(encoding="utf-8"))

    def deploy(self, candidate: dict[str, Any], actor: str, reason: str, validation_summary: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            cur = self.get_active()
            pol = copy.deepcopy(candidate)
            pol["version"] = self.next_version()
            pol["parent_version"] = cur.get("version")
            pol["created_at"] = datetime.now(timezone.utc).isoformat()
            pol["actor"] = actor
            pol["reason"] = reason
            pol["validation_summary"] = validation_summary
            pol["sha256"] = policy_sha256(pol)
            self._write_version_locked(pol)
            self._write_active_locked(pol)
            return copy.deepcopy(pol)

    def rollback(self, target_version: str | None = None, actor: str = "operator", reason: str = "manual_rollback") -> dict[str, Any]:
        with self._lock:
            cur = self.get_active()
            target = target_version or cur.get("parent_version")
            if not target:
                raise ValueError("No previous policy version available to roll back to")
            prev = self.load_version(target)
            # Verify integrity hash before activating rollback target
            expected_sha = prev.get("sha256")
            if expected_sha and policy_sha256(prev) != expected_sha:
                raise RuntimeError(f"Policy integrity check failed for {target}")
            self._write_active_locked(prev)
            return {
                "rolled_back_from": cur.get("version"),
                "active_version": prev.get("version"),
                "sha256": prev.get("sha256"),
                "actor": actor,
                "reason": reason,
            }


_TLS = threading.local()
_GLOBAL_LOCK = threading.Lock()
_GLOBAL_MANAGER: PolicyManager | None = None


def get_policy_manager(policy_dir: str | Path | None = None) -> PolicyManager:
    global _GLOBAL_MANAGER
    if policy_dir is not None:
        return PolicyManager(policy_dir)
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = PolicyManager(DEFAULT_POLICY_DIR)
        return _GLOBAL_MANAGER


def get_active_policy() -> dict[str, Any]:
    override = getattr(_TLS, "policy_override", None)
    if override is not None:
        return override
    return get_policy_manager().get_active()


@contextmanager
def policy_override_context(candidate_policy: dict[str, Any]):
    """Isolate candidate policy evaluation to the current thread without touching live policy."""
    prev = getattr(_TLS, "policy_override", None)
    _TLS.policy_override = candidate_policy
    try:
        yield
    finally:
        _TLS.policy_override = prev


def apply_patch_to_policy(base_policy: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Deterministically apply a bounded Correction Planner patch to a copy of base_policy."""
    pol = copy.deepcopy(base_policy)
    action = patch.get("action", "")

    if action == "update_verified_domain_intelligence":
        dom = str(patch.get("domain", "")).strip().lower()
        disposition = patch.get("disposition", "deny")
        if not dom or "." not in dom:
            raise ValueError("Valid domain is required for domain intelligence update")
        if disposition == "allow":
            pol["verified_deny_domains"] = [d for d in pol["verified_deny_domains"] if d != dom]
            if dom not in pol["verified_allow_domains"]:
                pol["verified_allow_domains"].append(dom)
        else:
            pol["verified_allow_domains"] = [d for d in pol["verified_allow_domains"] if d != dom]
            if dom not in pol["verified_deny_domains"]:
                pol["verified_deny_domains"].append(dom)

    elif action == "update_verified_threat_indicators":
        dom = str(patch.get("domain", "")).strip().lower()
        ind = str(patch.get("indicator", "")).strip().lower()
        disposition = patch.get("disposition", "deny")
        if dom and "." in dom:
            if disposition == "deny":
                pol["verified_allow_domains"] = [d for d in pol["verified_allow_domains"] if d != dom]
                if dom not in pol["verified_deny_domains"]:
                    pol["verified_deny_domains"].append(dom)
            else:
                pol["verified_deny_domains"] = [d for d in pol["verified_deny_domains"] if d != dom]
                if dom not in pol["verified_allow_domains"]:
                    pol["verified_allow_domains"].append(dom)
        if ind and len(ind) >= 6:
            if disposition == "deny" and ind not in pol["verified_deny_indicators"]:
                pol["verified_deny_indicators"].append(ind)
            elif disposition == "allow" and ind not in pol["verified_benign_phrases"]:
                pol["verified_benign_phrases"].append(ind)

    elif action == "adjust_risk_weights":
        channel = str(patch.get("channel", "url"))
        if channel not in ("url", "sms", "network"):
            raise ValueError(f"Unsupported channel for weight adjustment: {channel}")
        rule_code = str(patch.get("rule_code", "")).strip()
        new_weight = int(patch.get("new_weight", 20))
        clamped = max(WEIGHT_MIN, min(WEIGHT_MAX, new_weight))
        if not rule_code:
            raise ValueError("rule_code is required for weight adjustment")
        pol["rule_weights"].setdefault(channel, {})[rule_code] = clamped

    elif action == "fix_preprocessing_and_feature_extraction":
        pre = pol.setdefault("preprocessing", {})
        pre["unicode_normalization"] = True
        pre["embedded_url_handoff"] = True

    elif action == "recalibrate_thresholds":
        tkey = str(patch.get("threshold_key", ""))
        if tkey not in THRESHOLD_CLAMPS:
            raise ValueError(f"Unknown threshold key: {tkey}")
        lo, hi = THRESHOLD_CLAMPS[tkey]
        val = float(patch.get("new_value"))
        clamped = max(lo, min(hi, val))
        pol["thresholds"][tkey] = int(round(clamped)) if tkey == "sms_flag_score" else round(clamped, 4)

    elif action == "extract_indicators_and_queue_retrain":
        dom = str(patch.get("domain", "")).strip().lower()
        ind = str(patch.get("indicator", "")).strip().lower()
        if dom and "." in dom and dom not in pol["verified_deny_domains"]:
            pol["verified_deny_domains"].append(dom)
        if ind and len(ind) >= 6 and ind not in pol["verified_deny_indicators"]:
            pol["verified_deny_indicators"].append(ind)

    elif action == "queue_workstation_retraining":
        # Policy weights unchanged on-device; verified example queued for workstation retrain pack
        pass
    else:
        raise ValueError(f"Unsupported correction action: {action}")

    return pol
