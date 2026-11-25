"""SCI Stage 4: Prediction-Error Root-Cause Diagnosis Engine.

Maps verified False Positives (VERIFIED_FP) and False Negatives (VERIFIED_FN) to the exact
7 root causes defined in Section 4 of the SCI Architecture specification.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from app.sci.policy import get_active_policy


def _extract_domain(event: dict[str, Any]) -> str:
    meta = event.get("metadata") or {}
    host = str(meta.get("hostname", "") or meta.get("registrable_domain", "") or "").lower().strip()
    if host:
        return host
    raw = str(event.get("raw_input_preview", ""))
    m = re.search(r"https?://[^\s<>\"']+", raw, re.I)
    if m:
        try:
            return (urlparse(m.group(0)).hostname or "").lower()
        except Exception:
            pass
    return ""


def _extract_novel_indicator(raw: str, domain: str) -> str:
    if domain:
        return domain
    # Extract longest informative phrase (4-6 words) for SMS/text indicator
    cleaned = re.sub(r"https?://\S+", "", raw or "").strip()
    words = [w for w in re.findall(r"[A-Za-z0-9\u0900-\u097F*._-]{3,}", cleaned) if len(w) >= 3]
    return " ".join(words[:5]).lower() if words else (raw or "")[:40].lower()


def diagnose_prediction_error(
    event: dict[str, Any],
    verification: dict[str, Any],
    recent_verified_errors_in_channel: int = 0,
) -> dict[str, Any]:
    """Classify the exact root cause of a VERIFIED_FP or VERIFIED_FN."""
    outcome = verification.get("outcome", "UNCERTAIN")
    if outcome not in {"VERIFIED_FP", "VERIFIED_FN"}:
        return {
            "root_cause": "no_correction_needed",
            "channel": "none",
            "rationale": f"Outcome is {outcome}; no detection error correction required.",
            "details": {},
        }

    src = str(event.get("source", ""))
    channel = "url" if "url" in src else "sms" if "sms" in src else "network"
    meta = event.get("metadata") or {}
    evidence = event.get("evidence") or []
    ev_codes = [str(e.get("code", "")) for e in evidence if isinstance(e, dict) and int(e.get("weight", 0)) > 0]
    ml_score = meta.get("ml_score")
    ml_thr = meta.get("ml_threshold")
    raw = str(event.get("raw_input_preview", ""))
    domain = _extract_domain(event)
    policy = get_active_policy()

    # 1. Incorrect URL feature extraction / preprocessing anomaly
    if channel == "url" and (
        "INVALID_HOST" in ev_codes
        or "hxxp" in raw.lower()
        or bool(re.search(r"[\u200b-\u200f\u2044\u2215]", raw))
        or (not domain and "." in raw)
    ):
        return {
            "root_cause": "incorrect_url_feature_extraction",
            "channel": channel,
            "rationale": "URL parsing/normalization produced malformed host or unnormalized confusable/obfuscated tokens.",
            "details": {"domain": domain, "evidence_codes": ev_codes},
        }

    # 2. Incorrect malicious-domain classification (allowlist bypass or bad deny entry)
    if domain and (
        (outcome == "VERIFIED_FN" and (meta.get("ml_allowlisted") is True or domain in policy.get("verified_allow_domains", [])))
        or (outcome == "VERIFIED_FP" and domain in policy.get("verified_deny_domains", []))
    ):
        return {
            "root_cause": "incorrect_malicious_domain_classification",
            "channel": channel,
            "rationale": f"Domain '{domain}' was misclassified by current allow/deny domain intelligence.",
            "details": {"domain": domain, "disposition": "deny" if outcome == "VERIFIED_FN" else "allow"},
        }

    # 3. Excessive feature / rule weighting (VERIFIED_FP where rule fired despite low ML probability)
    if outcome == "VERIFIED_FP" and ev_codes:
        non_ml_codes = [c for c in ev_codes if not c.startswith("ML_")]
        if non_ml_codes and (ml_score is None or float(ml_score) < 0.55):
            top_ev = max(
                [e for e in evidence if isinstance(e, dict) and str(e.get("code", "")) in non_ml_codes],
                key=lambda x: int(x.get("weight", 0)),
            )
            return {
                "root_cause": "excessive_feature_weighting",
                "channel": channel,
                "rationale": f"Rule '{top_ev['code']}' (weight={top_ev['weight']}) over-penalized a verified benign input while ML score was low ({ml_score}).",
                "details": {
                    "rule_code": top_ev["code"],
                    "current_weight": int(top_ev["weight"]),
                    "suggested_weight": max(10, int(top_ev["weight"]) - 15),
                    "domain": domain,
                },
            }

    # 4. Poor probability calibration (borderline ML score within +/-0.12 of threshold)
    if ml_score is not None and ml_thr is not None:
        diff = abs(float(ml_score) - float(ml_thr))
        if diff <= 0.12:
            delta = 0.04 if outcome == "VERIFIED_FP" else -0.04
            tkey = f"{channel}_ml_threshold"
            return {
                "root_cause": "poor_probability_calibration",
                "channel": channel,
                "rationale": f"ML score {float(ml_score):.3f} was borderline near threshold {float(ml_thr):.3f}; calibration shift recommended.",
                "details": {
                    "threshold_key": tkey,
                    "current_threshold": float(ml_thr),
                    "suggested_threshold": round(float(ml_thr) + delta, 4),
                    "ml_score": float(ml_score),
                },
            }

    # 5. Missing domain reputation (URL or embedded-URL domain unknown to local intel)
    if domain and domain not in policy.get("verified_allow_domains", []) and domain not in policy.get("verified_deny_domains", []):
        if channel == "url":
            return {
                "root_cause": "missing_domain_reputation",
                "channel": channel,
                "rationale": f"Domain '{domain}' has no verified local reputation entry; updating verified domain intelligence resolves the error without retraining.",
                "details": {"domain": domain, "disposition": "deny" if outcome == "VERIFIED_FN" else "allow"},
            }

    # 6. Repeated model misclassification (>= 3 verified errors in this channel -> workstation retrain!)
    if recent_verified_errors_in_channel >= 3:
        return {
            "root_cause": "repeated_model_misclassification",
            "channel": channel,
            "rationale": f"Channel '{channel}' has accumulated {recent_verified_errors_in_channel} verified errors; queuing verified labelled examples for workstation retraining.",
            "details": {"channel": channel, "verified_error_count": recent_verified_errors_in_channel, "raw_input": raw},
        }

    # 7. New phishing / scam / malware technique (VERIFIED_FN with unseen pattern)
    if outcome == "VERIFIED_FN":
        ind = _extract_novel_indicator(raw, domain)
        return {
            "root_cause": "new_phishing_technique",
            "channel": channel,
            "rationale": f"Unseen attack pattern missed by current rules and ML; extracting verified indicator '{ind}' and queueing representative example for retraining.",
            "details": {"domain": domain, "indicator": ind, "raw_input": raw},
        }

    # Fallback for remaining benign FPs -> verified benign phrase / indicator override
    return {
        "root_cause": "incorrect_malicious_domain_classification",
        "channel": channel,
        "rationale": "Verified false positive without rule-weight/calibration match; adding bounded verified benign override.",
        "details": {"domain": domain, "indicator": _extract_novel_indicator(raw, ""), "disposition": "allow"},
    }
