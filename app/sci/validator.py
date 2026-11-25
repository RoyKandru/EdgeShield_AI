"""SCI Stage 7: Automated Regression & Security Validation Engine.

Replays any candidate SCI policy/intel/weight/calibration update in an isolated
thread-local context against the pinned Verified Golden Suite + verified feedback cases.
Blocks deployment if F1 drops, FPR exceeds cap, any critical threat regresses, or
the target verified error is not actually resolved.
"""
from __future__ import annotations

import time
from typing import Any

from app.detectors.network import analyse_flows
from app.detectors.sms import analyse_sms
from app.detectors.url import analyse_url
from app.sci.golden_suite import GOLDEN_CASES
from app.sci.policy import policy_override_context


def _is_flagged(channel: str, event) -> bool:
    if channel == "url":
        return event.prediction == "suspicious"
    return event.prediction not in {"no_obvious_indicators", "no_configured_anomaly"}


def _eval_one(channel: str, raw_input: Any) -> bool:
    if channel == "url":
        return _is_flagged("url", analyse_url(str(raw_input)))
    if channel == "sms":
        return _is_flagged("sms", analyse_sms(str(raw_input)))
    if channel == "network":
        flow = raw_input if isinstance(raw_input, dict) else {"raw_input": str(raw_input)}
        return _is_flagged("network", analyse_flows([flow])[0])
    return False


def evaluate_policy_on_suite(policy: dict[str, Any], extra_verified_cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    tp = fp = tn = fn = 0
    crit_total = crit_caught = 0
    failures: list[str] = []

    cases = list(GOLDEN_CASES)
    for idx, ex in enumerate(extra_verified_cases or []):
        ch = ex.get("channel")
        if ch in {"url", "sms"} and ex.get("raw_input"):
            cases.append({
                "id": f"ver-{idx}",
                "channel": ch,
                "expected_flagged": ex.get("verified_label") == "malicious",
                "critical": False,
                "input": ex["raw_input"],
            })

    with policy_override_context(policy):
        for c in cases:
            flagged = _eval_one(c["channel"], c["input"])
            exp = bool(c["expected_flagged"])
            if c.get("critical"):
                crit_total += 1
                if flagged == exp:
                    crit_caught += 1
            if exp and flagged:
                tp += 1
            elif not exp and not flagged:
                tn += 1
            elif not exp and flagged:
                fp += 1
                failures.append(f"FP:{c['id']}")
            else:
                fn += 1
                failures.append(f"FN:{c['id']}")

    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    crit_rec = crit_caught / crit_total if crit_total else 1.0
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "total_cases": len(cases),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "critical_threat_recall": round(crit_rec, 4),
        "failures": failures[:15],
        "elapsed_ms": elapsed_ms,
    }


def validate_candidate_policy(
    current_policy: dict[str, Any],
    candidate_policy: dict[str, Any],
    target_case: dict[str, Any] | None = None,
    extra_verified_cases: list[dict[str, Any]] | None = None,
    min_f1: float = 0.90,
    max_fpr: float = 0.05,
) -> dict[str, Any]:
    baseline = evaluate_policy_on_suite(current_policy, extra_verified_cases)
    candidate = evaluate_policy_on_suite(candidate_policy, extra_verified_cases)

    reasons: list[str] = []
    if candidate["f1"] < min_f1:
        reasons.append(f"Candidate F1 {candidate['f1']} < minimum {min_f1}")
    if candidate["f1"] < baseline["f1"] - 0.01:
        reasons.append(f"Candidate F1 regressed ({candidate['f1']} vs baseline {baseline['f1']})")
    if candidate["fpr"] > max_fpr:
        reasons.append(f"Candidate FPR {candidate['fpr']} > maximum {max_fpr}")
    if candidate["fpr"] > baseline["fpr"] + 0.01:
        reasons.append(f"Candidate FPR regressed ({candidate['fpr']} vs baseline {baseline['fpr']})")
    if candidate["critical_threat_recall"] < 1.0:
        reasons.append(f"Critical threat regression detected (recall={candidate['critical_threat_recall']})")

    target_resolved = True
    if target_case and target_case.get("channel") in {"url", "sms", "network"}:
        with policy_override_context(candidate_policy):
            now_flagged = _eval_one(target_case["channel"], target_case["input"])
        expected = bool(target_case["expected_flagged"])
        target_resolved = (now_flagged == expected)
        if not target_resolved:
            reasons.append(f"Candidate did not resolve target case (expected flagged={expected}, got {now_flagged})")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "target_resolved": target_resolved,
        "baseline": baseline,
        "candidate": candidate,
    }
