"""End-to-end tests for the 9-stage Evidence-Grounded Closed-Loop SCI Pipeline."""
from __future__ import annotations

import pytest
from app.detectors.url import analyse_url
from app.detectors.sms import analyse_sms
from app.sci.closed_loop import SCIClosedLoopController
from app.sci.engine import SCIEngine
from app.sci.planner import ROOT_CAUSE_ACTION_MAP, plan_correction
from app.sci.policy import PolicyManager, policy_override_context


def _make_controller(tmp_path):
    db = str(tmp_path / "sci_test.db")
    pdir = tmp_path / "sci_policy"
    engine = SCIEngine(db)
    pmgr = PolicyManager(pdir)
    ctrl = SCIClosedLoopController(db, engine, pmgr)
    return ctrl, engine, pmgr


def test_unverified_feedback_is_not_auto_trusted_and_becomes_uncertain(tmp_path):
    ctrl, _, _ = _make_controller(tmp_path)
    ev = analyse_url("https://unseen-novel-example99.org/page").model_dump(mode="json")
    res = ctrl.submit_error_cycle(
        event=ev,
        claimed_verdict="false_negative",
        claimed_label="malicious",
        reviewer="random-user",
        independent_checks=[],
        notes="I think this might be bad",
        auto_deploy_if_validated=True,
    )
    assert res["verification"]["outcome"] == "UNCERTAIN"
    assert res["verification"]["verified"] is False
    assert res["deployed"] is False
    assert res["stored_in"] == "sci_unverified_feedback"
    assert ctrl.verifier.dataset_counts()["unverified_feedback"] == 1
    assert ctrl.verifier.dataset_counts()["verified_feedback"] == 0


def test_anti_poisoning_blocks_whitelisting_critical_attack(tmp_path):
    ctrl, engine, _ = _make_controller(tmp_path)
    ev = analyse_url("https://api.telegram.org/bot998877:AAFake/sendMessage?chat_id=1&text=OTP").model_dump(mode="json")
    res = ctrl.submit_error_cycle(
        event=ev,
        claimed_verdict="false_positive",
        claimed_label="benign",
        reviewer="compromised-user",
        independent_checks=[{"source": "fake", "passed": True, "independent": True}],
        auto_deploy_if_validated=True,
    )
    assert res["verification"]["poisoning_blocked"] is True
    assert res["verification"]["outcome"] == "UNCERTAIN"
    assert res["deployed"] is False
    # Verify attempt was recorded in immutable SHA-256 ledger
    assert engine.verify_ledger()["valid"] is True
    assert len(engine.list_incidents()) >= 1


def test_llm_actor_cannot_modify_policy_or_deploy(tmp_path):
    ctrl, _, _ = _make_controller(tmp_path)
    ev = analyse_url("https://unseen-phish-domain77.net/docs").model_dump(mode="json")
    with pytest.raises(PermissionError):
        ctrl.submit_error_cycle(
            event=ev,
            claimed_verdict="false_negative",
            claimed_label="malicious",
            reviewer="llama-executorch-advisor",
            independent_checks=[{"source": "x", "passed": True, "independent": True}],
        )


def test_all_seven_root_causes_have_explicit_actions():
    expected_causes = {
        "missing_domain_reputation",
        "incorrect_malicious_domain_classification",
        "excessive_feature_weighting",
        "incorrect_url_feature_extraction",
        "poor_probability_calibration",
        "repeated_model_misclassification",
        "new_phishing_technique",
    }
    assert set(ROOT_CAUSE_ACTION_MAP.keys()) == expected_causes
    for cause in expected_causes:
        plan = plan_correction({"root_cause": cause, "channel": "url", "details": {"domain": "bad-test.org", "rule_code": "LONG_URL"}}, actor="sec-analyst")
        assert plan["action"] == ROOT_CAUSE_ACTION_MAP[cause]["action"]


def test_full_closed_loop_verify_diagnose_plan_validate_deploy_and_rollback(tmp_path):
    ctrl, engine, pmgr = _make_controller(tmp_path)
    novel_url = "https://quiet-novel-scam-9981.org/docs"

    # Before SCI correction: novel clean-looking domain is not flagged
    with policy_override_context(pmgr.get_active()):
        before = analyse_url(novel_url)
    assert before.prediction == "no_obvious_indicators"

    # Run full closed loop with independent offline verification check
    res = ctrl.submit_error_cycle(
        event=before.model_dump(mode="json"),
        claimed_verdict="false_negative",
        claimed_label="malicious",
        reviewer="soc-analyst-1",
        independent_checks=[{"source": "offline_openphish_feed", "passed": True, "independent": True, "detail": "Confirmed active phishing host"}],
        auto_deploy_if_validated=True,
    )
    assert res["verification"]["outcome"] == "VERIFIED_FN"
    assert res["diagnosis"]["root_cause"] == "missing_domain_reputation"
    assert res["plan"]["action"] == "update_verified_domain_intelligence"
    assert res["validation"]["passed"] is True
    assert res["validation"]["target_resolved"] is True
    assert res["deployed"] is True
    new_ver = res["deployment"]["deployed_version"]
    assert new_ver == "v1.0.1"

    # Immediately verify detector NOW catches the novel domain under deployed policy v1.0.1
    with policy_override_context(pmgr.get_active()):
        after = analyse_url(novel_url)
    assert after.prediction == "suspicious"
    assert any(e.code == "SCI_VERIFIED_DENY_DOMAIN" for e in after.evidence)

    # Rollback to v1.0.0 and verify restoration + immutable SHA-256 ledger integrity
    rb = ctrl.rollback(actor="soc-analyst-1", reason="test_rollback")
    assert rb["active_version"] == "v1.0.0"
    with policy_override_context(pmgr.get_active()):
        restored = analyse_url(novel_url)
    assert restored.prediction == "no_obvious_indicators"
    assert engine.verify_ledger()["valid"] is True
