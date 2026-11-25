from app.adaptive.service import AdaptiveEngine
from app.adaptive.registry import evaluate_candidate

def test_verified_feedback_trains_candidate(tmp_path):
    engine = AdaptiveEngine(str(tmp_path/"db.sqlite"), str(tmp_path/"models"))
    result = engine.accept_feedback({
        "event_id":"e1", "verdict":"correct", "verified":True,
        "reviewer":"analyst", "notes":"confirmed", "source":"sms",
        "content":"Your account is blocked click here", "label":"malicious"
    })
    assert result["eligible_for_learning"] is True
    assert result["learning"]["status"] == "candidate_only"
    assert engine.learner.predict("account blocked click")["available"] is True

def test_unverified_feedback_is_not_trained(tmp_path):
    engine = AdaptiveEngine(str(tmp_path/"db.sqlite"), str(tmp_path/"models"))
    result = engine.accept_feedback({
        "event_id":"e2", "verdict":"false_positive", "verified":False,
        "reviewer":"user", "notes":"not sure", "source":"sms",
        "content":"normal message", "label":"benign"
    })
    assert result["eligible_for_learning"] is False
    assert engine.learner.predict("normal message")["available"] is False

def test_network_baseline_and_drift(tmp_path):
    engine = AdaptiveEngine(str(tmp_path/"db.sqlite"), str(tmp_path/"models"))
    first = engine.observe_network({"outbound_bytes":1000})[0]
    second = engine.observe_network({"outbound_bytes":100000})[0]
    assert first["drift_detected"] is False
    assert second["drift_detected"] is True

def test_candidate_gate_requires_metrics():
    assert evaluate_candidate({"f1":0.9})["eligible"] is False
    good = {"precision":0.95,"recall":0.90,"f1":0.92,"false_positive_rate":0.03}
    assert evaluate_candidate(good)["eligible"] is True
