from app.sci.engine import SCIEngine
import pytest

def test_sci_record_and_hash_chain(tmp_path):
    engine = SCIEngine(str(tmp_path/"sci.sqlite"))
    record = engine.record_incident(
        component="url", symptom="URL model disagreement",
        evidence={"model_disagreement": True},
        correction_action="mark_uncertain",
        verification={"independent": True, "checks": [
            {"name":"schema_check","passed":True},
            {"name":"reference_check","passed":True}
        ]}
    )
    assert record["verification"]["status"] == "verified"
    assert len(record["record_hash"]) == 64
    assert engine.verify_ledger()["valid"] is True

def test_sci_unverified_without_independent_checks(tmp_path):
    engine = SCIEngine(str(tmp_path/"sci.sqlite"))
    record = engine.record_incident(
        component="sms", symptom="message feature mismatch",
        evidence={"encoding_error": True},
        correction_action="mark_uncertain",
        verification={"checks":[{"name":"local_check","passed":True}], "independent":False}
    )
    assert record["verification"]["status"] == "unverified"

def test_sci_rejects_arbitrary_action(tmp_path):
    engine = SCIEngine(str(tmp_path/"sci.sqlite"))
    with pytest.raises(ValueError):
        engine.record_incident(
            component="soc", symptom="thermal issue", evidence={},
            correction_action="run arbitrary shell command", verification={}
        )

def test_sci_detects_tampering(tmp_path):
    db = str(tmp_path/"sci.sqlite")
    engine = SCIEngine(db)
    engine.record_incident(component="runtime", symptom="runtime failure",
        evidence={}, correction_action="reject_invalid_input", verification={})
    import sqlite3
    with sqlite3.connect(db) as c:
        c.execute("UPDATE sci_ledger SET record_hash='bad'")
    assert engine.verify_ledger()["valid"] is False

def test_sci_rejects_sensor_component(tmp_path):
    engine = SCIEngine(str(tmp_path/"sci.sqlite"))
    with pytest.raises(ValueError):
        engine.record_incident(component="sensor", symptom="sensor fault", evidence={},
            correction_action="mark_uncertain", verification={})
