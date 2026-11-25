"""URL hybrid (ML + rules) runtime: fallback, sanity, crash-safety and parity with the evaluation."""
import json
import random
from pathlib import Path

import pytest

from app.config import settings
from app.detectors.url import analyse_url
from app.ml import features as F
from app.ml import url_runtime
from app.ml.url_features_v2 import registrable_domain

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / url_runtime.ARTEFACT
needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="url_model_v2.joblib not present")


@pytest.fixture(autouse=True)
def _fresh_cache():
    url_runtime.reset_cache()
    yield
    url_runtime.reset_cache()


def _rules_only(monkeypatch):
    monkeypatch.setattr(settings, "ml_detectors", False)
    url_runtime.reset_cache()


def test_fallback_when_model_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_dir", str(tmp_path / "nope"))
    e = analyse_url("http://192.0.2.10/urgent/bank/verify-account")
    assert e.model_version == "url-rules-v1"
    assert e.prediction == "suspicious"          # rules alone still flag this
    assert not any(x.code == "ML_SCORE" for x in e.evidence)


def test_fallback_on_corrupt_artefact(monkeypatch, tmp_path):
    (tmp_path / url_runtime.ARTEFACT).write_bytes(b"not a joblib file")
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    e = analyse_url("http://example.com/")
    assert e.model_version == "url-rules-v1"


def test_fallback_on_bundle_missing_keys(monkeypatch, tmp_path):
    import joblib
    joblib.dump({"kind": "url_v2"}, tmp_path / url_runtime.ARTEFACT)
    monkeypatch.setattr(settings, "model_dir", str(tmp_path))
    assert url_runtime.score_url("http://example.com/") is None


def test_ml_disabled_is_identical_to_rules(monkeypatch):
    _rules_only(monkeypatch)
    e = analyse_url("http://192.0.2.10/urgent/bank/verify-account")
    assert e.model_version == "url-rules-v1"
    assert set(e.metadata) == {"hostname", "scheme"}


@needs_model
def test_flagged_vs_benign_sanity():
    bad = analyse_url("http://192.0.2.10/urgent/bank/verify-account")
    good = analyse_url("https://www.wikipedia.org/")
    assert bad.model_version == good.model_version == url_runtime.MODEL_VERSION
    assert bad.prediction == "suspicious" and bad.risk_score > good.risk_score
    assert good.prediction == "no_obvious_indicators"
    assert good.metadata["ml_allowlisted"] is True
    ml = next(x for x in bad.evidence if x.code == "ML_SCORE")
    assert ml.weight > 0
    assert 0.0 <= bad.confidence <= 1.0


@needs_model
@pytest.mark.parametrize("u", ["http://[bad-host/login", "", "   ", "http://", "[[[", "http://a@b@c/\x00x",
                               "http://" + "a" * 5000 + ".com/" + "b" * 5000])
def test_malformed_urls_do_not_crash_with_ml(u):
    e = analyse_url(u)
    assert e.source == "url_detector"
    assert 0 <= e.risk_score <= 100


def _sample_urls(n=200):
    p = ROOT / "data" / "external" / "url" / "urls.json"
    if p.is_file():
        rows = json.loads(p.read_text(encoding="utf-8"))
        rng = random.Random(0)
        return [r["text"] for r in rng.sample(rows, n)]
    rng = random.Random(0)
    words = ["login", "secure", "bank", "shop", "news", "update", "paypal", "wiki", "mail", "pay"]
    tlds = ["com", "net", "xyz", "top", "org", "ly"]
    return [f"http{'s' if rng.random() < .5 else ''}://{rng.choice(words)}-{rng.choice(words)}"
            f"{rng.randint(0, 999)}.{rng.choice(tlds)}/{rng.choice(words)}?id={rng.randint(0, 9999)}"
            for _ in range(n)]


@needs_model
def test_runtime_parity_with_evaluation(monkeypatch):
    """Independent re-implementation of the evaluated scoring (scripts/url_improve.py) must agree
    with what analyse_url actually reports, on 200 URLs."""
    import joblib
    b = joblib.load(MODEL)
    models, thr = b["models"], float(b["threshold_hybrid"])
    allow, psl = frozenset(b["allowlist"]), frozenset(b["psl_rules"])
    urls = _sample_urls(200)

    # rules-only strong flags (second half of the hybrid)
    _rules_only(monkeypatch)
    strong = [analyse_url(u).prediction == "suspicious" for u in urls]
    monkeypatch.setattr(settings, "ml_detectors", True)
    url_runtime.reset_cache()

    n_flag = 0
    for u, st in zip(urls, strong):
        norm = F.normalise_url(u)
        s = sum(float(m.predict_proba([norm])[0, 1]) for m in models) / len(models)
        ml_flag = s >= thr and registrable_domain(F.url_host(norm), psl) not in allow
        e = analyse_url(u)
        assert e.metadata["ml_score"] == pytest.approx(s, abs=1e-4), u
        npu_flag = any(ev.code in {"NPU_URL_DUAL_HIGH", "NPU_URL_DUAL_CORROBORATIVE"} for ev in e.evidence)
        assert (e.prediction == "suspicious") == (ml_flag or st or npu_flag), u
        n_flag += e.prediction == "suspicious"
    assert 0 < n_flag < len(urls)  # the sample is mixed, so both branches were exercised
