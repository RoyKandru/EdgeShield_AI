"""Controlled adaptive-learning services for EdgeShield AI.

Feedback is never trusted as training truth by default. Only verified feedback
from an authorised reviewer is eligible for candidate-model updates.
Candidate models are not automatically promoted to production.
"""
from __future__ import annotations
import json, math, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


class AdaptiveStore:
    def __init__(self, db_path: str, model_dir: str):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self.connect() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY, event_id TEXT NOT NULL,
                verdict TEXT NOT NULL, verified INTEGER NOT NULL,
                reviewer TEXT NOT NULL, notes TEXT NOT NULL,
                created_at TEXT NOT NULL, source TEXT NOT NULL,
                content TEXT NOT NULL, label TEXT NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS network_baselines (
                metric TEXT PRIMARY KEY, mean REAL NOT NULL, variance REAL NOT NULL,
                samples INTEGER NOT NULL, updated_at TEXT NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY, pipeline TEXT NOT NULL, version TEXT NOT NULL,
                status TEXT NOT NULL, metrics TEXT NOT NULL, created_at TEXT NOT NULL,
                artifact_path TEXT NOT NULL)""")

    def add_feedback(self, item: dict[str, Any]) -> str:
        fid = str(uuid.uuid4())
        with self.connect() as c:
            c.execute("""INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fid, item["event_id"], item["verdict"], int(item["verified"]),
                 item["reviewer"], item["notes"], datetime.now(timezone.utc).isoformat(),
                 item["source"], item["content"], item["label"]))
        return fid

    def feedback_count(self, verified_only=False):
        with self.connect() as c:
            q = "SELECT COUNT(*) FROM feedback"
            if verified_only:
                q += " WHERE verified=1"
            return c.execute(q).fetchone()[0]

    def update_baseline(self, metric: str, value: float, alpha: float = 0.15):
        """EWMA mean/variance baseline; first observation initialises the metric."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as c:
            row = c.execute("SELECT mean, variance, samples FROM network_baselines WHERE metric=?",
                            (metric,)).fetchone()
            if row is None:
                mean, var, n = float(value), 0.0, 1
                drift = False
            else:
                old_mean, old_var, n = row
                sigma = max(math.sqrt(max(old_var, 0.0)), abs(old_mean) * 0.05, 1.0)
                drift = abs(value - old_mean) > 3.0 * sigma
                delta = value - old_mean
                mean = (1-alpha)*old_mean + alpha*value
                var = (1-alpha)*old_var + alpha*(delta**2)
                n += 1
            c.execute("""INSERT OR REPLACE INTO network_baselines
                (metric,mean,variance,samples,updated_at) VALUES (?,?,?,?,?)""",
                (metric, mean, var, n, now))
        return {"metric": metric, "observed": value, "baseline_mean": mean,
                "samples": n, "drift_detected": drift}

    def baselines(self):
        with self.connect() as c:
            rows = c.execute("SELECT metric,mean,variance,samples,updated_at FROM network_baselines").fetchall()
        return [dict(metric=r[0], mean=r[1], variance=r[2], samples=r[3], updated_at=r[4]) for r in rows]

    def register_model(self, pipeline, version, status, metrics, artifact_path):
        mid = str(uuid.uuid4())
        with self.connect() as c:
            c.execute("INSERT INTO model_registry VALUES (?,?,?,?,?,?,?)",
                (mid, pipeline, version, status, json.dumps(metrics), datetime.now(timezone.utc).isoformat(), artifact_path))
        return mid

    def models(self):
        with self.connect() as c:
            rows = c.execute("SELECT model_id,pipeline,version,status,metrics,created_at,artifact_path FROM model_registry ORDER BY created_at DESC").fetchall()
        return [dict(model_id=r[0], pipeline=r[1], version=r[2], status=r[3],
                     metrics=json.loads(r[4]), created_at=r[5], artifact_path=r[6]) for r in rows]


class IncrementalTextLearner:
    """Small local candidate learner; it does not replace production detectors."""
    CLASSES = np.array(["benign", "malicious"])

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.model_dir / "text_candidate.joblib"
        self.vectorizer = HashingVectorizer(n_features=2**14, alternate_sign=False,
                                            norm="l2", ngram_range=(1,2))
        self.model = None
        if self.path.exists():
            try:
                self.model = joblib.load(self.path)
            except Exception:
                self.model = None

    def learn(self, text: str, label: str):
        if label not in {"benign", "malicious"}:
            raise ValueError("label must be benign or malicious")
        x = self.vectorizer.transform([text])
        if self.model is None:
            self.model = SGDClassifier(loss="log_loss", random_state=42)
            self.model.partial_fit(x, np.array([label]), classes=self.CLASSES)
        else:
            self.model.partial_fit(x, np.array([label]))
        joblib.dump(self.model, self.path)
        return {"candidate_model": str(self.path), "trained_examples": int(getattr(self.model, "t_", 0)),
                "status": "candidate_only"}

    def predict(self, text: str):
        if self.model is None:
            return {"available": False, "reason": "No verified examples have trained a candidate yet"}
        x = self.vectorizer.transform([text])
        return {"available": True, "prediction": str(self.model.predict(x)[0]),
                "probabilities": {str(k): float(v) for k, v in zip(self.model.classes_, self.model.predict_proba(x)[0])}}


class AdaptiveEngine:
    def __init__(self, db_path: str, model_dir: str):
        self.store = AdaptiveStore(db_path, model_dir)
        self.learner = IncrementalTextLearner(model_dir)

    def accept_feedback(self, item: dict[str, Any]):
        fid = self.store.add_feedback(item)
        learning = None
        if item["verified"] and item["source"] in {"sms", "url"}:
            learning = self.learner.learn(item["content"], item["label"])
        return {"feedback_id": fid, "stored": True, "eligible_for_learning": item["verified"],
                "learning": learning, "warning": "Candidate model only; production promotion requires evaluation and approval."}

    def observe_network(self, metrics: dict[str, float]):
        return [self.store.update_baseline(k, float(v)) for k, v in metrics.items()]

    def status(self):
        return {"feedback_total": self.store.feedback_count(),
                "verified_feedback": self.store.feedback_count(True),
                "network_baselines": self.store.baselines(),
                "candidate_text_model": self.learner.predict("test message"),
                "registered_models": self.store.models()}
