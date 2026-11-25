import json
import sqlite3
from pathlib import Path
from typing import Any


class EventStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    threat_category TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS correlations (
                    correlation_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    indicator TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    risk_score INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    event_ids TEXT NOT NULL
                )
            """)

    def save_event(self, event: dict[str, Any]):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO events
                (event_id, timestamp, source, risk_level, risk_score, threat_category, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["event_id"], event["timestamp"], event["source"],
                    event["risk_level"], event["risk_score"], event["threat_category"],
                    json.dumps(event),
                ),
            )

    def save_events_batch(self, events: list[dict[str, Any]]):
        """Persist many events in a single atomic SQLite transaction (prevents per-flow commit stalls)."""
        if not events:
            return
        rows = [
            (
                ev["event_id"], ev["timestamp"], ev["source"],
                ev["risk_level"], ev["risk_score"], ev["threat_category"],
                json.dumps(ev),
            )
            for ev in events
        ]
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO events
                (event_id, timestamp, source, risk_level, risk_score, threat_category, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def get_event(self, event_id: str):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_events(self, limit: int = 100):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def save_correlation(self, corr: dict[str, Any]):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO correlations
                (correlation_id, timestamp, indicator, channels, risk_level, risk_score, summary, event_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    corr["correlation_id"], corr["timestamp"], corr["indicator"],
                    json.dumps(corr["channels"]), corr["risk_level"], corr["risk_score"],
                    corr["summary"], json.dumps(corr["event_ids"]),
                ),
            )

    def list_correlations(self, limit: int = 50):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM correlations ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "correlation_id": r["correlation_id"],
                "timestamp": r["timestamp"],
                "indicator": r["indicator"],
                "channels": json.loads(r["channels"]),
                "risk_level": r["risk_level"],
                "risk_score": r["risk_score"],
                "summary": r["summary"],
                "event_ids": json.loads(r["event_ids"]),
            })
        return out

    def stats(self):
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            high = conn.execute(
                "SELECT COUNT(*) FROM events WHERE risk_level IN ('high','critical')"
            ).fetchone()[0]
            by_source = conn.execute(
                "SELECT source, COUNT(*) n FROM events GROUP BY source"
            ).fetchall()
            corr_count = conn.execute("SELECT COUNT(*) FROM correlations").fetchone()[0]
        return {
            "total_events": total,
            "high_or_critical": high,
            "correlated_incidents": corr_count,
            "by_source": {r["source"]: r["n"] for r in by_source},
        }
