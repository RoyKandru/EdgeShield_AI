"""SCI Stage 3: Independent Offline Outcome Verification & Anti-Poisoning Engine.

Strict security invariants:
  - User feedback is NEVER automatically trusted as ground truth.
  - Unverified feedback is quarantined separately from verified feedback.
  - Supports explicit `UNCERTAIN` outcome when independent evidence is insufficient.
  - Blocks feedback poisoning (rate limiting, flip-flop conflict detection, and hard refusal
    to whitelist immutable critical threat signatures such as Telegram OTP exfil, APK droppers,
    USSD *401* hijacks, or C2 ports).
"""
from __future__ import annotations

import re
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.sci.policy import get_active_policy

IMMUTABLE_MALICIOUS_PATTERNS = (
    re.compile(r"api\.telegram\.org/bot[^/]+/sendmessage", re.I),
    re.compile(r"\*+(?:401|21)\*[0-9]{7,13}#?", re.I),
    re.compile(r"digital\s+arrest", re.I),
    re.compile(r"\.(?:apk|exe|scr|hta)(?:$|[?#\s])", re.I),
    re.compile(r"[\u2044\u2215]"),
)

OFFICIAL_SAFE_DOMAINS = {
    "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com", "axisbank.com",
    "uidai.gov.in", "epfindia.gov.in", "incometax.gov.in", "rbi.org.in", "trai.gov.in",
    "irctc.co.in", "npci.org.in", "wikipedia.org", "github.com", "google.com",
    "microsoft.com", "apple.com", "paypal.com",
}

_RATE_LOCK_HISTORY: dict[str, list[float]] = defaultdict(list)
MAX_SUBMISSIONS_PER_HOUR = 25


def _extract_host(raw: str) -> str:
    txt = (raw or "").strip()
    m = re.search(r"https?://[^\s<>\"']+", txt, re.I)
    cand = m.group(0) if m else (txt if "." in txt and " " not in txt else "")
    if not cand:
        return ""
    if not cand.lower().startswith(("http://", "https://")):
        cand = "http://" + cand
    try:
        return (urlparse(cand).hostname or "").lower()
    except Exception:
        return ""


def _is_official_domain(host: str) -> bool:
    if not host:
        return False
    for d in OFFICIAL_SAFE_DOMAINS:
        if host == d or host.endswith("." + d):
            return True
    return host.endswith((".gov.in", ".nic.in", ".ac.in", ".edu.in"))


def _event_was_flagged(event: dict[str, Any]) -> bool:
    pred = str(event.get("prediction", "no_obvious_indicators"))
    src = str(event.get("source", ""))
    if src == "url_detector":
        return pred == "suspicious"
    return pred not in {"no_obvious_indicators", "no_configured_anomaly"}


class OutcomeVerifier:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self) -> None:
        with self._connect() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS sci_unverified_feedback (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                claimed_verdict TEXT NOT NULL,
                claimed_label TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                notes TEXT NOT NULL,
                reason TEXT NOT NULL,
                raw_input TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS sci_verified_feedback (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                outcome TEXT NOT NULL,
                verified_label TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                signals_json TEXT NOT NULL,
                raw_input TEXT NOT NULL,
                indicator TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")

    def verify(
        self,
        event: dict[str, Any],
        claimed_verdict: str,  # "false_positive" | "false_negative" | "correct"
        claimed_label: str,    # "benign" | "malicious"
        reviewer: str,
        independent_checks: list[dict[str, Any]] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Verify an error claim using independent offline evidence; returns VERIFIED_* or UNCERTAIN."""
        now = time.time()
        hist = _RATE_LOCK_HISTORY[reviewer]
        hist[:] = [t for t in hist if now - t <= 3600.0]
        if len(hist) >= MAX_SUBMISSIONS_PER_HOUR:
            return {
                "outcome": "UNCERTAIN",
                "verified": False,
                "poisoning_blocked": True,
                "poisoning_reason": f"Rate limit exceeded for reviewer '{reviewer}' ({MAX_SUBMISSIONS_PER_HOUR}/hr).",
                "independent_signals": [],
                "confidence": 0.0,
                "indicator": "",
            }
        hist.append(now)

        raw_input = str(event.get("raw_input_preview", ""))
        src = str(event.get("source", "unknown"))
        channel = "url" if "url" in src else "sms" if "sms" in src else "network"
        meta = event.get("metadata") or {}
        host = str(meta.get("hostname", "") or _extract_host(raw_input)).lower()
        was_flagged = _event_was_flagged(event)
        policy = get_active_policy()

        # --- Anti-Poisoning Guard #1: Never allow whitelisting immutable critical threat patterns ---
        if claimed_label == "benign" and claimed_verdict == "false_positive":
            for pat in IMMUTABLE_MALICIOUS_PATTERNS:
                if pat.search(raw_input):
                    return {
                        "outcome": "UNCERTAIN",
                        "verified": False,
                        "poisoning_blocked": True,
                        "poisoning_reason": "Anti-poisoning guard refused false_positive claim on an immutable high-risk attack signature.",
                        "independent_signals": [{"source": "anti_poisoning_guard", "supports": "malicious", "detail": pat.pattern}],
                        "confidence": 0.0,
                        "indicator": host or raw_input[:60],
                    }
            if channel == "network" and int(meta.get("dst_port", 0) or 0) in {4444, 6667, 1337, 3333, 9050}:
                return {
                    "outcome": "UNCERTAIN",
                    "verified": False,
                    "poisoning_blocked": True,
                    "poisoning_reason": "Anti-poisoning guard refused whitelisting known malware C2/miner port.",
                    "independent_signals": [{"source": "anti_poisoning_guard", "supports": "malicious", "detail": f"dst_port={meta.get('dst_port')}"}],
                    "confidence": 0.0,
                    "indicator": str(meta.get("src_ip", "")),
                }

        # --- Anti-Poisoning Guard #2: Conflicting flip-flop check against verified history ---
        indicator = host or str(meta.get("src_ip", "")) or raw_input[:64].lower()
        with self._connect() as c:
            prev_ver = c.execute(
                "SELECT verified_label, COUNT(*) n FROM sci_verified_feedback WHERE indicator=? GROUP BY verified_label",
                (indicator,),
            ).fetchall()
        hist_labels = {r["verified_label"]: r["n"] for r in prev_ver}
        opposite = "malicious" if claimed_label == "benign" else "benign"
        if hist_labels.get(opposite, 0) >= 2 and not independent_checks:
            return {
                "outcome": "UNCERTAIN",
                "verified": False,
                "poisoning_blocked": True,
                "poisoning_reason": f"Claim contradicts {hist_labels[opposite]} prior verified records for '{indicator}' without independent checks.",
                "independent_signals": [{"source": "verified_history_conflict", "supports": opposite}],
                "confidence": 0.0,
                "indicator": indicator,
            }

        # --- Collect Independent Offline Verification Signals ---
        signals: list[dict[str, Any]] = []
        support_score = 0.0

        # Signal 1: Official / Local Verified Threat Intelligence
        if host:
            if _is_official_domain(host) or host in policy.get("verified_allow_domains", []):
                if claimed_label == "benign":
                    signals.append({"source": "local_verified_intel", "supports": "benign", "detail": f"'{host}' is in official/verified safe domain intelligence."})
                    support_score += 0.65
                else:
                    signals.append({"source": "local_verified_intel", "supports": "benign", "detail": f"Conflict: '{host}' is an official trusted domain."})
                    support_score -= 0.50
            elif host in policy.get("verified_deny_domains", []):
                if claimed_label == "malicious":
                    signals.append({"source": "local_verified_intel", "supports": "malicious", "detail": f"'{host}' is in local verified malicious intelligence."})
                    support_score += 0.65

        # Signal 2: Multi-Signal Disagreement / Consensus inside the Event
        ml_score = meta.get("ml_score")
        if ml_score is not None:
            ml_val = float(ml_score)
            if claimed_label == "benign" and ml_val < 0.35:
                signals.append({"source": "ml_consensus", "supports": "benign", "detail": f"ML model probability is low ({ml_val:.3f}), corroborating benign claim."})
                support_score += 0.45
            elif claimed_label == "malicious" and ml_val >= 0.65:
                signals.append({"source": "ml_consensus", "supports": "malicious", "detail": f"ML model probability is elevated ({ml_val:.3f}), corroborating malicious claim."})
                support_score += 0.45

        # Signal 3: Cross-channel correlation in DB (strong independent corroboration for malicious)
        if indicator:
            corr = None
            try:
                with self._connect() as c:
                    corr = c.execute(
                        "SELECT channels, risk_score FROM correlations WHERE indicator=? ORDER BY timestamp DESC LIMIT 1",
                        (indicator,),
                    ).fetchone()
            except sqlite3.OperationalError:
                corr = None
            if corr is not None and claimed_label == "malicious":
                signals.append({"source": "cross_channel_correlation", "supports": "malicious", "detail": f"Indicator '{indicator}' correlated across channels {corr['channels']}."})
                support_score += 0.60

        # Signal 4: Historical verified matches supporting this label
        if hist_labels.get(claimed_label, 0) >= 1:
            signals.append({"source": "verified_history", "supports": claimed_label, "detail": f"Matched {hist_labels[claimed_label]} prior verified {claimed_label} record(s)."})
            support_score += 0.40

        # Signal 5: Caller-supplied independent verification checks (e.g. offline feed, SOC analyst log, DNS/TLS cert proof)
        passed_ext = [
            chk for chk in (independent_checks or [])
            if isinstance(chk, dict) and chk.get("passed") is True and chk.get("independent") is True
        ]
        for chk in passed_ext[:4]:
            signals.append({
                "source": str(chk.get("source", "independent_check")),
                "supports": claimed_label,
                "detail": str(chk.get("detail", "Independent verification check passed.")),
            })
            support_score += 0.55

        # Decision gate: require >= 0.55 independent corroboration score; otherwise UNCERTAIN!
        if support_score < 0.55:
            return {
                "outcome": "UNCERTAIN",
                "verified": False,
                "poisoning_blocked": False,
                "poisoning_reason": "",
                "independent_signals": signals,
                "confidence": round(max(0.0, support_score), 3),
                "indicator": indicator,
                "reason": "Insufficient independent evidence; retained in unverified feedback queue.",
            }

        if claimed_label == "benign" and was_flagged:
            outcome = "VERIFIED_FP"
        elif claimed_label == "malicious" and not was_flagged:
            outcome = "VERIFIED_FN"
        elif claimed_label == "malicious" and was_flagged:
            outcome = "VERIFIED_TP"
        else:
            outcome = "VERIFIED_TN"

        return {
            "outcome": outcome,
            "verified": True,
            "poisoning_blocked": False,
            "poisoning_reason": "",
            "independent_signals": signals,
            "confidence": round(min(0.99, support_score), 3),
            "indicator": indicator,
            "reason": f"Independently verified via {len(signals)} corroborating offline signal(s).",
        }

    def store_unverified(self, record_id: str, event: dict[str, Any], claimed_verdict: str,
                         claimed_label: str, reviewer: str, notes: str, reason: str) -> None:
        src = str(event.get("source", "unknown"))
        channel = "url" if "url" in src else "sms" if "sms" in src else "network"
        with self._connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO sci_unverified_feedback VALUES (?,?,?,?,?,?,?,?,?,?)",
                (record_id, event.get("event_id", ""), channel, claimed_verdict, claimed_label,
                 reviewer, notes, reason, str(event.get("raw_input_preview", "")),
                 datetime.now(timezone.utc).isoformat()),
            )

    def store_verified(self, record_id: str, event: dict[str, Any], outcome: str,
                       verified_label: str, reviewer: str, signals: list[dict[str, Any]],
                       indicator: str) -> None:
        import json
        src = str(event.get("source", "unknown"))
        channel = "url" if "url" in src else "sms" if "sms" in src else "network"
        with self._connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO sci_verified_feedback VALUES (?,?,?,?,?,?,?,?,?,?)",
                (record_id, event.get("event_id", ""), channel, outcome, verified_label,
                 reviewer, json.dumps(signals), str(event.get("raw_input_preview", "")),
                 indicator, datetime.now(timezone.utc).isoformat()),
            )

    def count_recent_verified_errors(self, channel: str) -> int:
        with self._connect() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM sci_verified_feedback WHERE channel=? AND outcome IN ('VERIFIED_FP','VERIFIED_FN')",
                (channel,),
            ).fetchone()
        return int(row[0]) if row else 0

    def list_verified_feedback(self, limit: int = 100) -> list[dict[str, Any]]:
        import json
        with self._connect() as c:
            rows = c.execute(
                "SELECT * FROM sci_verified_feedback ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {**dict(r), "signals": json.loads(r["signals_json"])}
            for r in rows
        ]

    def dataset_counts(self) -> dict[str, int]:
        with self._connect() as c:
            unv = c.execute("SELECT COUNT(*) FROM sci_unverified_feedback").fetchone()[0]
            ver = c.execute("SELECT COUNT(*) FROM sci_verified_feedback").fetchone()[0]
            vfp = c.execute("SELECT COUNT(*) FROM sci_verified_feedback WHERE outcome='VERIFIED_FP'").fetchone()[0]
            vfn = c.execute("SELECT COUNT(*) FROM sci_verified_feedback WHERE outcome='VERIFIED_FN'").fetchone()[0]
        return {
            "unverified_feedback": int(unv),
            "verified_feedback": int(ver),
            "verified_fp": int(vfp),
            "verified_fn": int(vfn),
        }
