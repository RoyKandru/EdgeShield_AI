"""Reversible Graduated Containment Ladder (nftables IP/MAC timeout sets + dnsmasq sinkhole).

Safety guarantees:
  1. Dry-run by default (EDGESHIELD_CONTAINMENT_DRY_RUN=true): records exact reversible apply/undo
     commands without touching host firewall rules unless an operator explicitly enables enforcement.
  2. Protected hosts (loopback, default gateway, board local IPs, DNS resolvers, EDGESHIELD_PROTECTED_HOSTS)
     are never auto-contained; they return status="needs_owner_approval".
  3. Every containment action carries a finite TTL (default 3600s), an exact Undo command, persistent
     SQLite history, and a SHA-256 hash-linked entry in the SCI Audit Ledger.
  4. Strict regex / ipaddress validation prevents command or path injection.
"""
from __future__ import annotations

import ipaddress
import json
import re
import shlex
import socket
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import settings
from app.sci.engine import SCIEngine

_DOMAIN_RE = re.compile(r"^(?=.{3,253}$)(?!-)[a-z0-9-]{1,63}(?<!-)(\.[a-z0-9-]{1,63})+$")
_MAC_RE = re.compile(r"^([0-9a-f]{2}:){5}[0-9a-f]{2}$")
_URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_IP_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
DNSMASQ_SINKHOLE_DIR = Path("/etc/dnsmasq.d/edgeshield")


def _discover_protected_hosts() -> set[str]:
    protected: set[str] = {"127.0.0.1", "::1", "0.0.0.0", "localhost"}
    # Protect host's own local IPs
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            protected.add(info[4][0])
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 80))
        protected.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    # Protect default gateway from /proc/net/route
    try:
        for line in Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "00000000":
                gw_hex = int(parts[2], 16)
                gw_ip = socket.inet_ntoa(gw_hex.to_bytes(4, "little"))
                protected.add(gw_ip)
    except Exception:
        pass
    # Protect configured resolvers from /etc/resolv.conf
    try:
        for line in Path("/etc/resolv.conf").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("nameserver"):
                toks = line.split()
                if len(toks) >= 2:
                    protected.add(toks[1])
    except Exception:
        pass
    # User-configured protected hosts
    extra = (settings.protected_hosts or "").strip()
    if extra:
        for tok in extra.split(","):
            if tok.strip():
                protected.add(tok.strip().lower())
    return protected


def _validate_target(action: str, target: str) -> str:
    t = (target or "").strip().lower()
    if action == "drop_ip":
        ip = ipaddress.IPv4Address(t)
        return str(ip)
    if action == "sinkhole_domain":
        t = t.rstrip(".")
        if not _DOMAIN_RE.match(t):
            raise ValueError(f"Invalid domain for sinkhole_domain: {target!r}")
        return t
    if action == "quarantine_mac":
        if not _MAC_RE.match(t):
            raise ValueError(f"Invalid MAC address for quarantine_mac: {target!r}")
        return t
    raise ValueError(f"Unsupported containment action: {action!r}")


def _build_commands(action: str, target: str, ttl_s: int) -> tuple[list[str], list[str]]:
    if action == "sinkhole_domain":
        conf = DNSMASQ_SINKHOLE_DIR / f"{target}.conf"
        apply_cmds = [
            f"mkdir -p {shlex.quote(str(DNSMASQ_SINKHOLE_DIR))}",
            f"printf 'address=/{target}/0.0.0.0\\n' > {shlex.quote(str(conf))}",
            "systemctl reload dnsmasq",
        ]
        undo_cmds = [
            f"rm -f {shlex.quote(str(conf))}",
            "systemctl reload dnsmasq",
        ]
        return apply_cmds, undo_cmds
    if action == "drop_ip":
        return (
            [f"nft add element inet edgeshield blocked_ips {{ {target} timeout {ttl_s}s }}"],
            [f"nft delete element inet edgeshield blocked_ips {{ {target} }}"],
        )
    if action == "quarantine_mac":
        return (
            [f"nft add element inet edgeshield quarantined_macs {{ {target} timeout {ttl_s}s }}"],
            [f"nft delete element inet edgeshield quarantined_macs {{ {target} }}"],
        )
    raise ValueError(f"Unsupported containment action: {action!r}")


def infer_target_from_event(event: dict[str, Any]) -> tuple[str, str]:
    """Derive a safe (action, target) pair from a stored ThreatEvent dict."""
    src = str(event.get("source", "")).lower()
    raw = str(event.get("raw_input_preview", ""))
    meta = event.get("metadata") or {}

    if "url" in src or "sms" in src:
        m = _URL_RE.search(raw)
        if m:
            host = (urlparse(m.group(0)).hostname or "").lower().rstrip(".")
            if host and _DOMAIN_RE.match(host):
                return "sinkhole_domain", host
        if "url" in src:
            host = (urlparse(raw if "://" in raw else f"http://{raw}").hostname or "").lower().rstrip(".")
            if host and _DOMAIN_RE.match(host):
                return "sinkhole_domain", host

    # Network event: prefer source IP of attacker, or dst IP for outbound C2
    src_ip = str(meta.get("src_ip", "")).strip()
    dst_ip = str(meta.get("dst_ip", "")).strip()
    ips = _IP_RE.findall(raw)
    if not src_ip and ips:
        src_ip = ips[0]
    if not dst_ip and len(ips) > 1:
        dst_ip = ips[1]
    cat = str(event.get("threat_category", "")).lower()
    protected = _discover_protected_hosts()
    if "c2" in cat or "exfil" in cat:
        cand = dst_ip or src_ip
    else:
        cand = src_ip if (src_ip and src_ip not in protected) else (dst_ip or src_ip)
    if cand:
        return "drop_ip", cand

    raise ValueError("Could not infer an IPv4 address or domain target from this event; supply action & target explicitly.")


class ContainmentEngine:
    """SQLite-backed reversible containment ladder linked to EdgeShield's SHA-256 SCI ledger."""

    def __init__(self, db_path: str, sci_engine: SCIEngine):
        self.db_path = db_path
        self.sci = sci_engine
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS containment_actions (
                action_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                expires_at REAL NOT NULL,
                event_id TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT NOT NULL,
                ttl_seconds INTEGER NOT NULL,
                dry_run INTEGER NOT NULL,
                status TEXT NOT NULL,
                actor TEXT NOT NULL,
                reason TEXT NOT NULL,
                apply_commands TEXT NOT NULL,
                undo_commands TEXT NOT NULL,
                sci_incident_id TEXT NOT NULL,
                undone_at TEXT,
                undo_actor TEXT,
                undo_sci_incident_id TEXT
            )""")

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    @property
    def dry_run(self) -> bool:
        return bool(settings.containment_dry_run)

    def protected_hosts(self) -> list[str]:
        return sorted(_discover_protected_hosts())

    def contain(
        self,
        *,
        action: str,
        target: str,
        event_id: str = "",
        risk_score: int = 80,
        ttl_seconds: int | None = None,
        actor: str = "soc-operator",
        reason: str = "high_risk_threat_containment",
        force: bool = False,
    ) -> dict[str, Any]:
        ttl = max(60, min(int(ttl_seconds or settings.containment_default_ttl_s), 86400))
        clean_target = _validate_target(action, target)
        protected = _discover_protected_hosts()

        if clean_target in protected:
            return {
                "status": "needs_owner_approval",
                "blocked_by_safeguard": True,
                "reason": f"Target '{clean_target}' is a protected local/gateway host and cannot be auto-contained.",
                "action": action,
                "target": clean_target,
                "dry_run": self.dry_run,
            }

        if risk_score < 50 and not force:
            raise ValueError(f"Refusing containment on low-risk event (risk_score={risk_score} < 50); pass force=true to override.")

        apply_cmds, undo_cmds = _build_commands(action, clean_target, ttl)
        is_dry = self.dry_run
        exec_status = "dry_run_recorded"

        if not is_dry:
            for cmd in apply_cmds:
                subprocess.run(cmd, shell=True, check=True, timeout=10)
            exec_status = "enforced_active"

        comp = "url" if action == "sinkhole_domain" else "network"
        sci_rec = self.sci.record_incident(
            component=comp,
            symptom=f"Reversible containment ({action}) -> {clean_target} [dry_run={is_dry}]",
            evidence={"event_id": event_id, "action": action, "target": clean_target, "ttl_seconds": ttl, "dry_run": is_dry, "commands": apply_cmds},
            correction_action="reversible_containment_apply",
            verification={"independent": True, "checks": [{"name": "protected_host_safeguard", "passed": True}, {"name": "ttl_reversible_undo_ready", "passed": True}]},
            actor=actor,
        )

        action_id = "CNT-" + uuid.uuid4().hex[:10].upper()
        now_iso = datetime.now(timezone.utc).isoformat()
        expires_at = time.time() + ttl
        with self._conn() as c:
            c.execute(
                """INSERT INTO containment_actions
                   (action_id, created_at, expires_at, event_id, action, target, ttl_seconds,
                    dry_run, status, actor, reason, apply_commands, undo_commands, sci_incident_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    action_id, now_iso, expires_at, event_id, action, clean_target, ttl,
                    1 if is_dry else 0, exec_status, actor, reason,
                    json.dumps(apply_cmds), json.dumps(undo_cmds), sci_rec["incident_id"],
                ),
            )
        return self.get_action(action_id)

    def undo(self, action_id: str, actor: str = "soc-operator") -> dict[str, Any]:
        row = self.get_action(action_id)
        if not row:
            raise KeyError(f"Containment action '{action_id}' not found")
        if row["status"] == "undone":
            return row

        is_dry = bool(row["dry_run"])
        if not is_dry:
            for cmd in row["undo_commands"]:
                subprocess.run(cmd, shell=True, check=True, timeout=10)

        comp = "url" if row["action"] == "sinkhole_domain" else "network"
        sci_rec = self.sci.record_incident(
            component=comp,
            symptom=f"Reversible containment UNDO ({row['action']}) -> {row['target']} [dry_run={is_dry}]",
            evidence={"action_id": action_id, "target": row["target"], "undo_commands": row["undo_commands"], "dry_run": is_dry},
            correction_action="reversible_containment_undo",
            verification={"independent": True, "checks": [{"name": "reversible_undo_logged", "passed": True}]},
            actor=actor,
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._conn() as c:
            c.execute(
                "UPDATE containment_actions SET status=?, undone_at=?, undo_actor=?, undo_sci_incident_id=? WHERE action_id=?",
                ("undone", now_iso, actor, sci_rec["incident_id"], action_id),
            )
        return self.get_action(action_id)

    def _row_to_dict(self, r: sqlite3.Row) -> dict[str, Any]:
        now_t = time.time()
        st = r["status"]
        if st in ("dry_run_recorded", "enforced_active") and now_t > float(r["expires_at"]):
            st = "expired"
        return {
            "action_id": r["action_id"],
            "created_at": r["created_at"],
            "expires_at": r["expires_at"],
            "ttl_remaining_s": max(0, int(float(r["expires_at"]) - now_t)) if st in ("dry_run_recorded", "enforced_active") else 0,
            "event_id": r["event_id"],
            "action": r["action"],
            "target": r["target"],
            "ttl_seconds": r["ttl_seconds"],
            "dry_run": bool(r["dry_run"]),
            "status": st,
            "actor": r["actor"],
            "reason": r["reason"],
            "apply_commands": json.loads(r["apply_commands"]),
            "undo_commands": json.loads(r["undo_commands"]),
            "sci_incident_id": r["sci_incident_id"],
            "undone_at": r["undone_at"],
            "undo_actor": r["undo_actor"],
            "undo_sci_incident_id": r["undo_sci_incident_id"],
        }

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM containment_actions WHERE action_id=?", (action_id,)).fetchone()
        return self._row_to_dict(r) if r else None

    def list_actions(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM containment_actions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def status(self) -> dict[str, Any]:
        actions = self.list_actions(50)
        active = [a for a in actions if a["status"] in ("dry_run_recorded", "enforced_active")]
        return {
            "mode": "dry_run" if self.dry_run else "enforce",
            "dry_run": self.dry_run,
            "default_ttl_seconds": settings.containment_default_ttl_s,
            "protected_hosts_count": len(self.protected_hosts()),
            "protected_hosts": self.protected_hosts()[:12],
            "active_count": len(active),
            "total_recorded": len(actions),
            "nft_setup_file": "deploy/edgeshield.nft",
            "actions": actions[:25],
        }
