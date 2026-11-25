"""Continuous On-Device Live Network & Wire-Traffic Monitor for Arm Linux.

Runs bin/live_sniffer (cap_net_raw, zero root web server) in a background thread,
sliding over 2-second windows to passively detect:
  1. Network intrusions & malware behaviour (TCP SYN port scans, SSH/IoT brute-force,
     UDP floods, DNS tunneling, C2/cryptominer beacons, ICMP covert channels + ML flow scoring).
  2. Phishing URLs transmitted in live DNS queries and cleartext HTTP wire frames.
  3. Scam SMS / fraudulent payloads transmitted in live SIP MESSAGE / SMTP / SMPP wire frames.
  4. Per-device LAN/IoT traffic baselines (for Home/SME IoT behaviour monitoring).
"""
from __future__ import annotations

import logging
import math
import re
import struct
import subprocess
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from app.detectors.network import (
    MALWARE_C2_PORTS,
    STRATUM_MINER_PORTS,
    UDP_AMP_PORTS,
    analyse_flow_records_hybrid,
)
from app.detectors.sms import analyse_sms
from app.detectors.url import analyse_url
from app.events import create_event
from app.ml.flow_reducer import FlowReducer, Packet, parse_raw_frame
from app.schemas import EvidenceItem, ThreatEvent

log = logging.getLogger("edgeshield.live_monitor")

_HTTP_REQ_RE = re.compile(rb"^(?:GET|POST|PUT|HEAD)\s+([^\r\n ]+)\s+HTTP/", re.I)
_HTTP_HOST_RE = re.compile(rb"(?im)^Host:\s*([^\r\n]+)")
_HTTP_REF_RE = re.compile(rb"(?im)^Referer:\s*(https?://[^\r\n]+)")
_SIP_MSG_RE = re.compile(rb"^MESSAGE\s+sip:", re.I)


def detect_default_interface() -> str:
    """Detect the primary active LAN network interface from /proc/net/route."""
    try:
        lines = Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "00000000" and parts[0] != "lo":
                return parts[0]
        for line in lines:
            parts = line.split()
            if parts and parts[0] not in {"lo", "docker0", "can0"}:
                return parts[0]
    except Exception:
        pass
    return "enP2p33s0"


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _parse_dns_qname(payload: bytes) -> str | None:
    """Extract the first DNS query domain name from a UDP/53 packet payload."""
    if len(payload) < 13:
        return None
    flags = struct.unpack("!H", payload[2:4])[0]
    if flags & 0x8000:  # Response, not query
        return None
    pos = 12
    labels = []
    while pos < len(payload):
        ln = payload[pos]
        if ln == 0:
            break
        if ln > 63 or pos + 1 + ln > len(payload):
            return None
        labels.append(payload[pos + 1 : pos + 1 + ln].decode("ascii", errors="ignore"))
        pos += 1 + ln
    return ".".join(labels).lower() if labels else None


class LiveNetworkMonitor:
    def __init__(
        self,
        on_event: Callable[[ThreatEvent], Any],
        window_s: float = 2.0,
        cooldown_s: float = 12.0,
        ignored_ports: set[int] | None = None,
    ) -> None:
        self._on_event = on_event
        self.window_s = max(1.0, float(window_s))
        self.cooldown_s = cooldown_s
        self.ignored_ports = ignored_ports or {8000, 8001, 8002}
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.interface = ""
        self.started_at: float | None = None
        self.packets_captured = 0
        self.windows_processed = 0
        self.alerts_emitted = 0
        self.last_error = ""
        self._dedup: dict[str, float] = {}
        self._devices: dict[str, dict[str, Any]] = {}
        self._sniffer_bin = Path(__file__).resolve().parents[2] / "bin" / "live_sniffer"

    def status(self) -> dict[str, Any]:
        with self._lock:
            uptime = round(time.time() - self.started_at, 1) if (self._running and self.started_at) else 0.0
            return {
                "running": self._running,
                "interface": self.interface,
                "sniffer_binary": str(self._sniffer_bin),
                "sniffer_available": self._sniffer_bin.is_file(),
                "window_s": self.window_s,
                "uptime_seconds": uptime,
                "packets_captured": self.packets_captured,
                "windows_processed": self.windows_processed,
                "alerts_emitted": self.alerts_emitted,
                "tracked_devices": len(self._devices),
                "last_error": self.last_error,
            }

    def list_devices(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for ip, info in sorted(self._devices.items(), key=lambda kv: kv[1]["last_seen"], reverse=True)[:50]:
                out.append({
                    "ip": ip,
                    "packets": info["packets"],
                    "bytes": info["bytes"],
                    "distinct_ports": len(info["ports"]),
                    "protocols": sorted(info["protocols"]),
                    "last_seen": info["last_seen"],
                    "alerts": info["alerts"],
                })
            return out

    def start(self, interface: str = "auto") -> dict[str, Any]:
        with self._lock:
            if self._running:
                return self.status()
            if not self._sniffer_bin.is_file():
                self.last_error = f"live_sniffer binary missing at {self._sniffer_bin}"
                return self.status()
            iface = detect_default_interface() if (not interface or interface == "auto") else interface
            self.interface = iface
            self.last_error = ""
            try:
                self._proc = subprocess.Popen(
                    [str(self._sniffer_bin), iface],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except Exception as exc:
                self.last_error = str(exc)
                return self.status()
            self._running = True
            self.started_at = time.time()
            self._thread = threading.Thread(target=self._capture_loop, name="edgeshield-live-sniffer", daemon=True)
            self._thread.start()
            log.info("LiveNetworkMonitor started on interface %s", iface)
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._running = False
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return self.status()

    def _emit_dedup(self, dedup_key: str, event: ThreatEvent) -> None:
        now = time.time()
        last = self._dedup.get(dedup_key, 0.0)
        if now - last < self.cooldown_s:
            return
        self._dedup[dedup_key] = now
        # Prune old keys
        if len(self._dedup) > 500:
            self._dedup = {k: v for k, v in self._dedup.items() if now - v < 60.0}
        with self._lock:
            self.alerts_emitted += 1
            src_ip = str(event.metadata.get("src_ip", "") or "")
            if src_ip and src_ip in self._devices:
                self._devices[src_ip]["alerts"] += 1
        try:
            self._on_event(event)
        except Exception as exc:
            log.warning("Live monitor event persistence error: %s", exc)

    def _inspect_l7_wire(self, pkt: Packet, payload: bytes) -> None:
        if not payload:
            return
        # 1. Live DNS query inspection (phishing domains + DNS tunneling)
        if pkt.proto == 17 and pkt.dport == 53:
            qname = _parse_dns_qname(payload)
            if qname and "." in qname and not qname.endswith((".local", ".arpa", ".lan")):
                first_lbl = qname.split(".")[0]
                if len(first_lbl) >= 24 or (_shannon_entropy(first_lbl) >= 3.6 and len(first_lbl) >= 16) or len(qname) >= 65:
                    ev = create_event(
                        source="network_ids",
                        category="dns_tunneling",
                        prediction="anomalous",
                        confidence=0.92,
                        evidence=[EvidenceItem(
                            code="LIVE_DNS_TUNNEL_QUERY",
                            description=f"Live UDP/53 DNS query has high-entropy/oversized label ({len(first_lbl)} chars, H={_shannon_entropy(first_lbl):.2f}): {qname[:80]}",
                            weight=55,
                        )],
                        actions=[
                            f"Inspect host {pkt.src} for DNS-tunneling malware or data exfiltration.",
                            f"Block suspicious domain {qname[:60]} at the local DNS resolver.",
                        ],
                        model_version="live-wire-dpi-v1",
                        raw_input=f"DNS {pkt.src} -> {qname}",
                        metadata={"src_ip": pkt.src, "dst_ip": pkt.dst, "dst_port": 53, "protocol": "UDP", "dns_query": qname, "live_capture": True},
                        base_score=25,
                    )
                    self._emit_dedup(f"dnstun:{pkt.src}", ev)
                else:
                    # Pass non-tunnel DNS domain through URL phishing detector!
                    url_ev = analyse_url("http://" + qname)
                    if url_ev.prediction == "suspicious":
                        url_ev.metadata["live_capture"] = True
                        url_ev.metadata["src_ip"] = pkt.src
                        url_ev.metadata["wire_protocol"] = "DNS"
                        self._emit_dedup(f"dnsphish:{qname}", url_ev)
            return

        # 2. Live cleartext HTTP wire inspection for phishing URLs (exclude own EdgeShield API ports)
        if pkt.dport in {80, 8080} and pkt.dport not in self.ignored_ports:
            m_req = _HTTP_REQ_RE.match(payload)
            if m_req:
                path_b = m_req.group(1)
                m_host = _HTTP_HOST_RE.search(payload)
                m_ref = _HTTP_REF_RE.search(payload)
                urls_to_check = []
                if m_ref:
                    urls_to_check.append(m_ref.group(1).decode("utf-8", errors="ignore").strip())
                if m_host:
                    host_s = m_host.group(1).decode("utf-8", errors="ignore").strip().split(":")[0]
                    path_s = path_b.decode("utf-8", errors="ignore").strip()
                    if host_s and not host_s.startswith(("127.", "192.168.", "10.")):
                        urls_to_check.append(f"http://{host_s}{path_s}")
                for u in urls_to_check[:2]:
                    url_ev = analyse_url(u)
                    if url_ev.prediction == "suspicious":
                        url_ev.metadata["live_capture"] = True
                        url_ev.metadata["src_ip"] = pkt.src
                        url_ev.metadata["wire_protocol"] = "HTTP"
                        self._emit_dedup(f"httpphish:{u[:80]}", url_ev)
            return

        # 3. Live SIP MESSAGE / messaging wire inspection for Scam SMS (UDP/5060, TCP/UDP 25/2775)
        if pkt.dport in {5060, 25, 2775} and len(payload) >= 24:
            if _SIP_MSG_RE.match(payload) or b"X-EdgeShield-Wire-Channel: sms" in payload:
                sep = payload.find(b"\r\n\r\n")
                body_bytes = payload[sep + 4 :] if sep != -1 else payload
                msg_text = body_bytes.decode("utf-8", errors="ignore").strip()
                if len(msg_text) >= 12:
                    sms_ev = analyse_sms(msg_text)
                    if sms_ev.prediction != "no_obvious_indicators":
                        sms_ev.metadata["live_capture"] = True
                        sms_ev.metadata["src_ip"] = pkt.src
                        sms_ev.metadata["wire_protocol"] = "SIP/SMS"
                        self._emit_dedup(f"wiresms:{msg_text[:48]}", sms_ev)

    def _flush_window(
        self,
        packets: list[Packet],
        syn_ports: dict[tuple[str, str], set[int]],
        syn_to_port: dict[tuple[str, str, int], int],
        udp_counts: dict[tuple[str, str], list[int]],
        c2_hits: dict[tuple[str, str], set[int]],
        icmp_large: dict[tuple[str, str], list[int]],
    ) -> None:
        with self._lock:
            self.windows_processed += 1

        # A. Unanswered TCP SYN / multi-port scan detection (catches 65-port scan in real time!)
        for (src, dst), ports in syn_ports.items():
            if len(ports) >= 10:
                ev = create_event(
                    source="network_ids",
                    category="port_scan",
                    prediction="anomalous",
                    confidence=0.95,
                    evidence=[EvidenceItem(
                        code="LIVE_PORT_SCAN_SYN",
                        description=f"Live wire capture: {src} probed {len(ports)} distinct TCP ports on {dst} within {self.window_s:.0f}s.",
                        weight=60,
                    )],
                    actions=[
                        f"Block scanning source IP {src} on the edge firewall.",
                        "Review open services on this Arm board and disable unused ports.",
                    ],
                    model_version="live-network-hybrid-v2",
                    raw_input=f"LIVE SCAN {src} -> {dst} ({len(ports)} ports: {sorted(ports)[:8]}...)",
                    metadata={"src_ip": src, "dst_ip": dst, "distinct_ports": len(ports), "sample_ports": sorted(ports)[:20], "live_capture": True},
                    base_score=25,
                )
                self._emit_dedup(f"scan:{src}:{dst}", ev)

        # B. SSH / IoT / Telnet / SMB / Redis brute-force bursts (new SYN bursts to a single sensitive port)
        for (src, dst, dport), cnt in syn_to_port.items():
            if dport in {21, 22, 23, 2323, 445, 1883, 3306, 3389, 5555, 5900, 6379} and cnt >= 6:
                ev = create_event(
                    source="network_ids",
                    category="brute_force_attack",
                    prediction="anomalous",
                    confidence=0.90,
                    evidence=[EvidenceItem(
                        code="LIVE_SERVICE_BRUTE_FORCE",
                        description=f"Live wire capture: {src} initiated {cnt} rapid connection attempts to sensitive port {dport} on {dst}.",
                        weight=55,
                    )],
                    actions=[
                        f"Rate-limit or block {src} attempting rapid logins on port {dport}.",
                        "Disable password auth (use SSH keys) and close exposed IoT/DB ports.",
                    ],
                    model_version="live-network-hybrid-v2",
                    raw_input=f"LIVE BRUTE-FORCE {src} -> {dst}:{dport} ({cnt} SYN probes/{self.window_s:.0f}s)",
                    metadata={"src_ip": src, "dst_ip": dst, "dst_port": dport, "syn_count": cnt, "live_capture": True},
                    base_score=20,
                )
                self._emit_dedup(f"bf:{src}:{dport}", ev)

        # C. UDP amplification / flood bursts
        for (src, dst), dports in udp_counts.items():
            if len(dports) >= 45:
                top_ports = [p for p, _ in Counter(dports).most_common(4)]
                ev = create_event(
                    source="network_ids",
                    category="udp_flood",
                    prediction="anomalous",
                    confidence=0.92,
                    evidence=[EvidenceItem(
                        code="LIVE_UDP_FLOOD",
                        description=f"Live wire capture: {src} sent {len(dports)} UDP datagrams in {self.window_s:.0f}s to ports {top_ports}.",
                        weight=55,
                    )],
                    actions=[
                        f"Drop UDP flood traffic from {src} at the gateway firewall.",
                        "Verify IoT devices on the LAN are not compromised by a DDoS botnet.",
                    ],
                    model_version="live-network-hybrid-v2",
                    raw_input=f"LIVE UDP FLOOD {src} -> {dst} ({len(dports)} pkts to ports {top_ports})",
                    metadata={"src_ip": src, "dst_ip": dst, "udp_packets": len(dports), "top_ports": top_ports, "live_capture": True},
                    base_score=20,
                )
                self._emit_dedup(f"udpflood:{src}:{dst}", ev)

        # D. Malware C2 / Cryptominer / Reverse-Shell beacons
        for (src, dst), ports in c2_hits.items():
            ev = create_event(
                source="network_ids",
                category="malware_c2_beacon",
                prediction="anomalous",
                confidence=0.90,
                evidence=[EvidenceItem(
                    code="LIVE_C2_BEACON",
                    description=f"Live wire capture: traffic between {src} and {dst} targeted high-risk C2/miner/Tor port(s) {sorted(ports)}.",
                    weight=55,
                )],
                actions=[
                    f"Quarantine host {src} immediately and inspect running processes.",
                    f"Block outbound/inbound connections on C2 port(s) {sorted(ports)}.",
                ],
                model_version="live-network-hybrid-v2",
                raw_input=f"LIVE C2 BEACON {src} -> {dst} ports={sorted(ports)}",
                metadata={"src_ip": src, "dst_ip": dst, "c2_ports": sorted(ports), "live_capture": True},
                base_score=25,
            )
            self._emit_dedup(f"c2:{src}:{dst}", ev)

        # E. Oversized / rapid ICMP covert channel
        for (src, dst), sizes in icmp_large.items():
            if len(sizes) >= 4:
                avg_sz = sum(sizes) / len(sizes)
                ev = create_event(
                    source="network_ids",
                    category="icmp_covert_channel",
                    prediction="anomalous",
                    confidence=0.88,
                    evidence=[EvidenceItem(
                        code="LIVE_ICMP_COVERT",
                        description=f"Live wire capture: {src} sent {len(sizes)} oversized ICMP packets (avg {avg_sz:.0f} B) to {dst}.",
                        weight=50,
                    )],
                    actions=[
                        f"Inspect host {src} for ICMP tunneling tools or covert data exfiltration.",
                        "Rate-limit oversized ICMP Echo payloads on the local firewall.",
                    ],
                    model_version="live-network-hybrid-v2",
                    raw_input=f"LIVE ICMP COVERT {src} -> {dst} ({len(sizes)} pkts, avg {avg_sz:.0f}B)",
                    metadata={"src_ip": src, "dst_ip": dst, "icmp_packets": len(sizes), "avg_bytes": round(avg_sz, 1), "live_capture": True},
                    base_score=20,
                )
                self._emit_dedup(f"icmp:{src}:{dst}", ev)

        # F. Run network_model_v2r ML scorer on multi-packet flows in this window
        if len(packets) >= 4:
            try:
                from app.ml import network_runtime
                bundle = network_runtime.load_bundle()
                if bundle is not None:
                    red = FlowReducer()
                    for p in packets:
                        red.add(p)
                    recs = red.finish()
                    if recs:
                        scores = network_runtime.score_records(bundle, recs)
                        evs = analyse_flow_records_hybrid(recs, scores, bundle["threshold"])
                        for ev in evs:
                            if ev.prediction == "anomalous":
                                ev.metadata["live_capture"] = True
                                self._emit_dedup(f"ml:{ev.metadata.get('src_ip')}:{ev.metadata.get('dst_port')}", ev)
            except Exception as exc:
                log.debug("Live window ML pass error: %s", exc)

    def _capture_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stream = proc.stdout
        win_start = time.time()
        win_packets: list[Packet] = []
        syn_ports: dict[tuple[str, str], set[int]] = defaultdict(set)
        syn_to_port: dict[tuple[str, str, int], int] = defaultdict(int)
        udp_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
        c2_hits: dict[tuple[str, str], set[int]] = defaultdict(set)
        icmp_large: dict[tuple[str, str], list[int]] = defaultdict(list)

        try:
            while self._running:
                hdr = stream.read(16)
                if not hdr or len(hdr) < 16:
                    if proc.poll() is not None:
                        err = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
                        if err.strip():
                            self.last_error = err.strip()[:300]
                        break
                    continue
                ts_sec, ts_usec, cap_len, _orig = struct.unpack("=IIII", hdr)
                if cap_len <= 0 or cap_len > 65536:
                    continue
                frame = stream.read(cap_len)
                if len(frame) < cap_len:
                    break
                ts = ts_sec + ts_usec / 1e6
                parsed = parse_raw_frame(ts, frame, 1)
                if parsed is not None:
                    pkt, l4_bytes = parsed
                    with self._lock:
                        self.packets_captured += 1
                        # Update IoT/LAN device telemetry map
                        dev = self._devices.setdefault(pkt.src, {
                            "packets": 0, "bytes": 0, "ports": set(), "protocols": set(), "last_seen": ts, "alerts": 0
                        })
                        dev["packets"] += 1
                        dev["bytes"] += pkt.payload
                        if pkt.dport:
                            dev["ports"].add(pkt.dport)
                        dev["protocols"].add({6: "TCP", 17: "UDP", 1: "ICMP"}.get(pkt.proto, str(pkt.proto)))
                        dev["last_seen"] = ts

                    # Skip EdgeShield's own HTTP API traffic so viewing the dashboard never self-triggers
                    if pkt.sport in self.ignored_ports or pkt.dport in self.ignored_ports:
                        continue

                    win_packets.append(pkt)
                    # Track TCP SYN probes
                    if pkt.proto == 6 and (pkt.flags & 0x02) and not (pkt.flags & 0x10):
                        syn_ports[(pkt.src, pkt.dst)].add(pkt.dport)
                        syn_to_port[(pkt.src, pkt.dst, pkt.dport)] += 1
                    # Track UDP bursts
                    if pkt.proto == 17 and pkt.dport != 53 and pkt.dport != 5060:
                        udp_counts[(pkt.src, pkt.dst)].append(pkt.dport)
                    # Track C2 / cryptominer ports
                    if pkt.dport in MALWARE_C2_PORTS or pkt.dport in STRATUM_MINER_PORTS:
                        c2_hits[(pkt.src, pkt.dst)].add(pkt.dport)
                    # Track oversized ICMP covert payloads
                    if pkt.proto == 1 and pkt.payload >= 200:
                        icmp_large[(pkt.src, pkt.dst)].append(pkt.payload)

                    # L7 wire DPI for DNS phishing/tunnel, HTTP phishing, SIP/SMS scams
                    self._inspect_l7_wire(pkt, l4_bytes)

                now = time.time()
                if now - win_start >= self.window_s:
                    if win_packets or syn_ports or udp_counts or c2_hits or icmp_large:
                        self._flush_window(win_packets, syn_ports, syn_to_port, udp_counts, c2_hits, icmp_large)
                    win_packets.clear()
                    syn_ports.clear()
                    syn_to_port.clear()
                    udp_counts.clear()
                    c2_hits.clear()
                    icmp_large.clear()
                    win_start = now
        except Exception as exc:
            self.last_error = f"Capture loop exception: {exc}"
            log.warning("LiveNetworkMonitor stopped with error: %s", exc)
        finally:
            with self._lock:
                self._running = False
