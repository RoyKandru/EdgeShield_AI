import math
from collections import Counter
from pathlib import Path
from app.events import create_event
from app.schemas import EvidenceItem, ThreatEvent

MALWARE_C2_PORTS = {4444, 6667, 1337, 31337, 5555, 9050, 9051}
STRATUM_MINER_PORTS = {3333, 14444, 45700}
BRUTE_FORCE_PORTS = {21, 22, 23, 2323, 445, 1433, 1883, 3306, 3389, 5432, 5900, 6379, 27017}
UDP_AMP_PORTS = {19, 123, 161, 389, 1900, 5353, 11211}


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _rule_evidence(flow: dict) -> list[EvidenceItem]:
    """Production flow heuristics + malware-behaviour & stealth intrusion rules."""
    evidence = []
    packets = int(flow.get("packets", 0))
    duration = float(flow.get("duration_seconds", 0.0))
    dst_port = int(flow.get("dst_port", 0))
    syn_count = int(flow.get("syn_count", flow.get("syn_packets", 0)))
    rst_count = int(flow.get("rst_packets", 0))
    failed_conns = int(flow.get("failed_connections", 0))
    bytes_count = int(flow.get("bytes", 0))
    proto = str(flow.get("protocol", "TCP")).upper()
    sw_ndport = int(flow.get("sw_ndport", 0))
    dns_query = str(flow.get("dns_query", "") or "")
    bpp = bytes_count / max(packets, 1)

    if syn_count >= 20:
        evidence.append(EvidenceItem(
            code="SYN_BURST",
            description=f"Flow contains {syn_count} SYN packets; possible SYN flood or rapid scanning.",
            weight=35,
        ))
    if packets >= 1000 and duration <= 10:
        evidence.append(EvidenceItem(
            code="HIGH_PACKET_RATE",
            description="High packet volume was observed over a short flow duration.",
            weight=35,
        ))
    if dst_port in {22, 23, 2323, 3389, 445, 5900} and syn_count >= 8:
        evidence.append(EvidenceItem(
            code="SENSITIVE_PORT_ATTEMPT",
            description=f"Repeated connection attempts target sensitive remote-admin port {dst_port}.",
            weight=30,
        ))
    if bytes_count > 5_000_000 and duration < 5:
        evidence.append(EvidenceItem(
            code="HIGH_BYTE_BURST",
            description="Large byte volume was observed in a short duration.",
            weight=25,
        ))

    # --- Malware Behaviour & Stealth Intrusion Rules ---
    if dst_port in MALWARE_C2_PORTS and packets >= 2:
        evidence.append(EvidenceItem(
            code="C2_MALWARE_PORT",
            description=f"Traffic targets high-risk C2/reverse-shell/Tor port {dst_port} ({packets} packets).",
            weight=45,
        ))
    if dst_port in STRATUM_MINER_PORTS and packets >= 3:
        evidence.append(EvidenceItem(
            code="CRYPTOMINER_STRATUM",
            description=f"Traffic matches Stratum cryptomining pool port {dst_port}.",
            weight=45,
        ))
    if dst_port == 443 and duration >= 300 and packets >= 50 and bpp <= 120:
        evidence.append(EvidenceItem(
            code="STEALTH_C2_BEACON_443",
            description=f"Long-lived low-payload periodic HTTPS flow ({packets} pkts over {duration:.0f}s, {bpp:.0f} B/pkt) indicates C2 beaconing.",
            weight=45,
        ))
    if dst_port == 53:
        dns_bad = False
        if dns_query:
            first_lbl = dns_query.split(".")[0]
            if len(first_lbl) >= 24 or _shannon_entropy(first_lbl) >= 3.6 or len(dns_query) >= 65:
                dns_bad = True
        if dns_bad or (packets >= 35 and bpp >= 140) or bytes_count >= 25_000:
            evidence.append(EvidenceItem(
                code="DNS_TUNNELING_EXFIL",
                description=f"Anomalous high-volume/high-entropy DNS traffic ({packets} pkts, {bytes_count} B) suggests DNS tunneling.",
                weight=50,
            ))
    if proto in {"ICMP", "ICMPV6"} and (bpp >= 200 or (packets >= 40 and duration <= 10)):
        evidence.append(EvidenceItem(
            code="ICMP_COVERT_OR_FLOOD",
            description=f"Anomalous ICMP payload/rate ({packets} pkts, avg {bpp:.0f} B/pkt) suggests covert tunneling or ping flood.",
            weight=45,
        ))
    if proto == "UDP" and ((packets >= 80 and duration <= 15) or (dst_port in UDP_AMP_PORTS and packets >= 25)):
        evidence.append(EvidenceItem(
            code="UDP_FLOOD_AMP",
            description=f"High-rate UDP burst to port {dst_port} ({packets} pkts in {duration:.1f}s) suggests UDP flood or amplification.",
            weight=45,
        ))
    if rst_count >= 25:
        evidence.append(EvidenceItem(
            code="TCP_RST_FLOOD",
            description=f"Flow contains {rst_count} TCP RST packets indicating a reset flood or aggressive scan.",
            weight=40,
        ))
    if failed_conns >= 10 or (syn_count >= 10 and failed_conns >= 6) or (dst_port in BRUTE_FORCE_PORTS and (failed_conns >= 5 or (packets >= 40 and duration <= 15 and syn_count >= 4))):
        evidence.append(EvidenceItem(
            code="BRUTE_FORCE_OR_LATERAL",
            description=f"Repeated failed connections/probes targeting service port {dst_port} (failed={failed_conns}, syn={syn_count}).",
            weight=45,
        ))
    if dst_port in {80, 443, 8000, 8080} and duration >= 60 and packets >= 20 and bpp <= 70 and syn_count >= 5:
        evidence.append(EvidenceItem(
            code="SLOWLORIS_DOS",
            description=f"Prolonged low-byte HTTP flow ({packets} pkts, {bpp:.0f} B/pkt over {duration:.0f}s) matches Slowloris DoS.",
            weight=45,
        ))
    if dst_port in {8088, 8443, 9001, 21} and bytes_count >= 500_000 and duration >= 20:
        evidence.append(EvidenceItem(
            code="SLOW_EXFILTRATION",
            description=f"Sustained large outbound transfer ({bytes_count} B over {duration:.0f}s) on non-standard port {dst_port}.",
            weight=40,
        ))
    if proto == "UDP" and dst_port in {67, 68} and packets >= 15:
        evidence.append(EvidenceItem(
            code="ROGUE_DHCP_ACTIVITY",
            description=f"High-frequency DHCP traffic ({packets} pkts) indicates DHCP starvation or rogue server activity.",
            weight=40,
        ))
    if sw_ndport >= 10:
        evidence.append(EvidenceItem(
            code="MULTI_PORT_SCAN",
            description=f"Source contacted {sw_ndport} distinct destination ports within one minute.",
            weight=45,
        ))

    try:
        from app.sci.policy import get_active_policy
        _pol = get_active_policy()
        _src_ip = str(flow.get("src_ip", "")).lower()
        _dst_ip = str(flow.get("dst_ip", "")).lower()
        for _ind in _pol.get("verified_deny_indicators", []):
            if _ind and (_ind in (_src_ip, _dst_ip) or _ind in dns_query.lower()):
                evidence.append(EvidenceItem(code="SCI_VERIFIED_NET_IOC", description=f"Flow matches SCI verified network threat IOC '{_ind}'.", weight=55))
                break
        _nw = (_pol.get("rule_weights") or {}).get("network") or {}
        if _nw:
            for _ev in evidence:
                if _ev.code in _nw:
                    _ev.weight = int(_nw[_ev.code])
    except Exception:
        pass
    return evidence


def analyse_flows(flows: list[dict]) -> list[ThreatEvent]:
    from app.config import settings
    try:
        from app.sci.policy import get_active_policy
        _pol = get_active_policy()
    except Exception:
        _pol = {}
    npu_client = None
    npu_thr = float((_pol.get("thresholds") or {}).get("network_npu_threshold", 0.80))
    if settings.ml_detectors:
        try:
            from app.edge_models.network_npu import get_network_npu_client
            _c = get_network_npu_client()
            if _c.ready:
                npu_client = _c
        except Exception:
            npu_client = None

    events = []
    for flow in flows:
        dst_port = int(flow.get("dst_port", 0))
        evidence = _rule_evidence(flow)
        meta_out = dict(flow)
        npu_prob = None
        if npu_client is not None:
            try:
                nr = npu_client.infer_flow_dict(flow, threshold=npu_thr)
                npu_prob = float(nr["score"])
                meta_out.update({
                    "npu_net_score": round(npu_prob, 4),
                    "npu_net_latency_ms": round(float(nr["npu_latency_ms"]), 2),
                    "npu_net_core": nr.get("core", "NPU_CORE_0"),
                    "npu_net_backend": nr.get("backend", "rknn-npu"),
                    "npu_net_threshold": npu_thr,
                    "npu_net_model": nr.get("model_id", "network-ids-1dcnn-mlp-v2r26-rknn-v1"),
                })
                if npu_prob >= npu_thr:
                    evidence.append(EvidenceItem(
                        code="NPU_NET_IDS_ANOMALY",
                        description=(
                            f"RK3588 NPU_CORE_0 Network 1D-CNN+MLP scored this flow {npu_prob:.3f} "
                            f"(>= {npu_thr:.2f}, {nr['npu_latency_ms']:.2f}ms on NPU)."
                        ),
                        weight=35,
                    ))
            except Exception:
                pass
        if evidence:
            prediction, category, confidence, base = "anomalous", "network_anomaly", max(0.80, npu_prob or 0.80), 15
            actions = [
                "Review the source and destination endpoints immediately.",
                "Isolate or firewall-block the suspicious source IP if unauthorized.",
                "Preserve packet capture evidence for further investigation.",
            ]
        else:
            prediction, category, confidence, base = "no_configured_anomaly", "network_flow", 0.55, 0
            actions = ["Continue monitoring; absence of a rule match does not prove traffic is benign."]
            evidence = [EvidenceItem(
                code="NO_RULE_MATCH",
                description="No configured flow heuristic was triggered.",
                weight=0,
            )]

        events.append(create_event(
            source="network_ids",
            category=category,
            prediction=prediction,
            confidence=confidence,
            evidence=evidence,
            actions=actions,
            model_version="network-rules-v0.1",
            raw_input=f"{flow.get('src_ip')}:{flow.get('src_port')} -> {flow.get('dst_ip')}:{dst_port}",
            metadata=meta_out,
            base_score=base,
        ))
    return events


def read_pcap_flows(path: str) -> list[dict]:
    """Zero-dependency fast PCAP/PCAPNG flow reader."""
    from app.ml.flow_reducer import packets_from_pcap
    flow_map = {}
    for p in packets_from_pcap(path):
        proto = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6"}.get(p.proto, "OTHER")
        key = (p.src, p.dst, p.sport, p.dport, proto)
        rec = flow_map.setdefault(key, {
            "src_ip": p.src, "dst_ip": p.dst, "src_port": p.sport, "dst_port": p.dport,
            "protocol": proto, "packets": 0, "bytes": 0, "first_ts": p.ts,
            "last_ts": p.ts, "syn_count": 0, "rst_packets": 0,
        })
        rec["packets"] += 1
        rec["bytes"] += p.payload
        rec["first_ts"] = min(rec["first_ts"], p.ts)
        rec["last_ts"] = max(rec["last_ts"], p.ts)
        if p.proto == 6:
            if p.flags & 0x02:
                rec["syn_count"] += 1
            if p.flags & 0x04:
                rec["rst_packets"] += 1
    flows = []
    for rec in flow_map.values():
        rec["duration_seconds"] = max(0.0, rec.pop("last_ts") - rec.pop("first_ts"))
        flows.append(rec)
    return flows


def analyse_flow_records_hybrid(records, scores, threshold: float) -> list[ThreatEvent]:
    from app.config import settings
    try:
        from app.sci.policy import get_active_policy
        _pol = get_active_policy()
    except Exception:
        _pol = {}
    npu_thr = float((_pol.get("thresholds") or {}).get("network_npu_threshold", 0.80))
    npu_results = None
    if settings.ml_detectors and records:
        try:
            from app.edge_models.network_npu import get_network_npu_client
            _nc = get_network_npu_client()
            if _nc.ready:
                npu_results = _nc.infer_batch([r.vector() for r in records], threshold=npu_thr)
        except Exception:
            npu_results = None

    events = []
    for idx_r, (rec, score) in enumerate(zip(records, scores)):
        flow = rec.rule_flow()
        f = rec.features
        evidence = _rule_evidence(flow)
        ml_flag = bool(score >= threshold)
        npu_meta = {}
        if npu_results is not None and idx_r < len(npu_results):
            nr = npu_results[idx_r]
            npu_p = float(nr["score"])
            npu_meta = {
                "npu_net_score": round(npu_p, 4),
                "npu_net_latency_ms": round(float(nr["npu_latency_ms"]), 2),
                "npu_net_core": nr.get("core", "NPU_CORE_0"),
                "npu_net_backend": nr.get("backend", "rknn-npu"),
                "npu_net_threshold": npu_thr,
                "npu_net_model": nr.get("model_id", "network-ids-1dcnn-mlp-v2r26-rknn-v1"),
            }
            if npu_p >= npu_thr:
                evidence.append(EvidenceItem(
                    code="NPU_NET_IDS_ANOMALY",
                    description=(
                        f"RK3588 NPU_CORE_0 Network 1D-CNN+MLP scored this flow {npu_p:.3f} "
                        f"(>= {npu_thr:.2f}, {nr['npu_latency_ms']:.2f}ms on NPU)."
                    ),
                    weight=35,
                ))
        if ml_flag:
            evidence.append(EvidenceItem(
                code="ML_FLOW_MODEL",
                description=f"Network model v2 scored this flow {score:.3f} (alert threshold {threshold:.3f}).",
                weight=35,
            ))
            if f["sw_ndport"] >= 10 or f["sw_flows"] >= 100 or f["dw_nsrc"] >= 10:
                evidence.append(EvidenceItem(
                    code="HOST_WINDOW_CONTEXT",
                    description=(f"Within the same minute the source opened {int(f['sw_flows'])} flows to "
                                 f"{int(f['sw_ndport'])} distinct ports on {int(f['sw_ndst'])} host(s); the "
                                 f"destination saw {int(f['dw_nsrc'])} distinct source(s)."),
                    weight=10,
                ))
        if evidence:
            prediction, category, base = "anomalous", "network_anomaly", 15
            confidence = max(0.75, float(score)) if ml_flag else 0.80
            actions = [
                "Review the source and destination endpoints immediately.",
                "Isolate or block unauthorized scanning/C2 source IPs on the local firewall.",
                "Preserve packet capture evidence for further investigation.",
            ]
        else:
            prediction, category, base, confidence = "no_configured_anomaly", "network_flow", 0, 0.55
            actions = ["Continue monitoring; absence of a rule or model match does not prove traffic is benign."]
            evidence = [EvidenceItem(
                code="NO_RULE_MATCH",
                description=f"No flow heuristic fired and the model score {score:.3f} is below the threshold.",
                weight=0,
            )]
        events.append(create_event(
            source="network_ids",
            category=category,
            prediction=prediction,
            confidence=confidence,
            evidence=evidence,
            actions=actions,
            model_version="network-hybrid-v2",
            raw_input=f"{rec.src}:{rec.sport} -> {rec.dst}:{rec.dport}",
            metadata={
                **flow,
                "ml_score": float(score), "ml_threshold": float(threshold),
                "sw_flows": f["sw_flows"], "sw_ndport": f["sw_ndport"],
                "sw_ndst": f["sw_ndst"], "dw_nsrc": f["dw_nsrc"],
                **npu_meta,
            },
            base_score=base,
        ))
    return events


def analyse_pcap(path: str) -> list[ThreatEvent]:
    """Hybrid (ML OR malware-behaviour rules + unanswered SYN scan check)."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    from app.ml import network_runtime
    bundle = network_runtime.load_bundle()
    if bundle is None:
        return analyse_flows(read_pcap_flows(path))
    from app.ml.flow_reducer import FlowReducer, packets_from_pcap
    reducer = FlowReducer()
    for p in packets_from_pcap(path):
        reducer.add(p)
    records = reducer.finish()
    scores = network_runtime.score_records(bundle, records)
    events = analyse_flow_records_hybrid(records, scores, bundle["threshold"])

    # Catch stealth 1-packet unanswered SYN scans that MIN_PACKETS=2 omitted
    flagged_pairs = {(e.metadata.get("src_ip"), e.metadata.get("dst_ip")) for e in events if e.prediction == "anomalous"}
    for (src, dst), ports in reducer.syn_ports_by_src.items():
        if len(ports) >= 10 and (src, dst) not in flagged_pairs:
            events.append(create_event(
                source="network_ids",
                category="port_scan",
                prediction="anomalous",
                confidence=0.90,
                evidence=[EvidenceItem(
                    code="UNANSWERED_SYN_PORT_SCAN",
                    description=f"Source {src} sent unanswered TCP SYN probes to {len(ports)} distinct ports on {dst}.",
                    weight=55,
                )],
                actions=[
                    f"Investigate or block source IP {src} scanning {len(ports)} ports.",
                    "Close unused listening services on the edge device.",
                ],
                model_version="network-hybrid-v2",
                raw_input=f"{src} -> {dst} ({len(ports)} distinct SYN ports)",
                metadata={"src_ip": src, "dst_ip": dst, "distinct_ports": len(ports), "sample_ports": sorted(ports)[:20]},
                base_score=20,
            ))
    return events
