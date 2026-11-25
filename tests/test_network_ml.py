import sys
from pathlib import Path

import numpy as np
import pytest

from app.detectors import network as net
from app.ml import flow_reducer as fr
from app.ml import network_runtime as nrt

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / nrt.BUNDLE_FILE
needs_model = pytest.mark.skipif(not BUNDLE.exists(), reason="models/network_model_v2r.joblib not built")

SYN, ACK, FIN, RST = 0x02, 0x10, 0x01, 0x04


def flow_packets(src, dst, sport, dport, proto, t0, events):
    """events: (dt_seconds, 'f'|'b', payload_bytes, tcp_flags). First event must be 'f'."""
    out = []
    for dt, d, pl, fl in events:
        a, b, pa, pb = (src, dst, sport, dport) if d == "f" else (dst, src, dport, sport)
        out.append(fr.Packet(t0 + dt, a, b, pa, pb, proto, pl, fl))
    return out


# ---------------------------------------------------------------- flow construction

def test_forward_is_first_packet_direction_and_downup_floor():
    pk = flow_packets("10.0.0.1", "10.0.0.2", 5000, 80, 6, 100.0,
                      [(0, "f", 0, SYN), (0.01, "b", 0, SYN | ACK), (0.02, "b", 100, ACK),
                       (0.03, "b", 100, ACK), (0.04, "f", 50, ACK)])
    (rec,) = fr.reduce_packets(pk)
    f = rec.features
    assert (rec.src, rec.dst, rec.dport) == ("10.0.0.1", "10.0.0.2", 80)
    assert (f["fpk"], f["bpk"]) == (2, 3)
    assert f["downup"] == 1            # floor(3/2), verified against the CIC CSVs
    assert f["fby"] == 50 and f["bby"] == 200
    assert rec.syn_count == 2


def test_fin_closes_flow_and_single_packet_remainder_is_dropped():
    pk = flow_packets("10.0.0.1", "10.0.0.2", 5000, 80, 6, 0.0,
                      [(0, "f", 10, ACK), (0.1, "b", 10, ACK), (0.2, "f", 0, FIN | ACK)])
    pk += flow_packets("10.0.0.1", "10.0.0.2", 5000, 80, 6, 0.3, [(0, "b", 0, ACK)])  # post-FIN ACK
    recs = fr.reduce_packets(pk)
    assert len(recs) == 1 and recs[0].features["packets"] == 3   # trailing 1-packet flow not emitted


def test_timeout_splits_flow_and_keeps_orientation():
    pk = flow_packets("10.0.0.1", "10.0.0.2", 5000, 80, 6, 0.0, [(0, "f", 5, ACK), (1, "b", 5, ACK)])
    # next packet arrives >120 s after flow start and comes from the *server*
    pk += [fr.Packet(200.0, "10.0.0.2", "10.0.0.1", 80, 5000, 6, 7, ACK),
           fr.Packet(201.0, "10.0.0.1", "10.0.0.2", 5000, 80, 6, 7, ACK)]
    recs = fr.reduce_packets(pk)
    assert len(recs) == 2
    assert all((r.src, r.dst) == ("10.0.0.1", "10.0.0.2") for r in recs)   # orientation kept
    assert recs[1].features["bpk"] == 1 and recs[1].features["fpk"] == 1


# ------------------------------------------------------------ parity with training code

def _raw_columns(events, src, dst, dport, start):
    """CIC-style raw columns computed independently with numpy from the packet events."""
    f = np.array([pl for _, d, pl, _ in events if d == "f"], dtype=float)
    b = np.array([pl for _, d, pl, _ in events if d == "b"], dtype=float)
    allp = np.concatenate([f, b])
    ts = np.array([t for t, *_ in events])
    ft = np.array([t for t, d, *_ in events if d == "f"])
    bt = np.array([t for t, d, *_ in events if d == "b"])
    iat = np.diff(ts) * 1e6

    def st(a):
        return (a.max() if a.size else 0, a.min() if a.size else 0, a.mean() if a.size else 0,
                a.std(ddof=1) if a.size > 1 else 0.0)
    fm, fn, fa, fs = st(f)
    bm, bn, ba, bs = st(b)
    return {
        "Source IP": src, "Destination IP": dst, "Destination Port": dport,
        "Timestamp": str(int((start + ts[0]) // 60)), "Flow Duration": (ts[-1] - ts[0]) * 1e6,
        "Total Fwd Packets": len(f), "Total Backward Packets": len(b),
        "Total Length of Fwd Packets": f.sum(), "Total Length of Bwd Packets": b.sum(),
        "Fwd Packet Length Max": fm, "Fwd Packet Length Min": fn, "Fwd Packet Length Mean": fa,
        "Fwd Packet Length Std": fs, "Bwd Packet Length Max": bm, "Bwd Packet Length Min": bn,
        "Bwd Packet Length Mean": ba, "Bwd Packet Length Std": bs,
        "Flow IAT Mean": iat.mean(), "Flow IAT Std": iat.std(ddof=1) if iat.size > 1 else 0.0,
        "Flow IAT Max": iat.max(), "Flow IAT Min": iat.min(),
        "Fwd IAT Mean": (np.diff(ft) * 1e6).mean() if len(ft) > 1 else 0.0,
        "Bwd IAT Mean": (np.diff(bt) * 1e6).mean() if len(bt) > 1 else 0.0,
        "Down/Up Ratio": np.floor(len(b) / max(len(f), 1)),
        "Min Packet Length": allp.min(), "Max Packet Length": allp.max(),
        # columns the v2r model does not use (unverifiable semantics) are zero-filled
        **{k: 0 for k in ("FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
                          "ACK Flag Count", "URG Flag Count", "Average Packet Size", "Init_Win_bytes_forward",
                          "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
                          "Packet Length Mean", "Packet Length Std")},
        "Label": "BENIGN",
    }


def test_runtime_features_match_training_feature_code(tmp_path):
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT / "scripts"))
    import network_improve as ni

    specs = []   # (src, dst, sport, dport, proto, t0, events)
    # port scanner 10.0.0.9 -> 10.0.0.1: 25 two-packet flows, same minute
    for i in range(25):
        specs.append(("10.0.0.9", "10.0.0.1", 40000 + i, 1000 + i, 6, 600.0 + i * 0.01,
                      [(0, "f", 0, SYN), (0.0005, "b", 0, RST | ACK)]))
    # ordinary web flow, same minute
    specs.append(("10.0.0.5", "10.0.0.1", 51000, 80, 6, 610.0,
                  [(0, "f", 0, SYN), (0.01, "b", 0, SYN | ACK), (0.02, "f", 300, ACK), (0.05, "b", 1200, ACK),
                   (0.06, "b", 1400, ACK), (0.2, "f", 0, ACK)]))
    # UDP request/response in the next minute
    specs.append(("10.0.0.5", "10.0.0.53", 53000, 53, 17, 661.0, [(0, "f", 40, 0), (0.03, "b", 120, 0)]))
    # flow from another source hitting the same destination port in the same minute as the scan
    specs.append(("10.0.0.6", "10.0.0.1", 52000, 80, 6, 620.0,
                  [(0, "f", 20, ACK), (0.5, "b", 20, ACK), (1.5, "f", 20, ACK)]))

    pkts = [p for s in specs for p in flow_packets(*s)]
    pkts.sort(key=lambda p: p.ts)
    recs = fr.reduce_packets(pkts)
    assert len(recs) == len(specs)

    rows = []
    for r in recs:     # order by start, same as the reducer's output
        s = next(s for s in specs if (s[0], s[3], s[5]) == (r.src, r.dport, r.start))
        rows.append(_raw_columns(s[6], s[0], s[1], s[3], s[5]))
    csv = tmp_path / "flows.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    trained = ni.load_file(csv)       # the exact code that produced the training matrix
    for i, r in enumerate(recs):
        for name in fr.ALL_COMPUTED_FEATURES:
            got, want = r.features[name], float(trained.iloc[i][name])
            assert np.isclose(got, want, rtol=1e-4, atol=1e-5), (i, name, got, want)


# -------------------------------------------------------------------- hybrid logic

class _FakeModelBundle(dict):
    pass


def _rec(syn_count=0, **over):
    pk = [fr.Packet(0.0, "1.1.1.1", "2.2.2.2", 1000, 80, 6, 0, SYN),
          fr.Packet(0.1, "2.2.2.2", "1.1.1.1", 80, 1000, 6, 0, ACK)]
    (r,) = fr.reduce_packets(pk)
    r.syn_count = syn_count
    r.features.update(over)
    return r


def test_hybrid_flags_on_rule_alone_ml_alone_and_neither():
    thr = 0.8
    ev_rule, ev_ml, ev_none = net.analyse_flow_records_hybrid(
        [_rec(syn_count=25), _rec(), _rec()], np.array([0.0, 0.95, 0.1]), thr)
    assert ev_rule.prediction == "anomalous" and any(e.code == "SYN_BURST" for e in ev_rule.evidence)
    assert ev_ml.prediction == "anomalous" and any(e.code == "ML_FLOW_MODEL" for e in ev_ml.evidence)
    assert ev_none.prediction == "no_configured_anomaly"
    assert {ev_rule.model_version, ev_ml.model_version, ev_none.model_version} == {"network-hybrid-v2"}


def test_missing_or_mismatched_model_falls_back(tmp_path, monkeypatch):
    pytest.importorskip("scapy")
    from scapy.all import IP, TCP, Ether, wrpcap
    p = Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=5000, dport=22, flags="S")
    wrpcap(str(tmp_path / "t.pcap"), [p])
    nrt.reset_cache()
    monkeypatch.setattr(nrt.settings, "model_dir", str(tmp_path))           # no artefact there
    assert nrt.load_bundle() is None
    events = net.analyse_pcap(str(tmp_path / "t.pcap"))
    assert events and events[0].model_version == "network-rules-v0.1"

    import joblib                                                             # stale feature list
    joblib.dump({"features": ["x"], "model": None, "threshold": 0.5}, tmp_path / nrt.BUNDLE_FILE)
    nrt.reset_cache()
    assert nrt.load_bundle() is None
    monkeypatch.setattr(nrt.settings, "ml_detectors", False)
    nrt.reset_cache()
    assert nrt.load_bundle() is None
    nrt.reset_cache()


# ------------------------------------------------------------ end-to-end on a pcap

@needs_model
def test_port_scan_pcap_is_flagged_and_ordinary_traffic_is_not(tmp_path):
    pytest.importorskip("scapy")
    from scapy.all import IP, TCP, Ether, wrpcap
    nrt.reset_cache()
    pkts, t = [], 1_700_000_000.0

    def add(p, ts):
        p = wire_pad(p)                  # real captures of short frames carry Ethernet padding (60 B min)
        p.time = ts
        pkts.append(p)

    for port in range(1, 1501):          # 1,500 SYN->RST/ACK probes from one source in one minute (<~200 ports/min is below what the model learned; see NETWORK_V2R.md)
        a = Ether() / IP(src="10.9.9.9", dst="10.0.0.1") / TCP(sport=40000 + port, dport=port, flags="S")
        b = Ether() / IP(src="10.0.0.1", dst="10.9.9.9") / TCP(sport=port, dport=40000 + port, flags="RA")
        add(a, t + port * 0.01)
        add(b, t + port * 0.01 + 0.0004)
    t2 = t + 1200                         # ordinary traffic much later: three clients, a handful of flows
    for i, client in enumerate(("10.0.1.11", "10.0.1.12", "10.0.1.13")):
        sp, base = 50000 + i, t2 + i * 40.0
        seq = [("S", "f", 0), ("SA", "b", 0), ("A", "f", 0), ("PA", "f", 400), ("PA", "b", 1400),
               ("A", "f", 0), ("FA", "f", 0)]
        for k, (fl, d, pl) in enumerate(seq):
            ip = IP(src=client, dst="10.0.0.80") if d == "f" else IP(src="10.0.0.80", dst=client)
            tcp = (TCP(sport=sp, dport=443, flags=fl) if d == "f" else TCP(sport=443, dport=sp, flags=fl))
            p = Ether() / ip / tcp / (b"x" * pl)
            add(p, base + k * 0.05)
    path = tmp_path / "scan.pcap"
    wrpcap(str(path), pkts)

    events = net.analyse_pcap(str(path))
    assert all(e.model_version == "network-hybrid-v2" for e in events)
    scan = [e for e in events if e.metadata["src_ip"] == "10.9.9.9"]
    ordinary = [e for e in events if e.metadata["src_ip"].startswith("10.0.1.")]
    assert len(scan) == 1500 and ordinary
    assert sum(e.prediction == "anomalous" for e in scan) >= 1425       # >=95% of scan flows flagged
    assert all(e.prediction == "no_configured_anomaly" for e in ordinary)  # normal not flagged


def wire_pad(p):
    """Pad to the 60-byte Ethernet minimum frame, as a NIC/switch does on the wire."""
    from scapy.all import Padding
    n = len(p)
    return p / Padding(load=b"\x00" * (60 - n)) if n < 60 else p


def test_pcap_payload_includes_ethernet_padding_like_cicflowmeter(tmp_path):
    pytest.importorskip("scapy")
    from scapy.all import IP, TCP, Ether, wrpcap
    syn = Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=5000, dport=80, flags="S")
    rst = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=80, dport=5000, flags="RA")
    for i, p in enumerate((wire_pad(syn), wire_pad(rst))):
        p.time = 1000.0 + i * 0.001
    wrpcap(str(tmp_path / "padded.pcap"), [wire_pad(syn), wire_pad(rst)])
    pk = list(fr.packets_from_pcap(str(tmp_path / "padded.pcap")))
    assert [p.payload for p in pk] == [6, 6]        # 60 - 14 - 20 - 20, the value seen in the CIC CSVs
    wrpcap(str(tmp_path / "bare.pcap"), [syn, rst])
    assert [p.payload for p in fr.packets_from_pcap(str(tmp_path / "bare.pcap"))] == [0, 0]
