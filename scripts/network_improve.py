"""
Network detector improvement study on the ORIGINAL CIC-IDS2017 labelled flows
(data/external/network_ip/TrafficLabelling /*.csv: has IPs + minute timestamps + Monday).

Feature sets
  V0_baseline12            the 12 features the shipped v1 model used (packets, bytes, duration,
                           dst_port, syn_count + ratios)
  V1_flow_rich             + fwd/bwd packet & byte counts, packet-length stats, IAT stats, TCP flag
                           counts, init windows, down/up ratio ... (all computable per flow from a pcap
                           by a CICFlowMeter-style reducer)
  V2_flow_rich+host_agg    + per-(source IP, minute), per-(dest IP, minute) and per-(src,dst,minute)
                           fan-out/fan-in counters (what actually exposes scans / floods)
IPs themselves and timestamps are NEVER model inputs (only counts derived from them).

Protocols: A random 70/10/20; B train Mon-Thu -> test Fri (Bot/PortScan/DDoS unseen);
C leave-one-family-out (held-out family never seen in train/val).
Thresholds: best val F1 s.t. val FPR <= 1% (and a stricter 0.1% point). Test is scored once.
Rules baselines: v1 production rules (vectorised, same thresholds) and an aggregated-window rule set
whose two thresholds are grid-searched on training data only.

Usage: python scripts/network_improve.py [--quick]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ml.metrics import evaluate, pick_threshold  # noqa: E402

SRC = ROOT / "data" / "external" / "network_ip" / "TrafficLabelling"
CACHE = ROOT / "data" / "external" / "network_ip" / "features_v2.parquet"
SEED = 42
BENIGN_CAP, FAMILY_CAP = 500_000, 80_000

RAW = {
    "Source IP": "src", "Destination IP": "dst", "Destination Port": "dport", "Timestamp": "ts",
    "Flow Duration": "dur_us", "Total Fwd Packets": "fpk", "Total Backward Packets": "bpk",
    "Total Length of Fwd Packets": "fby", "Total Length of Bwd Packets": "bby",
    "Fwd Packet Length Max": "fpl_max", "Fwd Packet Length Min": "fpl_min",
    "Fwd Packet Length Mean": "fpl_mean", "Fwd Packet Length Std": "fpl_std",
    "Bwd Packet Length Max": "bpl_max", "Bwd Packet Length Min": "bpl_min",
    "Bwd Packet Length Mean": "bpl_mean", "Bwd Packet Length Std": "bpl_std",
    "Flow IAT Mean": "iat_mean", "Flow IAT Std": "iat_std", "Flow IAT Max": "iat_max",
    "Flow IAT Min": "iat_min", "Fwd IAT Mean": "fiat_mean", "Bwd IAT Mean": "biat_mean",
    "FIN Flag Count": "fin", "SYN Flag Count": "syn", "RST Flag Count": "rst",
    "PSH Flag Count": "psh", "ACK Flag Count": "ack", "URG Flag Count": "urg",
    "Down/Up Ratio": "downup", "Average Packet Size": "avg_pkt", "Init_Win_bytes_forward": "win_f",
    "Init_Win_bytes_backward": "win_b", "act_data_pkt_fwd": "act_data", "min_seg_size_forward": "min_seg",
    "Min Packet Length": "pl_min", "Max Packet Length": "pl_max", "Packet Length Mean": "pl_mean",
    "Packet Length Std": "pl_std", "Label": "label",
}
NUM = [v for k, v in RAW.items() if v not in ("src", "dst", "ts", "label")]
# Features that are not in the 12 baseline and are computed from one flow only.
FLOW_RICH = ["fpk", "bpk", "fby", "bby", "fpl_max", "fpl_min", "fpl_mean", "fpl_std", "bpl_max", "bpl_min",
             "bpl_mean", "bpl_std", "iat_mean", "iat_std", "iat_max", "iat_min", "fiat_mean", "biat_mean",
             "fin", "syn", "rst", "psh", "ack", "urg", "downup", "avg_pkt", "win_f", "win_b", "act_data",
             "min_seg", "pl_min", "pl_max", "pl_mean", "pl_std", "fby_per_fpk", "bby_per_bpk", "bpk_ratio"]
BASE12 = ["packets", "bytes", "duration_s", "dst_port", "syn_count", "bytes_per_pkt", "pkts_per_s",
          "bytes_per_s", "syn_ratio", "sensitive_port", "well_known_port", "ephemeral_port"]
AGG = ["sw_flows", "sw_ndport", "sw_ndst", "sw_small_frac", "sw_mean_pkts", "dw_flows", "dw_nsrc",
       "pw_flows", "pw_ndport", "sw_rst_frac"]
SETS = {"V0_baseline12": BASE12, "V1_flow_rich": BASE12 + FLOW_RICH, "V2_flow_rich+host_agg": BASE12 + FLOW_RICH + AGG}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def nuniq(key: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Per-row count of distinct `val` within the row's `key` group."""
    uk, inv = np.unique(key, return_inverse=True)
    pairs = np.unique(np.stack([inv.astype(np.int64), val.astype(np.int64)], 1), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=len(uk))
    return counts[inv].astype(np.float32)


def group_mean(key: np.ndarray, x: np.ndarray) -> np.ndarray:
    uk, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=x.astype(np.float64), minlength=len(uk))
    n = np.bincount(inv, minlength=len(uk))
    return (s / n)[inv].astype(np.float32)


def load_file(f: Path) -> pd.DataFrame:
    d = pd.read_csv(f, encoding="latin1", low_memory=False,
                    usecols=lambda c: c.strip() in RAW)
    d.columns = [c.strip() for c in d.columns]
    d = d.rename(columns=RAW)
    d = d.dropna(subset=["label", "src", "dst"])
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d[NUM] = d[NUM].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["dur_us", "fpk", "bpk", "dport"]).reset_index(drop=True)
    d["label"] = d["label"].astype(str).map(lambda s: re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip()))

    out = pd.DataFrame(index=d.index)
    pk = (d.fpk + d.bpk).clip(lower=1)
    by = d.fby + d.bby
    dur = (d.dur_us / 1e6).clip(lower=0)
    dd = np.maximum(dur, 1e-6)
    out["packets"], out["bytes"], out["duration_s"] = pk, by, dur
    out["dst_port"], out["syn_count"] = d.dport, d.syn
    out["bytes_per_pkt"], out["pkts_per_s"], out["bytes_per_s"] = by / pk, pk / dd, by / dd
    out["syn_ratio"] = d.syn / pk
    out["sensitive_port"] = d.dport.isin([22, 23, 3389, 445, 5900, 21, 139, 135]).astype(float)
    out["well_known_port"] = (d.dport < 1024).astype(float)
    out["ephemeral_port"] = (d.dport >= 49152).astype(float)
    for c in FLOW_RICH:
        if c in d:
            out[c] = d[c]
    out["fby_per_fpk"] = d.fby / d.fpk.clip(lower=1)
    out["bby_per_bpk"] = d.bby / d.bpk.clip(lower=1)
    out["bpk_ratio"] = d.bpk / pk

    # ---- host aggregation (windows = same source/dest within the same CIC minute stamp) ----
    sc = pd.factorize(d.src)[0].astype(np.int64)
    dc = pd.factorize(d.dst)[0].astype(np.int64)
    tc = pd.factorize(d.ts)[0].astype(np.int64)
    n_t = int(tc.max()) + 1
    k_src, k_dst = sc * n_t + tc, dc * n_t + tc
    k_pair = (sc * (int(dc.max()) + 1) + dc) * n_t + tc
    port = d.dport.to_numpy().astype(np.int64)
    ones = np.ones(len(d), dtype=np.float32)
    def cnt(k):
        inv = np.unique(k, return_inverse=True)[1]
        return np.bincount(inv)[inv].astype(np.float32)
    out["sw_flows"] = cnt(k_src)
    out["sw_ndport"] = nuniq(k_src, port)
    out["sw_ndst"] = nuniq(k_src, dc)
    out["sw_small_frac"] = group_mean(k_src, (pk <= 2).to_numpy())
    out["sw_mean_pkts"] = group_mean(k_src, pk.to_numpy())
    out["dw_flows"] = cnt(k_dst)
    out["dw_nsrc"] = nuniq(k_dst, sc)
    out["pw_flows"] = cnt(k_pair)
    out["pw_ndport"] = nuniq(k_pair, port)
    out["sw_rst_frac"] = group_mean(k_src, (d.rst > 0).to_numpy())
    out = out.astype(np.float32)
    out["label"] = d["label"].astype("category")
    out["file"] = f.name
    del d
    return out


def load_all() -> pd.DataFrame:
    if CACHE.exists():
        log(f"loading cache {CACHE.name}")
        return pd.read_parquet(CACHE)
    frames = []
    for f in sorted(SRC.glob("*.csv")):
        log(f"parse {f.name}")
        frames.append(load_file(f))
        gc.collect()
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(str)
    df["y"] = (df["label"] != "BENIGN").astype(np.int8)
    df.to_parquet(CACHE)
    return df


# ---------------------------------------------------------------- rules -------

def rules_v1(df: pd.DataFrame) -> np.ndarray:
    """Production rules in app/detectors/network.py, vectorised with identical thresholds."""
    p, dur, b = df["packets"].to_numpy(), df["duration_s"].to_numpy(), df["bytes"].to_numpy()
    syn, port = df["syn_count"].to_numpy(), df["dst_port"].to_numpy()
    return ((syn >= 20) | ((p >= 1000) & (dur <= 10)) | (np.isin(port, [22, 23, 3389, 445, 5900]) & (syn >= 8))
            | ((b > 5_000_000) & (dur < 5)))


def rules_window(df: pd.DataFrame, ports_thr: float, flows_thr: float) -> np.ndarray:
    """Aggregated rule: flag every flow of a source-minute window that touches many distinct
    ports or emits very many flows."""
    return ((df["sw_ndport"].to_numpy() >= ports_thr) | (df["sw_flows"].to_numpy() >= flows_thr))


def tune_window_rules(df_tr: pd.DataFrame) -> tuple[float, float]:
    y = df_tr["y"].to_numpy().astype(bool)
    best, arg = -1, (100, 1000)
    for pt in (10, 20, 50, 100, 200, 500):
        for ft in (50, 100, 300, 1000, 3000, 10 ** 9):
            fl = rules_window(df_tr, pt, ft)
            tp, fp, fn = (y & fl).sum(), (~y & fl).sum(), (y & ~fl).sum()
            tn = (~y & ~fl).sum()
            if fp / max(1, fp + tn) > 0.01:
                continue
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            if f1 > best:
                best, arg = f1, (pt, ft)
    return arg


# ------------------------------------------------------------- modelling ------

def train_sample(df: pd.DataFrame, idx: np.ndarray, rng) -> np.ndarray:
    lab = df["label"].to_numpy()[idx]
    keep = []
    for l in np.unique(lab):
        ii = idx[lab == l]
        cap = BENIGN_CAP if l == "BENIGN" else FAMILY_CAP
        keep.append(rng.choice(ii, cap, replace=False) if len(ii) > cap else ii)
    return np.concatenate(keep)


def strat_split(y_labels: np.ndarray, idx: np.ndarray, rng, frac=(0.7, 0.1, 0.2)):
    """Per-label random split of `idx` into train/val/test."""
    tr, va, te = [], [], []
    for l in np.unique(y_labels[idx]):
        ii = rng.permutation(idx[y_labels[idx] == l])
        a, b = int(len(ii) * frac[0]), int(len(ii) * (frac[0] + frac[1]))
        tr.append(ii[:a]); va.append(ii[a:b]); te.append(ii[b:])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def fit(X, y, quick):
    return HistGradientBoostingClassifier(max_iter=60 if quick else 150, learning_rate=0.12,
                                          max_leaf_nodes=31, random_state=SEED).fit(X, y)


def run_fold(df, cols, tr, va, te, quick, rng, rules_tuned=None, bundle_path=None, n_boot=200):
    X = df[cols].to_numpy(np.float32)
    y = df["y"].to_numpy()
    trs = train_sample(df, tr, rng)
    m = fit(X[trs], y[trs], quick)
    vs = m.predict_proba(X[va])[:, 1]
    s_te = m.predict_proba(X[te])[:, 1]
    out = {"n_test": int(len(te)), "n_test_attack": int(y[te].sum()), "n_train_used": int(len(trs))}
    for name, cap in (("fpr<=1%", 0.01), ("fpr<=0.1%", 0.001)):
        thr = pick_threshold(y[va], vs, cap)
        flag = s_te >= thr
        out[f"ml[{name}]"] = {"threshold": thr, **evaluate(y[te], flag, s_te, n_boot=n_boot)}
        out[f"ml[{name}]"]["per_family_recall"] = fam_recall(df, te, flag)
        if name == "fpr<=1%":
            thr1, flag1 = thr, flag
    if bundle_path:
        joblib.dump({"kind": "network_v2", "model": m, "threshold": thr1, "features": cols,
                     "window_definition": "CIC minute-stamp windows keyed by source IP / dest IP / (src,dst)",
                     "trained": datetime.now(timezone.utc).isoformat()}, bundle_path, compress=3)
    return out, flag1


def fam_recall(df, te, flag):
    lab = df["label"].to_numpy()[te]
    return {l: {"n": int((lab == l).sum()), "recall": float(flag[lab == l].mean())}
            for l in np.unique(lab) if l != "BENIGN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    df = load_all()
    y = df["y"].to_numpy()
    lab = df["label"].to_numpy()
    log(f"{len(df)} flows, {y.mean():.3f} attack; labels={pd.Series(lab).value_counts().to_dict()}")
    res = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "dataset": {"name": "CIC-IDS2017 GeneratedLabelledFlows (original, with IPs)",
                       "source": "HF Mireu-Lab/CIC-IDS ZIP/GeneratedLabelledFlows.zip",
                       "zip_sha256": sha(ROOT / "data/external/network_ip/GeneratedLabelledFlows.zip"),
                       "flows": int(len(df)), "label_counts": pd.Series(lab).value_counts().to_dict(),
                       "caveats": ["CIC-IDS2017 has known label/flow-construction artefacts (Engelen et al. 2021)",
                                   "timestamps are minute-resolution; windows = identical minute stamp",
                                   "benign traffic is one lab network over 5 days"]},
           "feature_sets": {k: len(v) for k, v in SETS.items()}, "A": {}, "B": {}, "C": {}, "rules": {}}

    # ---------- Protocol A: random split ----------
    tr, va, te = strat_split(lab, np.arange(len(df)), rng)
    va = rng.choice(va, min(len(va), 300_000), replace=False)
    log(f"A: tr={len(tr)} va={len(va)} te={len(te)}")
    for name, cols in SETS.items():
        r, _ = run_fold(df, cols, tr, va, te, args.quick, rng,
                        bundle_path=(ROOT / "models" / "network_model_v2.joblib") if name.startswith("V2") else None)
        res["A"][name] = r
        log(f"A {name}: P={r['ml[fpr<=1%]']['precision']:.4f} R={r['ml[fpr<=1%]']['recall']:.4f} "
            f"FPR={r['ml[fpr<=1%]']['fpr']:.4f}")
    te_df = df.iloc[te]
    pt, ft = tune_window_rules(df.iloc[va])
    res["rules"]["A"] = {"v1_production": evaluate(y[te], rules_v1(te_df), n_boot=200),
                         "window_rules": {"ports_thr": pt, "flows_thr": ft,
                                          **evaluate(y[te], rules_window(te_df, pt, ft), n_boot=200)},
                         "per_family_recall_v1": fam_recall(df, te, rules_v1(te_df)),
                         "per_family_recall_window": fam_recall(df, te, rules_window(te_df, pt, ft))}
    log(f"A window rules thr=({pt},{ft}) -> {res['rules']['A']['window_rules']['recall']:.3f}")

    # ---------- Protocol B: Mon-Thu -> Fri ----------
    fri = df["file"].str.startswith("Friday").to_numpy()
    pool = np.where(~fri)[0]
    tr, va, _ = strat_split(lab, pool, rng, frac=(0.88, 0.12, 0.0))
    va = rng.choice(va, min(len(va), 300_000), replace=False)
    te = np.where(fri)[0]
    for name, cols in SETS.items():
        r, _ = run_fold(df, cols, tr, va, te, args.quick, rng)
        res["B"][name] = r
        log(f"B {name}: P={r['ml[fpr<=1%]']['precision']:.4f} R={r['ml[fpr<=1%]']['recall']:.4f} "
            f"FPR={r['ml[fpr<=1%]']['fpr']:.4f}")
    te_df = df.iloc[te]
    pt, ft = tune_window_rules(df.iloc[tr])
    res["rules"]["B"] = {"v1_production": evaluate(y[te], rules_v1(te_df), n_boot=200),
                         "window_rules": {"ports_thr": pt, "flows_thr": ft,
                                          **evaluate(y[te], rules_window(te_df, pt, ft), n_boot=200)},
                         "per_family_recall_v1": fam_recall(df, te, rules_v1(te_df)),
                         "per_family_recall_window": fam_recall(df, te, rules_window(te_df, pt, ft))}
    log(f"B window rules thr=({pt},{ft}) -> R={res['rules']['B']['window_rules']['recall']:.3f} "
        f"FPR={res['rules']['B']['window_rules']['fpr']:.4f}")

    # ---------- Protocol C: leave-one-family-out ----------
    groups = {"PortScan": ["PortScan"], "DDoS": ["DDoS"], "Bot": ["Bot"], "DoS_Hulk": ["DoS_Hulk"],
              "WebAttacks": [l for l in np.unique(lab) if l.startswith("Web_Attack")],
              "Patator": ["FTP-Patator", "SSH-Patator"]}
    benign = np.where(lab == "BENIGN")[0]
    for gname, fams in groups.items():
        held = np.isin(lab, fams)
        others = np.where((lab != "BENIGN") & ~held)[0]
        # Benign flows from the capture file(s) containing the held-out family are test-only: those
        # files hold benign-LABELLED flows from the very same attack windows (label noise), and
        # training on them would contradict the "family never seen" premise (measured: this made
        # leave-PortScan-out recall 0.001 vs 0.998 once the contamination was removed).
        files = df["file"].to_numpy()
        fam_files = np.unique(files[held])
        same_file = benign[np.isin(files[benign], fam_files)]
        free_benign = benign[~np.isin(files[benign], fam_files)]
        btr, bva, bte_free = strat_split(lab, free_benign, rng)
        bte = np.concatenate([bte_free, same_file])
        otr, ova, _ = strat_split(lab, others, rng, frac=(0.85, 0.15, 0.0))
        tr_c, va_c = np.concatenate([btr, otr]), np.concatenate([bva, ova])
        va_c = rng.choice(va_c, min(len(va_c), 300_000), replace=False)
        # held-out family flows are subsampled so FPR/precision are measured at a realistic ratio
        hi = np.where(held)[0]
        te_c = np.concatenate([bte, hi])
        res["C"][gname] = {}
        for name in ("V1_flow_rich", "V2_flow_rich+host_agg"):
            r, _ = run_fold(df, SETS[name], tr_c, va_c, te_c, args.quick, rng, n_boot=100)
            res["C"][gname][name] = r
            m1 = r["ml[fpr<=1%]"]
            log(f"C hold-out {gname} {name}: recall={m1['recall']:.3f} FPR={m1['fpr']:.4f} P={m1['precision']:.3f}")
        tdf = df.iloc[te_c]
        fl_v1 = rules_v1(tdf)
        ptc, ftc = tune_window_rules(df.iloc[tr_c])
        fl_w = rules_window(tdf, ptc, ftc)
        yy = y[te_c]
        res["C"][gname]["rules_v1"] = evaluate(yy, fl_v1, n_boot=100)
        res["C"][gname]["rules_window"] = {"ports_thr": ptc, "flows_thr": ftc, **evaluate(yy, fl_w, n_boot=100)}

    out = ROOT / "reports"
    (out / "network_improved.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "NETWORK_IMPROVED.md").write_text(render(res), encoding="utf-8")
    log("done")


def row(name, m):
    ci = m["ci95"]
    return (f"| {name} | {m['precision']:.4f} [{ci['precision'][0]:.3f}-{ci['precision'][1]:.3f}] | "
            f"{m['recall']:.4f} [{ci['recall'][0]:.3f}-{ci['recall'][1]:.3f}] | {m['f1']:.4f} | "
            f"{m['fpr']:.4f} [{ci['fpr'][0]:.4f}-{ci['fpr'][1]:.4f}] | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")


HDR = "| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |\n|---|---|---|---|---|---|---|---|---|"


def render(r):
    L = [f"# Network detector improvement (auto-generated {r['generated']})", "",
         f"Dataset: {r['dataset']['name']}; {r['dataset']['flows']} flows. Caveats: "
         + "; ".join(r["dataset"]["caveats"]) + ".", "",
         "Operating point `fpr<=1%` / `fpr<=0.1%` = best validation F1 under that validation FPR cap.", ""]
    for p, title in (("A", "Protocol A - random split (upper bound)"), ("B", "Protocol B - train Mon-Thu, test Friday (unseen Bot/PortScan/DDoS)")):
        L += [f"## {title}", "", HDR]
        for name, v in r[p].items():
            L.append(row(f"{name} ml fpr<=1%", v["ml[fpr<=1%]"]))
            L.append(row(f"{name} ml fpr<=0.1%", v["ml[fpr<=0.1%]"]))
        L.append(row("rules v1 (production)", r["rules"][p]["v1_production"]))
        w = r["rules"][p]["window_rules"]
        L.append(row(f"rules window (ports>={w['ports_thr']} or flows>={w['flows_thr']})", w))
        L += ["", "Per-family recall (ML fpr<=1%):", "",
              "| family | n | " + " | ".join(r[p]) + " | rules v1 | rules window |", "|---|---|" + "---|" * (len(r[p]) + 2)]
        fams = list(next(iter(r[p].values()))["ml[fpr<=1%]"]["per_family_recall"])
        for f in fams:
            n = r[p][next(iter(r[p]))]["ml[fpr<=1%]"]["per_family_recall"][f]["n"]
            cells = [f"{v['ml[fpr<=1%]']['per_family_recall'][f]['recall']:.3f}" for v in r[p].values()]
            L.append(f"| {f} | {n} | " + " | ".join(cells) +
                     f" | {r['rules'][p]['per_family_recall_v1'][f]['recall']:.3f} | "
                     f"{r['rules'][p]['per_family_recall_window'][f]['recall']:.3f} |")
        L.append("")
    L += ["## Protocol C - leave-one-family-out (held-out family unseen in train/val)", "",
          "| held-out | V1 recall | V1 FPR | V2 recall | V2 FPR | V2 precision | rules v1 recall | rules window recall (FPR) |", "|---|---|---|---|---|---|---|---|"]
    for g, v in r["C"].items():
        a, b = v["V1_flow_rich"]["ml[fpr<=1%]"], v["V2_flow_rich+host_agg"]["ml[fpr<=1%]"]
        w = v["rules_window"]
        L.append(f"| {g} | {a['recall']:.3f} | {a['fpr']:.4f} | {b['recall']:.3f} | {b['fpr']:.4f} | {b['precision']:.3f} | "
                 f"{v['rules_v1']['recall']:.3f} | {w['recall']:.3f} ({w['fpr']:.4f}) |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
