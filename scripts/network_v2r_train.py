"""
Measured parity study + reproducible-feature network model ("v2r").

Why: models/network_model_v2.joblib uses 59 CIC-IDS2017 CSV columns. Several cannot be reproduced from
raw packets with verified semantics (checked against the CSVs):
  * fin/syn/rst/psh/ack/urg "Flag Count": not per-packet counts nor presence flags (RST is 0 in every
    flow, SYN presence is only ~2% of PortScan flows) -> semantics unknown.
  * avg_pkt, pl_mean, pl_std: off-by-one style CICFlowMeter artefacts (mean != total/n in ~77% of rows).
  * act_data, min_seg (UDP rows report TCP-like segment sizes), win_f/win_b (-1 for 35% of flows).
  * syn_count / syn_ratio / sw_rst_frac derive from the flag counts above.
This script retrains on ONLY the verified-reproducible features, head-to-head with the original 59-feature
model on identical splits and identical training rows, so the cost of dropping them is measured.

Usage: python scripts/network_v2r_train.py
Outputs: models/network_model_v2r.joblib, reports/network_v2r.json, reports/NETWORK_V2R.md
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import network_improve as ni  # noqa: E402
from app.ml.metrics import evaluate, pick_threshold  # noqa: E402

from app.ml.flow_reducer import ALL_COMPUTED_FEATURES, MODEL_FEATURES  # noqa: E402

SETS = {
    "V2_full59": ni.SETS["V2_flow_rich+host_agg"],
    "V2r_repro43": list(ALL_COMPUTED_FEATURES),     # all features the reducer computes (incl. packet sizes)
    "V2r_nolen26": list(MODEL_FEATURES),            # shipped: no packet-size features
}
SHIP = "V2r_nolen26"
REPRO = SETS["V2r_repro43"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def head_to_head(df, tr, va, te, rng, tag, ship=None):
    y = df["y"].to_numpy()
    lab = df["label"].to_numpy()
    trs = ni.train_sample(df, tr, rng)           # identical rows for every feature set
    out = {"n_test": int(len(te)), "n_test_attack": int(y[te].sum()), "n_train_used": int(len(trs))}
    rules = ni.rules_v1(df.iloc[te])
    for name, cols in SETS.items():
        X = df[cols].to_numpy(np.float32)
        m = ni.fit(X[trs], y[trs], quick=False)
        vs = m.predict_proba(X[va])[:, 1]
        st = m.predict_proba(X[te])[:, 1]
        thr = pick_threshold(y[va], vs, 0.01)
        flag = st >= thr
        r = {"threshold": thr, "n_features": len(cols),
             "ml": {**evaluate(y[te], flag, st, n_boot=200), "per_family_recall": ni.fam_recall(df, te, flag)},
             "hybrid_ml_or_rules_v1": evaluate(y[te], flag | rules, n_boot=200)}
        out[name] = r
        log(f"{tag} {name}: P={r['ml']['precision']:.4f} R={r['ml']['recall']:.4f} FPR={r['ml']['fpr']:.4f} "
            f"| hybrid R={r['hybrid_ml_or_rules_v1']['recall']:.4f} FPR={r['hybrid_ml_or_rules_v1']['fpr']:.4f}")
        if ship is not None and name == SHIP:
            joblib.dump({"kind": "network_v2r", "variant": SHIP, "model": m, "threshold": thr, "features": cols,
                         "window_definition": "minute windows (floor(flow_start/60s)) keyed by source IP / "
                                              "dest IP / (src,dst); flows with <2 packets are not emitted",
                         "flow_construction": "CICFlowMeter-style: bidirectional 5-tuple, 120 s timeout, "
                                              "FIN closes the flow, flows with <2 packets dropped",
                         "trained": datetime.now(timezone.utc).isoformat()}, ship, compress=3)
    return out


def main():
    rng = np.random.default_rng(ni.SEED)
    df = ni.load_all()
    lab = df["label"].to_numpy()
    res = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "feature_sets": {k: len(v) for k, v in SETS.items()}, "dropped_for_parity": sorted(
               set(ni.SETS["V2_flow_rich+host_agg"]) - set(REPRO)), "A": None, "B": None}
    tr, va, te = ni.strat_split(lab, np.arange(len(df)), rng)
    va = rng.choice(va, min(len(va), 300_000), replace=False)
    res["A"] = head_to_head(df, tr, va, te, rng, "A", ship=ROOT / "models" / "network_model_v2r.joblib")
    fri = df["file"].str.startswith("Friday").to_numpy()
    pool = np.where(~fri)[0]
    tr, va, _ = ni.strat_split(lab, pool, rng, frac=(0.88, 0.12, 0.0))
    va = rng.choice(va, min(len(va), 300_000), replace=False)
    res["B"] = head_to_head(df, tr, va, np.where(fri)[0], rng, "B")
    (ROOT / "reports" / "network_v2r.json").write_text(json.dumps(res, indent=2, default=float))
    (ROOT / "reports" / "NETWORK_V2R.md").write_text(render(res), encoding="utf-8")
    log("done")


def render(r):
    L = ["# Network v2 vs reproducible-feature v2r (measured parity cost)", "",
         f"Generated {r['generated']}. Same splits and identical training rows per protocol. Dropped for parity "
         f"({len(r['dropped_for_parity'])}): {', '.join(r['dropped_for_parity'])}.", ""]
    for p, title in (("A", "Protocol A random split (upper bound)"), ("B", "Protocol B train Mon-Thu, test Fri")):
        L += [f"## {title}", "", ni.HDR]
        for name in SETS:
            L.append(ni.row(f"{name} ml", r[p][name]["ml"]))
            L.append(ni.row(f"{name} ML OR rules v1", r[p][name]["hybrid_ml_or_rules_v1"]))
        fams = list(r[p]["V2_full59"]["ml"]["per_family_recall"])
        L += ["", "Per-family recall (ML):", "", "| family | n | " + " | ".join(SETS) + " |",
              "|---|---|" + "---|" * len(SETS)]
        for f in fams:
            n = r[p]["V2_full59"]["ml"]["per_family_recall"][f]["n"]
            L.append(f"| {f} | {n} | " + " | ".join(f"{r[p][k]['ml']['per_family_recall'][f]['recall']:.3f}" for k in SETS) + " |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
