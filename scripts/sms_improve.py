"""
Multi-corpus SMS evaluation: leave-one-corpus-out (LOCO) transfer + pooled held-out split.

Corpora (all fetched from Hugging Face; licence/provenance recorded in the report):
  uci        UCI SMS Spam Collection             real, en            CC BY 4.0
  india      CloveAI/india-spam-sms              TEMPLATED, en/IN    MIT   (indicative only)
  bengali_en shariul-islam/bengali-sms-smishing  English variety; research corpus, provenance
                                                 unclear (likely generated)  MIT  (indicative only)
  moz_pt     MOZNLP/MOZ-Smishing                 real crowd-sourced, pt    openrail-m
Excluded: africa-smishing (100% synthetic), DarkNeuron/*-01 (89% copy of UCI),
          alusci/sms-otp (synthetic OTP), flashflow-bank-sms (no licence; used only as a benign probe).

Templated corpora leak between random splits, so every split is grouped by a "skeleton"
(digits masked, non-initial capitalised words masked).

python scripts/sms_improve.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.detectors.sms import analyse_sms  # noqa: E402
from app.ml.metrics import evaluate, fmt_row, pick_threshold  # noqa: E402

EXT = ROOT / "data" / "external" / "sms"
HF = "https://huggingface.co/datasets"
SEED = 42
MAX_FPR = 0.01
HEADER = ("| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |\n"
          "|---|---|---|---|---|---|---|---|---|")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(url, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"fetch {url}")
        dest.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "edgeshield-eval"}), timeout=120).read())
    return dest


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def skeleton(t: str) -> str:
    t = re.sub(r"\d+", "0", t.strip())
    words = t.split()
    out = [words[0].lower()] if words else []
    out += ["X" if (w[:1].isupper() and not w.isupper()) else w.lower() for w in words[1:]]
    return " ".join(out)


def load_corpora() -> tuple[dict, dict]:
    meta, data = {}, {}
    p = EXT / "SMSSpamCollection"
    if not p.exists():
        sys.exit("run scripts/fetch_datasets.py sms first")
    rows = [l.rstrip("\n").split("\t", 1) for l in p.open(encoding="utf-8")]
    data["uci"] = pd.DataFrame([(r[1].strip(), int(r[0] == "spam")) for r in rows if len(r) == 2], columns=["text", "y"])
    meta["uci"] = {"licence": "CC BY 4.0", "provenance": "real (UK/Singapore, ~2004-2012)", "sha256": sha(p), "headline_ok": True}

    p = fetch(f"{HF}/CloveAI/india-spam-sms/resolve/main/data/train-00000-of-00001.parquet", EXT / "india_spam_sms.parquet")
    d = pd.read_parquet(p)
    data["india"] = d.rename(columns={"label": "y"})[["text", "y"]]
    meta["india"] = {"licence": "MIT", "provenance": "TEMPLATED (names/cities substituted), undocumented -> indicative only",
                     "sha256": sha(p), "headline_ok": False}

    parts = []
    for s in ("train", "validation", "test"):
        p = fetch(f"{HF}/shariul-islam/bengali-sms-smishing-dataset/resolve/main/data/{s}-00000-of-00001.parquet",
                  EXT / f"bengali_{s}.parquet")
        parts.append(pd.read_parquet(p))
    b = pd.concat(parts)
    b = b[b["source"] == "English"]
    data["bengali_en"] = pd.DataFrame({"text": b["text"], "y": (b["label"] == "smish").astype(int)})
    meta["bengali_en"] = {"licence": "MIT", "provenance": "English variety of SmishDetect-LLM corpus (Bangladesh); "
                          "promo+normal treated benign; provenance unclear -> indicative only",
                          "sha256": sha(p), "headline_ok": False}

    p = fetch(f"{HF}/MOZNLP/MOZ-Smishing/resolve/main/test.csv", EXT / "moz_smishing.csv")
    m = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")
    data["moz_pt"] = pd.DataFrame({"text": m["text"].astype(str), "y": (m["label"].str.lower().str.startswith("smish")).astype(int)})
    meta["moz_pt"] = {"licence": "CreativeML OpenRAIL-M", "provenance": "real crowd-sourced, Portuguese (Mozambique), mobile money",
                      "sha256": sha(p), "headline_ok": True}

    # de-duplicate within and across corpora (earlier corpus wins), drop conflicting labels
    seen = {}
    for k in data:
        df = data[k].copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 0].drop_duplicates("text")
        df = df[~df["text"].isin(seen)]
        for t in df["text"]:
            seen[t] = k
        df["skel"] = df["text"].map(skeleton)
        data[k] = df.reset_index(drop=True)
        meta[k].update({"rows": len(df), "malicious": int(df["y"].sum()),
                        "unique_skeletons": int(df["skel"].nunique())})
    return data, meta


def split(df, seed=SEED):
    """Group-by-skeleton 70/10/20. Returns index arrays into df."""
    idx = np.arange(len(df))
    g = df["skel"].to_numpy()
    tr, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(idx, groups=g))
    va, te = next(GroupShuffleSplit(1, test_size=2 / 3, random_state=seed).split(rest, groups=g[rest]))
    return tr, rest[va], rest[te]


def make_model(C=5.0):
    f = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([("f", f), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                                          solver="liblinear", random_state=SEED))])


def fit_with_threshold(train_df, val_df):
    m = make_model().fit(train_df["text"].tolist(), train_df["y"])
    thr = pick_threshold(val_df["y"], m.predict_proba(val_df["text"].tolist())[:, 1], MAX_FPR)
    return m, thr


def score_set(model, thr, df, n_boot=200):
    texts = df["text"].tolist()
    y = df["y"].to_numpy()
    s = model.predict_proba(texts)[:, 1]
    evs = [analyse_sms(t) for t in texts]
    strong = np.array([e.prediction == "likely_scam" for e in evs])
    anyf = np.array([e.prediction != "no_obvious_indicators" for e in evs])
    rs = np.array([e.risk_score for e in evs])
    return {"n": len(df), "n_pos": int(y.sum()),
            "ml": evaluate(y, s >= thr, s, n_boot=n_boot),
            "rules_any": evaluate(y, anyf, rs, n_boot=n_boot),
            "rules_strong": evaluate(y, strong, rs, n_boot=n_boot),
            "hybrid": evaluate(y, (s >= thr) | strong, np.maximum(s, strong), n_boot=n_boot)}


def main():
    data, meta = load_corpora()
    for k, v in meta.items():
        log(f"{k}: {v['rows']} rows, {v['malicious']} malicious, {v['unique_skeletons']} skeletons")
    splits = {k: split(d) for k, d in data.items()}
    names = list(data)
    res = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "corpora": meta,
           "threshold_rule": f"best val F1 s.t. val FPR<={MAX_FPR}; val drawn from TRAINING corpora only",
           "loco": {}, "baseline_uci_only": {}, "pooled_heldout": {}}

    # ---- BEFORE: UCI-only model transferred to every other corpus (threshold from UCI val)
    tr, va, te = splits["uci"]
    m0, thr0 = fit_with_threshold(data["uci"].iloc[tr], data["uci"].iloc[va])
    for k in names:
        tgt = data["uci"].iloc[te] if k == "uci" else data[k]
        res["baseline_uci_only"][k] = {"threshold": thr0, **score_set(m0, thr0, tgt)}
        log(f"UCI-only -> {k}: FPR={res['baseline_uci_only'][k]['ml']['fpr']:.3f} rec={res['baseline_uci_only'][k]['ml']['recall']:.3f}")

    # ---- LOCO: train on all other corpora (their full train+val splits), test on whole held-out corpus
    for held in names:
        trs = pd.concat([data[k].iloc[splits[k][0]] for k in names if k != held])
        vas = pd.concat([data[k].iloc[splits[k][1]] for k in names if k != held])
        m, thr = fit_with_threshold(trs, vas)
        res["loco"][held] = {"train_corpora": [k for k in names if k != held], "threshold": thr,
                             **score_set(m, thr, data[held])}
        r = res["loco"][held]["ml"]
        log(f"LOCO held={held}: P={r['precision']:.3f} R={r['recall']:.3f} FPR={r['fpr']:.3f}")

    # ---- pooled held-out: train on pooled train splits, test on each corpus' own (skeleton-grouped) test split
    trs = pd.concat([data[k].iloc[splits[k][0]] for k in names])
    vas = pd.concat([data[k].iloc[splits[k][1]] for k in names])
    mv2, thr2 = fit_with_threshold(trs, vas)
    for k in names:
        res["pooled_heldout"][k] = {"threshold": thr2, **score_set(mv2, thr2, data[k].iloc[splits[k][2]])}
    tes = pd.concat([data[k].iloc[splits[k][2]] for k in names])
    res["pooled_heldout"]["ALL_pooled"] = {"threshold": thr2, **score_set(mv2, thr2, tes)}

    # ---- benign probe: flashflow bank alerts (NO LICENCE; templated; probe only, never trained on)
    try:
        p = fetch(f"{HF}/4-alokk/flashflow-bank-sms/resolve/main/train.jsonl", EXT / "flashflow_bank_sms.jsonl")
        probe = [json.loads(l)["input"] for l in p.open(encoding="utf-8")]
        probe = list(dict.fromkeys(probe))
        s2 = mv2.predict_proba(probe)[:, 1]
        s0 = m0.predict_proba(probe)[:, 1]
        res["benign_probe_flashflow"] = {
            "note": "unlicensed/templated Indian bank alerts, assumed benign (no labels): flag rate is an "
                    "upper bound on FPR for transactional alerts; indicative only",
            "n": len(probe), "sha256": sha(p),
            "uci_only_flag_rate": float((s0 >= thr0).mean()), "v2_flag_rate": float((s2 >= thr2).mean()),
            "rules_flag_rate": float(np.mean([analyse_sms(t).prediction != "no_obvious_indicators" for t in probe]))}
    except Exception as exc:
        res["benign_probe_flashflow"] = {"error": str(exc)}

    # ---- ship v2 (the exact pooled model that was evaluated)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"kind": "sms", "model": mv2, "threshold": thr2, "corpora": names,
                 "warning": "trained on UCI + templated/indicative corpora; no real labelled Indian-English SMS exists in the data",
                 "trained": res["generated"]}, ROOT / "models" / "sms_model_v2.joblib", compress=3)

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "sms_improved.json").write_text(json.dumps(res, indent=2, default=float))
    (ROOT / "reports" / "SMS_IMPROVED.md").write_text(render(res), encoding="utf-8")
    log("done")


def render(r):
    L = ["# SMS detector: multi-corpus evaluation", f"Generated {r['generated']}. Threshold: {r['threshold_rule']}.", "",
         "## Corpora", "", "| corpus | rows | malicious | licence | provenance | headline-grade |", "|---|---|---|---|---|---|"]
    for k, v in r["corpora"].items():
        L.append(f"| {k} | {v['rows']} | {v['malicious']} | {v['licence']} | {v['provenance']} | {'yes' if v['headline_ok'] else 'NO (indicative)'} |")
    L += ["", "**No real, licensed, labelled Indian-English SMS corpus was found.** Results on `india` and `bengali_en` are "
          "indicative only (templated / unclear provenance).", ""]
    for title, key in (("BEFORE: UCI-only model transferred to each corpus", "baseline_uci_only"),
                       ("Leave-one-corpus-out (train on the other corpora, test on the whole held-out corpus)", "loco"),
                       ("Pooled model, each corpus' own skeleton-grouped held-out split (this is the shipped v2)", "pooled_heldout")):
        L += [f"## {title}", ""]
        for k, v in r[key].items():
            L += [f"### {k} (n={v['n']}, malicious={v['n_pos']}, thr={v['threshold']:.3f})", "", HEADER]
            for mk in ("ml", "rules_any", "rules_strong", "hybrid"):
                L.append(fmt_row(mk, v[mk]))
            L.append(f"\nML ROC-AUC {v['ml'].get('roc_auc', float('nan')):.4f}\n")
    L += ["## Benign probe (flashflow bank alerts; unlicensed, probe only)", "", "```json",
          json.dumps(r.get("benign_probe_flashflow"), indent=2), "```"]
    return "\n".join(L)


if __name__ == "__main__":
    main()
