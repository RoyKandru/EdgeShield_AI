"""
Train and evaluate the EdgeShield detectors against public labelled datasets and
compare them with the existing rule-based detectors on IDENTICAL held-out data.

    python scripts/fetch_datasets.py
    python scripts/train_eval_detectors.py [sms] [url] [network]

Outputs
  models/<name>.joblib            shipped artefact (the exact model that was evaluated)
  reports/detector_eval.json      all numbers, provenance, library versions
  reports/DETECTOR_EVALUATION.md  generated tables

Protocol (all detectors)
  * deterministic seeds; train / validation / test split BEFORE any fitting
  * model selection and the operating threshold are chosen on validation only
  * the test split is scored once; the shipped artefact is the model trained on
    the train split only, so what is reported is what is deployed
  * threshold rule: best F1 on validation subject to validation FPR <= 1 %
  * rule baselines are the unmodified production detectors in app/detectors
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.detectors.network import analyse_flows  # noqa: E402
from app.detectors.sms import analyse_sms  # noqa: E402
from app.detectors.url import analyse_url  # noqa: E402
from app.ml import features as F  # noqa: E402
from app.ml.metrics import evaluate, fmt_row, pick_threshold  # noqa: E402

EXT = ROOT / "data" / "external"
MAX_FPR = 0.01
SEED = 42
HEADER = "| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |\n|---|---|---|---|---|---|---|---|---|"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def three_way(idx_labels, seed=SEED, groups=None):
    """70/10/20 split. Group-aware when `groups` is given. Returns index arrays."""
    n = len(idx_labels)
    idx = np.arange(n)
    if groups is None:
        tr, rest = train_test_split(idx, test_size=0.30, random_state=seed, stratify=idx_labels)
        va, te = train_test_split(rest, test_size=2 / 3, random_state=seed, stratify=idx_labels[rest])
        return tr, va, te
    g = np.asarray(groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, rest = next(gss.split(idx, groups=g))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=2 / 3, random_state=seed)
    va_i, te_i = next(gss2.split(rest, groups=g[rest]))
    return tr, rest[va_i], rest[te_i]


def dedupe(texts, labels):
    """Drop exact duplicates; drop texts that appear with conflicting labels."""
    seen: dict[str, int] = {}
    conflict = set()
    for t, y in zip(texts, labels):
        if t in seen and seen[t] != y:
            conflict.add(t)
        seen.setdefault(t, y)
    keep = [(t, y) for t, y in seen.items() if t not in conflict]
    return keep, {"raw_rows": len(texts), "unique_rows": len(seen), "conflicting_texts_dropped": len(conflict)}


def split_summary(y, tr, va, te):
    return {k: {"n": int(len(i)), "n_pos": int(np.sum(y[i]))} for k, i in
            (("train", tr), ("val", va), ("test", te))}


# ---------------------------------------------------------------- SMS ---------

def run_sms(out_dir: Path) -> dict:
    log("SMS: loading UCI SMS Spam Collection")
    path = EXT / "sms" / "SMSSpamCollection"
    rows = [l.rstrip("\n").split("\t", 1) for l in path.open(encoding="utf-8")]
    rows = [r for r in rows if len(r) == 2]
    texts = [r[1].strip() for r in rows]
    labels = [1 if r[0] == "spam" else 0 for r in rows]
    uniq, dd = dedupe(texts, labels)
    X = [t for t, _ in uniq]
    y = np.array([l for _, l in uniq])
    tr, va, te = three_way(y)

    def make(C):
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
        ])
        return Pipeline([("f", feats),
                         ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                                    solver="liblinear", random_state=SEED))])

    cands = {}
    for C in (1.0, 5.0, 20.0, 100.0):
        m = make(C).fit([X[i] for i in tr], y[tr])
        s = m.predict_proba([X[i] for i in va])[:, 1]
        cands[C] = (average_precision_score(y[va], s), m, s)
        log(f"SMS: C={C} val PR-AUC={cands[C][0]:.4f}")
    bestC = max(cands, key=lambda c: cands[c][0])
    _, model, vs = cands[bestC]
    thr = pick_threshold(y[va], vs, MAX_FPR)

    Xte = [X[i] for i in te]
    s_te = model.predict_proba(Xte)[:, 1]
    res = {
        "dataset": {"name": "UCI SMS Spam Collection", "licence": "CC BY 4.0",
                    "sha256": sha256(path), **dd, "split": split_summary(y, tr, va, te),
                    "split_method": "stratified random 70/10/20 on de-duplicated messages"},
        "model": {"type": "tfidf(word1-2 + char_wb2-5) + LogisticRegression", "C": bestC,
                  "threshold": thr, "threshold_rule": f"best val F1 s.t. val FPR<={MAX_FPR}"},
        "test": {},
    }
    res["test"]["ml_tuned_threshold"] = evaluate(y[te], s_te >= thr, s_te)
    res["test"]["ml_threshold_0.5"] = evaluate(y[te], s_te >= 0.5, s_te)
    evs = [analyse_sms(t) for t in Xte]
    rs = np.array([e.risk_score for e in evs])
    res["test"]["rules_any_flag"] = evaluate(y[te], [e.prediction != "no_obvious_indicators" for e in evs], rs)
    strong = np.array([e.prediction == "likely_scam" for e in evs])
    res["test"]["rules_strong_flag"] = evaluate(y[te], strong, rs)
    # hybrid = exactly what the runtime does: ML flag OR high-confidence rule
    res["test"]["HYBRID_shipped"] = evaluate(y[te], (s_te >= thr) | strong, np.maximum(s_te, strong))

    res["cross_corpus"] = sms_cross_corpus(model, thr)

    out_dir.mkdir(exist_ok=True)
    joblib.dump({"kind": "sms", "model": model, "threshold": thr,
                 "trained": datetime.now(timezone.utc).isoformat()}, out_dir / "sms_model.joblib", compress=3)
    return res


def sms_cross_corpus(model, thr) -> dict:
    """Transfer test on CloveAI/india-spam-sms (MIT). CAVEAT: the corpus is visibly templated
    (names/cities substituted into fixed sentences) with undocumented provenance, so this is an
    indicative stress test of domain shift, NOT a headline accuracy figure."""
    try:
        p = _fetch("https://huggingface.co/datasets/CloveAI/india-spam-sms/resolve/main/data/"
                   "train-00000-of-00001.parquet", EXT / "sms" / "india_spam_sms.parquet")
        d = pd.read_parquet(p).drop_duplicates("text")
        d = d[d["text"].str.strip().astype(bool)]
        texts, yy = d["text"].tolist(), d["label"].to_numpy()
        s = model.predict_proba(texts)[:, 1]
        evs = [analyse_sms(t) for t in texts]
        strong = np.array([e.prediction == "likely_scam" for e in evs])
        anyf = np.array([e.prediction != "no_obvious_indicators" for e in evs])
        return {"dataset": "CloveAI/india-spam-sms (templated; indicative only)", "sha256": sha256(p),
                "unique_rows": int(len(d)), "malicious": int(yy.sum()),
                "ml_uci_threshold": evaluate(yy, s >= thr, s, n_boot=100),
                "rules_any_flag": evaluate(yy, anyf, n_boot=100),
                "rules_strong_flag": evaluate(yy, strong, n_boot=100),
                "HYBRID_shipped": evaluate(yy, (s >= thr) | strong, n_boot=100)}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------- URL ---------

def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"fetch {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "edgeshield-eval/1.0"})
        dest.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return dest


def load_tranco() -> pd.DataFrame:
    z = _fetch("https://tranco-list.eu/top-1m.csv.zip", EXT / "url" / "tranco-top-1m.csv.zip")
    with zipfile.ZipFile(z) as zf:
        return pd.read_csv(zf.open(zf.namelist()[0]), header=None, names=["rank", "domain"])


def run_url(out_dir: Path, max_train: int, tranco_train: int) -> dict:
    log("URL: loading phishing-dataset urls.json")
    path = EXT / "url" / "urls.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    aug_note = "none"
    if tranco_train:
        # The corpus' benign URLs mostly carry paths while its phishing URLs are mostly bare
        # domains; without legitimate bare domains the model learns "bare domain => phishing"
        # (measured: 95% FPR on popular domains). Add Tranco ranks 1..N as benign, in both forms.
        top = load_tranco()
        top = top[top["rank"] <= tranco_train]["domain"].tolist()
        raw = raw + [{"text": d, "label": 0} for d in top] + [{"text": f"https://{d}/", "label": 0} for d in top]
        aug_note = f"Tranco ranks 1-{tranco_train} added as benign (bare + https root forms)"
    norm_to_raw: dict[str, tuple[str, int]] = {}
    conflict = set()
    for r in raw:
        t = r["text"]
        n = F.normalise_url(t)
        if not n or " " in n:
            continue
        y = int(r["label"])
        if n in norm_to_raw and norm_to_raw[n][1] != y:
            conflict.add(n)
        norm_to_raw.setdefault(n, (t, y))
    items = [(n, *v) for n, v in norm_to_raw.items() if n not in conflict]
    N = np.array([i[0] for i in items])
    R = [i[1] for i in items]
    y = np.array([i[2] for i in items])
    groups = [F.registered_domain(n) for n in N]
    dd = {"augmentation": aug_note, "raw_rows": len(raw), "unique_normalised_rows": len(norm_to_raw),
          "conflicting_dropped": len(conflict), "rows_used": len(items)}
    tr, va, te = three_way(y, groups=groups)
    log(f"URL: rows={len(items)} train={len(tr)} val={len(va)} test={len(te)}")
    rng = np.random.default_rng(SEED)
    if len(tr) > max_train:
        tr = rng.choice(tr, max_train, replace=False)
        log(f"URL: training subsampled to {len(tr)} rows (RAM cap)")

    def sgd(alpha):
        return Pipeline([
            ("h", HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2 ** 20,
                                    alternate_sign=False, norm=None, dtype=np.float32,
                                    preprocessor=F.normalise_url)),
            ("t", TfidfTransformer(sublinear_tf=True)),
            ("clf", SGDClassifier(loss="log_loss", alpha=alpha, max_iter=20, tol=None,
                                  class_weight="balanced", random_state=SEED)),
        ])

    def hgb():
        return Pipeline([("x", __import__("sklearn.preprocessing", fromlist=["FunctionTransformer"])
                          .FunctionTransformer(F.url_feature_matrix)),
                         ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                                max_leaf_nodes=63, random_state=SEED))])

    Ntr, Nva, Nte = N[tr].tolist(), N[va].tolist(), N[te].tolist()
    comps = {}
    for a in (1e-6, 5e-6):
        m = sgd(a).fit(Ntr, y[tr]); s = m.predict_proba(Nva)[:, 1]
        comps[f"char_sgd_alpha{a:g}"] = (m, s)
        log(f"URL: char_sgd alpha={a:g} val PR-AUC={average_precision_score(y[va], s):.4f}")
    t0 = time.time()
    m = hgb().fit(Ntr, y[tr]); s = m.predict_proba(Nva)[:, 1]
    comps["lexical_hgb"] = (m, s)
    log(f"URL: lexical_hgb val PR-AUC={average_precision_score(y[va], s):.4f} ({time.time()-t0:.0f}s)")

    best_char = max([k for k in comps if k.startswith("char")], key=lambda k: average_precision_score(y[va], comps[k][1]))
    ens_s = (comps[best_char][1] + comps["lexical_hgb"][1]) / 2
    cand_scores = {best_char: comps[best_char][1], "lexical_hgb": comps["lexical_hgb"][1], "ensemble": ens_s}
    ap = {k: average_precision_score(y[va], v) for k, v in cand_scores.items()}
    log(f"URL: val PR-AUC {ap}")
    chosen = max(ap, key=ap.get)
    thr = pick_threshold(y[va], cand_scores[chosen], MAX_FPR)

    def score(names, X):
        return np.mean([comps[k][0].predict_proba(X)[:, 1] for k in names], axis=0)

    names = [best_char, "lexical_hgb"] if chosen == "ensemble" else [chosen]
    s_te = score(names, Nte)
    res = {
        "dataset": {"name": "ealvaradob/phishing-dataset (urls.json)", "licence": "Apache-2.0",
                    "sha256": sha256(path), **dd, "split": split_summary(y, tr, va, te),
                    "split_method": "70/10/20 grouped by registered domain (no domain in two splits); "
                                    "scheme and 'www.' stripped before features"},
        "model": {"candidates_val_pr_auc": ap, "chosen": chosen, "components": names,
                  "threshold": thr, "threshold_rule": f"best val F1 s.t. val FPR<={MAX_FPR}"},
        "test": {},
    }
    res["test"]["ml_tuned_threshold"] = evaluate(y[te], s_te >= thr, s_te)
    res["test"]["ml_threshold_0.5"] = evaluate(y[te], s_te >= 0.5, s_te)
    Rte = [R[i] for i in te]
    evs = [analyse_url(u) for u in Rte]
    rs = np.array([e.risk_score for e in evs])
    res["test"]["rules_any_flag"] = evaluate(y[te], [e.prediction != "no_obvious_indicators" for e in evs], rs)
    strong = np.array([e.prediction == "suspicious" for e in evs])
    res["test"]["rules_strong_flag"] = evaluate(y[te], strong, rs)
    res["test"]["HYBRID_shipped"] = evaluate(y[te], (s_te >= thr) | strong, np.maximum(s_te, strong))

    # ---- external sanity sets, independent of the training corpus ----------------
    res["external"] = external_url_checks(names, comps, thr)

    out_dir.mkdir(exist_ok=True)
    joblib.dump({"kind": "url", "models": [comps[k][0] for k in names], "threshold": thr,
                 "trained": datetime.now(timezone.utc).isoformat()}, out_dir / "url_model.joblib", compress=3)
    return res


def external_url_checks(names, comps, thr) -> dict:
    """FPR on modern popular domains (Tranco) and recall on today's OpenPhish feed."""
    out = {}
    try:
        df = load_tranco()
        # ranks 50,001-60,000: popular, modern, legitimate-by-construction (not a safety guarantee)
        dom = df[(df["rank"] > 50000) & (df["rank"] <= 60000)]["domain"].tolist()
        sets = {"tranco_bare_domain": dom, "tranco_https_root": [f"https://{d}/" for d in dom]}
        for k, urls in sets.items():
            nrm = [F.normalise_url(u) for u in urls]
            s = np.mean([comps[n][0].predict_proba(nrm)[:, 1] for n in names], axis=0)
            rules = [analyse_url(u).prediction != "no_obvious_indicators" for u in urls]
            out[k] = {"n": len(urls), "ml_fpr": float((s >= thr).mean()), "rules_fpr": float(np.mean(rules)),
                      "note": "all treated as benign; ranks 50001-60000 of Tranco list"}
    except Exception as exc:  # network is optional
        out["tranco_error"] = str(exc)
    try:
        p = _fetch("https://openphish.com/feed.txt",
                   EXT / "url" / f"openphish_{datetime.now(timezone.utc):%Y%m%d}.txt")
        urls = [u.strip() for u in p.read_text(encoding="utf-8").splitlines() if u.strip()]
        nrm = [F.normalise_url(u) for u in urls]
        s = np.mean([comps[n][0].predict_proba(nrm)[:, 1] for n in names], axis=0)
        rules = [analyse_url(u).prediction != "no_obvious_indicators" for u in urls]
        out["openphish_live"] = {"n": len(urls), "ml_recall": float((s >= thr).mean()),
                                 "rules_recall": float(np.mean(rules)), "sha256": sha256(p),
                                 "fetched": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        out["openphish_error"] = str(exc)
    return out


# ------------------------------------------------------------ NETWORK ---------

def load_cic() -> pd.DataFrame:
    frames = []
    for f in sorted((EXT / "network").glob("*.csv")):
        d = pd.read_csv(f, encoding="latin1", low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        pk = d["Total_Fwd_Packets"] + d["Total_Backward_Packets"]
        by = d["Total_Length_of_Fwd_Packets"] + d["Total_Length_of_Bwd_Packets"]
        out = pd.DataFrame({
            "packets": pk, "bytes": by, "duration_s": d["Flow_Duration"] / 1e6,
            "dst_port": d["Destination_Port"], "syn_count": d["SYN_Flag_Count"],
            "label": d["Label"].astype(str).map(lambda s: re.sub(r"[^A-Za-z0-9_-]+", "_", s)),
            "file": f.name,
        })
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["duration_s"] = df["duration_s"].clip(lower=0)
    df["y"] = (df["label"] != "BENIGN").astype(int)
    return df


def feats(df: pd.DataFrame) -> np.ndarray:
    return np.asarray([F.flow_features(*r) for r in
                       df[["packets", "bytes", "duration_s", "dst_port", "syn_count"]].itertuples(index=False)],
                      dtype=np.float32)


def rules_flags(df: pd.DataFrame):
    flows = [{"packets": p, "bytes": b, "duration_seconds": d, "dst_port": int(dp), "syn_count": s}
             for p, b, d, dp, s in df[["packets", "bytes", "duration_s", "dst_port", "syn_count"]].itertuples(index=False)]
    evs = analyse_flows(flows)
    return (np.array([e.prediction == "anomalous" for e in evs]), np.array([e.risk_score for e in evs]))


def per_family(df: pd.DataFrame, flagged: np.ndarray) -> dict:
    out = {}
    for fam, g in df[df["y"] == 1].groupby("label"):
        out[fam] = {"n": int(len(g)), "recall": float(flagged[g.index.to_numpy()].mean())}
    return out


def fit_hgb(Xtr, ytr):
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.1, max_leaf_nodes=63,
                                          random_state=SEED).fit(Xtr, ytr)


def run_network(out_dir: Path) -> dict:
    log("NET: loading CIC-IDS2017 flows")
    df = load_cic().reset_index(drop=True)
    log(f"NET: {len(df)} flows, attack share {df['y'].mean():.3f}")
    X = feats(df)
    y = df["y"].to_numpy()
    manifest = {p.name: sha256(p) for p in sorted((EXT / "network").glob("*.csv"))}
    res = {"dataset": {"name": "CIC-IDS2017 (per-flow CSV, HF mirror Mireu-Lab/CIC-IDS)",
                       "licence": "UNB research dataset - check terms before redistribution",
                       "files_sha256": manifest, "flows": int(len(df)),
                       "label_counts": df["label"].value_counts().to_dict(),
                       "features": list(F.FLOW_FEATURE_NAMES),
                       "note": "Monday (benign-only) file not present on mirror. No IPs in this mirror: "
                               "per-flow detection only."},
           "protocols": {}}

    def protocol(name, tr, va, te, descr, ship=False):
        log(f"NET[{name}]: train={len(tr)} val={len(va)} test={len(te)}")
        model = fit_hgb(X[tr], y[tr])
        vs = model.predict_proba(X[va])[:, 1]
        thr = pick_threshold(y[va], vs, MAX_FPR)
        s_te = model.predict_proba(X[te])[:, 1]
        dte = df.iloc[te].reset_index(drop=True)
        ml_flag = s_te >= thr
        rflag, rscore = rules_flags(dte)
        r = {"description": descr, "threshold": thr,
             "split": split_summary(y, tr, va, te),
             "test": {"ml_tuned_threshold": evaluate(y[te], ml_flag, s_te),
                      "ml_threshold_0.5": evaluate(y[te], s_te >= 0.5, s_te),
                      "rules": evaluate(y[te], rflag, rscore),
                      "HYBRID_shipped": evaluate(y[te], ml_flag | rflag, np.maximum(s_te, rflag))},
             "per_family_recall": {"ml_tuned_threshold": per_family(dte, ml_flag),
                                   "rules": per_family(dte, rflag)}}
        res["protocols"][name] = r
        return model, thr

    # A: in-distribution random split (optimistic: near-duplicate flows from the same attack burst)
    tr, va, te = three_way(y)
    modelA, thrA = protocol("A_random_split", tr, va, te,
                            "stratified random 70/10/20 over all days. In-distribution; flows from the "
                            "same attack burst are near-duplicates, so this is an UPPER bound.")
    out_dir.mkdir(exist_ok=True)
    joblib.dump({"kind": "network", "model": modelA, "threshold": thrA,
                 "features": list(F.FLOW_FEATURE_NAMES),
                 "trained": datetime.now(timezone.utc).isoformat()}, out_dir / "network_model.joblib", compress=3)

    # B: temporal / unseen-attack generalisation: train Tue-Thu, test Friday (Bot, PortScan, DDoS unseen)
    is_fri = df["file"].str.startswith("Friday").to_numpy()
    trainpool = np.where(~is_fri)[0]
    te_b = np.where(is_fri)[0]
    tr_b, va_b = train_test_split(trainpool, test_size=0.12, random_state=SEED, stratify=y[trainpool])
    protocol("B_train_Tue-Thu_test_Fri", tr_b, va_b, te_b,
             "train on Tuesday-Thursday, test on Friday. Bot, PortScan and DDoS are NEVER seen in "
             "training: measures generalisation to new attack families.")
    return res


# ------------------------------------------------------------- REPORT ---------

def render_markdown(r: dict) -> str:
    L = [f"# Detector evaluation (auto-generated {r['generated']})", "",
         f"Operating point: best validation F1 with validation FPR <= {MAX_FPR:.0%}. "
         "Positive class = malicious. FPR = FP/(FP+TN). 95% CIs: 300-sample bootstrap over test items.", ""]
    for name in ("sms", "url"):
        if name not in r:
            continue
        d = r[name]
        L += [f"## {name.upper()}", f"Dataset: {d['dataset']['name']} ({d['dataset']['licence']}); "
              f"{d['dataset']['split_method']}.", "",
              f"Test items: {d['dataset']['split']['test']['n']} "
              f"({d['dataset']['split']['test']['n_pos']} malicious). Model: {d['model'].get('type', d['model'].get('chosen'))}, "
              f"threshold {d['model']['threshold']:.4f}.", "", HEADER]
        for k, m in d["test"].items():
            L.append(fmt_row(k, m))
        L.append("")
        for k, m in d["test"].items():
            if "roc_auc" in m:
                L.append(f"- {k}: ROC-AUC {m['roc_auc']:.4f}, PR-AUC {m['pr_auc']:.4f}")
        if name == "url" and d.get("external"):
            L += ["", "### External sanity sets (independent of training corpus)", "",
                  "```json", json.dumps(d["external"], indent=2), "```"]
        L.append("")
    if "cross_corpus" in r.get("sms", {}) and "error" not in r["sms"]["cross_corpus"]:
        c = r["sms"]["cross_corpus"]
        L += ["### SMS cross-corpus transfer (model trained on UCI only; threshold NOT re-tuned)",
              f"{c['dataset']}; {c['unique_rows']} unique rows, {c['malicious']} malicious.", "", HEADER]
        for k in ("ml_uci_threshold", "rules_any_flag", "rules_strong_flag", "HYBRID_shipped"):
            L.append(fmt_row(k, c[k]))
        L.append("")
    if "network" in r:
        d = r["network"]
        L += ["## NETWORK (CIC-IDS2017, per-flow)", f"Flows: {d['dataset']['flows']}. "
              f"Features: {', '.join(d['dataset']['features'])}.", ""]
        for pn, p in d["protocols"].items():
            L += [f"### {pn}", p["description"], "",
                  f"Test flows: {p['split']['test']['n']} ({p['split']['test']['n_pos']} attack), "
                  f"threshold {p['threshold']:.4f}", "", HEADER]
            for k, m in p["test"].items():
                L.append(fmt_row(k, m))
            L += ["", "Per-family recall (ML tuned / rules):", ""]
            ml, ru = p["per_family_recall"]["ml_tuned_threshold"], p["per_family_recall"]["rules"]
            for fam in ml:
                L.append(f"- {fam} (n={ml[fam]['n']}): {ml[fam]['recall']:.3f} / {ru[fam]['recall']:.3f}")
            L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["sms", "url", "network"])
    ap.add_argument("--url-max-train", type=int, default=300_000)
    ap.add_argument("--tranco-train", type=int, default=50_000,
                    help="benign Tranco ranks 1..N added to URL training (0 = off)")
    args = ap.parse_args()
    out_dir, rep_dir = ROOT / "models", ROOT / "reports"
    rep_dir.mkdir(exist_ok=True)
    jpath = rep_dir / "detector_eval.json"
    result = json.loads(jpath.read_text()) if jpath.exists() else {}
    result["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["env"] = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
                     "seed": SEED, "max_fpr_for_threshold": MAX_FPR}
    if "sms" in args.which:
        result["sms"] = run_sms(out_dir)
    if "url" in args.which:
        result["url"] = run_url(out_dir, args.url_max_train, args.tranco_train)
    if "network" in args.which:
        result["network"] = run_network(out_dir)
    jpath.write_text(json.dumps(result, indent=2, default=float))
    (rep_dir / "DETECTOR_EVALUATION.md").write_text(render_markdown(result), encoding="utf-8")
    log(f"wrote {jpath} and DETECTOR_EVALUATION.md")


if __name__ == "__main__":
    main()
