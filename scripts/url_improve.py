"""
URL detector v2: reduce false positives on popular domains and raise recall on fresh phishing.

Changes vs scripts/train_eval_detectors.py (v1):
  1. fresher phishing: Phishing.Database ACTIVE list (sampled), with every URL / registered
     domain that appears in the OpenPhish check set REMOVED (check set stays independent)
  2. Tranco ranks 1-50k as benign (bare + root forms), as in v1
  3. host-only char-ngram component added to the ensemble (targets bare-domain behaviour)
  4. threshold calibrated on a validation set that includes Tranco ranks 60,001-70,000
     (disjoint from the 50,001-60,000 check set) and a lower FPR cap
  5. documented hybrid: Tranco allowlist (ranks <=50k registered domains, minus any domain seen
     as phishing in training data) -> benign, otherwise ML. Evaluated both with and without.

Run: python scripts/url_improve.py
Outputs: reports/url_improved.json, reports/URL_IMPROVED.md, models/url_model_v2.joblib
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.detectors.url import analyse_url  # noqa: E402
from app.ml import features as F  # noqa: E402
from app.ml.metrics import evaluate, fmt_row, pick_threshold  # noqa: E402

EXT = ROOT / "data" / "external" / "url"
SEED = 42
MAX_TRAIN = 250_000
MAX_FPR = 0.005
PHISHDB_URL = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
HEADER = ("| model | precision [95% CI] | recall [95% CI] | F1 | FPR [95% CI] | TP | FP | TN | FN |\n"
          "|---|---|---|---|---|---|---|---|---|")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(url, dest):
    if not dest.exists():
        log(f"fetch {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "edgeshield-eval/1.0"})
        dest.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    return dest


from app.ml.url_features_v2 import host_only, parse_psl, registrable_domain  # noqa: E402

PSL = parse_psl(fetch('https://publicsuffix.org/list/public_suffix_list.dat', EXT / 'public_suffix_list.dat').read_text(encoding='utf-8'))


def rd(norm: str) -> str:
    return registrable_domain(F.url_host(norm), PSL)


def host_feature_matrix_model(alpha):
    return Pipeline([
        ("h", HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=2 ** 19,
                                alternate_sign=False, norm=None, dtype=np.float32,
                                preprocessor=host_only)),
        ("t", TfidfTransformer(sublinear_tf=True)),
        ("clf", SGDClassifier(loss="log_loss", alpha=alpha, max_iter=20, tol=None,
                              class_weight="balanced", random_state=SEED)),
    ])


def full_char_model(alpha):
    return Pipeline([
        ("h", HashingVectorizer(analyzer="char", ngram_range=(3, 5), n_features=2 ** 20,
                                alternate_sign=False, norm=None, dtype=np.float32,
                                preprocessor=F.normalise_url)),
        ("t", TfidfTransformer(sublinear_tf=True)),
        ("clf", SGDClassifier(loss="log_loss", alpha=alpha, max_iter=20, tol=None,
                              class_weight="balanced", random_state=SEED)),
    ])


def lexical_model():
    return Pipeline([("x", FunctionTransformer(F.url_feature_matrix)),
                     ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                            max_leaf_nodes=63, random_state=SEED))])


def ens_score(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def main():
    t_all = time.time()
    # ---------------- check sets (independent of training) -------------------------------
    op_path = sorted(EXT.glob("openphish_*.txt"))[-1]
    op_urls = [u.strip() for u in op_path.read_text(encoding="utf-8").splitlines() if u.strip()]
    op_norm = {F.normalise_url(u) for u in op_urls}
    op_dom = {rd(n) for n in op_norm}
    log(f"OpenPhish check set: {len(op_urls)} urls, {len(op_dom)} domains ({op_path.name})")

    z = fetch("https://tranco-list.eu/top-1m.csv.zip", EXT / "tranco-top-1m.csv.zip")
    with zipfile.ZipFile(z) as zf:
        tr = pd.read_csv(zf.open(zf.namelist()[0]), header=None, names=["rank", "domain"])
    tr_train = tr[tr["rank"] <= 50_000]["domain"].tolist()
    tr_check = tr[(tr["rank"] > 50_000) & (tr["rank"] <= 60_000)]["domain"].tolist()   # final check
    tr_val = tr[(tr["rank"] > 60_000) & (tr["rank"] <= 70_000)]["domain"].tolist()      # threshold calibration

    # ---------------- training pool ------------------------------------------------------
    raw = json.loads((EXT / "urls.json").read_text(encoding="utf-8"))
    pool: dict[str, tuple[str, int, str]] = {}   # norm -> (raw, label, source)
    conflict = set()

    def add(text, y, src):
        n = F.normalise_url(text)
        if not n or " " in n or len(n) > 300:
            return
        if n in op_norm or rd(n) in op_dom:      # keep OpenPhish check independent
            return
        if n in pool and pool[n][1] != y:
            conflict.add(n)
        pool.setdefault(n, (text, y, src))

    for r in raw:
        add(r["text"], int(r["label"]), "corpus")
    for d in tr_train:
        add(d, 0, "tranco_train")
        add(f"https://{d}/", 0, "tranco_train")
    n_corpus = len(pool)
    p = fetch(PHISHDB_URL, EXT / "phishing_database_active.txt")
    lines = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    rng = np.random.default_rng(SEED)
    if len(lines) > 150_000:
        lines = [lines[i] for i in rng.choice(len(lines), 150_000, replace=False)]
    for l in lines:
        add(l, 1, "phishdb")
    for n in conflict:
        pool.pop(n, None)
    items = [(n, *v) for n, v in pool.items()]
    N = np.array([i[0] for i in items], dtype=object)
    R = [i[1] for i in items]
    y = np.array([i[2] for i in items])
    src = np.array([i[3] for i in items])
    groups = np.array([rd(n) for n in N])
    log(f"pool={len(items)} (corpus+tranco={n_corpus}); by source: {pd.Series(src).value_counts().to_dict()}; "
        f"conflicts dropped={len(conflict)}")

    # domain-grouped 70/10/20
    idx = np.arange(len(N))
    a, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(idx, groups=groups))
    b, c = next(GroupShuffleSplit(1, test_size=2 / 3, random_state=SEED).split(rest, groups=groups[rest]))
    tr_i, va_i, te_i = a, rest[b], rest[c]
    if len(tr_i) > MAX_TRAIN:
        tr_i = rng.choice(tr_i, MAX_TRAIN, replace=False)
    log(f"train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    # allowlist = Tranco<=50k registered domains minus any domain seen as phishing in TRAIN split
    seen_phish_dom = set(groups[tr_i][y[tr_i] == 1])
    allow = {rd(F.normalise_url(d)) for d in tr_train} - seen_phish_dom
    log(f"allowlist: {len(allow)} domains ({len(seen_phish_dom & {rd(F.normalise_url(d)) for d in tr_train})} removed as seen-phishing)")

    # ---------------- models -------------------------------------------------------------
    Ntr = N[tr_i].tolist()
    comps = {}
    for name, mk in (("char", lambda: full_char_model(1e-6)), ("host", lambda: host_feature_matrix_model(1e-6)),
                     ("lexical", lexical_model)):
        t0 = time.time()
        comps[name] = mk().fit(Ntr, y[tr_i])
        log(f"fit {name} ({time.time()-t0:.0f}s)")

    # validation = pool val split + Tranco-val (bare + root) as extra benign (deployment-like)
    tv = [F.normalise_url(d) for d in tr_val] + [F.normalise_url(f"https://{d}/") for d in tr_val]
    Nva = N[va_i].tolist() + tv
    yva = np.r_[y[va_i], np.zeros(len(tv), dtype=int)]
    dom_va = np.r_[groups[va_i], np.array([rd(t) for t in tv])]
    sets = {"char+lexical (v1)": ["char", "lexical"], "char+host+lexical (v2)": ["char", "host", "lexical"]}
    vs_all = {k: ens_score([comps[n] for n in v], Nva) for k, v in sets.items()}
    ap = {k: average_precision_score(yva, s) for k, s in vs_all.items()}
    log(f"val PR-AUC {ap}")
    chosen = max(ap, key=ap.get)
    names = sets[chosen]
    models = [comps[n] for n in names]

    def allow_mask(norm_list):
        return np.array([rd(n) in allow for n in norm_list])

    # threshold: for the hybrid, calibrate on what the model sees after the allowlist
    vs = vs_all[chosen]
    thr_ml = pick_threshold(yva, vs, MAX_FPR)
    am = allow_mask(Nva)
    thr_hyb = thr_ml  # an allowlist only removes flags, so it must not loosen the threshold
    log(f"chosen={chosen} thr_ml={thr_ml:.4f} thr_hybrid={thr_hyb:.4f}")

    # ---------------- evaluation ---------------------------------------------------------
    Nte = N[te_i].tolist()
    s_te = ens_score(models, Nte)
    am_te = allow_mask(Nte)
    yte = y[te_i]
    ste_src = src[te_i]
    res = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "config": {"max_train": MAX_TRAIN, "max_fpr_val": MAX_FPR, "chosen": chosen, "components": names,
                      "val_pr_auc": ap, "threshold_ml": thr_ml, "threshold_hybrid": thr_hyb,
                      "allowlist_size": len(allow),
                      "pool_by_source": pd.Series(src).value_counts().to_dict(),
                      "openphish_overlap_removed_domains": len(op_dom),
                      "split_sizes": {"train": int(len(tr_i)), "val": int(len(va_i)), "test": int(len(te_i))}},
           "test_mixed": {}, "test_by_source": {}, "external": {}}
    res["test_mixed"]["ml_only"] = evaluate(yte, s_te >= thr_ml, s_te)
    res["test_mixed"]["hybrid_allowlist+ml"] = evaluate(yte, (s_te >= thr_hyb) & ~am_te, np.where(am_te, 0, s_te))
    for sname in ("corpus", "phishdb", "tranco_train"):
        m = ste_src == sname
        if m.sum() and 0 < yte[m].sum() < m.sum():
            res["test_by_source"][f"{sname}_ml_only"] = evaluate(yte[m], s_te[m] >= thr_ml, s_te[m], n_boot=100)
            res["test_by_source"][f"{sname}_hybrid"] = evaluate(
                yte[m], (s_te[m] >= thr_hyb) & ~am_te[m], n_boot=100)
        elif m.sum():
            ps = s_te[m] >= thr_ml
            res["test_by_source"][sname] = {"n": int(m.sum()), "positives": int(yte[m].sum()),
                                            "flag_rate_ml_only": float(ps.mean()),
                                            "flag_rate_hybrid": float(((s_te[m] >= thr_hyb) & ~am_te[m]).mean())}
    # rule baseline on same test
    Rte = [R[i] for i in te_i]
    rf = np.array([analyse_url(u).prediction != "no_obvious_indicators" for u in Rte])
    res["test_mixed"]["rules_any_flag"] = evaluate(yte, rf)

    def flag(norm_list, hybrid):
        s = ens_score(models, norm_list)
        if hybrid:
            return (s >= thr_hyb) & ~allow_mask(norm_list)
        return s >= thr_ml

    ext = {}
    for k, urls in (("tranco_bare_domain", tr_check), ("tranco_https_root", [f"https://{d}/" for d in tr_check])):
        nl = [F.normalise_url(u) for u in urls]
        ext[k] = {"n": len(urls), "note": "Tranco ranks 50,001-60,000 (disjoint from training/allowlist/threshold calibration)",
                  "ml_only_fpr": float(flag(nl, False).mean()), "hybrid_fpr": float(flag(nl, True).mean()),
                  "rules_fpr": float(np.mean([analyse_url(u).prediction != "no_obvious_indicators" for u in urls]))}
    nl = [F.normalise_url(u) for u in op_urls]
    ext["openphish_live"] = {"n": len(op_urls), "file": op_path.name,
                             "ml_only_recall": float(flag(nl, False).mean()),
                             "hybrid_recall": float(flag(nl, True).mean()),
                             "rules_recall": float(np.mean([analyse_url(u).prediction != "no_obvious_indicators" for u in op_urls])),
                             "fraction_on_allowlisted_domains": float(allow_mask(nl).mean())}
    res["external"] = ext

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump({"kind": "url_v2", "models": models, "components": names, "threshold_ml": thr_ml,
                 "threshold_hybrid": thr_hyb, "allowlist": sorted(allow), "psl_rules": sorted(PSL),
                 "trained": res["generated"]}, out / "url_model_v2.joblib", compress=3)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "url_improved.json").write_text(json.dumps(res, indent=2, default=float))
    L = ["# URL detector v2 evaluation (auto-generated)", "",
         f"Config: {json.dumps(res['config'])}", "", "## Held-out mixed test (domain-grouped)", "", HEADER]
    for k, m in res["test_mixed"].items():
        L.append(fmt_row(k, m))
    L += ["", "## Test split by source", ""]
    L += [f"- {k}: `{json.dumps({a: b for a, b in v.items() if a in ('precision', 'recall', 'f1', 'fpr', 'n', 'positives', 'flag_rate_ml_only', 'flag_rate_hybrid')}, default=float)}`"
          for k, v in res["test_by_source"].items()]
    L += ["", "## External checks", "", "```json", json.dumps(ext, indent=2), "```"]
    (ROOT / "reports" / "URL_IMPROVED.md").write_text("\n".join(L), encoding="utf-8")
    log(f"done in {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
