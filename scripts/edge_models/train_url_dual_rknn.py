#!/usr/bin/env python3
"""
Train & Export Phishing URL Dual-Branch Neural Detector for RK3588 NPU_CORE_1
combining Prasad, Dondeti & Dondeti (2026, Procedia Computer Science 283) Lexical MLP
with Character 1D-CNN (EdgeShield v1.4.0 & Prasad Sec. 7 recommendation).

- Strict disk-space check before download; datasets are downloaded ONLY on workstation
  and automatically deleted immediately after training & RKNN export!
- Strict domain-grouped 70/15/15 split (zero shared registered domain across splits).
- 100% native RK3588 NPU FP16 ops: Conv1d + ReLU + MaxPool1d + Gemm + Concat.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRATCH = Path(__file__).resolve().parent
DATA_DIR = SCRATCH / "data"
ART_DIR = SCRATCH / "artifacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CHAR_MAX_LEN = 128
URL_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
CHAR_VOCAB = len(URL_ALPHABET) + 2  # 0=PAD, 1=UNK, 2..60=alphabet -> 61 channels
CHAR_LOOKUP = {c: i + 2 for i, c in enumerate(URL_ALPHABET)}

_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "ow.ly", "rb.gy"}
SUSPICIOUS_TERMS = (
    "verify", "urgent", "login", "signin", "update", "secure", "account", "bank", "kyc",
    "wallet", "password", "bonus", "prize", "confirm", "suspend", "billing", "paypal",
)
URL_FEATURE_NAMES = (
    "len", "host_len", "path_len", "query_len", "dots", "host_dots", "hyphens_host",
    "digits_ratio", "host_digits_ratio", "slashes", "at", "pct", "eq", "amp", "ip_host",
    "punycode", "shortener", "susp_terms", "host_entropy", "url_entropy", "has_path",
    "has_query", "tld_len", "longest_label", "numeric_labels",
)


def normalise_url(raw: str) -> str:
    u = html.unescape((raw or "").strip()).lower()
    u = _SCHEME.sub("", u)
    if u.startswith("www."):
        u = u[4:]
    return u


def url_host(norm: str) -> str:
    authority = re.split(r"[/?#]", norm, maxsplit=1)[0]
    authority = authority.rsplit("@", 1)[-1]
    return authority.split(":", 1)[0]


def registered_domain(norm: str) -> str:
    host = url_host(norm)
    if _IPV4.match(host):
        return host
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def extract_url_features(raw: str) -> list[float]:
    """Exact 25 offline lexical/structural features matching app/ml/features.py."""
    n = normalise_url(raw)
    host = url_host(n)
    rest = n[len(host):] if n.startswith(host) else ""
    path, _, query = rest.partition("?")
    labels = host.split(".") if host else [""]
    digits = sum(c.isdigit() for c in n)
    host_digits = sum(c.isdigit() for c in host)
    return [
        float(len(n)), float(len(host)), float(len(path)), float(len(query)),
        float(n.count(".")), float(host.count(".")), float(host.count("-")),
        digits / max(1, len(n)), host_digits / max(1, len(host)),
        float(n.count("/")), float(n.count("@")), float(n.count("%")),
        float(n.count("=")), float(n.count("&")),
        1.0 if _IPV4.match(host) else 0.0,
        1.0 if "xn--" in host else 0.0,
        1.0 if host in SHORTENERS else 0.0,
        float(sum(t in n for t in SUSPICIOUS_TERMS)),
        _entropy(host), _entropy(n),
        1.0 if len(path) > 1 else 0.0,
        1.0 if query else 0.0,
        float(len(labels[-1])),
        float(max(len(x) for x in labels)),
        float(sum(x.isdigit() for x in labels)),
    ]


def encode_url_onehot(norm_url: str, max_len: int = CHAR_MAX_LEN) -> np.ndarray:
    x = np.zeros((CHAR_VOCAB, max_len), dtype=np.float32)
    s = (norm_url or "")[:max_len]
    for i, ch in enumerate(s):
        idx = CHAR_LOOKUP.get(ch, 1)
        x[idx, i] = 1.0
    if len(s) < max_len:
        x[0, len(s):] = 1.0  # PAD channel
    return x


class URLDualBranchDataset(Dataset):
    def __init__(self, norm_urls: list[str], lex_scaled: np.ndarray, labels: np.ndarray):
        self.norm_urls = norm_urls
        self.lex = lex_scaled.astype(np.float32)
        self.y = labels.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        oh = encode_url_onehot(self.norm_urls[idx], CHAR_MAX_LEN)
        return torch.from_numpy(self.lex[idx]), torch.from_numpy(oh), torch.tensor(self.y[idx], dtype=torch.float32)


class URLDualBranchNPU(nn.Module):
    """Prasad et al. (2026) Lexical MLP + Character 1D-CNN fusion — 100% RK3588 NPU ops."""
    def __init__(self, n_lex: int = 25, char_vocab: int = CHAR_VOCAB):
        super().__init__()
        self.lex_mlp = nn.Sequential(
            nn.Linear(n_lex, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Input char_oh: [B, 61, 128] -> /2 (64) -> /2 (32) -> /8 (4) => 64*4=256
        self.char_cnn = nn.Sequential(
            nn.Conv1d(char_vocab, 48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(32 + 64, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, lex_feat: torch.Tensor, char_oh: torch.Tensor) -> torch.Tensor:
        h_lex = self.lex_mlp(lex_feat)
        h_cnn = self.char_cnn(char_oh)
        return self.head(torch.cat([h_lex, h_cnn], dim=1))  # [B, 1]


def check_disk_space_gb(path: Path, min_gb: float = 2.0) -> float:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    print(f"[Disk Check] {path}: {free_gb:.2f} GB free (minimum required: {min_gb:.1f} GB)")
    if free_gb < min_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB < {min_gb:.1f} GB")
    return free_gb


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"Downloading {url} -> {dest.name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 EdgeShield-Trainer/1.4"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f"  saved {dest.stat().st_size / (1024*1024):.2f} MB")
    return dest


def build_targeted_aug(rng: random.Random) -> list[tuple[str, int]]:
    """Small targeted augmentation for Indian domains + raw-IP/obfuscated phishing (distinct from Golden Suite)."""
    rows: list[tuple[str, int]] = []
    benign_paths = ["", "/", "/about", "/docs", "/help", "/login", "/account/settings", "/pricing", "/support/faq"]
    benign_hosts = [
        "google.com", "github.com", "microsoft.com", "apple.com", "amazon.in", "wikipedia.org",
        "stackoverflow.com", "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com",
        "axisbank.com", "rbi.org.in", "incometax.gov.in", "parivahan.gov.in", "uidai.gov.in",
        "npci.org.in", "irctc.co.in", "digilocker.gov.in", "cybercrime.gov.in", "gov.in",
    ]
    for h in benign_hosts:
        for p in benign_paths:
            rows.append((f"https://{h}{p}", 0))
            if not p:
                rows.append((h, 0))
    # Raw-IP phishing + brand-spoof subdomains + suspicious TLDs (Prasad Section 4.1.2 attack taxonomy)
    brands = ["paypal", "appleid", "microsoft", "sbi-yono", "hdfc-netbanking", "paytm-kyc", "amazon-pay", "netflix", "facebook", "bankofamerica"]
    bad_tlds = ["xyz", "top", "tk", "cf", "ga", "ml", "click", "cfd", "sbs", "online", "site", "vip"]
    paths = ["/login.php", "/verify/kyc", "/account/update", "/secure/signin.html", "/auth/confirm?id=9821", "/billing/suspend"]
    for _ in range(800):
        ip = f"{rng.choice([45, 91, 103, 185, 192, 203])}.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}"
        rows.append((f"http://{ip}{rng.choice(paths)}", 1))
    for _ in range(1200):
        b = rng.choice(brands)
        tld = rng.choice(bad_tlds)
        sub = rng.choice(["secure", "verify", "login", "update", "account", "support", "auth", "kyc"])
        rows.append((f"https://{sub}-{b}-{rng.randint(10,999)}.{tld}{rng.choice(paths)}", 1))
        rows.append((f"http://{b}.com.{sub}-portal.{tld}{rng.choice(paths)}", 1))
    return rows


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray, tau: float) -> dict:
    preds = (probs >= tau).astype(int)
    fn = int(((y_true == 1) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    tn = int(((y_true == 0) & (preds == 0)).sum())
    tp = int(((y_true == 1) & (preds == 1)).sum())
    return {
        "threshold": round(float(tau), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4) if len(set(y_true)) > 1 else 1.0,
        "ap": round(float(average_precision_score(y_true, probs)), 4) if len(set(y_true)) > 1 else 1.0,
        "fpr": round(float(fp / max(1, fp + tn)), 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 0. Strict disk space check BEFORE any download
    check_disk_space_gb(SCRATCH, min_gb=2.0)

    try:
        tranco_zip = fetch("https://tranco-list.eu/top-1m.csv.zip", DATA_DIR / "tranco-top-1m.csv.zip")
        urls_json = fetch("https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/urls.json", DATA_DIR / "urls.json")
        openphish_txt = fetch("https://openphish.com/feed.txt", DATA_DIR / "openphish_live.txt")

        # Load OpenPhish live feed (kept 100% disjoint for external zero-day test!)
        openphish_urls = [u.strip() for u in openphish_txt.read_text(encoding="utf-8", errors="replace").splitlines() if u.strip()]
        openphish_domains = {registered_domain(normalise_url(u)) for u in openphish_urls}

        # Load Tranco Top-1M
        with zipfile.ZipFile(tranco_zip) as zf:
            lines = zf.open(zf.namelist()[0]).read().decode("utf-8", errors="replace").splitlines()
        tranco_domains = [ln.split(",", 1)[1].strip() for ln in lines if "," in ln]
        tranco_train_doms = tranco_domains[:35000]
        tranco_ext_doms = tranco_domains[50000:60000]  # 10,000 completely unseen benign domains!
        tranco_ext_set = set(tranco_ext_doms)

        # Load ealvaradob/phishing-dataset
        raw_corpus = json.loads(urls_json.read_text(encoding="utf-8"))
        print(f"Loaded ealvaradob urls.json: {len(raw_corpus)} raw rows")

        rng = random.Random(SEED)
        # Assemble deduplicated, conflict-free pool (excluding external OpenPhish & Tranco 50k-60k domains!)
        norm_map: dict[str, int] = {}
        conflicts: set[str] = set()

        def add_item(raw_u: str, lbl: int):
            nu = normalise_url(raw_u)
            if not nu or " " in nu or len(nu) < 4:
                return
            dom = registered_domain(nu)
            if dom in openphish_domains or dom in tranco_ext_set:
                return
            if nu in norm_map and norm_map[nu] != lbl:
                conflicts.add(nu)
                return
            norm_map.setdefault(nu, lbl)

        for r in raw_corpus:
            add_item(r.get("text", ""), int(r.get("label", 0)))

        benign_suffixes = ["", "/", "/index.html", "/about", "/contact", "/docs", "/products", "/login"]
        for d in tranco_train_doms:
            add_item(d, 0)
            add_item(f"https://{d}/", 0)
            if rng.random() < 0.25:
                add_item(f"https://{d}{rng.choice(benign_suffixes)}", 0)

        for raw_u, lbl in build_targeted_aug(rng):
            add_item(raw_u, lbl)

        items = [(nu, y) for nu, y in norm_map.items() if nu not in conflicts]
        rng.shuffle(items)
        print(f"Deduplicated clean pool (OpenPhish & Tranco-50k-60k excluded): {len(items)} rows | conflicts dropped: {len(conflicts)}")

        # Cap pool to 160,000 balanced samples for fast, thorough training while preserving full domain diversity
        pos_items = [x for x in items if x[1] == 1]
        neg_items = [x for x in items if x[1] == 0]
        print(f"Class pool counts: benign={len(neg_items)}, phishing={len(pos_items)}")
        pos_cap = pos_items[:75000]
        neg_cap = neg_items[:85000]
        pooled = pos_cap + neg_cap
        rng.shuffle(pooled)

        norm_all = [x[0] for x in pooled]
        y_all = np.array([x[1] for x in pooled], dtype=np.int64)
        groups_all = np.array([registered_domain(u) for u in norm_all])

        # Strict domain-grouped 70 / 15 / 15 split
        idx = np.arange(len(pooled))
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
        tr_idx, rest_idx = next(gss1.split(idx, y_all, groups=groups_all))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
        va_sub, te_sub = next(gss2.split(rest_idx, y_all[rest_idx], groups=groups_all[rest_idx]))
        va_idx, te_idx = rest_idx[va_sub], rest_idx[te_sub]
        print(f"Domain-grouped split: train={len(tr_idx)} (pos={y_all[tr_idx].sum()}) | val={len(va_idx)} (pos={y_all[va_idx].sum()}) | test={len(te_idx)} (pos={y_all[te_idx].sum()})")

        # Extract 25 lexical features & fit StandardScaler STRICTLY on training split
        print("Extracting 25 offline lexical features...")
        t_fe = time.perf_counter()
        lex_all = np.asarray([extract_url_features(u) for u in norm_all], dtype=np.float32)
        mean = lex_all[tr_idx].mean(axis=0)
        std = lex_all[tr_idx].std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        lex_scaled = np.clip((lex_all - mean) / std, -8.0, 8.0).astype(np.float32)
        print(f"Feature extraction done in {time.perf_counter() - t_fe:.1f}s")

        ds_tr = URLDualBranchDataset([norm_all[i] for i in tr_idx], lex_scaled[tr_idx], y_all[tr_idx])
        ds_va = URLDualBranchDataset([norm_all[i] for i in va_idx], lex_scaled[va_idx], y_all[va_idx])
        ds_te = URLDualBranchDataset([norm_all[i] for i in te_idx], lex_scaled[te_idx], y_all[te_idx])

        dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True, num_workers=2)
        dl_va = DataLoader(ds_va, batch_size=512, shuffle=False, num_workers=2)
        dl_te = DataLoader(ds_te, batch_size=512, shuffle=False, num_workers=2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training URLDualBranchNPU on {device} ...")

        n_tot = len(tr_idx)
        n_pos = int(y_all[tr_idx].sum())
        n_neg = n_tot - n_pos
        w1 = n_tot / (2.0 * n_pos)
        w0 = n_tot / (2.0 * n_neg)
        print(f"WBCE weights: w1={w1:.4f}, w0={w0:.4f}")

        model = URLDualBranchNPU(n_lex=len(URL_FEATURE_NAMES), char_vocab=CHAR_VOCAB).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

        epochs = 10
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        def predict_dl(dl):
            model.eval()
            ps, ys = [], []
            with torch.no_grad():
                for lx, ch, yb in dl:
                    logits = model(lx.to(device), ch.to(device)).squeeze(-1)
                    ps.append(torch.sigmoid(logits).cpu().numpy())
                    ys.append(yb.numpy())
            return np.concatenate(ys), np.concatenate(ps)

        best_ap = -1.0
        best_state = None
        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            model.train()
            tot_loss = 0.0
            for lx, ch, yb in dl_tr:
                lx, ch, yb = lx.to(device), ch.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(lx, ch).squeeze(-1)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                loss = (bce * (yb * w1 + (1.0 - yb) * w0)).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                tot_loss += float(loss.item()) * len(yb)
            sched.step()
            yv, pv = predict_dl(dl_va)
            m = evaluate_probs(yv, pv, 0.50)
            if m["ap"] > best_ap:
                best_ap = m["ap"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {ep:02d}/{epochs} | loss={tot_loss/n_tot:.4f} | val AP={m['ap']:.4f} AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} FPR={m['fpr']:.4f}")

        train_secs = round(time.perf_counter() - t0, 2)
        model.load_state_dict(best_state)
        model.eval()

        # Calibrate high-precision (FPR <= 1.0%) and cost-sensitive (5:1) thresholds on validation split
        yv, pv = predict_dl(dl_va)
        taus = np.linspace(0.10, 0.96, 87)
        hi_tau, hi_f1 = 0.85, -1.0
        for t in taus:
            m = evaluate_probs(yv, pv, t)
            if m["fpr"] <= 0.01 and m["f1"] > hi_f1:
                hi_f1 = m["f1"]
                hi_tau = float(t)
        cost_tau, best_cost = 0.50, float("inf")
        for t in taus:
            preds = (pv >= t).astype(int)
            fn = int(((yv == 1) & (preds == 0)).sum())
            fp = int(((yv == 0) & (preds == 1)).sum())
            c = 5.0 * fn + 1.0 * fp
            if c < best_cost:
                best_cost = c
                cost_tau = float(t)
        print(f"Calibrated thresholds: hi_tau (FPR<=1%)={hi_tau:.3f} | cost_tau (5:1)={cost_tau:.3f}")

        # Evaluate on held-out domain-grouped test split
        yt, pt = predict_dl(dl_te)
        test_at_05 = evaluate_probs(yt, pt, 0.50)
        test_at_hi = evaluate_probs(yt, pt, hi_tau)
        test_at_cost = evaluate_probs(yt, pt, cost_tau)
        print("Held-out Test (0.50):", test_at_05)
        print(f"Held-out Test (hi={hi_tau:.2f}):", test_at_hi)
        print(f"Held-out Test (cost={cost_tau:.2f}):", test_at_cost)

        # Evaluate on COMPLETELY DISJOINT external sanity sets: OpenPhish live + Tranco 50k-60k!
        def score_urls_pt(raw_list: list[str]) -> np.ndarray:
            nlist = [normalise_url(u) for u in raw_list]
            lx = np.asarray([extract_url_features(u) for u in nlist], dtype=np.float32)
            lx = np.clip((lx - mean) / std, -8.0, 8.0).astype(np.float32)
            ds = URLDualBranchDataset(nlist, lx, np.zeros(len(nlist), dtype=np.float32))
            _, ps = predict_dl(DataLoader(ds, batch_size=512, shuffle=False))
            return ps

        op_scores = score_urls_pt(openphish_urls)
        tr_https_scores = score_urls_pt([f"https://{d}/" for d in tranco_ext_doms[:5000]])
        tr_bare_scores = score_urls_pt(tranco_ext_doms[:5000])
        ext_checks = {
            "openphish_live": {
                "n": len(openphish_urls),
                "recall_at_050": round(float((op_scores >= 0.50).mean()), 4),
                "recall_at_hi": round(float((op_scores >= hi_tau).mean()), 4),
                "recall_at_cost": round(float((op_scores >= cost_tau).mean()), 4),
            },
            "tranco_50k_55k_https": {
                "n": 5000,
                "fpr_at_050": round(float((tr_https_scores >= 0.50).mean()), 4),
                "fpr_at_hi": round(float((tr_https_scores >= hi_tau).mean()), 4),
            },
            "tranco_50k_55k_bare": {
                "n": 5000,
                "fpr_at_050": round(float((tr_bare_scores >= 0.50).mean()), 4),
                "fpr_at_hi": round(float((tr_bare_scores >= hi_tau).mean()), 4),
            },
        }
        print("External Disjoint Checks:", json.dumps(ext_checks, indent=2))

        # Spot-check Prasad paper Fig. 16 failure cases!
        spot_urls = [
            "https://www.google.com",
            "https://github.com/OpenAI",
            "https://onlinesbi.sbi/",
            "http://192.168.0.1/login.php",
            "http://paypal-login-update.com",
            "https://secure-facebook-com-login-alert.ga",
            "http://accounts.google.com.mailverify.cc",
        ]
        spot_p = score_urls_pt(spot_urls)
        print("Spot Checks (incl. Prasad Fig. 16 cases):")
        for u, p in zip(spot_urls, spot_p):
            print(f"  p={p:.4f} | {u}")

        # Export ONNX -> RKNN FP16 (RK3588 NPU_CORE_1)
        model_cpu = model.cpu().eval()
        d_lex = torch.randn(1, len(URL_FEATURE_NAMES), dtype=torch.float32)
        d_ch = torch.zeros(1, CHAR_VOCAB, CHAR_MAX_LEN, dtype=torch.float32)
        d_ch[0, 0, :] = 1.0
        onnx_path = ART_DIR / "url_hybrid_fp32.onnx"
        torch.onnx.export(
            model_cpu,
            (d_lex, d_ch),
            str(onnx_path),
            input_names=["lex_feat", "char_oh"],
            output_names=["logit"],
            opset_version=12,
            do_constant_folding=True,
        )
        torch.save(model_cpu.state_dict(), ART_DIR / "url_hybrid_dual.pt")

        from rknn.api import RKNN
        rknn_path = ART_DIR / "url_hybrid_fp16.rknn"
        rknn = RKNN(verbose=False)
        try:
            assert rknn.config(target_platform="rk3588", optimization_level=3) == 0
            assert rknn.load_onnx(model=str(onnx_path)) == 0
            assert rknn.build(do_quantization=False) == 0
            assert rknn.export_rknn(str(rknn_path)) == 0
            assert rknn.init_runtime() == 0

            # Simulator parity check on 20 held-out URLs
            diffs = []
            for i in range(20):
                lx_t, ch_t, _ = ds_te[i]
                sim_logit = float(np.asarray(rknn.inference(inputs=[lx_t.numpy()[None, :], ch_t.numpy()[None, :, :]])[0]).reshape(-1)[0])
                sim_prob = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, sim_logit))))
                diffs.append(abs(sim_prob - float(pt[i])))
            max_diff = max(diffs)
            mean_diff = float(np.mean(diffs))
            print(f"RKNN FP16 Simulator Parity (20 test URLs): max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
        finally:
            rknn.release()

        rknn_sha = hashlib.sha256(rknn_path.read_bytes()).hexdigest()[:16]
        meta = {
            "model_id": "url-dualbranch-mlp-charcnn-rknn-v1",
            "paper": {
                "title": "Phishing URL Detection in Real-Time: Machine Learning and Deep Learning Approaches on a Custom Threat Intelligence Dataset",
                "authors": "Y. Bhanu Prasad, Venkatesulu Dondeti, Ravitez Dondeti",
                "venue": "Procedia Computer Science 283 (2026) 4859-4879",
                "upgrade": "25-feature offline Lexical MLP (Prasad 2026) + Character 1D-CNN (Sec. 7 & EdgeShield v1.4.0) fused inside RK3588 NPU_CORE_1 FP16",
            },
            "architecture": {
                "name": "URLDualBranchNPU (Lexical MLP 25->64->32 + Char1DCNN 61x128->64 + Fusion Head 96->48->1)",
                "npu_target": "rk3588",
                "npu_core_id": 1,
                "npu_precision": "fp16",
                "n_params": n_params,
                "n_lex_features": len(URL_FEATURE_NAMES),
                "char_vocab": CHAR_VOCAB,
                "char_max_len": CHAR_MAX_LEN,
                "rknn_bytes": rknn_path.stat().st_size,
                "rknn_sha256_prefix": rknn_sha,
            },
            "preprocessing": {
                "feature_names": list(URL_FEATURE_NAMES),
                "mean": [round(float(x), 6) for x in mean],
                "std": [round(float(x), 6) for x in std],
                "alphabet": URL_ALPHABET,
                "char_max_len": CHAR_MAX_LEN,
            },
            "training": {
                "datasets": ["ealvaradob/phishing-dataset (urls.json)", "Tranco Top-35k Benign", "Targeted IP/Brand-Spoof Augmentation"],
                "train_rows": len(tr_idx),
                "val_rows": len(va_idx),
                "test_rows": len(te_idx),
                "split_method": "domain-grouped 70/15/15 (zero shared registered domain)",
                "wbce_weights": {"w0": round(w0, 4), "w1": round(w1, 4)},
                "train_seconds": train_secs,
            },
            "calibration": {
                "high_precision_threshold": round(hi_tau, 3),
                "cost_sensitive_threshold": round(cost_tau, 3),
            },
            "test_metrics": {
                "at_050": test_at_05,
                "at_high_precision": test_at_hi,
                "at_cost_5_1": test_at_cost,
                "external_disjoint": ext_checks,
                "rknn_fp16_sim_parity_max_diff": round(max_diff, 6),
                "rknn_fp16_sim_parity_mean_diff": round(mean_diff, 6),
            },
        }
        meta_path = ART_DIR / "url_hybrid_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Exported RKNN artifacts: {rknn_path.name} ({rknn_path.stat().st_size/1024:.1f} KB), {meta_path.name} ({meta_path.stat().st_size/1024:.1f} KB)")
    finally:
        # ALWAYS delete downloaded training datasets immediately after training!
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR, ignore_errors=True)
            print(f"[Cleanup] Deleted temporary training dataset directory: {DATA_DIR}")
        check_disk_space_gb(SCRATCH, min_gb=1.0)


if __name__ == "__main__":
    main()
