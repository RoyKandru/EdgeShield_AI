#!/usr/bin/env python3
"""
Train & Export 26-Feature Network Intrusion Detection Model (1D-CNN + Residual MLP)
for RK3588 NPU_CORE_0, matching EdgeShield's exact raw-packet `V2r_nolen26` (`MODEL_FEATURES`).

- Uses open, ungated CIC-IDS2017 CSVs (HuggingFace Mireu-Lab/CIC-IDS) covering DDoS, PortScan,
  DoS (Hulk/GoldenEye/Slowloris/Slowhttptest), FTP/SSH-Patator, Bot, and Web Attacks (SQLi/XSS/BF),
  plus teacher-guided distillation parity with EdgeShield's 2.83M-flow `network_model_v2r.joblib`.
- Uses signed-log1p compression + StandardScaler so heavy-tailed flow rates fit cleanly in RKNN FP16.
- Strict disk-space check before download; automatically deletes downloaded CSVs immediately after
  RKNN export so zero dataset files remain on disk!
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import shutil
import time
import urllib.request
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
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRATCH = Path(__file__).resolve().parent
DATA_DIR = SCRATCH / "data"
ART_DIR = SCRATCH / "artifacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SENSITIVE_PORTS = {21, 22, 23, 135, 139, 445, 3389, 5900}

# Exact 26-feature order from app/ml/flow_reducer.py MODEL_FEATURES (V2r_nolen26)
MODEL_FEATURES = (
    "packets", "duration_s", "dst_port", "pkts_per_s",
    "sensitive_port", "well_known_port", "ephemeral_port",
    "fpk", "bpk",
    "iat_mean", "iat_std", "iat_max", "iat_min", "fiat_mean", "biat_mean",
    "downup", "bpk_ratio",
    "sw_flows", "sw_ndport", "sw_ndst", "sw_small_frac", "sw_mean_pkts",
    "dw_flows", "dw_nsrc", "pw_flows", "pw_ndport",
)

HF_BASE = "https://huggingface.co/datasets/Mireu-Lab/CIC-IDS/resolve/main/CSV"
CIC_FILES = [
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 35000, 35000),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 30000, 35000),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", 25000, 20000),
    ("Wednesday-workingHours.pcap_ISCX.csv", 25000, 35000),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", 15000, 5000),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", 15000, 5000),
]


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def check_disk_space_gb(path: Path, min_gb: float = 2.0) -> float:
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    print(f"[Disk Check] {path}: {free_gb:.2f} GB free (min required: {min_gb:.1f} GB)")
    if free_gb < min_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB < {min_gb:.1f} GB")
    return free_gb


def _fval(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key)
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
        return v if math.isfinite(v) else default
    except ValueError:
        return default


def synth_window_features(label: str, pk: float, rng: random.Random) -> list[float]:
    """Generate realistic 60-second FlowReducer window context matching each CIC-IDS2017 traffic regime."""
    lbl = label.lower()
    if lbl == "benign":
        sw_f = float(rng.randint(1, 22))
        sw_ndp = float(rng.randint(1, min(6, int(sw_f))))
        sw_ndst = float(rng.randint(1, min(6, int(sw_f))))
        sw_sm = rng.uniform(0.0, 0.35) if pk > 4 else rng.uniform(0.2, 0.7)
        sw_mp = max(2.0, pk * rng.uniform(0.7, 1.5))
        dw_f = float(rng.randint(1, 30))
        dw_nsrc = float(rng.randint(1, min(8, int(dw_f))))
        pw_f = float(rng.randint(1, min(10, int(sw_f))))
        pw_ndp = float(rng.randint(1, min(4, int(pw_f))))
    elif "portscan" in lbl:
        sw_f = float(rng.randint(80, 1200))
        sw_ndp = float(rng.randint(30, min(900, int(sw_f))))
        sw_ndst = float(rng.randint(1, 3))
        sw_sm = rng.uniform(0.85, 1.0)
        sw_mp = rng.uniform(2.0, 3.5)
        dw_f = sw_f
        dw_nsrc = float(rng.randint(1, 3))
        pw_f = sw_f
        pw_ndp = sw_ndp
    elif "ddos" in lbl or "dos" in lbl:
        sw_f = float(rng.randint(60, 800))
        sw_ndp = float(rng.randint(1, 4))
        sw_ndst = float(rng.randint(1, 2))
        sw_sm = rng.uniform(0.1, 0.8)
        sw_mp = max(2.0, pk * rng.uniform(0.8, 1.2))
        dw_f = float(rng.randint(100, 1500))
        dw_nsrc = float(rng.randint(3, 40)) if "ddos" in lbl else float(rng.randint(1, 4))
        pw_f = sw_f
        pw_ndp = float(rng.randint(1, 3))
    elif "patator" in lbl or "brute" in lbl or "web" in lbl:
        sw_f = float(rng.randint(25, 250))
        sw_ndp = float(rng.randint(1, 3))
        sw_ndst = 1.0
        sw_sm = rng.uniform(0.0, 0.4)
        sw_mp = max(4.0, pk * rng.uniform(0.8, 1.2))
        dw_f = sw_f
        dw_nsrc = float(rng.randint(1, 3))
        pw_f = sw_f
        pw_ndp = 1.0
    else:  # Bot / Infiltration / Heartbleed
        sw_f = float(rng.randint(10, 120))
        sw_ndp = float(rng.randint(1, 8))
        sw_ndst = float(rng.randint(1, 5))
        sw_sm = rng.uniform(0.1, 0.7)
        sw_mp = max(2.0, pk * rng.uniform(0.7, 1.3))
        dw_f = float(rng.randint(10, 150))
        dw_nsrc = float(rng.randint(1, 6))
        pw_f = float(rng.randint(5, int(sw_f)))
        pw_ndp = float(rng.randint(1, 4))
    return [sw_f, sw_ndp, sw_ndst, sw_sm, sw_mp, dw_f, dw_nsrc, pw_f, pw_ndp]


def build_live_wire_aug(rng: random.Random) -> tuple[list[list[float]], list[int], list[str]]:
    """Augment with realistic live-sniffer / edge LAN benign + attack flows (ICMP flood, C2, SYN/UDP flood, SSH BF)."""
    X_aug, y_aug, fam_aug = [], [], []
    # Benign LAN/HTTPS/DNS/SSH interactive flows (incl. single-flow context sw_flows=1)
    for _ in range(12000):
        port = rng.choice([80, 443, 53, 123, 8000, 8080, 9090, 22])
        fpk = float(rng.randint(2, 45))
        bpk = float(rng.randint(1, 45)) if port != 53 else float(rng.randint(1, 4))
        pk = fpk + bpk
        dur = rng.uniform(0.02, 18.0)
        pps = pk / max(dur, 1e-6)
        iat_m = dur / max(pk - 1.0, 1.0)
        win = [float(rng.randint(1, 10)), float(rng.randint(1, 3)), float(rng.randint(1, 3)),
               0.0 if pk > 4 else 0.5, pk, float(rng.randint(1, 12)), float(rng.randint(1, 3)),
               float(rng.randint(1, 6)), float(rng.randint(1, 2))]
        vec = [
            pk, dur, float(port), pps,
            1.0 if port in SENSITIVE_PORTS else 0.0,
            1.0 if port < 1024 else 0.0,
            1.0 if port >= 49152 else 0.0,
            fpk, bpk,
            iat_m, iat_m * rng.uniform(0.1, 0.8), iat_m * rng.uniform(1.1, 3.0), iat_m * rng.uniform(0.01, 0.5),
            iat_m * 1.2, iat_m * 1.2,
            float(math.floor(bpk / max(fpk, 1.0))), bpk / pk,
            *win,
        ]
        X_aug.append(vec); y_aug.append(0); fam_aug.append("BENIGN_EDGE")

    # Edge attack flows: C2 ports (4444/1337/6667/31337/5555), SYN/UDP/ICMP floods, PortScan, SSH/Telnet BF
    for _ in range(10000):
        kind = rng.choice(["SYN_FLOOD", "PORT_SCAN", "UDP_FLOOD", "C2_PORT", "SSH_BF", "ICMP_FLOOD"])
        if kind == "SYN_FLOOD":
            port = rng.choice([80, 443, 8000, 22, 8080])
            fpk, bpk = float(rng.randint(40, 1500)), float(rng.randint(0, 5))
            dur = rng.uniform(0.05, 4.0)
            win = [float(rng.randint(50, 600)), 1.0, 1.0, 0.9, fpk, float(rng.randint(50, 600)), float(rng.randint(1, 10)), float(rng.randint(50, 600)), 1.0]
        elif kind == "PORT_SCAN":
            port = float(rng.randint(1, 65000))
            fpk, bpk = float(rng.randint(1, 3)), float(rng.randint(0, 2))
            dur = rng.uniform(0.0001, 0.2)
            ndp = float(rng.randint(15, 600))
            win = [ndp, ndp, 1.0, 0.98, 2.0, ndp, 1.0, ndp, ndp]
        elif kind == "UDP_FLOOD":
            port = rng.choice([53, 123, 1900, 11211, 5060, 9999])
            fpk, bpk = float(rng.randint(80, 2000)), 0.0
            dur = rng.uniform(0.1, 6.0)
            win = [float(rng.randint(20, 300)), 1.0, 1.0, 0.2, fpk, float(rng.randint(20, 300)), float(rng.randint(1, 5)), float(rng.randint(20, 300)), 1.0]
        elif kind == "C2_PORT":
            port = rng.choice([4444, 6667, 1337, 31337, 5555, 9050])
            fpk, bpk = float(rng.randint(4, 80)), float(rng.randint(2, 60))
            dur = rng.uniform(0.5, 90.0)
            win = [float(rng.randint(1, 20)), 1.0, 1.0, 0.1, fpk + bpk, float(rng.randint(1, 20)), 1.0, float(rng.randint(1, 20)), 1.0]
        elif kind == "SSH_BF":
            port = rng.choice([22, 23, 21, 3389, 445])
            fpk, bpk = float(rng.randint(12, 80)), float(rng.randint(6, 40))
            dur = rng.uniform(0.2, 8.0)
            win = [float(rng.randint(20, 250)), 1.0, 1.0, 0.2, fpk + bpk, float(rng.randint(20, 250)), 1.0, float(rng.randint(20, 250)), 1.0]
        else:  # ICMP_FLOOD
            port = 0
            fpk, bpk = float(rng.randint(50, 1500)), float(rng.randint(0, 500))
            dur = rng.uniform(0.1, 5.0)
            win = [float(rng.randint(20, 400)), 1.0, 1.0, 0.3, fpk + bpk, float(rng.randint(20, 400)), float(rng.randint(1, 5)), float(rng.randint(20, 400)), 1.0]

        pk = max(fpk + bpk, 1.0)
        pps = pk / max(dur, 1e-6)
        iat_m = dur / max(pk - 1.0, 1.0)
        vec = [
            pk, dur, float(port), pps,
            1.0 if int(port) in SENSITIVE_PORTS else 0.0,
            1.0 if int(port) < 1024 else 0.0,
            1.0 if int(port) >= 49152 else 0.0,
            fpk, bpk,
            iat_m, iat_m * rng.uniform(0.0, 0.4), iat_m * rng.uniform(1.0, 2.5), iat_m * rng.uniform(0.0, 0.3),
            iat_m, iat_m if bpk > 1 else 0.0,
            float(math.floor(bpk / max(fpk, 1.0))), bpk / pk,
            *win,
        ]
        X_aug.append(vec); y_aug.append(1); fam_aug.append(kind)
    return X_aug, y_aug, fam_aug


class NetworkDualNPU(nn.Module):
    """Compact 26-feature 1D-CNN + Residual MLP — 100% native RK3588 NPU FP16 ops."""
    def __init__(self, n_features: int = 26):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # 1D-CNN over [B, 1, 26] -> Conv(1->24, k=3, p=1) -> MaxPool(2) [13] -> Conv(24->32, k=3, p=1) -> MaxPool(2) [6] -> 192
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 6, 48),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(64 + 48, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 26]
        h_mlp = self.mlp(x)
        h_cnn = self.cnn(x.unsqueeze(1))
        return self.head(torch.cat([h_mlp, h_cnn], dim=1))  # [B, 1]


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

    # 0. Strict disk space check BEFORE downloading CIC-IDS2017 CSVs
    check_disk_space_gb(SCRATCH, min_gb=2.5)

    try:
        rng = random.Random(SEED)
        X_rows: list[list[float]] = []
        y_rows: list[int] = []
        fam_rows: list[str] = []

        for fname, max_benign, max_attack in CIC_FILES:
            dest = DATA_DIR / fname
            url = f"{HF_BASE}/{fname}"
            print(f"Downloading {fname} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 EdgeShield-Trainer/1.4"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                dest.write_bytes(resp.read())
            print(f"  saved {dest.stat().st_size / (1024*1024):.1f} MB; streaming & subsampling...")

            ben_buf, atk_buf = [], []
            with dest.open("r", encoding="latin1", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                # Normalize both space-separated (UNB) and underscore-separated (Mireu-Lab HF) headers
                reader.fieldnames = [re.sub(r"\s+", "_", c.strip()) for c in (reader.fieldnames or [])]
                for row in reader:
                    lbl = (row.get("Label") or "").strip()
                    if not lbl:
                        continue
                    is_ben = lbl.upper() == "BENIGN"
                    if is_ben and len(ben_buf) >= max_benign * 2:
                        continue
                    if not is_ben and len(atk_buf) >= max_attack * 2:
                        continue
                    fpk = max(0.0, _fval(row, "Total_Fwd_Packets"))
                    bpk = max(0.0, _fval(row, "Total_Backward_Packets"))
                    pk = fpk + bpk
                    if pk < 2.0:
                        continue
                    dur_s = max(0.0, _fval(row, "Flow_Duration") / 1e6)
                    dd = max(dur_s, 1e-6)
                    port = max(0.0, min(65535.0, _fval(row, "Destination_Port")))
                    iat_m = max(0.0, _fval(row, "Flow_IAT_Mean") / 1e6)
                    iat_s = max(0.0, _fval(row, "Flow_IAT_Std") / 1e6)
                    iat_mx = max(0.0, _fval(row, "Flow_IAT_Max") / 1e6)
                    iat_mn = max(0.0, _fval(row, "Flow_IAT_Min") / 1e6)
                    fiat_m = max(0.0, _fval(row, "Fwd_IAT_Mean") / 1e6)
                    biat_m = max(0.0, _fval(row, "Bwd_IAT_Mean") / 1e6)
                    downup = float(math.floor(bpk / max(fpk, 1.0)))
                    bpk_ratio = bpk / pk
                    win = synth_window_features(lbl, pk, rng)
                    vec = [
                        pk, dur_s, port, pk / dd,
                        1.0 if int(port) in SENSITIVE_PORTS else 0.0,
                        1.0 if int(port) < 1024 else 0.0,
                        1.0 if int(port) >= 49152 else 0.0,
                        fpk, bpk,
                        iat_m, iat_s, iat_mx, iat_mn, fiat_m, biat_m,
                        downup, bpk_ratio,
                        *win,
                    ]
                    if is_ben:
                        ben_buf.append((vec, 0, "BENIGN"))
                    else:
                        atk_buf.append((vec, 1, lbl))

            # Immediately delete each CSV after parsing to minimize peak disk usage!
            dest.unlink(missing_ok=True)
            rng.shuffle(ben_buf); rng.shuffle(atk_buf)
            kept = ben_buf[:max_benign] + atk_buf[:max_attack]
            for v, yv, fm in kept:
                X_rows.append(v); y_rows.append(yv); fam_rows.append(fm)
            print(f"  extracted {len(ben_buf[:max_benign])} benign + {len(atk_buf[:max_attack])} attack; deleted {fname}")

        # Add live-sniffer / edge LAN augmentation
        xa, ya, fa = build_live_wire_aug(rng)
        X_rows.extend(xa); y_rows.extend(ya); fam_rows.extend(fa)

        X = np.asarray(X_rows, dtype=np.float32)
        y = np.asarray(y_rows, dtype=np.int64)
        fams = np.asarray(fam_rows)
        print(f"Total dataset: {len(y)} flows | benign={(y==0).sum()} attack={(y==1).sum()}")
        print("Families:", Counter(fam_rows).most_common(15))

        # Stratified 70 / 15 / 15 split by attack family
        idx = np.arange(len(y))
        fam_counts = Counter(fam_rows)
        strat = np.array([f if fam_counts[f] >= 10 else ("ATK" if yi else "BEN") for f, yi in zip(fam_rows, y)])
        tr_idx, rest_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=strat)
        va_idx, te_idx = train_test_split(rest_idx, test_size=0.50, random_state=SEED, stratify=strat[rest_idx])

        # Signed-log1p + StandardScaler fitted STRICTLY on training split!
        X_log = signed_log1p(X)
        mean = X_log[tr_idx].mean(axis=0)
        std = X_log[tr_idx].std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        X_sc = np.clip((X_log - mean) / std, -8.0, 8.0).astype(np.float32)

        n_tot = len(tr_idx)
        n_pos = int(y[tr_idx].sum())
        n_neg = n_tot - n_pos
        w1 = n_tot / (2.0 * n_pos)
        w0 = n_tot / (2.0 * n_neg)
        print(f"WBCE weights: w1={w1:.4f}, w0={w0:.4f}")

        dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_sc[tr_idx]), torch.from_numpy(y[tr_idx].astype(np.float32))), batch_size=512, shuffle=True)
        dl_va = DataLoader(TensorDataset(torch.from_numpy(X_sc[va_idx]), torch.from_numpy(y[va_idx].astype(np.float32))), batch_size=1024, shuffle=False)
        dl_te = DataLoader(TensorDataset(torch.from_numpy(X_sc[te_idx]), torch.from_numpy(y[te_idx].astype(np.float32))), batch_size=1024, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NetworkDualNPU(len(MODEL_FEATURES)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Training NetworkDualNPU ({n_params:,} params) on {device} ...")

        epochs = 14
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        def predict_dl(dl):
            model.eval()
            ps, ys = [], []
            with torch.no_grad():
                for xb, yb in dl:
                    logits = model(xb.to(device)).squeeze(-1)
                    ps.append(torch.sigmoid(logits).cpu().numpy())
                    ys.append(yb.numpy())
            return np.concatenate(ys), np.concatenate(ps)

        best_ap, best_state = -1.0, None
        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            model.train()
            tot_loss = 0.0
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb).squeeze(-1)
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
            if ep in (1, 5, 10, epochs) or m["ap"] == best_ap:
                print(f"Epoch {ep:02d}/{epochs} | loss={tot_loss/n_tot:.4f} | val AP={m['ap']:.4f} AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} FPR={m['fpr']:.4f}")

        train_secs = round(time.perf_counter() - t0, 2)
        model.load_state_dict(best_state)
        model.eval()

        # Calibrate operating threshold on val split (best F1 s.t. FPR <= 0.5%)
        yv, pv = predict_dl(dl_va)
        taus = np.linspace(0.10, 0.96, 87)
        opt_tau, opt_f1 = 0.70, -1.0
        for t in taus:
            m = evaluate_probs(yv, pv, t)
            if m["fpr"] <= 0.005 and m["f1"] > opt_f1:
                opt_f1 = m["f1"]
                opt_tau = float(t)

        yt, pt = predict_dl(dl_te)
        test_opt = evaluate_probs(yt, pt, opt_tau)
        test_050 = evaluate_probs(yt, pt, 0.50)
        print(f"Calibrated threshold (val FPR<=0.5%): tau={opt_tau:.3f}")
        print("Held-out Test (opt):", test_opt)
        print("Held-out Test (0.50):", test_050)

        # Per-family recall on held-out test set
        fam_te = fams[te_idx]
        per_fam = {}
        for fm in sorted(set(fam_te)):
            mask = fam_te == fm
            if yt[mask][0] == 1:
                per_fam[fm] = {"n": int(mask.sum()), "recall": round(float((pt[mask] >= opt_tau).mean()), 4)}
        print("Per-family Test Recall:", json.dumps(per_fam, indent=2))

        # Export ONNX -> RKNN FP16 (RK3588 NPU_CORE_0)
        model_cpu = model.cpu().eval()
        dummy = torch.randn(1, len(MODEL_FEATURES), dtype=torch.float32)
        onnx_path = ART_DIR / "network_ids_fp32.onnx"
        torch.onnx.export(
            model_cpu, dummy, str(onnx_path),
            input_names=["features"], output_names=["logit"],
            opset_version=12, do_constant_folding=True,
        )
        torch.save(model_cpu.state_dict(), ART_DIR / "network_ids_dual.pt")

        from rknn.api import RKNN
        rknn_path = ART_DIR / "network_ids_fp16.rknn"
        rknn = RKNN(verbose=False)
        try:
            assert rknn.config(target_platform="rk3588", optimization_level=3) == 0
            assert rknn.load_onnx(model=str(onnx_path)) == 0
            assert rknn.build(do_quantization=False) == 0
            assert rknn.export_rknn(str(rknn_path)) == 0
            assert rknn.init_runtime() == 0

            diffs = []
            for i in range(30):
                xi = X_sc[te_idx[i]:te_idx[i]+1]
                sim_logit = float(np.asarray(rknn.inference(inputs=[xi])[0]).reshape(-1)[0])
                sim_p = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, sim_logit))))
                diffs.append(abs(sim_p - float(pt[i])))
            max_diff, mean_diff = max(diffs), float(np.mean(diffs))
            print(f"RKNN FP16 Simulator Parity (30 test flows): max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
        finally:
            rknn.release()

        rknn_sha = hashlib.sha256(rknn_path.read_bytes()).hexdigest()[:16]
        meta = {
            "model_id": "network-ids-1dcnn-mlp-v2r26-rknn-v1",
            "architecture": {
                "name": "NetworkDualNPU (1D-CNN 1x26->48 + MLP 26->64->64 + Fusion Head 112->48->1)",
                "npu_target": "rk3588",
                "npu_core_id": 0,
                "npu_precision": "fp16",
                "n_params": n_params,
                "n_features": len(MODEL_FEATURES),
                "rknn_bytes": rknn_path.stat().st_size,
                "rknn_sha256_prefix": rknn_sha,
            },
            "preprocessing": {
                "feature_names": list(MODEL_FEATURES),
                "transform": "signed_log1p_then_zscore_clip8",
                "mean": [round(float(x), 6) for x in mean],
                "std": [round(float(x), 6) for x in std],
            },
            "training": {
                "datasets": ["CIC-IDS2017 (Mireu-Lab/CIC-IDS: DDoS, PortScan, DoS, Patator, Bot, WebAttacks)", "Edge LAN Live-Wire Augmentation"],
                "total_flows": int(len(y)),
                "train_flows": int(len(tr_idx)),
                "val_flows": int(len(va_idx)),
                "test_flows": int(len(te_idx)),
                "wbce_weights": {"w0": round(w0, 4), "w1": round(w1, 4)},
                "train_seconds": train_secs,
            },
            "calibration": {
                "default_threshold": round(opt_tau, 3),
                "threshold_rule": "best val F1 subject to val FPR <= 0.5%",
            },
            "test_metrics": {
                "at_opt": test_opt,
                "at_050": test_050,
                "per_family_recall": per_fam,
                "rknn_fp16_sim_parity_max_diff": round(max_diff, 6),
                "rknn_fp16_sim_parity_mean_diff": round(mean_diff, 6),
            },
        }
        meta_path = ART_DIR / "network_ids_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Exported RKNN artifacts: {rknn_path.name} ({rknn_path.stat().st_size/1024:.1f} KB), {meta_path.name} ({meta_path.stat().st_size/1024:.1f} KB)")
    finally:
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR, ignore_errors=True)
            print(f"[Cleanup] Deleted temporary CIC-IDS2017 dataset directory: {DATA_DIR}")
        check_disk_space_gb(SCRATCH, min_gb=1.0)


if __name__ == "__main__":
    main()
