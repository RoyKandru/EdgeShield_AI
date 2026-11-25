#!/usr/bin/env python3
"""Train compact binary detector models from labelled CSVs and export fixed-shape ONNX.

Input formats:
  network.csv: all columns in app.ml.flow_reducer.MODEL_FEATURES plus label (0 benign, 1 attack)
  url.csv: url,label
  sms.csv: message,label

Splits are stratified and deterministic. For rigorous work, use domain-grouped URL splits and
capture-day/grouped network splits as in Item 2; this simple script is a bring-up baseline only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.edge_models.models import Network1DCNN, CharacterCNN
from app.edge_models.preprocess import URL_ALPHABET, SMS_ALPHABET, one_hot_chars
from app.ml.flow_reducer import MODEL_FEATURES

def train(model, X, y, epochs=18, batch=128):
    xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xt)).numpy()
    return probs

def export(model, sample, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, sample, str(out), input_names=["input"], output_names=["logit"],
                      opset_version=12, do_constant_folding=True, dynamo=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("pipeline", choices=["network","url","sms"])
    ap.add_argument("csv"); ap.add_argument("--out", default="models/torch"); ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--max-len", type=int, default=512); args=ap.parse_args()
    df=pd.read_csv(args.csv); y=df["label"].astype(int).to_numpy()
    if set(np.unique(y)) - {0,1}: raise SystemExit("label must contain only 0/1")
    if args.pipeline == "network":
        missing=[f for f in MODEL_FEATURES if f not in df.columns]
        if missing: raise SystemExit(f"Missing network features: {missing}")
        X=df[list(MODEL_FEATURES)].astype("float32").to_numpy()
        # Per-feature robust clipping then standardisation; stats are recorded in manifest.
        med=np.nanmedian(X,axis=0); X=np.where(np.isfinite(X),X,med)
        X=np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mean=X.mean(0); std=X.std(0); std[std<1e-6]=1.0
        X=np.clip((X-mean)/std,-8,8)[:,None,:]
        model=Network1DCNN(len(MODEL_FEATURES))
        artifact={"feature_names":list(MODEL_FEATURES),"mean":mean.tolist(),"std":std.tolist(),"max_len":None}
    else:
        col="url" if args.pipeline=="url" else "message"
        alphabet=URL_ALPHABET if args.pipeline=="url" else SMS_ALPHABET
        X=np.stack([one_hot_chars(str(s),alphabet,args.max_len) for s in df[col].fillna("")]).astype("float32")
        model=CharacterCNN(len(alphabet)+2)
        artifact={"alphabet":alphabet,"max_len":args.max_len}
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    train(model,Xtr,ytr,args.epochs)
    with torch.no_grad():
        probs=torch.sigmoid(model(torch.tensor(Xte,dtype=torch.float32))).numpy()
    threshold=.5
    # Choose threshold on held-out validation data in production; this baseline threshold is not calibrated.
    print(classification_report(yte,(probs>=threshold).astype(int),digits=4))
    try: print("ROC-AUC",roc_auc_score(yte,probs))
    except ValueError: pass
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    onnx=out/f"{args.pipeline}_compact_fp32.onnx"
    export(model,torch.tensor(X[:1],dtype=torch.float32),onnx)
    torch.save(model.state_dict(),out/f"{args.pipeline}_compact.pt")
    artifact.update({"pipeline":args.pipeline,"onnx":str(onnx),"threshold":threshold,
                     "input_shape":list(X.shape[1:]),"label_mapping":{"0":"benign","1":"malicious"},
                     "threshold_calibrated":False})
    (out/f"{args.pipeline}_training_manifest.json").write_text(json.dumps(artifact,indent=2))
    print(f"Exported {onnx}; do not deploy until grouped validation, calibration and RKNN parity checks pass.")
if __name__=="__main__": main()
