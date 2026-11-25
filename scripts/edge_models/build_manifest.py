#!/usr/bin/env python3
"""Assemble deployment manifest from training manifests and exported RKNN artifacts."""
import argparse, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument("--training-dir",default="models/torch")
p.add_argument("--rknn-dir",default="models/rknn"); p.add_argument("--out",default="models/rknn/manifest.json")
a=p.parse_args(); td=Path(a.training_dir); rd=Path(a.rknn_dir)
cores={"network":0,"url":1,"sms":2}; files={"network":"network_1dcnn_int8.rknn","url":"url_charcnn_int8.rknn","sms":"sms_charcnn_int8.rknn"}
models={}
for name in ("network","url","sms"):
    m=json.loads((td/f"{name}_training_manifest.json").read_text())
    artifact=rd/files[name]
    if not artifact.is_file(): raise SystemExit(f"Missing RKNN artifact: {artifact}")
    spec={"artifact":str(artifact),"core_id":cores[name],"input_shape":m["input_shape"],"threshold":m["threshold"],"max_len":m.get("max_len"),"feature_names":m.get("feature_names"),"mean":m.get("mean"),"std":m.get("std"),"model_version":f"{name}-compact-int8-v1"}
    models[name]={k:v for k,v in spec.items() if v is not None}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"schema_version":1,"soc":"RK3588J","models":models},indent=2))
print(out)
