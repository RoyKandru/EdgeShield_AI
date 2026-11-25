#!/usr/bin/env python3
"""Convert a fixed-shape ONNX model to RK3588 INT8 with RKNN Toolkit2.
Run on a supported development host with rknn-toolkit2 installed, not in the board venv.
Calibration file is a text file containing one representative .npy tensor path per line.
"""
import argparse
from pathlib import Path
from rknn.api import RKNN

p=argparse.ArgumentParser(); p.add_argument("onnx"); p.add_argument("--dataset",required=True)
p.add_argument("--output",required=True); p.add_argument("--name",default="edgeshield")
a=p.parse_args()
r=RKNN(verbose=True)
try:
    rc=r.config(target_platform="rk3588", quantized_dtype="asymmetric_quantized-8")
    if rc: raise SystemExit(f"config failed: {rc}")
    rc=r.load_onnx(model=str(a.onnx), inputs=["input"])
    if rc: raise SystemExit(f"load_onnx failed: {rc}")
    rc=r.build(do_quantization=True, dataset=a.dataset)
    if rc: raise SystemExit(f"build failed: {rc}")
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    rc=r.export_rknn(a.output)
    if rc: raise SystemExit(f"export failed: {rc}")
    print(f"Exported {a.output}; verify on board with RKNNLite and held-out parity suite.")
finally: r.release()
