#!/usr/bin/env python3
"""Concurrent three-context benchmark harness; requires built RKNN artifacts and rknnlite."""
import argparse, concurrent.futures, json, statistics, time
from pathlib import Path
import numpy as np
from app.edge_models.service import EdgeInferenceService
from app.edge_models.preprocess import URL_ALPHABET, SMS_ALPHABET, one_hot_chars
from app.ml.flow_reducer import MODEL_FEATURES

p=argparse.ArgumentParser(); p.add_argument("--iterations",type=int,default=200); a=p.parse_args()
s=EdgeInferenceService()
inputs={"network":np.zeros((1,len(MODEL_FEATURES)),np.float32),
        "url":one_hot_chars("https://example.com/login",URL_ALPHABET,512),
        "sms":one_hot_chars("Your account is blocked. Verify now",SMS_ALPHABET,512)}
results={k:[] for k in inputs}
def run(name):
    vals=[]
    for _ in range(a.iterations):
        t=time.perf_counter(); s.pool.infer(name,inputs[name],.5); vals.append((time.perf_counter()-t)*1000)
    return name, vals
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for name,vals in ex.map(run,inputs): results[name]=vals
    print(json.dumps({k:{"n":len(v),"median_ms":statistics.median(v),"p95_ms":float(np.percentile(v,95)),"throughput_per_s":1000/statistics.mean(v)} for k,v in results.items()},indent=2))
    print("Note: this measures concurrent host requests; inspect RKNN/driver counters to prove physical overlap.")
finally: s.close()
