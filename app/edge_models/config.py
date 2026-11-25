from __future__ import annotations
import json, os
from pathlib import Path

DEFAULT_SPECS = {
  "network": {"artifact":"./models/rknn/network_1dcnn_int8.rknn", "input_shape":[1, 26], "core_id":0, "threshold":0.5},
  "url": {"artifact":"./models/rknn/url_charcnn_int8.rknn", "input_shape":[len("abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%")+2, 512], "core_id":1, "threshold":0.5},
  "sms": {"artifact":"./models/rknn/sms_charcnn_int8.rknn", "input_shape":[len("abcdefghijklmnopqrstuvwxyz0123456789 .,!?@#$%&*()-_/:;+'\"")+2, 512], "core_id":2, "threshold":0.5},
}

def load_specs(path: str = "./models/rknn/manifest.json"):
    p = Path(path)
    if p.is_file():
        raw = json.loads(p.read_text())
        return raw.get("models", raw)
    return DEFAULT_SPECS
