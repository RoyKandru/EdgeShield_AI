"""Deterministic preprocessing shared by training and device inference."""
from __future__ import annotations
import numpy as np

# Fixed ASCII-oriented vocabulary; unknown characters map to 1, padding is 0.
URL_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
SMS_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?@#$%&*()-_/:;+'\""

def encode_chars(text: str, alphabet: str, max_len: int) -> np.ndarray:
    lookup = {c: i + 2 for i, c in enumerate(alphabet)}
    ids = np.zeros(max_len, dtype=np.int64)
    s = (text or "").lower()[:max_len]
    for i, ch in enumerate(s):
        ids[i] = lookup.get(ch, 1)
    return ids

def one_hot_chars(text: str, alphabet: str, max_len: int) -> np.ndarray:
    """Return NCHW-ready [vocab, length] float32; channels include PAD and UNK."""
    ids = encode_chars(text, alphabet, max_len)
    x = np.zeros((len(alphabet) + 2, max_len), dtype=np.float32)
    x[ids, np.arange(max_len)] = 1.0
    return x

def network_vector(features: dict, names: list[str]) -> np.ndarray:
    x = np.asarray([float(features.get(k, 0.0)) for k in names], dtype=np.float32)
    # Fixed transform is deliberately simple and reproducible. Training script stores means/stds.
    return x
