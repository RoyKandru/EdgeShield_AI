"""
Feature extraction shared by training/evaluation and the runtime detectors.

Anything the model sees at training time MUST come from these functions, so the
runtime cannot drift from what was evaluated.
"""
from __future__ import annotations

import html
import math
import re
from collections import Counter

# ---------------------------------------------------------------- URL ---------

_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "ow.ly", "rb.gy"}
SUSPICIOUS_TERMS = (
    "verify", "urgent", "login", "signin", "update", "secure", "account", "bank", "kyc",
    "wallet", "password", "bonus", "prize", "confirm", "suspend", "billing", "paypal",
)


def normalise_url(raw: str) -> str:
    """Lower-case, unescape, drop scheme and leading 'www.'.

    The public phishing corpora are inconsistent about writing 'http://' / 'www.'
    per class, so a model must not be allowed to learn them as signal.
    """
    u = html.unescape(raw.strip()).lower()
    u = _SCHEME.sub("", u)
    if u.startswith("www."):
        u = u[4:]
    return u


def url_host(norm: str) -> str:
    """Host of a normalised URL (no userinfo, no port)."""
    authority = re.split(r"[/?#]", norm, maxsplit=1)[0]
    authority = authority.rsplit("@", 1)[-1]
    return authority.split(":", 1)[0]


def registered_domain(norm: str) -> str:
    """Approximate registrable domain (last two labels; used only for split grouping)."""
    host = url_host(norm)
    if _IPV4.match(host):
        return host
    return ".".join(host.split(".")[-2:])


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


URL_FEATURE_NAMES = (
    "len", "host_len", "path_len", "query_len", "dots", "host_dots", "hyphens_host",
    "digits_ratio", "host_digits_ratio", "slashes", "at", "pct", "eq", "amp", "ip_host",
    "punycode", "shortener", "susp_terms", "host_entropy", "url_entropy", "has_path",
    "has_query", "tld_len", "longest_label", "numeric_labels",
)


def url_features(raw: str) -> list[float]:
    n = normalise_url(raw)
    host = url_host(n)
    rest = n[len(host):] if n.startswith(host) else ""
    path, _, query = rest.partition("?")
    labels = host.split(".") if host else [""]
    digits = sum(c.isdigit() for c in n)
    host_digits = sum(c.isdigit() for c in host)
    return [
        len(n), len(host), len(path), len(query), n.count("."), host.count("."),
        host.count("-"), digits / max(1, len(n)), host_digits / max(1, len(host)),
        n.count("/"), n.count("@"), n.count("%"), n.count("="), n.count("&"),
        1.0 if _IPV4.match(host) else 0.0, 1.0 if "xn--" in host else 0.0,
        1.0 if host in SHORTENERS else 0.0, sum(t in n for t in SUSPICIOUS_TERMS),
        _entropy(host), _entropy(n), 1.0 if len(path) > 1 else 0.0,
        1.0 if query else 0.0, len(labels[-1]), max(len(x) for x in labels),
        sum(x.isdigit() for x in labels),
    ]


def url_feature_matrix(urls):
    """sklearn-friendly (picklable, module-level) wrapper: list[str] -> float32 matrix."""
    import numpy as np
    return np.asarray([url_features(u) for u in urls], dtype=np.float32)


# --------------------------------------------------------------- FLOWS --------

SENSITIVE_PORTS = {22, 23, 3389, 445, 5900, 21, 139, 135}

# Only fields the runtime flow schema actually provides (see detectors/network.py).
FLOW_FEATURE_NAMES = (
    "packets", "bytes", "duration_s", "dst_port", "syn_count", "bytes_per_pkt",
    "pkts_per_s", "bytes_per_s", "syn_ratio", "sensitive_port", "well_known_port",
    "ephemeral_port",
)


def flow_features(packets: float, bytes_: float, duration_s: float, dst_port: float,
                  syn_count: float) -> list[float]:
    d = max(duration_s, 1e-6)
    p = max(packets, 1.0)
    port = int(dst_port)
    return [
        packets, bytes_, duration_s, dst_port, syn_count, bytes_ / p, packets / d,
        bytes_ / d, syn_count / p, 1.0 if port in SENSITIVE_PORTS else 0.0,
        1.0 if port < 1024 else 0.0, 1.0 if port >= 49152 else 0.0,
    ]


def flow_features_from_dict(flow: dict) -> list[float]:
    return flow_features(
        float(flow.get("packets", 0)), float(flow.get("bytes", 0)),
        float(flow.get("duration_seconds", 0.0)), float(flow.get("dst_port", 0)),
        float(flow.get("syn_count", 0)),
    )
