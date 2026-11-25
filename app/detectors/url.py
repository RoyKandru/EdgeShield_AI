import ipaddress
import re
from urllib.parse import urlparse
from app.events import create_event
from app.ml.url_runtime import MODEL_VERSION, score_url
from app.schemas import EvidenceItem, ThreatEvent

SUSPICIOUS_TERMS = (
    "verify", "urgent", "login", "update", "secure", "account",
    "bank", "kyc", "wallet", "password", "bonus", "prize",
    "signin", "otp", "claim", "refund", "webmail", "oauth", "suspended",
)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly",
    "rb.gy", "shorturl.at", "tiny.cc", "ow.ly", "rebrand.ly", "t.ly", "v.gd",
}
KNOWN_BAD_EXAMPLE_DOMAINS = {"example-phishing.invalid", "malware.test.invalid"}

HIGH_RISK_FILE_TLDS = {"zip", "mov"}
ABUSE_TLDS = {
    "top", "xyz", "click", "tk", "ml", "ga", "cf", "gq",
    "cfd", "sbs", "buzz", "icu", "pw", "work", "loan", "rest", "cyou", "cam",
}
TUNNEL_SUFFIXES = (
    ".ngrok-free.app", ".ngrok.io", ".ngrok.app", ".trycloudflare.com",
    ".serveo.net", ".loca.lt", ".lhr.life",
)
PAAS_SUFFIXES = (
    ".pages.dev", ".workers.dev", ".web.app", ".firebaseapp.com",
    ".vercel.app", ".netlify.app", ".github.io", ".azurewebsites.net",
    ".sites.google.com", ".blogspot.com", ".duckdns.org",
)
DROPPER_EXT_RE = re.compile(r"\.(?:apk|exe|scr|msi|bat|ps1|vbs|hta|jar)(?:$|[?#])", re.I)
CONFUSABLE_SLASH_RE = re.compile(r"[\u2044\u2215\uff0f\uff0e\u3002]")
REDIRECT_PARAM_RE = re.compile(r"(?:redirect|redir|next|url|target|goto|dest)=(?:https?|%68%74%74%70|%2f%2f)", re.I)
OCTAL_HEX_IP_RE = re.compile(r"^(?:0x[0-9a-f]+|0[0-7]{2,}(?:\.0?[0-7]+){3}|\d{8,10})$", re.I)

BRAND_OFFICIAL_DOMAINS = {
    "sbi": {"sbi.co.in", "onlinesbi.sbi", "sbi.co", "sbicard.com"},
    "yono": {"sbi.co.in", "onlinesbi.sbi"},
    "hdfc": {"hdfcbank.com", "hdfc.com", "hdfcsec.com"},
    "icici": {"icicibank.com", "icicidirect.com"},
    "axis": {"axisbank.com"},
    "paytm": {"paytm.com", "paytmbank.com"},
    "phonepe": {"phonepe.com"},
    "uidai": {"uidai.gov.in"},
    "aadhaar": {"uidai.gov.in"},
    "epfo": {"epfindia.gov.in", "gov.in"},
    "incometax": {"incometax.gov.in", "incometaxindia.gov.in"},
    "rbi": {"rbi.org.in"},
    "trai": {"trai.gov.in"},
    "irctc": {"irctc.co.in"},
    "npci": {"npci.org.in"},
    "paypal": {"paypal.com", "paypal.me"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "outlook.com"},
    "apple": {"apple.com", "icloud.com"},
    "amazon": {"amazon.com", "amazon.in", "aws.amazon.com"},
    "google": {"google.com", "google.co.in", "googleapis.com", "gstatic.com"},
    "whatsapp": {"whatsapp.com", "wa.me"},
    "netflix": {"netflix.com"},
}

OFFICIAL_ALLOW_SUFFIXES = (
    ".gov.in", ".nic.in", ".ac.in", ".edu.in", ".rbi.org.in", ".npci.org.in",
    ".sbi.co.in", ".onlinesbi.sbi", ".hdfcbank.com", ".icicibank.com", ".axisbank.com",
    ".paytm.com", ".phonepe.com", ".google.com", ".microsoft.com", ".apple.com",
    ".paypal.com", ".wikipedia.org", ".github.com", ".amazon.in", ".amazon.com",
)


def _is_official_host(host: str, allowed_set: set[str]) -> bool:
    for d in allowed_set:
        if host == d or host.endswith("." + d):
            return True
    return False


def _is_trusted_suffix(host: str) -> bool:
    for s in OFFICIAL_ALLOW_SUFFIXES:
        bare = s.lstrip(".")
        if host == bare or host.endswith(s):
            return True
    return False


def analyse_url(raw_url: str) -> ThreatEvent:
    raw_stripped = raw_url.strip()
    candidate = re.sub(r"^hxxps?://", lambda m: "https://" if "s" in m.group(0).lower() else "http://", raw_stripped, flags=re.I)
    if not re.match(r"^https?://", candidate, re.I):
        candidate = "http://" + candidate

    try:
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
        scheme = parsed.scheme
        path_q = ((parsed.path or "") + "?" + (parsed.query or "")).lower()
    except ValueError:
        host, scheme, path_q = "", "", ""
    lower_cand = candidate.lower()

    evidence: list[EvidenceItem] = []
    suspicious_points = 0

    # --- SCI Verified Local Domain & Indicator Intelligence (Stage 6/8) ---
    try:
        from app.sci.policy import get_active_policy
        _pol = get_active_policy()
    except Exception:
        _pol = {}
    _url_w = (_pol.get("rule_weights") or {}).get("url") or {}
    _sci_denied = False
    _sci_allowed = False
    if host and any(host == d or host.endswith("." + d) for d in _pol.get("verified_deny_domains", [])):
        _w = int(_url_w.get("SCI_VERIFIED_DENY_DOMAIN", 65))
        evidence.append(EvidenceItem(
            code="SCI_VERIFIED_DENY_DOMAIN",
            description=f"Domain '{host}' is listed in local SCI verified malicious domain intelligence.",
            weight=_w,
        ))
        suspicious_points += _w
        _sci_denied = True
    for _ind in _pol.get("verified_deny_indicators", []):
        if _ind and _ind in lower_cand:
            _w = int(_url_w.get("SCI_VERIFIED_INDICATOR", 60))
            evidence.append(EvidenceItem(
                code="SCI_VERIFIED_INDICATOR",
                description=f"URL matches local SCI verified malicious indicator '{_ind}'.",
                weight=_w,
            ))
            suspicious_points += _w
            _sci_denied = True
            break
    if host and not _sci_denied and any(host == d or host.endswith("." + d) for d in _pol.get("verified_allow_domains", [])):
        _sci_allowed = True

    if not host or "." not in host:
        evidence.append(EvidenceItem(
            code="INVALID_HOST", description="URL host is missing or malformed.", weight=35
        ))
        suspicious_points += 35

    # Unicode slash / dot authority spoofing (e.g., sbi.co.in\u2044evil.top)
    if CONFUSABLE_SLASH_RE.search(raw_stripped):
        evidence.append(EvidenceItem(
            code="UNICODE_CONFUSABLE_URI",
            description="URL contains Unicode slash/dot lookalike characters used to spoof trusted domains.",
            weight=60,
        ))
        suspicious_points += 60

    # Punycode / IDN homograph
    if "xn--" in host or any(ord(ch) > 127 for ch in host):
        evidence.append(EvidenceItem(
            code="IDN_HOMOGRAPH",
            description="Domain uses Punycode or non-ASCII homograph characters that imitate trusted brands.",
            weight=60,
        ))
        suspicious_points += 60

    # IP address / octal / hex / decimal IPv4 / IPv6 host
    ip_host = False
    try:
        ipaddress.ip_address(host)
        ip_host = True
    except ValueError:
        if host and OCTAL_HEX_IP_RE.match(host):
            ip_host = True
    if ip_host:
        w = 60 if any(t in path_q for t in SUSPICIOUS_TERMS) or DROPPER_EXT_RE.search(path_q) else 35
        evidence.append(EvidenceItem(
            code="IP_HOST",
            description="URL uses a raw or obfuscated IP address instead of a domain name.",
            weight=w,
        ))
        suspicious_points += w

    # Telegram Bot API credential / OTP exfiltration (bypasses Tranco allowlist!)
    if host == "api.telegram.org" and "/bot" in path_q and any(k in path_q for k in ("sendmessage", "senddocument", "chat_id", "otp", "text=")):
        evidence.append(EvidenceItem(
            code="TELEGRAM_BOT_EXFIL",
            description="URL abuses the Telegram Bot API endpoint to exfiltrate stolen OTPs or credentials.",
            weight=65,
        ))
        suspicious_points += 65

    # Ephemeral reverse-tunnel phishing host (ngrok, trycloudflare, serveo)
    if any(host.endswith(suf) for suf in TUNNEL_SUFFIXES):
        evidence.append(EvidenceItem(
            code="TUNNEL_PHISH_HOST",
            description="URL uses an ephemeral tunneling service commonly abused to host disposable phishing kits.",
            weight=60,
        ))
        suspicious_points += 60

    # Cloud PaaS hosting abused with brand/phishing keywords
    if any(host.endswith(suf) for suf in PAAS_SUFFIXES):
        if any(t in lower_cand for t in SUSPICIOUS_TERMS) or any(b in host for b in BRAND_OFFICIAL_DOMAINS) or DROPPER_EXT_RE.search(path_q):
            evidence.append(EvidenceItem(
                code="PAAS_BRAND_PHISH",
                description="URL hosts credential/brand lure content on a free cloud PaaS subdomain.",
                weight=60,
            ))
            suspicious_points += 60

    # High-risk file-extension TLDs (.zip, .mov)
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in HIGH_RISK_FILE_TLDS:
        evidence.append(EvidenceItem(
            code="HIGH_RISK_FILE_TLD",
            description=f"Domain uses the .{tld} top-level domain frequently abused to disguise phishing URLs as filenames.",
            weight=60,
        ))
        suspicious_points += 60

    # Malware dropper extensions (.apk, .exe, .scr, .msi) on untrusted host
    if DROPPER_EXT_RE.search(path_q) and not _is_trusted_suffix(host):
        evidence.append(EvidenceItem(
            code="MALWARE_DROPPER_EXT",
            description="URL points directly to an executable or Android APK dropper on an untrusted host.",
            weight=60,
        ))
        suspicious_points += 60

    if len(candidate) > 100:
        evidence.append(EvidenceItem(
            code="LONG_URL", description="URL is unusually long.", weight=10
        ))
        suspicious_points += 10

    if host in SHORTENERS:
        # Shorteners obscure final phishing destination; elevate strongly when path/slug is present
        w = 55 if (parsed.path and len(parsed.path.strip("/")) >= 2) else 25
        evidence.append(EvidenceItem(
            code="URL_SHORTENER", description="URL uses a link-shortening domain that hides the true destination.", weight=w
        ))
        suspicious_points += w

    labels = host.split(".")
    if len(labels) >= 4 and not _is_trusted_suffix(host):
        evidence.append(EvidenceItem(
            code="MANY_SUBDOMAINS", description="URL contains multiple subdomain labels.", weight=15
        ))
        suspicious_points += 15

    if "@" in candidate:
        evidence.append(EvidenceItem(
            code="AT_SYMBOL", description="URL contains an @ symbol that obscures the real destination host.", weight=60
        ))
        suspicious_points += 60

    if REDIRECT_PARAM_RE.search(lower_cand):
        evidence.append(EvidenceItem(
            code="OPEN_REDIRECT_CHAIN",
            description="URL chains an open-redirect parameter pointing to an external destination.",
            weight=55,
        ))
        suspicious_points += 55

    # Brand / Government impersonation on non-official domain
    if host and not _is_trusted_suffix(host):
        for brand, official_domains in BRAND_OFFICIAL_DOMAINS.items():
            if brand in host and not _is_official_host(host, official_domains):
                # Require either hyphen/subdomain/abuse-TLD/phishing keyword to avoid benign substring coincidences
                if (tld in ABUSE_TLDS or "-" in host or len(labels) >= 3 or any(t in lower_cand for t in SUSPICIOUS_TERMS)):
                    evidence.append(EvidenceItem(
                        code="BRAND_IMPERSONATION",
                        description=f"Domain references '{brand}' on an unofficial host '{host}'.",
                        weight=60,
                    ))
                    suspicious_points += 60
                    break

    matched_terms = [term for term in SUSPICIOUS_TERMS if term in lower_cand]
    if matched_terms:
        evidence.append(EvidenceItem(
            code="SUSPICIOUS_TERMS",
            description="URL contains potentially deceptive terms: " + ", ".join(matched_terms[:5]),
            weight=15,
        ))
        suspicious_points += 15

    # Abuse TLD paired with deceptive path/host terms
    if tld in ABUSE_TLDS and (matched_terms or any(w in lower_cand for w in ("portal", "claim", "mail", "invite", "challan", "support", "auth", "sim-"))):
        evidence.append(EvidenceItem(
            code="ABUSE_TLD_PHISH",
            description=f"High-abuse .{tld} domain combined with phishing or credential-harvesting keywords.",
            weight=45,
        ))
        suspicious_points += 45

    if host in KNOWN_BAD_EXAMPLE_DOMAINS:
        evidence.append(EvidenceItem(
            code="LOCAL_TEST_INDICATOR",
            description="Host matches a reserved demonstration indicator.",
            weight=35,
        ))
        suspicious_points += 35

    if _url_w:
        re_pts = 0
        for _ev in evidence:
            if _ev.code in _url_w:
                _ev.weight = int(_url_w[_ev.code])
            re_pts += max(0, _ev.weight)
        suspicious_points = re_pts
    _hard_bad = any(e.code in {"TELEGRAM_BOT_EXFIL", "MALWARE_DROPPER_EXT", "UNICODE_CONFUSABLE_URI", "IDN_HOMOGRAPH", "SCI_VERIFIED_DENY_DOMAIN", "SCI_VERIFIED_INDICATOR"} for e in evidence)
    if _sci_allowed and not _hard_bad:
        suspicious_points = 0
        evidence = [EvidenceItem(code="SCI_VERIFIED_ALLOW_DOMAIN", description=f"Domain '{host}' is in local SCI verified safe domain intelligence.", weight=0)]

    if suspicious_points >= 55:
        prediction, category, confidence, base = "suspicious", "phishing_suspected", 0.82, 15
    elif suspicious_points >= 25:
        prediction, category, confidence, base = "review", "url_needs_review", 0.62, 5
    else:
        prediction, category, confidence, base = "no_obvious_indicators", "no_obvious_url_indicators", 0.55, 0

    model_version = "url-rules-v0.1"
    metadata = {"hostname": host, "scheme": scheme}

    ml = score_url(raw_url)
    if ml is not None:
        _cal_thr = (_pol.get("thresholds") or {}).get("url_ml_threshold")
        _eff_ml_flag = ((ml.score >= float(_cal_thr)) and not ml.allowlisted) if _cal_thr is not None else ml.flagged
        if _sci_allowed and not _hard_bad:
            _eff_ml_flag = False
        strong_rule = prediction == "suspicious"
        flagged = _eff_ml_flag or strong_rule
        model_version = MODEL_VERSION
        metadata.update({
            "ml_score": round(ml.score, 4),
            "ml_threshold": round(ml.threshold, 4),
            "ml_allowlisted": ml.allowlisted,
            "registrable_domain": ml.domain,
        })
        if flagged:
            prediction, category, base = "suspicious", "phishing_suspected", 15
            confidence = max(0.65, min(0.99, ml.score)) if ml.flagged else max(confidence, 0.82)
        elif prediction == "review":
            pass
        else:
            prediction, category, confidence, base = "no_obvious_indicators", "no_obvious_url_indicators", 0.55, 0
        note = " (allowlisted popular domain)" if ml.allowlisted else ""
        evidence.append(EvidenceItem(
            code="ML_SCORE",
            description=f"ML phishing score {ml.score:.3f} vs threshold {ml.threshold:.3f}{note}.",
            weight=int(round(min(ml.score, 1.0) * 35)) if ml.flagged else 0,
        ))

    # RK3588 NPU_CORE_1 Prasad (2026) Dual-Branch Lexical MLP + Char 1D-CNN Fusion
    # Respects enable_ml_url / model_dir contract (only active when ML is enabled/available)
    if ml is not None:
      try:
        from app.edge_models.url_dual_npu import get_url_dual_npu_client
        npu_url_client = get_url_dual_npu_client()
        if npu_url_client.ready:
            _th = _pol.get("thresholds") or {}
            hi_tau = float(_th.get("url_npu_high_threshold", 0.96))
            cor_tau = float(_th.get("url_npu_cost_threshold", 0.70))
            npu_res = npu_url_client.infer(raw_url, threshold=hi_tau)
            npu_prob = float(npu_res["score"])
            npu_ms = float(npu_res["npu_latency_ms"])
            metadata.update({
                "npu_url_score": round(npu_prob, 4),
                "npu_url_latency_ms": round(npu_ms, 2),
                "npu_url_core": npu_res.get("core", "NPU_CORE_1"),
                "npu_url_backend": npu_res.get("backend", "rknn-npu"),
                "npu_url_threshold_high": hi_tau,
                "npu_url_model": npu_res.get("model_id", "url-dualbranch-mlp-charcnn-rknn-v1"),
            })
            _is_allow = (ml is not None and ml.allowlisted) or _sci_allowed or _is_trusted_suffix(host)
            if not _is_allow or _hard_bad:
                if npu_prob >= hi_tau:
                    evidence.append(EvidenceItem(
                        code="NPU_URL_DUAL_HIGH",
                        description=(
                            f"RK3588 NPU_CORE_1 Dual-Branch URL model (Prasad 2026 MLP + CharCNN) scored "
                            f"{npu_prob:.3f} (>= tau_hi={hi_tau:.2f}, {npu_ms:.2f}ms on NPU)."
                        ),
                        weight=45,
                    ))
                    prediction, category, base = "suspicious", "phishing_suspected", 15
                    confidence = max(confidence, min(0.98, npu_prob))
                elif npu_prob >= cor_tau and suspicious_points >= 15:
                    evidence.append(EvidenceItem(
                        code="NPU_URL_DUAL_CORROBORATIVE",
                        description=(
                            f"RK3588 NPU_CORE_1 Dual-Branch URL model scored {npu_prob:.3f} "
                            f"(>= {cor_tau:.2f}, {npu_ms:.2f}ms on NPU), corroborating lexical indicators."
                        ),
                        weight=25,
                    ))
                    prediction, category, base = "suspicious", "phishing_suspected", 15
                    confidence = max(confidence, min(0.92, npu_prob))
      except Exception as _u_exc:
        pass

    if not evidence:
        evidence.append(EvidenceItem(
            code="NO_RULE_MATCH",
            description="No configured heuristic indicators were found; this does not prove the URL is safe.",
            weight=0,
        ))

    actions = (
        ["Do not open the link or download any files from it.",
         "Navigate to the organisation's official website or app manually.",
         "Do not enter passwords, UPI PINs, payment details, or OTPs.",
         "Report suspicious financial phishing links in India via 1930 or cybercrime.gov.in."]
        if prediction != "no_obvious_indicators"
        else ["No configured indicator was detected; remain cautious and verify the sender and destination."]
    )

    return create_event(
        source="url_detector",
        category=category,
        prediction=prediction,
        confidence=confidence,
        evidence=evidence,
        actions=actions,
        model_version=model_version,
        raw_input=raw_url,
        metadata=metadata,
        base_score=base,
    )
