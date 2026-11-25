import logging
import re
import threading
import unicodedata
from pathlib import Path
from app.events import create_event
from app.schemas import EvidenceItem, ThreatEvent

log = logging.getLogger("edgeshield.detectors.sms")

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
_SPACED_WORD_RE = re.compile(r"(?<![A-Za-z0-9])((?:[A-Za-z0-9]\s+){2,}[A-Za-z0-9])(?![A-Za-z0-9])")
_DOTTED_ACRO_RE = re.compile(r"\b([A-Za-z])\.([A-Za-z])\.([A-Za-z])(?:\.([A-Za-z]))?\b")
_LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_URL_EXTRACT_RE = re.compile(
    r"(?:hxxps?|https?)://[^\s<>\"'\)\]]+|www\.[^\s<>\"'\)\]]+|"
    r"\b(?:[a-z0-9-]+\.)+(?:top|xyz|zip|mov|click|pages\.dev|workers\.dev|ngrok-free\.app|bit\.ly|rb\.gy|tinyurl\.com|tk|ml|ga|cf|gq|cfd|sbs|co\.in|com|in)(?:/[^\s<>\"'\)\]]*)?",
    re.I,
)

# Official cyber-safety advisory guard (checked FIRST so RBI warnings naming "AnyDesk/OTP/UPI PIN" never FP!)
_OFFICIAL_SAFETY_ADVISORY_RE = re.compile(
    r"(?:rbi\s+kehta\s+hai|cyber\s*dost|stay\s+cyber\s+safe|never\s+asks?\s+for\s+(?:your\s+)?(?:otp|pin|password|cvv|kyc)|"
    r"visited\s+.{0,40}bank\s+branch.{0,120}completed.{0,60}kyc)",
    re.I,
)

# Genuine bank transactional OTP / debit-credit alert guard
_BENIGN_PROTECTIVE_RE = re.compile(
    r"(?:do\s+not\s+share|never\s+share|don'?t\s+share|not\s+for\s+sharing|"
    r"(?:debited|credited|spent|paid)\s+(?:by|for|with|from)?\s*(?:inr|rs\.?|₹)\s*[\d,]+.{0,80}(?:avl\.?\s*bal|available\s+bal|a/c|acct|upi\s*ref|imps|neft)|"
    r"(?:inr|rs\.?|₹)\s*[\d,]+(?:\.\d+)?\s+(?:has\s+been\s+|is\s+)?(?:debited|credited|spent|paid)\s+(?:from|to|on)\s+(?:your\s+)?(?:a/c|acct|account|card))",
    re.I,
)

# Explicit demand to send/share OTP (NEGATIVE lookbehind excludes "Do NOT share this OTP" / "Never share your OTP"!)
_EXPLICIT_OTP_DEMAND_RE = re.compile(
    r"(?<!not\s)(?<!never\s)(?<!don't\s)(?<!dont\s)\b(?:please\s+)?(?:share|send|reply|forward|tell|enter|provide)\s+"
    r"(?:the\s+|this\s+|your\s+|that\s+)?(?:6[- ]?digit\s+)?(?:otp|pin|cvv|password|code)",
    re.I,
)

STRONG_SCAM_PATTERNS = [
    ("DIGITAL_ARREST_COERCION", re.compile(
        r"(?:digital\s+arrest|arrest\s+warrant|non[- ]?bailable\s+warrant|"
        r"(?:trai|cbi|ed|enforcement\s+directorate|narcotics|mumbai\s+police|delhi\s+police|cyber\s+police|customs|fedex|dhl).{0,90}"
        r"(?:illegal|money\s+laundering|drugs|seized|skype|video\s+call|disconnect(?:ed)?\s+in\s+\d|arrest|fir\s+registered|case\s+registered))", re.I),
     55, "Message uses Digital Arrest / law-enforcement / TRAI / customs impersonation and coercion."),

    ("MALICIOUS_APK_LURE", re.compile(
        r"\.apk\b|(?:wedding|invitation|rto|challan|e-?challan|pm\s*kisan|bank\s+support).{0,60}(?:install|download|open).{0,30}(?:app|apk|card)", re.I),
     55, "Message lures the recipient into installing a malicious Android APK file."),

    ("ECHALLAN_TRAFFIC_SCAM", re.compile(
        r"(?:parivahan|e-?challan|fastag|traffic\s+challan).{0,90}(?:unpaid|pay\s+today|licence\s+will\s+be\s+seized|suspended|penalty|blacklisted|click|http)", re.I),
     50, "Message impersonates Parivahan / e-Challan / FASTag demanding urgent payment or threatening licence seizure."),

    ("INSTANT_LOAN_EXTORTION", re.compile(
        r"(?:pre[- ]?approved|instant)\s+loan.{0,90}(?:zero\s+cibil|no\s+cibil|processing\s+fee|5\s+mins?|instant\s+approval|pay\s+rs\.?\s*[\d,]+)", re.I),
     50, "Message promotes a predatory instant-loan / advance processing-fee scam."),

    ("REMOTE_ACCESS_RAT", re.compile(
        r"\b(?:anydesk|teamviewer|quicksupport|rustdesk|ammyy|ultraviewer|airdroid)\b|share\s+(?:your\s+)?screen\s+(?:code|id)|9[- ]?digit\s+(?:anydesk|support)\s+(?:code|id)", re.I),
     55, "Message requests installing a remote-access tool (AnyDesk/TeamViewer/QuickSupport)."),

    ("USSD_CALL_FORWARD", re.compile(
        r"\*+(?:401|21|61|62|67)\*[0-9]{7,13}#?", re.I),
     60, "Message asks the user to dial a USSD call-forwarding code (*401*/**21*) that hijacks OTP calls and SMS."),

    ("UTILITY_DISCONNECT_SCAM", re.compile(
        r"(?:electricity|power\s+supply|light|gas\s+connection|water\s+supply|bescom|msedcl|bses|tata\s+power|adani|indane|igl).{0,80}"
        r"(?:disconnect(?:ed)?|cut\s+off|suspend(?:ed)?|tonight|9:30|unpaid\s+bill|last\s+month\s+bill.{0,30}not\s+updated)", re.I),
     55, "Message threatens imminent electricity/gas/utility disconnection to force an urgent scam callback or payment."),

    ("PREPAID_TASK_JOB_SCAM", re.compile(
        r"(?:earn|make|income|salary)\s+(?:up\s*to\s+)?(?:rs\.?|₹|inr|\$)\s*\d{3,6}\s*(?:daily|per\s+day|everyday|weekly)|"
        r"(?:like|subscribe)\s+(?:youtube|instagram)\s+(?:videos?|channels?).{0,50}(?:earn|paid|rs|₹|telegram)|"
        r"(?:telegram|whatsapp)\s+(?:task|part[- ]?time|prepaid\s+task|receptionist|hotel\s+review).{0,50}(?:earn|bonus|deposit|rs|₹)", re.I),
     50, "Message promotes a prepaid part-time task / YouTube-like / Telegram job scam."),

    ("INVESTMENT_PIG_BUTCHERING", re.compile(
        r"(?:vip|exclusive|institutional)\s+(?:stock|ipo|trading|crypto|wealth|equity)\s+(?:group|club|signals?|tips?|account)|"
        r"(?:guaranteed|assured)\s+(?:returns?|profit|200%|300%|500%|\d{2,3}%\s+daily)|"
        r"upper\s+circuit\s+(?:stock|pick|tip)|(?:usdt|crypto|binance).{0,50}(?:guaranteed|daily\s+profit|vip\s+group|join\s+telegram)|"
        r"(?:matrimony|dating\s+app).{0,120}(?:exchange|insider\s+node|crypto|usdt|trading|profit|invest)", re.I),
     50, "Message pushes a fraudulent VIP stock/IPO/crypto investment or romance pig-butchering scheme."),

    ("WHATSAPP_SIM_HIJACK", re.compile(
        r"(?:whatsapp|telegram)\s+(?:6[- ]?digit|verification|registration)\s+code|"
        r"accidentally\s+sent\s+(?:you\s+|my\s+)?(?:a\s+|my\s+)?(?:6[- ]?digit\s+)?(?:whatsapp\s+)?(?:verification\s+)?(?:code|otp)|"
        r"(?:sim\s+card\s+will\s+stop|20[- ]?digit\s+sim|approve\s+(?:the\s+)?e-?sim|"
        r"(?:sim|e-?sim)\s+(?:will\s+be\s+blocked|upgrade\s+to\s+5g|swap|deactivated).{0,70}(?:forward|send|dial|click|call|sms\s+\d|approve))", re.I),
     55, "Message attempts a WhatsApp 6-digit code hijack or SIM/eSIM swap takeover."),

    ("QR_UPI_RECEIVE_SCAM", re.compile(
        r"(?:scan\s+(?:the\s+|this\s+)?qr|enter\s+(?:your\s+)?(?:upi\s+)?pin|approve\s+(?:the\s+)?collect\s+request).{0,40}"
        r"(?:receive|get|claim|accept|credit)\s+(?:money|payment|refund|cashback|amount|rs|₹)", re.I),
     55, "Message tricks the victim into scanning a QR code or entering a UPI PIN claiming it is needed to receive money."),

    ("CALLBACK_AUTODEBIT_TRAP", re.compile(
        r"(?:auto[- ]?debit(?:ed)?|auto[- ]?renewal|subscription\s+renewed|mcafee|norton|geek\s*squad).{0,110}"
        r"(?:call|dial|contact|cancel|if\s+you\s+did\s+not|if\s+not\s+authorized|helpline|in\s+\d+\s*(?:min|hour))", re.I),
     50, "Message fabricates an auto-debit or subscription charge to lure an urgent callback or cancellation scam."),

    ("FAMILY_OR_EXEC_IMPERSONATION", re.compile(
        r"(?:lost\s+my\s+phone|temporary\s+number|personal\s+number).{0,120}(?:don'?t\s+call|cannot\s+take\s+calls|urgently|emergency|gift\s+card|rs\.?\s*[\d,]+|transfer)|"
        r"(?:i\s+am\s+in\s+a\s+meeting|stuck\s+in\s+a\s+board\s+meeting|urgent\s+wire\s+transfer|"
        r"(?:buy|purchase)\s+(?:\d+\s+)?(?:apple|amazon|google\s+play|itunes)\s+gift\s+cards?|"
        r"(?:ceo|cfo|managing\s+director|hr\s+team|payroll).{0,90}(?:gift\s+card|wire\s+transfer|updated\s+bank\s+account|do\s+not\s+call))", re.I),
     50, "Message exhibits family-emergency ('Hi Dad, lost my phone') or executive BEC gift-card impersonation."),

    ("HINDI_DEVANAGARI_SCAM", re.compile(
        r"(?:ओटीपी|केवाईसी|खाता\s*(?:बंद|ब्लॉक|निलंबित)|गिरफ्तार|डिजिटल\s*अरेस्ट|लॉटरी|इनाम\s*जीता|"
        r"बिजली\s*(?:कनेक्शन|काट|बंद)|तुरंत\s*(?:लिंक|क्लिक|अपडेट|भेजें|कॉल)|पुलिस\s*(?:केस|वारंट))"),
     55, "Message contains high-risk Hindi (Devanagari) scam, KYC, OTP, arrest, or utility-cutoff coercion."),

    ("REGIONAL_ROMAN_SCAM", re.compile(
        r"(?:aapka\s+(?:bank|sbi|hdfc|khata|sim|bijli)|khata\s+band|turant\s+(?:kyc|otp|link|update|bheje|call)|"
        r"bijli\s+(?:kat|band)|inaam\s+jeeta|lottery\s+lagi|giraftar|"
        r"ungal\s+(?:bank|account|kyc|otp|sim)|udane\s+(?:click|send|share|call)|"
        r"mee\s+(?:bank|account|kyc|otp|sim)|ventane\s+(?:click|send|share|call))", re.I),
     50, "Message uses regional Indian (Hinglish/Tanglish/Tenglish) social-engineering scam phrasing."),
]

BASE_PATTERNS = [
    ("OTP_REQUEST", re.compile(r"\b(otp|one[- ]time password|verification code)\b", re.I), 25,
     "Message refers to an OTP or verification code."),
    ("CREDENTIAL_REQUEST", re.compile(r"\b(password|pin|cvv|card number|login details|netbanking\s+id)\b", re.I), 30,
     "Message refers to sensitive credentials or payment information."),
    ("URGENCY", re.compile(r"\b(urgent|urgently|immediately|within\s+\d+\s*hours?|suspended|blocked|expire|deactivated|frozen|final\s+warning)\b", re.I), 15,
     "Message uses urgency or account-suspension language."),
    ("FINANCIAL_LURE", re.compile(r"\b(prize|lottery|cashback|refund|won|reward|jackpot|crore|lakh\s+won|free\s+bonus|claim\s+now)\b", re.I), 20,
     "Message contains a financial reward, lottery, or refund lure."),
    ("KYC_REQUEST", re.compile(r"\b(kyc|verify\s+your\s+account|update\s+your\s+account|pan\s+update|re-?kyc|update\s+pan)\b", re.I), 25,
     "Message requests account, PAN, or KYC verification."),
    ("LINK_PRESENT", re.compile(r"https?://|hxxps?://|www\.", re.I), 20,
     "Message contains a web link."),
    ("PAYMENT_PRESSURE", re.compile(r"\b(pay\s+now|transfer\s+(?:now|immediately|money)|send\s+money|upi\s+id|bank\s+account|deposit\s+rs)\b", re.I), 15,
     "Message requests or pressures a financial transaction."),
    ("OTP_DEMAND_ACTION", _EXPLICIT_OTP_DEMAND_RE, 45,
     "Message explicitly instructs the recipient to share or enter an OTP, PIN, or password."),
]

_ml_lock = threading.Lock()
_ml_bundle = None
_ml_tried = False


def _load_sms_ml():
    global _ml_bundle, _ml_tried
    with _ml_lock:
        if _ml_tried:
            return _ml_bundle
        _ml_tried = True
        try:
            from app.config import settings
            if not settings.ml_detectors:
                return None
            p = Path(settings.model_dir) / "sms_model_v2.joblib"
            if p.is_file():
                import joblib
                b = joblib.load(p)
                if isinstance(b, dict) and "model" in b:
                    _ml_bundle = b
        except Exception as exc:
            log.warning("SMS ML bundle unavailable (%s); continuing with hybrid rules", exc)
        return _ml_bundle


def _normalize_sms(message: str) -> tuple[str, str, bool]:
    had_obf = False
    no_zw = _ZERO_WIDTH_RE.sub("", message)
    if len(no_zw) != len(message):
        had_obf = True
    nfkc = unicodedata.normalize("NFKC", no_zw)
    collapsed = _SPACED_WORD_RE.sub(lambda m: re.sub(r"\s+", "", m.group(1)), nfkc)
    if collapsed != nfkc:
        had_obf = True
    collapsed = _DOTTED_ACRO_RE.sub(lambda m: "".join(g for g in m.groups() if g), collapsed)
    leet = collapsed.translate(_LEET_MAP)
    if re.search(r"\b(?:0tp|acc0unt|b1ocked|p4ssword|k\.?y\.?c)\b", nfkc, re.I):
        had_obf = True
    return collapsed, leet, had_obf


def _extract_urls(text: str) -> list[str]:
    found = []
    for m in _URL_EXTRACT_RE.findall(text):
        u = m.rstrip(".,;:!?")
        if u.lower().startswith("hxxp"):
            u = "http" + u[4:]
        if u not in found:
            found.append(u)
    return found[:3]


def analyse_sms(message: str) -> ThreatEvent:
    clean_msg, leet_msg, had_obf = _normalize_sms(message)
    embedded_urls = _extract_urls(clean_msg)

    # 0. Official safety advisories ("RBI Kehta Hai: Never share OTP/AnyDesk...") & benign branch KYC notes
    if _OFFICIAL_SAFETY_ADVISORY_RE.search(clean_msg) and not embedded_urls and not had_obf:
        return create_event(
            source="sms_detector",
            category="unclassified_message",
            prediction="no_obvious_indicators",
            confidence=0.85,
            evidence=[EvidenceItem(
                code="OFFICIAL_SAFETY_ADVISORY",
                description="Message matches a legitimate cyber-safety awareness advisory or benign personal finance note with no external lure links.",
                weight=0,
            )],
            actions=["Verify unexpected requests independently; never share credentials or OTPs."],
            model_version="sms-hybrid-v2",
            raw_input=message,
            metadata={"matched_rule_count": 0, "protective_bank_guard": True},
            base_score=0,
        )

    evidence: list[EvidenceItem] = []
    score = 0
    strong_hit = False

    # 0b. SCI Verified Local Threat Intelligence & Policy Overrides (Stage 6/8)
    try:
        from app.sci.policy import get_active_policy
        _pol = get_active_policy()
    except Exception:
        _pol = {}
    _sms_w = (_pol.get("rule_weights") or {}).get("sms") or {}
    _low_clean = clean_msg.lower()
    for _bp in _pol.get("verified_benign_phrases", []):
        if _bp and _bp in _low_clean and not embedded_urls and not had_obf:
            return create_event(
                source="sms_detector", category="unclassified_message",
                prediction="no_obvious_indicators", confidence=0.88,
                evidence=[EvidenceItem(code="SCI_VERIFIED_BENIGN_PHRASE", description=f"Matched SCI verified benign phrase '{_bp}'.", weight=0)],
                actions=["Verify unexpected requests independently; never share credentials or OTPs."],
                model_version="sms-hybrid-v2", raw_input=message,
                metadata={"matched_rule_count": 0, "protective_bank_guard": True, "sci_policy_version": _pol.get("version", "v1.0.0")},
                base_score=0,
            )
    for _ind in _pol.get("verified_deny_indicators", []):
        if _ind and _ind in _low_clean:
            _w = int(_sms_w.get("SCI_VERIFIED_SCAM_INDICATOR", 55))
            evidence.append(EvidenceItem(code="SCI_VERIFIED_SCAM_INDICATOR", description=f"Message matches SCI verified scam indicator '{_ind}'.", weight=_w))
            score += _w
            strong_hit = True
            break

    # 1. Strong Indian/Global/Multilingual scam archetypes
    for code, pattern, weight, description in STRONG_SCAM_PATTERNS:
        if pattern.search(clean_msg) or pattern.search(leet_msg):
            evidence.append(EvidenceItem(code=code, description=description, weight=weight))
            score += weight
            strong_hit = True

    # 2. Extract and cross-analyze embedded URLs using url-hybrid-v2
    embedded_phish_hits: list[str] = []
    if embedded_urls:
        try:
            from app.detectors.url import analyse_url
            for u in embedded_urls:
                url_ev = analyse_url(u)
                if url_ev.prediction == "suspicious":
                    embedded_phish_hits.append(u)
                    evidence.append(EvidenceItem(
                        code="EMBEDDED_PHISHING_URL",
                        description=f"SMS embeds a suspected phishing/malicious link ({u[:80]}, URL risk {url_ev.risk_score}/100).",
                        weight=45,
                    ))
                    score += 45
                    strong_hit = True
                    break
        except Exception:
            pass

    # 3. Protective genuine bank OTP / debit-credit guard (negative lookbehind avoids false match on "Do NOT share this OTP")
    is_protective_benign = (
        bool(_BENIGN_PROTECTIVE_RE.search(clean_msg))
        and not strong_hit
        and not had_obf
        and not embedded_urls
        and not _EXPLICIT_OTP_DEMAND_RE.search(clean_msg)
    )

    if had_obf:
        evidence.append(EvidenceItem(
            code="TEXT_OBFUSCATION_EVASION",
            description="Message uses zero-width characters, spaced letters, or leetspeak to evade scam filters.",
            weight=30,
        ))
        score += 30

    if not is_protective_benign:
        for code, pattern, weight, description in BASE_PATTERNS:
            if pattern.search(clean_msg) or pattern.search(leet_msg):
                evidence.append(EvidenceItem(code=code, description=description, weight=weight))
                score += weight

    # 4. Calibrated ML + RK3588 NPU_CORE_2 BiGRU+MaskedPool+WBCE (Phap 2026) Fusion
    model_version = "sms-rules-v2.0"
    ml_score = None
    bundle = _load_sms_ml()
    if bundle is not None:
        model_version = "sms-hybrid-v2"
        try:
            prob = float(bundle["model"].predict_proba([clean_msg])[0, 1])
            ml_score = round(prob, 4)
            if not is_protective_benign and prob >= float((_pol.get("thresholds") or {}).get("sms_ml_threshold", 0.85)):
                evidence.append(EvidenceItem(
                    code="ML_SMS_MODEL",
                    description=f"On-device TF-IDF SMS baseline scored this message {prob:.3f} (high lexical scam confidence).",
                    weight=35,
                ))
                score += 35
        except Exception:
            pass

    npu_bigru_info: dict = {}
    try:
        from app.edge_models.sms_bigru_npu import get_sms_bigru_npu_client
        npu_client = get_sms_bigru_npu_client()
        if npu_client.ready:
            _th = _pol.get("thresholds") or {}
            cost_tau = float(_th.get("sms_bigru_threshold", 0.22))
            hi_tau = float(_th.get("sms_bigru_high_threshold", 0.56))
            npu_res = npu_client.infer(clean_msg, threshold=cost_tau)
            npu_prob = float(npu_res["score"])
            npu_ms = float(npu_res["npu_latency_ms"])
            model_version = "sms-hybrid-v3-npu-bigru"
            npu_bigru_info = {
                "npu_bigru_score": round(npu_prob, 4),
                "npu_bigru_latency_ms": round(npu_ms, 2),
                "npu_bigru_core": npu_res.get("core", "NPU_CORE_2"),
                "npu_bigru_backend": npu_res.get("backend", "rknn-npu"),
                "npu_bigru_threshold_cost": cost_tau,
                "npu_bigru_threshold_high": hi_tau,
                "npu_bigru_model": npu_res.get("model_id", "sms-bigru-maskedpool-wbce-rknn-v1"),
            }
            if not is_protective_benign:
                if npu_prob >= hi_tau:
                    evidence.append(EvidenceItem(
                        code="NPU_SMS_BIGRU_HIGH",
                        description=(
                            f"RK3588 NPU_CORE_2 BiGRU+MaskedPool+WBCE (Phap 2026) scored {npu_prob:.3f} "
                            f"(>= tau_hi={hi_tau:.2f}, {npu_ms:.1f}ms on NPU - high-confidence smishing)."
                        ),
                        weight=45,
                    ))
                    score += 45
                elif npu_prob >= cost_tau:
                    evidence.append(EvidenceItem(
                        code="NPU_SMS_BIGRU_COST_SENSITIVE",
                        description=(
                            f"RK3588 NPU_CORE_2 BiGRU+MaskedPool+WBCE (Phap 2026) scored {npu_prob:.3f} "
                            f"(>= tau*={cost_tau:.2f} [5:1 cost-calibrated], {npu_ms:.1f}ms on NPU)."
                        ),
                        weight=20,
                    ))
                    score += 20
    except Exception as _npu_exc:
        log.debug("RKNN SMS BiGRU inference skipped: %s", _npu_exc)

    if is_protective_benign and score == 0:
        evidence = [EvidenceItem(
            code="PROTECTIVE_BANK_CONTEXT",
            description="Message matches legitimate bank/transactional OTP phrasing with protective 'do not share' guidance.",
            weight=0,
        )]

    _best_ml = max(ml_score or 0.0, float(npu_bigru_info.get("npu_bigru_score", 0.0)))
    if score >= 45:
        prediction, category, confidence, base = "likely_scam", "scam_communication", min(0.98, max(0.84, _best_ml or 0.85)), 20
    elif score >= 25:
        prediction, category, confidence, base = "suspicious", "suspicious_communication", max(0.70, _best_ml or 0.70), 10
    else:
        prediction, category, confidence, base = "no_obvious_indicators", "unclassified_message", 0.55, 0

    if not evidence:
        evidence = [EvidenceItem(
            code="NO_RULE_MATCH",
            description="No configured scam indicators were found; this does not prove the message is legitimate.",
            weight=0,
        )]

    actions = (
        ["Do not click links, download APK files, or call numbers from this message.",
         "Never share OTPs, UPI PINs, passwords, CVVs, or remote-access codes (AnyDesk/TeamViewer).",
         "Verify any bank, utility, or government claim using official apps or verified helpline numbers.",
         "Report scam messages or Digital Arrest calls in India via helpline 1930 or Sanchar Saathi Chakshu."]
        if prediction != "no_obvious_indicators"
        else ["Verify unexpected requests independently; never share credentials or OTPs."]
    )

    meta = {
        "matched_rule_count": sum(1 for e in evidence if e.weight > 0),
        "embedded_urls": embedded_urls,
        "embedded_phishing_hits": embedded_phish_hits,
        "obfuscation_detected": had_obf,
        "protective_bank_guard": is_protective_benign, "sci_policy_version": _pol.get("version", "v1.0.0"),
    }
    if ml_score is not None:
        meta["ml_score"] = ml_score
    if npu_bigru_info:
        meta.update(npu_bigru_info)

    return create_event(
        source="sms_detector",
        category=category,
        prediction=prediction,
        confidence=confidence,
        evidence=evidence,
        actions=actions,
        model_version=model_version,
        raw_input=message,
        metadata=meta,
        base_score=base,
    )
