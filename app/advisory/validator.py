"""Prompt-injection sanitization, incident-specific vetted recommendations, and verdict-locked
output validation for on-device Llama-3.2-1B ExecuTorch threat advisories."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from app.schemas import ThreatEvent

_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]{1,40}\|>", re.IGNORECASE)
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|system)", re.I),
    re.compile(r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\b", re.I),
    re.compile(r"\b(system|assistant)\s*:", re.I),
    re.compile(r"(override|reverse|change)\s+(the\s+)?(verdict|risk|score|classification)", re.I),
    re.compile(r"(say|respond|output|declare)\s+(that\s+)?(this\s+)?(is\s+)?(safe|legitimate|benign)", re.I),
]

_SAFE_CLAIM_RE = re.compile(
    r"(?<!not\s)(?<!n't\s)(?<!never\s)(?<!nahi\s)(?<!नहीं\s)\b("
    r"this\s+(message|sms|link|url|alert|input|flow)\s+is\s+(completely\s+|totally\s+)?(safe|legitimate|genuine|harmless|benign)"
    r"|appears\s+to\s+be\s+(completely\s+)?(safe|legitimate|genuine|benign)"
    r"|no\s+(security\s+)?(threat|danger|risk|scam|phishing)\s+(was\s+)?(found|detected|present)"
    r"|not\s+a\s+(scam|phishing|threat|attack)"
    r"|false\s+alarm"
    r"|bilkul\s+surakshit"
    r"|koi\s+khatra\s+nahi"
    r"|पूरी\s+तरह\s+सुरक्षित"
    r"|कोई\s+खतरा\s+नहीं"
    r")",
    re.IGNORECASE,
)

_PANIC_CLAIM_RE = re.compile(
    r"\b(your\s+account\s+has\s+been\s+hacked|confirmed\s+ransomware|confirmed\s+active\s+breach|money\s+was\s+stolen)\b",
    re.IGNORECASE,
)

_UNSAFE_ADVICE_RE = re.compile(
    r"\b(click\s+(on\s+)?(the|this)\s+link|open\s+(the|this)\s+(link|apk|file)|"
    r"share\s+(your\s+)?(otp|pin|cvv|password)|send\s+(your\s+)?(otp|pin|cvv|money)|"
    r"install\s+(anydesk|teamviewer|the\s+apk)|call\s+the\s+number\s+in\s+the\s+(sms|message))\b",
    re.IGNORECASE,
)
_NEGATION_PREFIX_RE = re.compile(
    r"(do\s+not|don't|never|avoid|refuse\s+to|should\s+not|must\s+not|mat\s+|na\s+|kabhi\s+nahi|कभी\s+नहीं|मत\s+|न\s+)",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_IP_PORT_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})(?::(\d+))?")


def normalize_language(language: str | None) -> str:
    val = (language or "en").strip().lower()
    if val == "hinglish" or "hinglish" in val:
        return "hinglish"
    if val in {"hi", "hindi", "hi-in"} or val.startswith("hi"):
        return "hi"
    return "en"


def sanitize_event_for_llm(event: ThreatEvent) -> tuple[dict[str, Any], bool]:
    """Pass the REAL sanitized SMS / URL / Network flow directly to Llama so every
    explanation is incident-specific, while stripping any prompt-injection payload."""
    raw = str(event.raw_input_preview or "")
    injection_detected = bool(_CONTROL_TOKEN_RE.search(raw)) or any(p.search(raw) for p in _INJECTION_PATTERNS)

    cleaned_raw = _CONTROL_TOKEN_RE.sub(" ", raw)
    cleaned_raw = re.sub(r"[\r\n\t]+", " ", cleaned_raw).strip()

    if injection_detected:
        # Strip injection sentences while keeping any non-injection URL/scam context
        stripped = cleaned_raw
        for p in _INJECTION_PATTERNS:
            stripped = p.sub("[blocked-instruction]", stripped)
        pos_reasons = [item.description for item in event.evidence if item.weight > 0][:2]
        reason_str = "; ".join(pos_reasons) if pos_reasons else event.threat_category
        preview = f"{reason_str} (prompt-injection attempt stripped)"
    else:
        # Keep the real input front-and-center for Llama's 512-token context window!
        preview = cleaned_raw
        if not preview:
            pos_reasons = [item.description for item in event.evidence if item.weight > 0][:2]
            preview = "; ".join(pos_reasons) if pos_reasons else event.threat_category

    ev_dict = event.model_dump(mode="json")
    ev_dict["raw_input_preview"] = preview[:280]
    return ev_dict, injection_detected


def _extract_host_and_apk(text: str) -> tuple[str, str]:
    m = _URL_RE.search(text or "")
    if not m:
        return "", ""
    u = m.group(0).rstrip(".,);")
    try:
        parsed = urlparse(u)
        host = parsed.netloc or ""
        path = parsed.path or ""
        apk = ""
        if ".apk" in path.lower():
            apk = path.rsplit("/", 1)[-1]
        return host, apk
    except Exception:
        return "", ""


def vetted_recommendations(event: ThreatEvent, language: str = "en") -> list[str]:
    """Incident-specific, non-destructive safety actions tailored to the actual event."""
    lang = normalize_language(language)
    is_threat = event.risk_score >= 45 or event.risk_level in {"high", "critical"}
    src = (event.source or "").lower()
    raw = str(event.raw_input_preview or "")
    low = raw.lower()
    host, apk = _extract_host_and_apk(raw)

    if lang == "hi":
        if not is_threat:
            return [
                "कोई स्पष्ट खतरा नहीं मिला, फिर भी अनजान लिंक या कॉलर से सावधान रहें।",
                "कभी भी किसी को OTP, UPI PIN, CVV या पासवर्ड साझा न करें।",
            ]
        recs: list[str] = []
        if "sms" in src:
            if any(k in low for k in ("digital arrest", "cyber police", "trai", "cbi", "skype", "warrant")):
                recs.append("पुलिस या TRAI कभी वीडियो कॉल पर 'डिजिटल अरेस्ट' नहीं करती — कॉल काटें और नंबर ब्लॉक करें।")
            if apk:
                recs.append(f"संदिग्ध ऐप फ़ाइल '{apk}' को कभी डाउनलोड या इंस्टॉल न करें — यह बैंकिंग/OTP चोरी करने वाला मैलवेयर हो सकता है।")
            if host:
                recs.append(f"अनजान डोमेन '{host}' को न खोलें और वहां कोई KYC, PAN या OTP दर्ज न करें।")
            if any(k in low for k in ("anydesk", "teamviewer", "quick support", "electricity", "bescom")):
                recs.append("AnyDesk/TeamViewer जैसा रिमोट-एक्सेस ऐप न इंस्टॉल करें; बिजली/सेवा बिल केवल आधिकारिक ऐप में जांचें।")
            if not recs:
                recs.append("इस संदेश में दिए लिंक पर क्लिक न करें और न ही अनजान नंबर पर कॉल या OTP साझा करें।")
            recs.append("धोखाधड़ी की रिपोर्ट भारत के राष्ट्रीय साइबर क्राइम हेल्पलाइन 1930 या cybercrime.gov.in पर करें।")
            return recs[:4]
        if "url" in src:
            if apk:
                recs.append(f"'{host or raw}' से '{apk}' डाउनलोड या इंस्टॉल न करें; अज्ञात स्रोतों से APK इंस्टॉलेशन बंद रखें।")
            elif host:
                recs.append(f"फ़िशिंग डोमेन '{host}' को न खोलें और वहां लॉगिन पासवर्ड, OTP या भुगतान जानकारी न डालें।")
            else:
                recs.append("इस संदिग्ध लिंक को न खोलें और ब्राउज़र में स्वयं आधिकारिक वेबसाइट का पता टाइप करें।")
            recs.append("यदि विवरण पहले ही दर्ज हो गए हैं, तो तुरंत पासवर्ड बदलें और बैंक / 1930 हेल्पलाइन को सूचित करें।")
            return recs[:4]
        # Network HI
        ips = _IP_PORT_RE.findall(raw)
        if ips:
            src_ip = ips[0][0]
            dst_part = f"{ips[1][0]}:{ips[1][1]}" if len(ips) > 1 and ips[1][1] else (ips[1][0] if len(ips) > 1 else "")
            recs.append(f"स्रोत IP {src_ip} से {dst_part or 'लक्ष्य डिवाइस'} पर हो रहे अनधिकृत ट्रैफ़िक की तुरंत फ़ायरवॉल/लॉग जांच करें।")
        else:
            recs.append(f"चिह्नित नेटवर्क गतिविधि ({event.threat_category}) के लिए स्थानीय फ़ायरवॉल और सक्रिय कनेक्शन जांचें।")
        recs.append("अनावश्यक खुले पोर्ट बंद करें और प्रभावित डिवाइस को संदिग्ध ट्रैफ़िक जारी रहने पर LAN से अलग करें।")
        return recs[:4]

    if lang == "hinglish":
        if not is_threat:
            return [
                "Koi clear threat indicator nahi mila, phir bhi unknown links ya unexpected calls se savdhan rahein.",
                "Kabhi bhi kisi ko OTP, UPI PIN, CVV ya password SMS, call ya unverified link par share na karein.",
            ]
        recs = []
        if "sms" in src:
            if any(k in low for k in ("digital arrest", "cyber police", "trai", "cbi", "skype", "warrant")):
                recs.append("Police, CBI ya TRAI kabhi Skype/video call par 'Digital Arrest' nahi karti — turant call kaatein aur sender ko block karein.")
            if apk:
                recs.append(f"Suspicious file '{apk}' ko bilkul download ya install mat karein — aise APK links mein banking/OTP churane wala spyware hota hai.")
            if host:
                recs.append(f"Unknown domain '{host}' ko mat kholein aur wahan KYC, PAN, banking credentials ya OTP bilkul enter na karein.")
            if any(k in low for k in ("anydesk", "teamviewer", "quick support", "electricity", "bescom", "bill")):
                recs.append("Caller ke kehne par AnyDesk/TeamViewer jaisa remote-access app kabhi install mat karein; bill sirf official utility app mein check karein.")
            if any(k in low for k in ("*401*", "call forward", "ussd")):
                recs.append("USSD code jaise *401* bilkul dial mat karein — isse aapki incoming OTP calls silently attacker ko forward ho sakti hain.")
            if not recs:
                recs.append("Is SMS mein diye gaye link par click mat karein, na unknown number par call karein aur na hi OTP/UPI PIN share karein.")
            recs.append("Cyber fraud ki report India helpline 1930, cybercrime.gov.in ya Sanchar Saathi Chakshu par turant karein.")
            return recs[:4]
        if "url" in src:
            if apk:
                recs.append(f"'{host or 'is URL'}' se '{apk}' bilkul download ya install mat karein aur phone mein 'Install unknown apps' band rakhein.")
            elif host:
                recs.append(f"Phishing domain '{host}' ko mat kholein aur wahan netbanking password, UPI PIN ya OTP bilkul enter na karein.")
            else:
                recs.append("Is suspicious link ko mat kholein aur browser mein khud official website ka address type karein.")
            if any(k in low for k in ("telegram.org/bot", "chat_id=")):
                recs.append("Yeh URL Telegram Bot data-exfiltration endpoint lag raha hai — agar koi token ya OTP expose hua hai to turant rotate karein.")
            recs.append("Official app kholein ya verified domain khud type karein; fraud ki report helpline 1930 / cybercrime.gov.in par karein.")
            return recs[:4]
        # Network Hinglish
        ips = _IP_PORT_RE.findall(raw)
        meta = event.metadata or {}
        dst_port = meta.get("dst_port") or (int(ips[1][1]) if len(ips) > 1 and ips[1][1] else None)
        src_ip = meta.get("src_ip") or (ips[0][0] if ips else "")
        dst_ip = meta.get("dst_ip") or (ips[1][0] if len(ips) > 1 else "")
        if dst_port == 22 or "ssh" in low or "brute" in (event.threat_category or "").lower():
            recs.append(f"Source IP {src_ip or 'unknown host'} se TCP/22 par failed SSH logins (/var/log/auth.log) check karein aur key-only auth enforce karein.")
        elif dst_port in (4444, 1337, 6667, 3333, 9050) or "c2" in (event.threat_category or "").lower():
            recs.append(f"{src_ip or 'Local host'} aur {dst_ip or 'remote IP'}:{dst_port or '?'} ke beech outbound C2/beacon traffic block karein aur running processes (`ss -tp`) check karein.")
        elif dst_port == 53 or "dns" in low or "exfil" in low:
            recs.append(f"{src_ip or 'Local host'} se suspicious DNS queries (port 53) tunneling/exfiltration ke liye inspect karein aur external domain block karein.")
        elif "scan" in (event.threat_category or "").lower() or "syn" in low:
            recs.append(f"Scanner IP {src_ip or 'flagged source'} ke probing traffic ko firewall par rate-limit/block karein aur sirf zaroori ports open rakhein.")
        else:
            dst_label = f"{dst_ip}:{dst_port}" if dst_ip and dst_port else (dst_ip or "target")
            recs.append(f"{src_ip or 'Flagged host'} -> {dst_label} ke active connections (`ss -tupn`) aur firewall logs ({event.threat_category}) turant inspect karein.")
        recs.append("PCAP/flow evidence save rakhein aur agar suspicious traffic jari rahe to affected device ko temporarily LAN se isolate karein.")
        return recs[:4]

    # English incident-specific recommendations
    if not is_threat:
        return [
            "No threat indicators triggered; stay cautious with unexpected requests.",
            "Never share OTPs, UPI PINs, CVVs, or passwords over SMS, calls, or unverified links.",
        ]

    recs = []
    if "sms" in src:
        if any(k in low for k in ("digital arrest", "cyber police", "trai", "cbi", "skype", "warrant")):
            recs.append("Police, CBI, and TRAI never conduct 'Digital Arrest' over Skype/video calls — disconnect and block the sender.")
        if apk:
            recs.append(f"Do NOT download or install '{apk}' — unsolicited APK links often carry Android banking/OTP-stealing spyware.")
        if host:
            recs.append(f"Do NOT visit '{host}' or enter KYC, PAN, banking credentials, or OTPs on it.")
        if any(k in low for k in ("anydesk", "teamviewer", "electricity", "bescom", "bill")):
            recs.append("Never install AnyDesk/TeamViewer on a caller's instruction; check bills only via your official utility app.")
        if any(k in low for k in ("*401*", "call forward", "ussd")):
            recs.append("Do NOT dial USSD codes like *401* — they silently forward your incoming OTP calls to an attacker.")
        if not recs:
            recs.append("Do NOT click links, reply with OTP/UPI PIN, or call phone numbers listed inside this message.")
        recs.append("Report cyber fraud in India via helpline 1930, cybercrime.gov.in, or Sanchar Saathi Chakshu.")
        return recs[:4]

    if "url" in src:
        if apk:
            recs.append(f"Block download of '{apk}' from '{host or 'this URL'}' and keep 'Install unknown apps' disabled.")
        elif host:
            recs.append(f"Do NOT visit '{host}' or enter netbanking credentials, UPI PINs, or OTPs on that domain.")
        if any(k in low for k in ("telegram.org/bot", "chat_id=")):
            recs.append("This URL looks like a Telegram Bot exfiltration endpoint — rotate any exposed tokens/OTPs immediately.")
        recs.append("Open the official service app or type the verified domain manually; report fraud at 1930 / cybercrime.gov.in.")
        return recs[:4]

    # Network EN
    ips = _IP_PORT_RE.findall(raw)
    meta = event.metadata or {}
    dst_port = meta.get("dst_port") or (int(ips[1][1]) if len(ips) > 1 and ips[1][1] else None)
    src_ip = meta.get("src_ip") or (ips[0][0] if ips else "")
    dst_ip = meta.get("dst_ip") or (ips[1][0] if len(ips) > 1 else "")

    if dst_port == 22 or "ssh" in low or "brute" in (event.threat_category or "").lower():
        recs.append(f"Check SSH auth logs (/var/log/auth.log) for failed logins from {src_ip or 'the source host'} on TCP/22 and enforce key-only auth.")
    elif dst_port in (4444, 1337, 6667, 3333, 9050) or "c2" in (event.threat_category or "").lower():
        recs.append(f"Block outbound C2/beacon traffic between {src_ip or 'local host'} and {dst_ip or 'remote IP'}:{dst_port or '?'} and inspect running processes (`ss -tp`).")
    elif dst_port == 53 or "dns" in low or "exfil" in low:
        recs.append(f"Inspect suspicious DNS queries from {src_ip or 'local host'} for tunneling/exfiltration and block the external domain.")
    elif "scan" in (event.threat_category or "").lower() or "syn" in low:
        recs.append(f"Rate-limit or firewall-block probing traffic from {src_ip or 'the scanner IP'} and verify only required ports are exposed.")
    else:
        recs.append(f"Inspect active connections (`ss -tupn`) involving {src_ip or 'flagged host'} -> {dst_ip or 'target'} ({event.threat_category}).")
    recs.append("Preserve PCAP/flow evidence and temporarily isolate the host from the LAN if anomalous traffic continues.")
    return recs[:4]


def _has_unsafe_imperative(text: str) -> bool:
    for m in _UNSAFE_ADVICE_RE.finditer(text):
        prefix = text[max(0, m.start() - 36) : m.start()]
        if not _NEGATION_PREFIX_RE.search(prefix):
            return True
    return False


def _has_repetition_loop(text: str) -> bool:
    words = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    if len(words) < 8:
        return False
    # Single-word run >= 5
    run = 1
    for a, b in zip(words, words[1:]):
        if a == b:
            run += 1
            if run >= 5:
                return True
        else:
            run = 1
    # Multi-word phrase loop (2..6 words repeated back-to-back)
    for k in range(2, 14):
        for i in range(0, len(words) - 2 * k + 1):
            if words[i : i + k] == words[i + k : i + 2 * k]:
                return True
    return False



_FAKE_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-]{6,14}\d)")


def _polish_sentences(text: str) -> str:
    """Drop sentences with hallucinated phone numbers (except 1930) or near-duplicate phrasing."""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    kept: list[str] = []
    kept_sets: list[set[str]] = []
    for s in parts:
        st = s.strip()
        if not st:
            continue
        # Drop any sentence containing a hallucinated phone number (unless it is India helpline 1930)
        phones = _FAKE_PHONE_RE.findall(st)
        if any(re.sub(r"\D", "", p) != "1930" and len(re.sub(r"\D", "", p)) >= 7 for p in phones):
            continue
        # Near-duplicate sentence filter (Jaccard >= 0.72)
        wset = set(re.findall(r"[a-z0-9ऀ-ॿ]{3,}", st.lower()))
        if wset and any((len(wset & prev) / max(1, len(wset | prev))) >= 0.72 for prev in kept_sets):
            continue
        kept.append(st)
        kept_sets.append(wset)
    return " ".join(kept).strip()


def validate_advisory_output(
    event: ThreatEvent,
    explanation: str,
    recommendations: list[str],
    language: str = "en",
) -> dict[str, Any]:
    """Validate LLM output against Pipeline A's locked verdict and safety rules."""
    lang = normalize_language(language)
    notes: list[str] = []

    cleaned = _CONTROL_TOKEN_RE.sub("", str(explanation or "")).strip()
    cleaned = re.sub(r"^(assistant|advisor|system)\s*:\s*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    valid = True
    if _has_repetition_loop(cleaned):
        valid = False
        notes.append("rejected_token_repetition_loop")

    cleaned = _polish_sentences(cleaned)

    if len(cleaned) < 48:
        valid = False
        notes.append("rejected_too_short_or_empty")

    is_threat = event.risk_score >= 45 or event.risk_level in {"high", "critical"}
    is_benign = event.risk_score == 0 and event.risk_level == "low"

    if is_threat and _SAFE_CLAIM_RE.search(cleaned):
        valid = False
        notes.append("rejected_verdict_contradiction_claimed_safe_on_threat")

    if is_benign and _PANIC_CLAIM_RE.search(cleaned):
        valid = False
        notes.append("rejected_verdict_contradiction_hallucinated_breach_on_benign")

    if _has_unsafe_imperative(cleaned):
        valid = False
        notes.append("rejected_unsafe_user_instruction")

    # Keep safe Llama-generated incident tips FIRST, then append incident-specific vetted guardrails
    safe_llm_recs: list[str] = []
    for rec in recommendations or []:
        r_clean = _CONTROL_TOKEN_RE.sub("", str(rec)).strip()
        if not r_clean or len(r_clean) < 10:
            continue
        if _has_unsafe_imperative(r_clean):
            notes.append("dropped_unsafe_llm_recommendation")
            continue
        if r_clean not in safe_llm_recs:
            safe_llm_recs.append(r_clean)

    vetted = vetted_recommendations(event, lang)
    merged_recs: list[str] = []
    if valid:
        for r_item in safe_llm_recs[:2]:
            if r_item not in merged_recs:
                merged_recs.append(r_item)
    for r_item in vetted:
        if len(merged_recs) >= 4:
            break
        if r_item not in merged_recs:
            merged_recs.append(r_item)

    if valid and not notes:
        notes.append("verdict_locked_passed")

    return {
        "valid": valid,
        "language": lang,
        "explanation": cleaned,
        "recommendations": merged_recs[:4],
        "notes": notes,
    }
