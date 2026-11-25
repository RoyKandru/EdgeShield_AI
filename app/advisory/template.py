from app.advisory.base import AdvisoryProvider
from app.advisory.validator import normalize_language, vetted_recommendations
from app.schemas import AdvisoryResponse, ThreatEvent

_RISK_HI = {
    "low": "कम (Low)",
    "medium": "मध्यम (Medium)",
    "high": "उच्च (High)",
    "critical": "अत्यधिक गंभीर (Critical)",
}

_SRC_HI = {
    "sms": "SMS / संदेश",
    "url": "URL / लिंक",
    "network": "नेटवर्क ट्रैफ़िक",
}

_RISK_HINGLISH = {
    "low": "Low (Kam)",
    "medium": "Medium (Madhyam)",
    "high": "High (Uchh)",
    "critical": "Critical (Ati-Gambhir)",
}

_SRC_HINGLISH = {
    "sms": "SMS / message",
    "url": "URL / link",
    "network": "network traffic",
}


class TemplateAdvisory(AdvisoryProvider):
    """Deterministic grounded offline fallback supporting English (en), Hindi (hi), and Hinglish (hinglish)."""

    def explain(
        self,
        event: ThreatEvent,
        language: str = "en",
        *,
        fallback_used: bool = False,
        validation_notes: list[str] | None = None,
        provider_name: str = "deterministic-template",
        metrics: dict | None = None,
    ) -> AdvisoryResponse:
        lang = normalize_language(language)
        pos_evidence = [item.description for item in event.evidence if item.weight > 0]
        evidence_text = "; ".join(pos_evidence[:4])

        if lang == "hi":
            src_key = "sms" if "sms" in event.source else "url" if "url" in event.source else "network"
            src_label = _SRC_HI[src_key]
            risk_hi = _RISK_HI.get(event.risk_level, event.risk_level)
            if event.risk_score == 0 and event.risk_level == "low":
                explanation = (
                    f"स्थानीय {src_label} विश्लेषण में इस इनपुट में कोई स्पष्ट स्कैम, फ़िशिंग या नेटवर्क हमले का संकेत नहीं मिला "
                    f"(जोखिम स्कोर: {event.risk_score}/100, स्तर: {risk_hi})। फिर भी किसी अनजान व्यक्ति के साथ OTP या पासवर्ड साझा न करें।"
                )
            else:
                ev_part = evidence_text or event.prediction
                explanation = (
                    f"सावधान! स्थानीय {src_label} जांच ने इस इनपुट को '{event.threat_category}' के रूप में चिह्नित किया है "
                    f"(जोखिम स्तर: {risk_hi}, स्कोर: {event.risk_score}/100)। "
                    f"पाए गए मुख्य संकेत: {ev_part}। नीचे दिए गए सुरक्षित कदमों का पालन करें।"
                )
            note = (
                "सत्यापित हिंदी सुरक्षा सलाह (Verdict-Locked Guardrail Fallback)।"
                if fallback_used
                else "स्थानीय हिंदी टेम्पलेट सलाह (ExecuTorch Llama कॉन्फ़िगर न होने पर सुरक्षित फ़ॉलबैक)।"
            )
        elif lang == "hinglish":
            src_key = "sms" if "sms" in event.source else "url" if "url" in event.source else "network"
            src_label = _SRC_HINGLISH[src_key]
            risk_hl = _RISK_HINGLISH.get(event.risk_level, event.risk_level)
            if event.risk_score == 0 and event.risk_level == "low":
                explanation = (
                    f"Local {src_label} scan mein is input mein koi clear scam, phishing ya network attack ka signal nahi mila "
                    f"(Risk Score: {event.risk_score}/100, Level: {risk_hl}). Phir bhi kisi unknown person ke saath OTP, UPI PIN ya password kabhi share na karein."
                )
            else:
                ev_part = evidence_text or event.prediction
                explanation = (
                    f"Savdhan! Local {src_label} analysis ne is input ko '{event.threat_category}' ke roop mein flag kiya hai "
                    f"(Risk Level: {risk_hl}, Score: {event.risk_score}/100). "
                    f"Main verified signals: {ev_part}. Neeche diye gaye safe steps turant follow karein."
                )
            note = (
                "Verified Hinglish safety advisory (Verdict-Locked Guardrail Fallback)."
                if fallback_used
                else "Local Hinglish template advisory (offline deterministic fallback)."
            )
        else:
            if not evidence_text:
                evidence_text = "No configured indicator was triggered."
            explanation = (
                f"The local {event.source} analysis classified this input as "
                f"'{event.prediction}' ({event.threat_category}) with a {event.risk_level} risk score of {event.risk_score}/100. "
                f"Verified evidence: {evidence_text}."
            )
            note = (
                "Verdict-locked deterministic fallback triggered by advisory output validator."
                if fallback_used
                else "Deterministic offline fallback. Configure Llama-3.2-1B via ExecuTorch for LLM narratives."
            )

        return AdvisoryResponse(
            event_id=event.event_id,
            provider=provider_name,
            language=lang,
            explanation=explanation,
            recommendations=vetted_recommendations(event, lang),
            generated_locally=True,
            runtime_note=note,
            validated=True,
            fallback_used=fallback_used,
            prompt_injection_guarded=True,
            validation_notes=list(validation_notes or ["verdict_locked_template"]),
            metrics=dict(metrics or {}),
        )
