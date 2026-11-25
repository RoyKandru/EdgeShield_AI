"""Opt-in Isolated Link Sandbox & HTML Phishing Form Inspector.

Strict Offline Policy:
  - Live network fetching is DISABLED by default (EDGESHIELD_SANDBOX_ENABLED=false)
    so EdgeShield operates 100% offline with zero outbound packets.
  - Pure-offline HTML snapshot inspection (passing `html_content`) always works locally.
  - When explicitly enabled by an operator (EDGESHIELD_SANDBOX_ENABLED=true), live fetch
    enforces SSRF protection (blocks RFC1918, loopback, link-local, metadata IPs), strips
    cookies, executes zero JavaScript, caps redirects (5) & body size (64 KB), and detects
    credential/OTP/UPI/Aadhaar/PAN phishing forms, Telegram bot exfil, and brand spoofing.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

from app.config import settings

_BRANDS = {
    "sbi": ("sbi.co.in", "onlinesbi.sbi"),
    "hdfc": ("hdfcbank.com",),
    "icici": ("icicibank.com",),
    "axis bank": ("axisbank.com",),
    "paytm": ("paytm.com",),
    "phonepe": ("phonepe.com",),
    "epfo": ("epfindia.gov.in",),
    "uidai": ("uidai.gov.in",),
    "aadhaar": ("uidai.gov.in",),
    "rbi": ("rbi.org.in",),
    "income tax": ("incometax.gov.in",),
    "paypal": ("paypal.com",),
    "amazon": ("amazon.in", "amazon.com"),
}

_SENSITIVE_FIELD_RE = re.compile(
    r"\b(otp|upi\s*pin|mpin|cvv|card\s*number|aadhaar|aadhar|pan\s*number|netbanking|password|passcode|debit\s*card|atm\s*pin)\b",
    re.IGNORECASE,
)
_TELEGRAM_EXFIL_RE = re.compile(r"api\.telegram\.org/bot[0-9a-zA-Z:_-]+", re.IGNORECASE)


class _FormAndTitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title_chunks: list[str] = []
        self.form_actions: list[str] = []
        self.has_password_input = False
        self.input_labels: list[str] = []
        self.text_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        ad = {k.lower(): (v or "") for k, v in attrs}
        t = tag.lower()
        if t == "title":
            self.in_title = True
        elif t == "form":
            if ad.get("action"):
                self.form_actions.append(ad["action"])
        elif t == "input":
            itype = ad.get("type", "").lower()
            if itype == "password":
                self.has_password_input = True
            for key in ("name", "id", "placeholder", "aria-label"):
                if ad.get(key):
                    self.input_labels.append(ad[key])

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        s = data.strip()
        if not s:
            return
        if self.in_title:
            self.title_chunks.append(s)
        elif len(self.text_chunks) < 200:
            self.text_chunks.append(s[:160])


def _assert_public_host(host: str) -> None:
    if not host:
        raise ValueError("Missing hostname")
    h = host.strip().lower().rstrip(".")
    if h in {"localhost", "metadata.google.internal"} or h.endswith((".local", ".internal", ".lan")):
        raise ValueError(f"SSRF safeguard blocked non-public host: {host}")
    try:
        ip = ipaddress.ip_address(h)
        if not ip.is_global:
            raise ValueError(f"SSRF safeguard blocked non-global IP: {host}")
        return
    except ValueError as exc:
        if "SSRF safeguard" in str(exc):
            raise
    infos = socket.getaddrinfo(h, None, proto=socket.IPPROTO_TCP)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError(f"SSRF safeguard blocked resolved non-global IP {ip} for {host}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def analyze_html_snapshot(url: str, html: str, redirect_chain: list[str] | None = None) -> dict[str, Any]:
    """Pure offline static HTML inspection (never executes JavaScript)."""
    parsed = urlparse(url)
    final_host = (parsed.hostname or "").lower().rstrip(".")
    parser = _FormAndTitleParser()
    parser.feed((html or "")[:65536])

    title = " ".join(parser.title_chunks).strip()[:140]
    combined = " ".join([title] + parser.input_labels + parser.text_chunks)
    sensitive_hits = sorted({m.group(0).lower() for m in _SENSITIVE_FIELD_RE.finditer(combined)})
    tg_hits = _TELEGRAM_EXFIL_RE.findall(html or "")

    findings: list[dict[str, Any]] = []
    score = 0

    if tg_hits:
        score += 55
        findings.append({"code": "SANDBOX_TELEGRAM_BOT_EXFIL", "weight": 55, "detail": "Page submits or embeds a Telegram Bot API exfiltration endpoint."})

    if parser.has_password_input:
        score += 20
        findings.append({"code": "SANDBOX_PASSWORD_FORM", "weight": 20, "detail": "Landing page renders a password input field."})

    if sensitive_hits:
        w = min(45, 15 * len(sensitive_hits))
        score += w
        findings.append({"code": "SANDBOX_SENSITIVE_FIELDS", "weight": w, "detail": f"Requests sensitive financial/identity fields: {', '.join(sensitive_hits[:5])}."})

    # Check brand impersonation: title claims a known Indian/global brand on an unrelated domain
    low_title = title.lower()
    for brand, legit_domains in _BRANDS.items():
        if brand in low_title:
            if not any(final_host == d or final_host.endswith("." + d) for d in legit_domains):
                score += 40
                findings.append({"code": "SANDBOX_BRAND_IMPERSONATION", "weight": 40, "detail": f"Page title references '{brand}' on non-official host '{final_host}'."})
                break

    # Cross-domain form submission
    for act in parser.form_actions:
        if act.startswith(("http://", "https://")):
            act_host = (urlparse(act).hostname or "").lower().rstrip(".")
            if act_host and act_host != final_host:
                score += 25
                findings.append({"code": "SANDBOX_CROSS_DOMAIN_FORM", "weight": 25, "detail": f"Form posts data to external domain '{act_host}'."})
                break

    chain = redirect_chain or [url]
    if len(chain) >= 3:
        score += 15
        findings.append({"code": "SANDBOX_MULTI_REDIRECT", "weight": 15, "detail": f"Multi-hop redirect chain ({len(chain)} hops) ending at '{final_host}'."})

    score = min(100, score)
    verdict = "phishing_confirmed_by_page" if score >= 55 else ("suspicious_page" if score >= 25 else "no_page_phishing_cues")
    return {
        "final_url": url,
        "final_host": final_host,
        "redirect_chain": chain,
        "page_title": title,
        "has_password_input": parser.has_password_input,
        "sensitive_fields_detected": sensitive_hits,
        "telegram_exfil_detected": bool(tg_hits),
        "sandbox_risk_score": score,
        "verdict": verdict,
        "findings": findings,
    }


def inspect_url_sandbox(url: str, *, html_content: str | None = None, allow_network: bool = False) -> dict[str, Any]:
    """Inspect a link offline (if html_content given) or via SSRF-safe isolated fetch when enabled."""
    if html_content is not None:
        res = analyze_html_snapshot(url, html_content, redirect_chain=[url])
        res["mode"] = "offline_html_snapshot"
        return res

    if not allow_network or not settings.sandbox_enabled:
        raise PermissionError(
            "Live network link sandbox is disabled (strict offline mode). "
            "Pass html_content for offline analysis, or set EDGESHIELD_SANDBOX_ENABLED=true and allow_network=true."
        )

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported by the link sandbox")

    opener = urllib.request.build_opener(_NoRedirect())
    cur = url
    chain = [cur]
    body = ""
    status_code = 0

    for _ in range(5):
        p = urlparse(cur)
        _assert_public_host(p.hostname or "")
        req = urllib.request.Request(
            cur,
            headers={"User-Agent": "EdgeShield-IsolatedSandbox/1.4 (no-js; no-cookies)", "Accept": "text/html,*/*;q=0.1"},
        )
        try:
            with opener.open(req, timeout=4.0) as resp:
                status_code = int(getattr(resp, "status", 200))
                body = resp.read(65536).decode("utf-8", errors="replace")
                break
        except urllib.error.HTTPError as he:
            status_code = int(he.code)
            loc = he.headers.get("Location")
            if status_code in (301, 302, 303, 307, 308) and loc:
                cur = urljoin(cur, loc)
                chain.append(cur)
                continue
            body = he.read(65536).decode("utf-8", errors="replace") if he.fp else ""
            break

    res = analyze_html_snapshot(cur, body, redirect_chain=chain)
    res["mode"] = "isolated_live_fetch"
    res["http_status"] = status_code
    return res
