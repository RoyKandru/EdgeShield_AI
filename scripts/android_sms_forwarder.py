#!/usr/bin/env python3
"""Zero-dependency Android Termux -> EdgeShield RK3588J Gateway SMS Forwarder.

Runs offline over your local home/office Wi-Fi LAN: polls recent SMS via `termux-sms-list`
(Termux:API) and forwards new incoming messages to EdgeShield's NPU Core 2 SMS detector
(`POST http://<GATEWAY_IP>:8000/api/sms/analyze`). Alerts locally via `termux-notification`
if risk_score >= 50.

Usage (in Termux on Android):
  pkg install python termux-api
  python3 android_sms_forwarder.py --gateway 192.168.1.6 --poll 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request


def fetch_recent_sms(limit: int = 10) -> list[dict]:
    out = subprocess.check_output(["termux-sms-list", "-l", str(limit), "-t", "inbox"], text=True, timeout=10)
    return json.loads(out)


def notify_phone(title: str, content: str) -> None:
    try:
        subprocess.run(["termux-notification", "--title", title, "--content", content[:240], "--priority", "high"], check=False)
    except Exception:
        print(f"[ALERT] {title}: {content}")


def scan_on_gateway(gateway: str, port: int, message: str, token: str = "") -> dict:
    url = f"http://{gateway}:{port}/api/sms/analyze"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-EdgeShield-Token"] = token
    req = urllib.request.Request(url, data=json.dumps({"message": message}).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Forward Android SMS to EdgeShield AI LAN Gateway")
    ap.add_argument("--gateway", required=True, help="EdgeShield RK3588J LAN IP (e.g. 192.168.1.6)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default="", help="Optional X-EdgeShield-Token")
    ap.add_argument("--poll", type=float, default=10.0, help="Poll interval in seconds")
    args = ap.parse_args()

    seen: set[int] = set()
    print(f"EdgeShield Mobile Forwarder active -> http://{args.gateway}:{args.port} (LAN-only, offline)")
    while True:
        try:
            for sms in fetch_recent_sms():
                sid = int(sms.get("_id", 0))
                if sid in seen:
                    continue
                seen.add(sid)
                body = str(sms.get("body", "")).strip()
                if not body:
                    continue
                res = scan_on_gateway(args.gateway, args.port, body, args.token)
                score = int(res.get("risk_score", 0))
                if score >= 50:
                    notify_phone(
                        f"EdgeShield Alert ({res.get('risk_level','high').upper()} {score}/100)",
                        f"{res.get('threat_category')} from {sms.get('number','unknown')}: {body[:140]}",
                    )
        except Exception as exc:
            print(f"[warn] poll error: {exc}")
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
