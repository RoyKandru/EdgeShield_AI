# Android Smartphone Companion Integration (LAN-Only SMS Forwarding)

EdgeShield AI is an **Arm64 Linux Edge Security Gateway** (RK3588J) protecting an entire home, SMB, or IoT LAN. While network intrusions, DNS phishing, HTTP URLs, and SIP/wire SMS are captured passively off the LAN NIC (`bin/live_sniffer`), cellular SMS arriving on an Android phone on the same Wi-Fi LAN can be forwarded directly to EdgeShield's **NPU Core 2 (`SplitBiGRU+MaskedPool`)** with zero cloud involvement.

## Option 1: Termux + Termux:API (Zero Dependencies)
```bash
pkg install python termux-api
python3 scripts/android_sms_forwarder.py --gateway 192.168.1.6 --port 8000
```
New incoming SMS messages are scored locally by the RK3588J gateway over Wi-Fi; threats (`risk_score >= 50`) trigger an immediate high-priority Android notification.

## Option 2: MacroDroid / Tasker Local Webhook
- **Trigger:** SMS Received (Any Sender)
- **Action:** HTTP POST → `http://<BOARD_LAN_IP>:8000/api/sms/analyze`
- **Headers:** `Content-Type: application/json` *(add `X-EdgeShield-Token: <token>` if auth enabled)*
- **Body:** `{"message": "{sms_message}"}`
