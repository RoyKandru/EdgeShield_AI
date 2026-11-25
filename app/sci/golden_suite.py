"""Pinned Verified Golden Regression Suite for SCI Stage 7 Validation Engine.

Every proposed SCI policy/intel/weight/calibration correction MUST pass this
verified benign + malicious test bench on the Arm board before deployment.
Runs in <0.35s on RK3588J.
"""
from __future__ import annotations

GOLDEN_CASES: list[dict] = [
    # --- Verified Benign URLs (must NOT flag) ---
    {"id": "g-url-b1", "channel": "url", "expected_flagged": False, "critical": False, "input": "https://www.sbi.co.in/"},
    {"id": "g-url-b2", "channel": "url", "expected_flagged": False, "critical": False, "input": "https://uidai.gov.in/en/my-aadhaar/get-aadhaar"},
    {"id": "g-url-b3", "channel": "url", "expected_flagged": False, "critical": False, "input": "https://www.wikipedia.org/"},
    {"id": "g-url-b4", "channel": "url", "expected_flagged": False, "critical": False, "input": "https://www.hdfcbank.com/personal/pay/cards"},
    {"id": "g-url-b5", "channel": "url", "expected_flagged": False, "critical": False, "input": "https://github.com/pytorch/executorch"},
    {"id": "g-url-b6", "channel": "url", "expected_flagged": False, "critical": False, "input": "https://incometax.gov.in/iec/foportal/"},
    # --- Verified Malicious URLs (must flag; critical=True blocks any regression) ---
    {"id": "g-url-m1", "channel": "url", "expected_flagged": True, "critical": True, "input": "https://sim-sbi-yono-portal.pages.dev/kyc/verify-otp"},
    {"id": "g-url-m2", "channel": "url", "expected_flagged": True, "critical": True, "input": "https://api.telegram.org/bot99887766:AAFake/sendMessage?chat_id=1&text=OTP"},
    {"id": "g-url-m3", "channel": "url", "expected_flagged": True, "critical": True, "input": "http://wedding-invite-download.xyz/ShubhVivah_Card.apk"},
    {"id": "g-url-m4", "channel": "url", "expected_flagged": True, "critical": True, "input": "https://rbi-refund-claim.zip/claim"},
    {"id": "g-url-m5", "channel": "url", "expected_flagged": True, "critical": True, "input": "https://paypa1-account-alert.ngrok-free.app/signin"},
    {"id": "g-url-m6", "channel": "url", "expected_flagged": True, "critical": True, "input": "https://sbi.co.in\u2044evil-kyc.top/login"},
    {"id": "g-url-m7", "channel": "url", "expected_flagged": True, "critical": True, "input": "http://192.0.2.10/urgent/bank/verify-account"},
    {"id": "g-url-m8", "channel": "url", "expected_flagged": True, "critical": True, "input": "https://rb.gy/sbi-kyc-9x2"},

    # --- Verified Benign SMS (must NOT flag) ---
    {"id": "g-sms-b1", "channel": "sms", "expected_flagged": False, "critical": False, "input": "849201 is your OTP for Rs. 450.00 transaction at Swiggy on HDFC Bank Card XX0000. Valid for 10 mins. Do NOT share this OTP with anyone."},
    {"id": "g-sms-b2", "channel": "sms", "expected_flagged": False, "critical": False, "input": "Your one-time password for logging into IRCTC is 539104. Never share your OTP or password with anyone, including bank officials."},
    {"id": "g-sms-b3", "channel": "sms", "expected_flagged": False, "critical": False, "input": "RBI Kehta Hai: Never share your OTP, UPI PIN, CVV, or download screen-sharing apps like AnyDesk for receiving money. Stay cyber safe."},
    {"id": "g-sms-b4", "channel": "sms", "expected_flagged": False, "critical": False, "input": "Rs 1,250.00 debited from A/c XX4321 on 23-Nov via UPI Ref 4328192011. Avl Bal Rs 24,800.50."},
    {"id": "g-sms-b5", "channel": "sms", "expected_flagged": False, "critical": False, "input": "Hey, are we still meeting for lunch at 1 PM near Indiranagar metro?"},
    {"id": "g-sms-b6", "channel": "sms", "expected_flagged": False, "critical": False, "input": "Your Amazon order #402-1928374 is out for delivery today by 7 PM."},
    # --- Verified Malicious SMS (must flag) ---
    {"id": "g-sms-m1", "channel": "sms", "expected_flagged": True, "critical": True, "input": "TRAI Alert: Your mobile will be disconnected in 2 hours. Join Skype with Mumbai Cyber Police or Digital Arrest warrant will be issued."},
    {"id": "g-sms-m2", "channel": "sms", "expected_flagged": True, "critical": True, "input": "Dear SBI customer, your YONO KYC is suspended today. Update PAN at https://sim-sbi-yono-portal.pages.dev/kyc and share OTP."},
    {"id": "g-sms-m3", "channel": "sms", "expected_flagged": True, "critical": True, "input": "BESCOM Electricity Alert: Power will be disconnected tonight at 9:30 PM because last month bill was not updated. Install AnyDesk now."},
    {"id": "g-sms-m4", "channel": "sms", "expected_flagged": True, "critical": True, "input": "Alert: Dial *401*9876543210# now to confirm courier parcel delivery."},
    {"id": "g-sms-m5", "channel": "sms", "expected_flagged": True, "critical": True, "input": "चेतावनी: आपका एसबीआई बैंक खाता आज बंद हो जाएगा। तुरंत केवाईसी अपडेट करें http://hdfc-kyc-update.top/kyc"},
    {"id": "g-sms-m6", "channel": "sms", "expected_flagged": True, "critical": True, "input": "U R G E N T: Your acc0unt is b1ocked. Sh\u200bare your 0 T P now at https://rbi-refund-claim.zip/verify"},
    {"id": "g-sms-m7", "channel": "sms", "expected_flagged": True, "critical": True, "input": "Namaste! Open family wedding invitation card: http://wedding-invite-download.xyz/ShubhVivah_Card.apk"},
    {"id": "g-sms-m8", "channel": "sms", "expected_flagged": True, "critical": True, "input": "Congrats! Earn Rs 5000 daily liking YouTube videos from home. Join Telegram prepaid task group."},

    # --- Verified Benign Network Flows (must NOT flag) ---
    {"id": "g-net-b1", "channel": "network", "expected_flagged": False, "critical": False, "input": {"src_ip": "192.168.1.10", "dst_ip": "93.184.216.34", "src_port": 51200, "dst_port": 443, "protocol": "TCP", "packets": 18, "bytes": 14000, "duration_seconds": 3.2, "syn_count": 1}},
    {"id": "g-net-b2", "channel": "network", "expected_flagged": False, "critical": False, "input": {"src_ip": "192.168.1.10", "dst_ip": "8.8.8.8", "src_port": 54321, "dst_port": 53, "protocol": "UDP", "packets": 2, "bytes": 160, "duration_seconds": 0.04, "syn_count": 0}},
    {"id": "g-net-b3", "channel": "network", "expected_flagged": False, "critical": False, "input": {"src_ip": "192.168.1.10", "dst_ip": "192.168.1.6", "src_port": 50211, "dst_port": 22, "protocol": "TCP", "packets": 25, "bytes": 4200, "duration_seconds": 120.0, "syn_count": 1, "failed_connections": 0}},
    # --- Verified Malicious Network Flows (must flag) ---
    {"id": "g-net-m1", "channel": "network", "expected_flagged": True, "critical": True, "input": {"src_ip": "198.51.100.9", "dst_ip": "192.168.1.6", "src_port": 42000, "dst_port": 22, "protocol": "TCP", "packets": 60, "bytes": 4200, "duration_seconds": 2.0, "syn_count": 30, "failed_connections": 18}},
    {"id": "g-net-m2", "channel": "network", "expected_flagged": True, "critical": True, "input": {"src_ip": "192.168.1.40", "dst_ip": "203.0.113.77", "src_port": 51020, "dst_port": 4444, "protocol": "TCP", "packets": 40, "bytes": 4800, "duration_seconds": 30.0, "syn_count": 2}},
    {"id": "g-net-m3", "channel": "network", "expected_flagged": True, "critical": True, "input": {"src_ip": "192.168.1.40", "dst_ip": "198.51.100.53", "src_port": 54012, "dst_port": 53, "protocol": "UDP", "packets": 80, "bytes": 32000, "duration_seconds": 10.0, "dns_query": "4a9f8c2e1b7d6a0f9e3c5b2a8d1f4e7c.exfil.sim-c2.xyz"}},
    {"id": "g-net-m4", "channel": "network", "expected_flagged": True, "critical": True, "input": {"src_ip": "192.168.1.40", "dst_ip": "203.0.113.88", "src_port": 53010, "dst_port": 3333, "protocol": "TCP", "packets": 50, "bytes": 6200, "duration_seconds": 60.0, "syn_count": 1}},
    {"id": "g-net-m5", "channel": "network", "expected_flagged": True, "critical": True, "input": {"src_ip": "198.51.100.44", "dst_ip": "192.168.1.6", "src_port": 0, "dst_port": 0, "protocol": "ICMP", "packets": 15, "bytes": 9600, "duration_seconds": 1.0, "syn_count": 0}},
]
