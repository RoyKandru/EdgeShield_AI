import json
from pathlib import Path

flows = [
    {
        "src_ip": "192.168.1.10", "dst_ip": "192.168.1.1",
        "src_port": 50122, "dst_port": 443, "protocol": "TCP",
        "packets": 24, "bytes": 18000, "duration_seconds": 8.2, "syn_count": 1
    },
    {
        "src_ip": "192.168.1.55", "dst_ip": "192.168.1.1",
        "src_port": 41000, "dst_port": 22, "protocol": "TCP",
        "packets": 35, "bytes": 4200, "duration_seconds": 1.2, "syn_count": 24
    },
]
Path("data/sample_flows.json").write_text(json.dumps(flows, indent=2))
print("Wrote data/sample_flows.json")
