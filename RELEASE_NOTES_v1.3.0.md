# EdgeShield AI v1.3.0

- Added native C++20 HTTP service and dashboard.
- Added URL, SMS and network flow heuristic endpoints.
- Added classic Ethernet/IPv4 TCP/UDP PCAP reader CLI.
- Added Linux System Monitor with CPU per-core sampling, RAM, root storage, thermal zone 0, load, uptime and network interface counters.
- Added SCI incident ledger with SHA-256 when OpenSSL is available and explicit non-cryptographic fallback otherwise.
- Retained Python prototype and prior scripts for compatibility.
- Added deployment and hardware-validation documentation.

Known gaps: trained/calibrated detectors; live capture/PCAPNG/IPv6; mandatory Llama+ExecuTorch C++ runtime; RKNN C++ model backend; board-specific NPU/GPU counters; authenticated/TLS API hardening.
