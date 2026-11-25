# EdgeBLE 6B / RK3588J — Board Bring-up

The native C++ service is intended to build directly on Debian 12 ARM64. Use the Debian system compiler and CMake; Python/Miniforge is not required for the native runtime.

## 1. Check board and free storage

```bash
uname -m
cat /etc/os-release
free -h
df -h /
```

Expected architecture: `aarch64`. Ensure several hundred MB of free storage for build tools and artifacts; do not start large model downloads while the root filesystem is nearly full.

## 2. Install build tools

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
# Optional, recommended for SHA-256 ledger:
sudo apt-get install -y libssl-dev
```

## 3. Build and test

```bash
cd ~/edgeshield-ai
chmod +x scripts/build_native.sh scripts/run_native.sh
./scripts/build_native.sh
```

## 4. Run local service

```bash
./build/edgeshieldd serve 8080
```

Open `http://127.0.0.1:8080` on the board. To reach it from a trusted LAN client, explicitly use `./build/edgeshieldd serve 8080 0.0.0.0` and configure a firewall. The service has no authentication/TLS.

## 5. Smoke-test APIs

```bash
curl http://127.0.0.1:8080/api/health
curl http://127.0.0.1:8080/api/system/metrics
curl -X POST http://127.0.0.1:8080/api/analyze/url \
  -H 'Content-Type: application/json' \
  -d '{"url":"http://secure-login-example.com/verify"}'
curl -X POST http://127.0.0.1:8080/api/analyze/sms \
  -H 'Content-Type: application/json' \
  -d '{"message":"Urgent: verify your account and share OTP"}'
curl http://127.0.0.1:8080/api/sci/incidents
```

## 6. Validate PCAP input

```bash
./build/edgeshieldd pcap /path/to/sample.pcap
```

Supported subset: classic PCAP, Ethernet link type, IPv4, TCP/UDP and one 802.1Q VLAN tag. PCAPNG, IPv6 and live capture are not supported by this reader.

## 7. Capture benchmark evidence

Record board model, OS/kernel, compiler, build flags, input corpus hash, number of samples, warm-up count, repetition count, median/p95 latency, throughput, CPU/RAM before and during test, thermal state and whether inference was CPU or accelerator. Do not extrapolate desktop measurements to RK3588J.

## 8. Mandatory model gate

The current code does not include a working Llama + ExecuTorch C++ model runner. Before claiming challenge compliance, the SoC/AI owner must validate the official challenge-approved Llama-family model, exact tokenizer, `.pte` export, ExecuTorch C++ runtime build, generation loop and offline execution on the board. Record model/runtime versions and measured latency/RAM. RKNN runtime validation is a separate accelerator path and does not replace this requirement.
