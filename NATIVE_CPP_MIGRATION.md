# Native C++ Core — v1.2.0 migration foundation

This release adds a C++20 core foundation; it does **not** claim that all existing Python subsystems have been ported.

## Build on Debian 12 / ARM64
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
cd native
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## Smoke tests
```bash
./build/edgeshield_native_demo url 'http://secure-login-example.com/verify'
./build/edgeshield_native_demo sms 'Urgent: verify your account and share OTP'
./build/edgeshield_native_demo monitor unused
```

## What is native now
- C++20 shared contracts for threat events and evidence.
- C++ URL and SMS lexical baseline detectors (demonstrator only; not validated classifiers).
- C++ risk-fusion starter.
- Linux System Monitor starter reading memory, root filesystem and thermal zone where exposed.
- CMake build and native smoke tests.

## What remains Python / not yet ported
- Existing FastAPI API and dashboard, SQLite persistence, PCAP/Scapy network pipeline, SCI ledger/engine, adaptive candidate learner and ExecuTorch adapter remain in Python.
- The C++ System Monitor is not yet wired into the dashboard API.
- The native network detector, full event persistence, service supervisor, RKNN C++ inference and Llama/ExecuTorch C++ inference are not implemented here.

## Migration rules
1. Keep the v1.1.0 Python app runnable as a reference/compatibility prototype.
2. Freeze versioned API/event contracts before replacing implementations.
3. Port one module at a time behind interfaces; maintain golden test vectors across Python and C++.
4. Use C++ for production edge runtime, data plane, SCI, telemetry and accelerator adapters.
5. Keep Python for training, dataset engineering, experiments and benchmark orchestration.
6. Do not claim model accuracy, NPU use, ExecuTorch readiness or challenge compliance without board evidence.
