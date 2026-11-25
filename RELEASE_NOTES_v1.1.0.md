# EdgeShield AI v1.1.0 — System Monitor

## Added
- New System Monitor tab with live CPU, RAM, swap, root filesystem, network interface throughput, CPU frequency, thermal sensors, load average and uptime.
- Best-effort NPU/GPU availability and utilisation reporting from Linux devfreq/sysfs interfaces.
- Read-only `/api/system/metrics` endpoint.
- Two-second polling while the System Monitor tab is open; polling stops when leaving the tab.

## Important telemetry semantics
- CPU and network rates are delta-based; the first sample is a baseline.
- NPU/GPU utilisation is displayed only if the kernel exports a readable counter. A device node alone is not a utilisation measurement.
- Thermal and frequency values depend on the board kernel's exported interfaces.
- The monitor does not alter clocks, governors, power modes, fans or hardware state.
