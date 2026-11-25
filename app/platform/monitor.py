"""Read-only Linux resource telemetry for the EdgeShield system monitor.

All measurements are best-effort and sourced from /proc, /sys and the filesystem.
Unavailable hardware counters are reported as unavailable; they are never fabricated.
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import time
from pathlib import Path
from typing import Any

_PROC_STAT = Path("/proc/stat")
_PROC_MEMINFO = Path("/proc/meminfo")
_PROC_NETDEV = Path("/proc/net/dev")
_PROC_UPTIME = Path("/proc/uptime")
_SYS = Path("/sys")
_last_cpu: dict[str, tuple[int, int]] = {}
_last_net: dict[str, tuple[int, int, float]] = {}
_last_sample = 0.0


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (OSError, PermissionError):
        return None


def _cpu_counters() -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    raw = _read_text(_PROC_STAT)
    if not raw:
        return result
    for line in raw.splitlines():
        parts = line.split()
        if not parts or not (parts[0] == "cpu" or parts[0].startswith("cpu") and parts[0][3:].isdigit()):
            continue
        try:
            values = [int(x) for x in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            result[parts[0]] = (sum(values), idle)
        except (ValueError, IndexError):
            continue
    return result


def _cpu_metrics() -> dict[str, Any]:
    global _last_cpu
    current = _cpu_counters()
    usage: dict[str, float | None] = {}
    for name, (total, idle) in current.items():
        previous = _last_cpu.get(name)
        if previous:
            dt, di = total - previous[0], idle - previous[1]
            usage[name] = round(max(0.0, min(100.0, 100.0 * (dt - di) / dt)), 1) if dt > 0 else None
        else:
            usage[name] = None
    _last_cpu = current
    cores = [v for k, v in usage.items() if k != "cpu"]
    known = [v for v in cores if v is not None]
    return {
        "usage_percent": round(sum(known) / len(known), 1) if known else None,
        "per_core_percent": usage,
        "logical_cores": len(cores),
        "note": "CPU utilisation is calculated from /proc/stat deltas; first sample is warming up."
    }


def _memory_metrics() -> dict[str, Any]:
    values: dict[str, int] = {}
    raw = _read_text(_PROC_MEMINFO) or ""
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            values[key] = int(value.strip().split()[0]) * 1024
        except (ValueError, IndexError):
            pass
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = max(0, total - available)
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    return {
        "total_bytes": total, "used_bytes": used, "available_bytes": available,
        "usage_percent": round(100 * used / total, 1) if total else None,
        "swap_total_bytes": swap_total, "swap_used_bytes": max(0, swap_total-swap_free),
        "swap_usage_percent": round(100 * (swap_total-swap_free) / swap_total, 1) if swap_total else 0.0,
    }


def _disk_metrics() -> dict[str, Any]:
    usage = shutil.disk_usage("/")
    return {
        "mount": "/", "total_bytes": usage.total, "used_bytes": usage.used,
        "free_bytes": usage.free,
        "usage_percent": round(100 * usage.used / usage.total, 1) if usage.total else None,
    }


def _temperature_metrics() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    # Linux thermal zones (millidegrees Celsius).
    for path in sorted((_SYS / "class/thermal").glob("thermal_zone*")):
        raw = _read_text(path / "temp")
        if raw is None:
            continue
        try:
            value = float(raw)
            if abs(value) > 1000:
                value /= 1000
            if -20 <= value <= 150:
                found.append({"sensor": path.name, "temperature_c": round(value, 1), "source": "thermal_zone"})
        except ValueError:
            pass
    # hwmon provides labels on many board vendor kernels.
    for path in sorted((_SYS / "class/hwmon").glob("hwmon*")):
        for inp in sorted(path.glob("temp*_input")):
            raw = _read_text(inp)
            if raw is None:
                continue
            try:
                value = float(raw) / 1000
                label = _read_text(path / inp.name.replace("_input", "_label")) or inp.name
                item = {"sensor": label, "temperature_c": round(value, 1), "source": "hwmon"}
                if not any(x["sensor"] == label and x["temperature_c"] == item["temperature_c"] for x in found):
                    found.append(item)
            except ValueError:
                pass
    return found


def _frequency_metrics() -> dict[str, Any]:
    values = []
    for path in (_SYS / "devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        raw = _read_text(path)
        try:
            if raw:
                values.append(int(raw) / 1000)
        except ValueError:
            pass
    return {
        "average_mhz": round(sum(values) / len(values), 0) if values else None,
        "per_core_mhz": [round(x, 0) for x in values],
    }


def _devfreq_metrics(kind: str) -> dict[str, Any]:
    candidates = []
    for path in (_SYS / "class/devfreq").glob("*"):
        name = path.name.lower()
        if kind in name:
            load = _read_text(path / "load")
            cur = _read_text(path / "cur_freq")
            maxf = _read_text(path / "max_freq")
            busy = None
            if load:
                # Accept only explicit percentages or two-counter busy/total formats.
                import re
                percent_match = re.search(r"(\\d+(?:\\.\\d+)?)\\s*%", load)
                if percent_match:
                    busy = round(max(0.0, min(100.0, float(percent_match.group(1)))), 1)
                else:
                    counters = re.search(r"(\\d+)\\s*[/, ]\\s*(\\d+)", load)
                    if counters:
                        busy_ticks, total_ticks = int(counters.group(1)), int(counters.group(2))
                        if total_ticks > 0 and busy_ticks <= total_ticks:
                            busy = round(100.0 * busy_ticks / total_ticks, 1)
            candidates.append({
                "device": path.name, "available": True, "busy_percent": busy,
                "current_frequency_hz": int(cur) if cur and cur.isdigit() else None,
                "max_frequency_hz": int(maxf) if maxf and maxf.isdigit() else None,
                "raw_load": load,
            })
    node_candidates = {
        "npu": [Path("/dev/dri/renderD129"), Path("/dev/rknpu")],
        "gpu": [Path("/dev/dri/renderD128"), Path("/dev/mali0")],
    }
    node_present = any(p.exists() for p in node_candidates[kind])
    if not candidates:
        return {"available": node_present, "busy_percent": None, "devices": [],
                "note": "Device node detected; utilisation counter is not exposed by this kernel." if node_present
                else "No matching device or devfreq telemetry interface detected."}
    return {"available": True, "busy_percent": next((x["busy_percent"] for x in candidates if x["busy_percent"] is not None), None),
            "devices": candidates, "note": "Busy percentage is shown only when exported by the kernel."}


def _network_metrics() -> list[dict[str, Any]]:
    global _last_net
    now = time.monotonic()
    result = []
    raw = _read_text(_PROC_NETDEV) or ""
    current: dict[str, tuple[int, int]] = {}
    for line in raw.splitlines()[2:]:
        if ":" not in line:
            continue
        name, stats = line.split(":", 1)
        parts = stats.split()
        if len(parts) < 9:
            continue
        try:
            current[name.strip()] = (int(parts[0]), int(parts[8]))
        except ValueError:
            continue
    for name, (rx, tx) in current.items():
        old = _last_net.get(name)
        rx_rate = tx_rate = None
        if old and now > old[2]:
            elapsed = now - old[2]
            rx_rate = round(max(0, rx - old[0]) / elapsed, 1)
            tx_rate = round(max(0, tx - old[1]) / elapsed, 1)
        result.append({"interface": name, "rx_bytes": rx, "tx_bytes": tx,
                       "rx_bytes_per_sec": rx_rate, "tx_bytes_per_sec": tx_rate})
    _last_net = {k: (v[0], v[1], now) for k, v in current.items()}
    return result


def _load_uptime() -> dict[str, Any]:
    raw = _read_text(_PROC_UPTIME)
    uptime = float(raw.split()[0]) if raw else None
    load = _read_text(Path("/proc/loadavg"))
    loads = []
    if load:
        try:
            loads = [float(x) for x in load.split()[:3]]
        except ValueError:
            loads = []
    return {"uptime_seconds": round(uptime, 1) if uptime is not None else None, "load_average": loads}


def collect_system_metrics() -> dict[str, Any]:
    """Return one read-only resource snapshot. Call periodically for delta metrics."""
    global _last_sample
    now = time.time()
    cpu = _cpu_metrics()
    memory = _memory_metrics()
    disk = _disk_metrics()
    temps = _temperature_metrics()
    npu = _devfreq_metrics("npu")
    gpu = _devfreq_metrics("gpu")
    return {
        "timestamp": now,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "memory": memory,
        "storage": disk,
        "temperature": {"sensors": temps, "max_c": max((x["temperature_c"] for x in temps), default=None)},
        "frequency": _frequency_metrics(),
        "npu": npu,
        "gpu": gpu,
        "network": _network_metrics(),
        "system": _load_uptime(),
        "sampling_interval_seconds": round(now - _last_sample, 2) if _last_sample else None,
        "limitations": [
            "Hardware counters depend on kernel and vendor driver exposure.",
            "NPU/GPU availability does not imply that utilisation counters are available.",
            "Temperature sensors are reported only when exported through Linux thermal or hwmon interfaces.",
            "Metrics are read-only; this endpoint does not change CPU governors, clocks, fan policy or device state."
        ],
    }
