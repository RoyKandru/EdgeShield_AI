#pragma once
#include <optional>
#include <string>
#include <vector>
namespace edgeshield {
struct CpuCoreMetric { std::string name; std::optional<double> utilisation_percent; };
struct NetworkInterfaceMetric { std::string name; unsigned long long rx_bytes{0}, tx_bytes{0}; };
struct SystemMetrics {
 std::optional<double> cpu_utilisation_percent; std::vector<CpuCoreMetric> cpu_cores;
 std::optional<unsigned long long> memory_total_kb, memory_available_kb;
 std::optional<double> root_storage_utilisation_percent, cpu_temperature_celsius;
 std::optional<double> load1, load5, load15; std::optional<unsigned long long> uptime_seconds;
 std::vector<NetworkInterfaceMetric> network_interfaces;
 std::string platform{"linux"}; int logical_cpu_count{0};
};
class SystemMonitor { public: SystemMetrics sample(unsigned int cpu_sample_ms=120) const; };
}
