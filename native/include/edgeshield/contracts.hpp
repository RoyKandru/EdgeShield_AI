#pragma once
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
namespace edgeshield {
enum class Source { Network, Url, Sms, System };
enum class Severity { Info, Low, Medium, High, Critical };
struct Evidence { std::string key, value, explanation; };
struct ThreatEvent {
  std::string event_id, schema_version{"1.1"};
  std::chrono::system_clock::time_point timestamp{std::chrono::system_clock::now()};
  Source source{Source::Url}; std::string category;
  Severity severity{Severity::Info};
  double confidence{0.0}; // heuristic strength, not calibrated probability
  double risk_score{0.0};
  std::vector<Evidence> evidence;
  std::vector<std::string> recommended_actions;
  std::string detector_version{"1.3.0"};
};
struct DetectionResult { bool flagged{false}; ThreatEvent event; };
struct NetworkFlow {
  std::string src_ip, dst_ip, protocol{"tcp"};
  std::uint16_t src_port{0}, dst_port{0};
  std::uint64_t packets{0}, bytes{0}, syn_packets{0}, rst_packets{0}, failed_connections{0};
  double duration_seconds{0.0};
};
}
