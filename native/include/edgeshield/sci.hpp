#pragma once
#include "edgeshield/contracts.hpp"
#include <mutex>
#include <string>
#include <vector>
namespace edgeshield {
struct SciIncident { std::string incident_id, fault_domain, diagnosis, correction, verification_status, chain_checksum; };
class SciEngine {
 public: explicit SciEngine(std::string ledger_path="data/sci_ledger.jsonl");
  SciIncident record(const ThreatEvent& event, std::string fault_domain, std::string diagnosis, std::string correction);
  bool verify(const std::string& incident_id, bool caller_check_passed);
  std::vector<SciIncident> list() const;
 private: std::string path_; mutable std::mutex mutex_; std::vector<SciIncident> incidents_; std::string previous_checksum_;
  void load(); void persist(const SciIncident& incident);
};
}
