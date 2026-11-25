#pragma once
#include "edgeshield/contracts.hpp"
#include <string>
#include <vector>
namespace edgeshield {
struct PcapSummary { std::uint64_t packets_seen{0}; std::vector<NetworkFlow> flows; };
class PcapReader { public: PcapSummary read_ethernet_ipv4(const std::string& path) const; };
}
