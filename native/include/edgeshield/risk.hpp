#pragma once
#include "edgeshield/contracts.hpp"
#include <vector>
namespace edgeshield {
class RiskFusion {
 public: static double score(const std::vector<ThreatEvent>& events);
};
}
