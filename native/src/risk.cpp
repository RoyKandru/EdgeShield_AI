#include "edgeshield/risk.hpp"
#include <algorithm>
namespace edgeshield {
double RiskFusion::score(const std::vector<ThreatEvent>& events) {
  // Conservative max-plus-corroboration rule; risk is not classifier confidence.
  double maximum=0.0; std::size_t corroborating=0;
  for(const auto& e:events){ maximum=std::max(maximum,e.risk_score); if(e.risk_score>=40.0) ++corroborating; }
  return std::min(100.0, maximum + (corroborating>1 ? 10.0*static_cast<double>(corroborating-1) : 0.0));
}
}
