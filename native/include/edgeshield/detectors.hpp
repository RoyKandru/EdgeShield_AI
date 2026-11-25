#pragma once
#include "edgeshield/contracts.hpp"
#include <string_view>
namespace edgeshield {
class ITextDetector { public: virtual ~ITextDetector()=default; virtual DetectionResult analyze(std::string_view input) const=0; };
class UrlDetector final: public ITextDetector { public: DetectionResult analyze(std::string_view input) const override; };
class SmsDetector final: public ITextDetector { public: DetectionResult analyze(std::string_view input) const override; };
class NetworkDetector { public: DetectionResult analyze(const NetworkFlow& flow) const; };
}
