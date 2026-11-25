#include "edgeshield/sci.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <functional>
#ifdef EDGESHIELD_WITH_OPENSSL
#include <openssl/sha.h>
#endif
namespace edgeshield { namespace {
std::string digest(const std::string& s){
#ifdef EDGESHIELD_WITH_OPENSSL
 unsigned char hash[SHA256_DIGEST_LENGTH];SHA256(reinterpret_cast<const unsigned char*>(s.data()),s.size(),hash);std::ostringstream o;for(auto c:hash)o<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);return o.str();
#else
 // Development fallback only: not cryptographic and not suitable for tamper-proof audit claims.
 std::ostringstream o;o<<std::hex<<std::hash<std::string>{}(s);return "NONCRYPTO-"+o.str();
#endif
}
std::string safe(const std::string&s){std::string o=s;for(char&c:o)if(c=='\n'||c=='\r'||c=='|')c=' ';return o;}
}
SciEngine::SciEngine(std::string p):path_(std::move(p)){load();}
void SciEngine::load(){std::ifstream f(path_);std::string line;while(std::getline(f,line)){std::istringstream s(line);SciIncident i;std::getline(s,i.incident_id,'|');std::getline(s,i.fault_domain,'|');std::getline(s,i.diagnosis,'|');std::getline(s,i.correction,'|');std::getline(s,i.verification_status,'|');std::getline(s,i.chain_checksum,'|');if(!i.incident_id.empty()){incidents_.push_back(i);previous_checksum_=i.chain_checksum;}}}
void SciEngine::persist(const SciIncident&i){std::ofstream f(path_,std::ios::app);if(!f)return;f<<safe(i.incident_id)<<'|'<<safe(i.fault_domain)<<'|'<<safe(i.diagnosis)<<'|'<<safe(i.correction)<<'|'<<safe(i.verification_status)<<'|'<<safe(i.chain_checksum)<<'\n';}
SciIncident SciEngine::record(const ThreatEvent&e,std::string domain,std::string diagnosis,std::string correction){std::lock_guard<std::mutex>g(mutex_);SciIncident i;i.incident_id=e.event_id.empty()?"sci-"+std::to_string(incidents_.size()+1):"sci-"+e.event_id;i.fault_domain=std::move(domain);i.diagnosis=std::move(diagnosis);i.correction=std::move(correction);i.verification_status="unverified";i.chain_checksum=digest(previous_checksum_+i.incident_id+i.fault_domain+i.diagnosis+i.correction+i.verification_status);previous_checksum_=i.chain_checksum;incidents_.push_back(i);persist(i);return i;}
bool SciEngine::verify(const std::string&id,bool passed){std::lock_guard<std::mutex>g(mutex_);for(auto&i:incidents_)if(i.incident_id==id){i.verification_status=passed?"caller_check_passed":"caller_check_failed";i.chain_checksum=digest(previous_checksum_+i.incident_id+i.verification_status);previous_checksum_=i.chain_checksum;persist(i);return passed;}return false;}
std::vector<SciIncident> SciEngine::list()const{std::lock_guard<std::mutex>g(mutex_);return incidents_;}
}
