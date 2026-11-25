#include "edgeshield/detectors.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
namespace edgeshield { namespace {
std::string lower(std::string_view s){std::string o(s);std::transform(o.begin(),o.end(),o.begin(),[](unsigned char c){return char(std::tolower(c));});return o;}
ThreatEvent base(Source src,std::string cat){ThreatEvent e;e.source=src;e.category=std::move(cat);return e;}
void finish(ThreatEvent& e,int hits,double weight){e.confidence=std::min(.98,weight*hits);e.risk_score=std::min(100.0,weight*100.0*hits);e.severity=hits>=4?Severity::Critical:(hits>=3?Severity::High:(hits>=2?Severity::Medium:(hits?Severity::Low:Severity::Info)));}
}
DetectionResult UrlDetector::analyze(std::string_view input) const {
 auto s=lower(input);auto e=base(Source::Url,"url_phishing_heuristic");int hits=0;
 const std::vector<std::string> markers={"@","xn--","bit.ly/","tinyurl.com/","t.co/","redirect=","%40","login-","verify-","secure-","account-","password","wallet","kyc","upi"};
 for(const auto&m:markers)if(s.find(m)!=std::string::npos){++hits;e.evidence.push_back({"url_marker",m,"Suspicious URL lexical feature"});}
 if(s.rfind("http://",0)==0){++hits;e.evidence.push_back({"scheme","http","URL does not use HTTPS"});}
 auto scheme=s.find("://");auto host_start=scheme==std::string::npos?0:scheme+3;auto host_end=s.find_first_of("/?#:",host_start);auto host=s.substr(host_start,host_end==std::string::npos?std::string::npos:host_end-host_start);
 if(!host.empty() && host.find('.')==std::string::npos){++hits;e.evidence.push_back({"host",host,"Host does not resemble a fully qualified domain"});}
 if(host.size()>45){++hits;e.evidence.push_back({"host_length",std::to_string(host.size()),"Unusually long hostname"});}
 finish(e,hits,.17);e.recommended_actions=hits?std::vector<std::string>{"Do not open the URL until independently verified","Check the registered domain through a trusted channel","Do not enter credentials, OTPs or payment details"}:std::vector<std::string>{"Continue to treat unknown URLs cautiously"};
 return {hits>0,std::move(e)};
}
DetectionResult SmsDetector::analyze(std::string_view input) const {
 auto s=lower(input);auto e=base(Source::Sms,"sms_scam_heuristic");int hits=0;
 const std::vector<std::string> markers={"urgent","verify your account","otp","one-time password","click here","claim your prize","blocked","bank account","send money","upi","kyc","suspended","won a prize","act now","refund","update pan"," aadhaar","link your"};
 for(const auto&m:markers)if(s.find(m)!=std::string::npos){++hits;e.evidence.push_back({"message_marker",m,"Potential social-engineering phrase"});}
 if(s.find("http://")!=std::string::npos||s.find("https://")!=std::string::npos){++hits;e.evidence.push_back({"embedded_url","present","Message contains a URL; inspect separately"});}
 if(s.find("otp")!=std::string::npos||s.find("pin")!=std::string::npos||s.find("password")!=std::string::npos)e.recommended_actions={"Never share OTPs, PINs or passwords","Contact the organisation using its official number"};
 finish(e,hits,.15);if(e.recommended_actions.empty())e.recommended_actions=hits?std::vector<std::string>{"Do not follow links or payment instructions in the message","Verify the request using an official channel"}:std::vector<std::string>{"Do not disclose sensitive information in response to unsolicited messages"};
 return {hits>0,std::move(e)};
}
DetectionResult NetworkDetector::analyze(const NetworkFlow& f) const {
 auto e=base(Source::Network,"network_flow_anomaly_heuristic");e.evidence.push_back({"flow_tuple",f.src_ip+":"+std::to_string(f.src_port)+" -> "+f.dst_ip+":"+std::to_string(f.dst_port),"Observed flow endpoints"});e.evidence.push_back({"protocol",f.protocol,"Observed transport protocol"});int hits=0;
 if(f.dst_port==23||f.dst_port==2323||f.dst_port==445||f.dst_port==3389){++hits;e.evidence.push_back({"destination_port",std::to_string(f.dst_port),"Sensitive or commonly targeted service port"});}
 if(f.syn_packets>20 && f.packets>0 && double(f.syn_packets)/double(f.packets)>.75){++hits;e.evidence.push_back({"syn_ratio",std::to_string(double(f.syn_packets)/double(f.packets)),"High SYN packet ratio may indicate scanning or connection flooding"});}
 if(f.failed_connections>10){++hits;e.evidence.push_back({"failed_connections",std::to_string(f.failed_connections),"Repeated failed connection attempts"});}
 if(f.packets>0 && f.bytes/f.packets>1500){++hits;e.evidence.push_back({"mean_packet_bytes",std::to_string(f.bytes/f.packets),"Unusually high average bytes per packet"});}
 if(f.duration_seconds>0 && f.bytes>100000000 && f.duration_seconds<10){++hits;e.evidence.push_back({"throughput_burst","high","Large short-duration transfer"});}
 finish(e,hits,.19);e.recommended_actions=hits?std::vector<std::string>{"Inspect the source and destination endpoints","Review firewall and service exposure","Preserve flow and packet evidence before taking action"}:std::vector<std::string>{"No configured network heuristic was triggered"};
 return {hits>0,std::move(e)};
}
}
