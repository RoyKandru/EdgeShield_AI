#include "edgeshield/detectors.hpp"
#include "edgeshield/risk.hpp"
#include "edgeshield/system_monitor.hpp"
#include "edgeshield/sci.hpp"
#include <cassert>
#include <cstdio>
int main(){
 auto u=edgeshield::UrlDetector{}.analyze("http://secure-login-example.com/verify");assert(u.flagged&&!u.event.evidence.empty());
 auto s=edgeshield::SmsDetector{}.analyze("Urgent! Verify your account and share OTP");assert(s.flagged);
 auto safe=edgeshield::SmsDetector{}.analyze("Meeting moved to 3 PM");assert(!safe.flagged);
 edgeshield::NetworkFlow f;f.dst_port=23;f.packets=30;f.syn_packets=25;f.failed_connections=15;auto n=edgeshield::NetworkDetector{}.analyze(f);assert(n.flagged);
 assert(edgeshield::RiskFusion::score({u.event,s.event})>=u.event.risk_score);
 auto m=edgeshield::SystemMonitor{}.sample(10);assert(m.platform=="linux");assert(m.memory_total_kb.has_value());
 std::remove("/tmp/edgeshield-test-ledger.jsonl");edgeshield::SciEngine sci("/tmp/edgeshield-test-ledger.jsonl");auto i=sci.record(u.event,"url","test diagnosis","manual review");assert(i.verification_status=="unverified");assert(sci.verify(i.incident_id,true));assert(sci.list().size()==1);std::remove("/tmp/edgeshield-test-ledger.jsonl");
}
