#include "edgeshield/pcap_reader.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <tuple>
namespace edgeshield { namespace {
std::uint16_t u16(const unsigned char*p,bool little){return little?std::uint16_t(p[0]|(p[1]<<8)):std::uint16_t((p[0]<<8)|p[1]);}
std::uint32_t u32(const unsigned char*p,bool little){return little?(std::uint32_t(p[0])|(std::uint32_t(p[1])<<8)|(std::uint32_t(p[2])<<16)|(std::uint32_t(p[3])<<24)):(std::uint32_t(p[0])<<24)|(std::uint32_t(p[1])<<16)|(std::uint32_t(p[2])<<8)|p[3];}
std::string ip(const unsigned char*p){return std::to_string(p[0])+"."+std::to_string(p[1])+"."+std::to_string(p[2])+"."+std::to_string(p[3]);}
struct Key{std::string a,b,proto;std::uint16_t ap,bp;bool operator<(const Key&o)const{return std::tie(a,b,proto,ap,bp)<std::tie(o.a,o.b,o.proto,o.ap,o.bp);}};
}
PcapSummary PcapReader::read_ethernet_ipv4(const std::string&path)const{std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot open PCAP file: "+path);std::array<unsigned char,24>gh{};f.read(reinterpret_cast<char*>(gh.data()),gh.size());if(f.gcount()!=static_cast<std::streamsize>(gh.size()))throw std::runtime_error("PCAP global header is truncated");bool little=false;const unsigned char le[]={0xd4,0xc3,0xb2,0xa1},be[]={0xa1,0xb2,0xc3,0xd4};if(std::equal(std::begin(le),std::end(le),gh.begin()))little=true;else if(!std::equal(std::begin(be),std::end(be),gh.begin()))throw std::runtime_error("Unsupported PCAP magic; PCAPNG is not supported by this reader");if(u32(gh.data()+20,little)!=1)throw std::runtime_error("Only Ethernet link type PCAP is supported");PcapSummary out;std::map<Key,NetworkFlow> flows;std::array<unsigned char,16>rh{};while(f.read(reinterpret_cast<char*>(rh.data()),rh.size())){auto incl=u32(rh.data()+8,little);if(incl>16*1024*1024)throw std::runtime_error("Unreasonable PCAP packet length");std::vector<unsigned char>p(incl);f.read(reinterpret_cast<char*>(p.data()),incl);if(f.gcount()!=static_cast<std::streamsize>(incl))throw std::runtime_error("Truncated packet record");++out.packets_seen;if(incl<14)continue;std::size_t off=14;auto eth=u16(p.data()+12,false);if(eth==0x8100){if(incl<18)continue;eth=u16(p.data()+16,false);off=18;}if(eth!=0x0800||incl<off+20)continue;auto ihl=(p[off]&15)*4;if(ihl<20||incl<off+ihl)continue;auto proto=p[off+9];if(proto!=6&&proto!=17)continue;auto total=u16(p.data()+off+2,false);auto src=ip(p.data()+off+12),dst=ip(p.data()+off+16);auto l4=off+ihl;if(incl<l4+4)continue;auto sp=u16(p.data()+l4,false),dp=u16(p.data()+l4+2,false);std::string pr=proto==6?"tcp":"udp";Key k{src,dst,pr,sp,dp};auto it=flows.find(k);if(it==flows.end()){NetworkFlow nf;nf.src_ip=src;nf.dst_ip=dst;nf.src_port=sp;nf.dst_port=dp;nf.protocol=pr;it=flows.emplace(k,nf).first;}auto&nf=it->second;++nf.packets;nf.bytes+=total;if(proto==6&&incl>=l4+14){auto flags=p[l4+13];if(flags&0x02)++nf.syn_packets;if(flags&0x04)++nf.rst_packets;}}
 for(auto&kv:flows){auto&v=kv.second;out.flows.push_back(v);}return out;}
}
