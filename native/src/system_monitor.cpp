#include "edgeshield/system_monitor.hpp"
#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>
#include <sys/statvfs.h>
#include <dirent.h>
namespace edgeshield { namespace {
struct CpuTicks{unsigned long long total{0},idle{0};};
std::vector<std::pair<std::string,CpuTicks>> read_cpu(){std::vector<std::pair<std::string,CpuTicks>> out;std::ifstream f("/proc/stat");std::string line;while(std::getline(f,line)){std::istringstream ss(line);std::string name;if(!(ss>>name)||name.rfind("cpu",0)!=0)break;CpuTicks t;unsigned long long user=0,nice=0,system=0,idle=0,iowait=0,irq=0,softirq=0,steal=0;ss>>user>>nice>>system>>idle>>iowait>>irq>>softirq>>steal;t.total=user+nice+system+idle+iowait+irq+softirq+steal;t.idle=idle+iowait;out.push_back({name,t});}return out;}
std::optional<double> util(CpuTicks a,CpuTicks b){if(b.total<=a.total)return std::nullopt;auto dt=b.total-a.total,di=b.idle-a.idle;if(dt==0)return std::nullopt;return 100.0*double(dt-di)/double(dt);}
}
SystemMetrics SystemMonitor::sample(unsigned int ms) const {
 SystemMetrics m;auto a=read_cpu();std::this_thread::sleep_for(std::chrono::milliseconds(ms));auto b=read_cpu();if(!a.empty()&&!b.empty()){m.cpu_utilisation_percent=util(a[0].second,b[0].second);for(std::size_t i=1;i<std::min(a.size(),b.size());++i)m.cpu_cores.push_back({b[i].first,util(a[i].second,b[i].second)});}
 std::ifstream mem("/proc/meminfo");std::string key,unit;unsigned long long val;while(mem>>key>>val>>unit){if(key=="MemTotal:")m.memory_total_kb=val;if(key=="MemAvailable:")m.memory_available_kb=val;}
 struct statvfs fs{};if(statvfs("/",&fs)==0&&fs.f_blocks)m.root_storage_utilisation_percent=100.0*double(fs.f_blocks-fs.f_bfree)/double(fs.f_blocks);
 std::ifstream load("/proc/loadavg");double l1,l5,l15;if(load>>l1>>l5>>l15){m.load1=l1;m.load5=l5;m.load15=l15;}
 std::ifstream up("/proc/uptime");double seconds;if(up>>seconds)m.uptime_seconds=static_cast<unsigned long long>(seconds);
 std::ifstream thermal("/sys/class/thermal/thermal_zone0/temp");long temp=0;if(thermal>>temp)m.cpu_temperature_celsius=double(temp)/1000.0;
 std::ifstream net("/proc/net/dev");std::string line;std::getline(net,line);std::getline(net,line);while(std::getline(net,line)){auto colon=line.find(':');if(colon==std::string::npos)continue;std::string name=line.substr(0,colon);name.erase(0,name.find_first_not_of(" \t"));std::istringstream ss(line.substr(colon+1));NetworkInterfaceMetric n;n.name=name;unsigned long long rx_packets,rx_err,rx_drop,rx_fifo,rx_frame,rx_compressed,rx_multicast,tx_packets,tx_err,tx_drop,tx_fifo,tx_colls,tx_carrier,tx_compressed;if(ss>>n.rx_bytes>>rx_packets>>rx_err>>rx_drop>>rx_fifo>>rx_frame>>rx_compressed>>rx_multicast>>n.tx_bytes>>tx_packets>>tx_err>>tx_drop>>tx_fifo>>tx_colls>>tx_carrier>>tx_compressed)m.network_interfaces.push_back(n);}
 m.logical_cpu_count=static_cast<int>(m.cpu_cores.size());return m;
}
}
