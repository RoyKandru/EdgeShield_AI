/* Minimal zero-dependency Linux AF_PACKET raw sniffer helper for EdgeShield AI.
 * Streams standard PCAP packet records (16-byte header + raw Ethernet frame) to stdout.
 * Grant non-root capture cleanly via: sudo setcap cap_net_raw,cap_net_admin=eip bin/live_sniffer
 */
#include <arpa/inet.h>
#include <errno.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

static volatile sig_atomic_t g_running = 1;
static void on_sig(int sig) { (void)sig; g_running = 0; }

int main(int argc, char** argv) {
    const char* iface = (argc > 1 && argv[1][0] != '\0') ? argv[1] : "";
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    signal(SIGPIPE, on_sig);

    int fd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (fd < 0) {
        fprintf(stderr, "live_sniffer: socket(AF_PACKET) failed: %s\n", strerror(errno));
        return 1;
    }

    if (iface[0] != '\0' && strcmp(iface, "any") != 0) {
        struct ifreq ifr;
        memset(&ifr, 0, sizeof(ifr));
        snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", iface);
        if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) {
            fprintf(stderr, "live_sniffer: interface '%s' not found: %s\n", iface, strerror(errno));
            close(fd);
            return 2;
        }
        struct sockaddr_ll sll;
        memset(&sll, 0, sizeof(sll));
        sll.sll_family = AF_PACKET;
        sll.sll_ifindex = ifr.ifr_ifindex;
        sll.sll_protocol = htons(ETH_P_ALL);
        if (bind(fd, (struct sockaddr*)&sll, sizeof(sll)) < 0) {
            fprintf(stderr, "live_sniffer: bind(%s) failed: %s\n", iface, strerror(errno));
            close(fd);
            return 3;
        }
    }

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 300000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    uint8_t buf[65536];
    while (g_running) {
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
            break;
        }
        struct timeval now;
        gettimeofday(&now, NULL);
        uint32_t hdr[4];
        hdr[0] = (uint32_t)now.tv_sec;
        hdr[1] = (uint32_t)now.tv_usec;
        hdr[2] = (uint32_t)n;
        hdr[3] = (uint32_t)n;
        if (fwrite(hdr, 1, 16, stdout) != 16) break;
        if (fwrite(buf, 1, (size_t)n, stdout) != (size_t)n) break;
        fflush(stdout);
    }

    close(fd);
    return 0;
}
