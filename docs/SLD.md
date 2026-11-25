# EdgeShield AI — Low-Level Design v1

## Native module map

| Module | Public contract | Implementation |
|---|---|---|
| URL | `ITextDetector::analyze(string_view)` | `UrlDetector` |
| SMS | `ITextDetector::analyze(string_view)` | `SmsDetector` |
| Network | `NetworkDetector::analyze(NetworkFlow)` | flow heuristic |
| PCAP | `PcapReader::read_ethernet_ipv4(path)` | classic PCAP Ethernet/IPv4 TCP/UDP |
| Event | `ThreatEvent`, `Evidence`, `DetectionResult` | shared contract |
| Risk | `RiskFusion::score(events)` | max score + corroboration |
| SCI | `record`, `verify`, `list` | JSONL append-only record and linked digest |
| Monitor | `SystemMonitor::sample(ms)` | procfs/sysfs/statvfs |
| HTTP | `HttpServer::run()` | POSIX socket server |

## REST endpoints

- `GET /api/health`
- `GET /api/system/metrics`
- `GET /api/sci/incidents`
- `POST /api/analyze/url` with `{ "url": "..." }`
- `POST /api/analyze/sms` with `{ "message": "..." }`
- `POST /api/analyze/network` with normalized flow fields
- `POST /api/sci/verify` with `{ "incident_id": "...", "passed": true|false }`

## Persistence

SCI entries are appended to `data/sci_ledger.jsonl`. The chain digest includes the prior digest and current record fields. OpenSSL SHA-256 is enabled at build time when found. A non-cryptographic fallback is visibly prefixed and must not be used to claim cryptographic integrity.

## Error and safety behaviour

- Missing metric sources produce null/unavailable values.
- No detector match is not a guarantee that an input is safe.
- No automatic firewall changes, process killing, device reset or firmware operation is performed.
- HTTP server is unauthenticated; loopback bind is default.
- PCAPNG, IPv6 and unsupported link types fail explicitly rather than silently decoding incorrectly.
