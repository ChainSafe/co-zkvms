## Flow

### Network Topology

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Worker 0 │  │ Worker 1 │  │ Worker 2 │
│ (party)  │  │ (party)  │  │ (party)  │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │ QUIC        │ QUIC        │ QUIC
     ├─────────────┼─────────────┤  ← ring topology (peer-to-peer, unchanged)
     │             │             │
     │ TCP+TLS     │ TCP+TLS     │ TCP+TLS
     └─────────────┼─────────────┘
                   ↓
          ┌────────────────┐
          │  Host Proxy    │  EC2 host (untrusted, full network+fs)
          │  TCP :9000     │  sees only TLS ciphertext
          │    ↕ bytes     │
          │  vsock CID:3   │
          └───────┬────────┘
                  ↓ vsock
          ┌────────────────┐
          │  Coordinator   │  Nitro Enclave (no network, no fs, no SSH)
          │  Enclave       │  ephemeral TLS identity
          └────────────────┘
```

- **Worker <-> Worker**: QUIC over UDP (ring topology, direct connections, unchanged)
- **Worker <-> Coordinator**: TLS 1.3 over TCP (star topology, through host proxy)
- **Host Proxy <-> Enclave**: vsock (kernel-level VM socket, reliable ordered stream)

### Why TLS (not QUIC) for the Enclave Transport

QUIC is UDP-based (datagrams). vsock is a reliable ordered stream socket (like TCP).
Bridging QUIC over vsock would require custom `AsyncUdpSocket` impl, datagram framing
over streams, and UDP NAT in the proxy. Raw TLS over TCP streams is simpler, correct,
and sufficient — QUIC's advantages (multiplexing, congestion control, 0-RTT) are
unnecessary over reliable vsock.

### Connection Establishment (per worker)

1. **Worker** opens TCP connection to host proxy (e.g. `host-proxy.example.com:9000`)
2. **Host proxy** accepts TCP, opens vsock to enclave (CID 3, port 9000), pipes bytes bidirectionally
3. **Worker <-> Enclave** TLS 1.3 handshake through the proxy (proxy sees ciphertext only)
4. **Enclave** sends length-prefixed attestation document over TLS
5. **Worker** verifies attestation (binds ephemeral TLS pubkey to enclave measurement)
6. **Worker** sends `(party_id, worker_id)` identification over TLS
7. Encrypted channel established

### MPC Protocol Communication

- **Worker -> Coordinator**: `TlsCoordinatorClient::send()` -> TLS encrypt -> TCP -> host proxy -> vsock -> enclave TLS decrypt
- **Coordinator -> Worker**: enclave writes to vsock -> host proxy -> TCP -> `TlsCoordinatorClient::recv()` -> TLS decrypt

### E2EE Flow (Enclave Side)

```
┌─────────────────────────────────────────────────────────────┐
│ ENCLAVE (coordinator.rs main)                                │
│                                                              │
│  1. Boot                                                     │
│  2. Generate ephemeral ECDSA P-256 keypair (rcgen)           │
│  3. [if aws_nitro] Request NSM attestation doc               │
│     - public_key = ephemeral pubkey bytes                    │
│     - NSM returns signed attestation document                │
│  4. Build self-signed X.509 cert from ephemeral key          │
│  5. Create rustls::ServerConfig with ephemeral cert + key    │
│  6. Listen on vsock, accept 3 TLS connections                │
│  7. For each connection:                                     │
│     - TLS handshake                                          │
│     - Send attestation doc (length-prefixed first message)   │
│     - Read (party_id, worker_id) identification              │
│  8. Drive proof via Rep3NetworkCoordinator trait              │
└─────────────────────────────────────────────────────────────┘
```

### Security Properties

- **Key ephemerality**: fresh ECDSA P-256 keypair every enclave boot, never touches disk (no disk in enclave)
- **Attestation binding**: NSM attestation doc contains the ephemeral pubkey, proving it was generated inside *this* enclave image
- **Forward secrecy**: TLS 1.3 (rustls default) provides PFS via ephemeral ECDHE
- **Host is untrusted**: TCP <-> vsock proxy sees only TLS ciphertext, has no keys
- **Enclave isolation**: no network, no filesystem, no SSH — only vsock to parent EC2 instance

### Component Locations

| Component | Binary | Runs on | Crate |
|-----------|--------|---------|-------|
| Worker (MPC party) | app-specific | Any machine | `co-jolt2` + `mpc-net` |
| Host Proxy | `host_proxy` | EC2 host | `co-jolt-coordinator` (feature: `aws_nitro`) |
| Coordinator | `coordinator` | Nitro Enclave | `co-jolt-coordinator` (feature: `aws_nitro`) |
| TLS Client | library | Worker process | `mpc-net` (feature: `tls`) |

### Configuration

**Workers** (TOML config):
```toml
[coordinator]
dns_name = "host-proxy.example.com:9000"
protocol = "tls"
# cert_path absent — ephemeral cert verified via attestation
```

**Host Proxy** (env vars only, no config file):
- `TCP_LISTEN_ADDR` — TCP bind address (default: `0.0.0.0:9000`)
- `VSOCK_CID` — enclave CID (default: `3`)
- `VSOCK_PORT` — enclave vsock port (default: `9000`)

**Coordinator Enclave** (no filesystem — env vars or hardcoded):
- Vsock listen port
- Number of expected worker connections
