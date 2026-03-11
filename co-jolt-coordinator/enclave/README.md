# AWS Nitro Enclave Build & Local Simulation

## Prerequisites

- **OS**: Linux x86_64 with KVM support (Ubuntu 20.04+)
- **RAM**: 4GB+ available
- **Storage**: 10GB free
- **Software**: Docker, QEMU (`qemu-kvm`, `qemu-system-x86`, `cloud-localds`), Go 1.19+
- **SSH key**: `ssh-keygen -t rsa -b 4096 -f ~/.ssh/dev-vm -N ""`

## Build the musl binary

```bash
cd enclave
make build
```

This builds the coordinator as a static `x86_64-unknown-linux-musl` binary inside an Alpine Docker container.

## Local Nitro simulation (fystack)

Uses [fystack/nitro-enclaves-simulation](https://github.com/fystack/nitro-enclaves-simulation) — a QEMU-based local environment that emulates the Nitro Enclave with vsock, LocalStack KMS, and a dummy NSM device.

Startup order is critical:

```bash
# 1. Boot the QEMU VM (wait for login prompt, then Ctrl+A X to detach)
make setup-vm

# 2. Build musl binary, copy into VM, start it
make start-enclave

# 3. Start LocalStack + vsock proxy
make start-vsock-proxy
```

### Debugging

```bash
make ssh-vm       # SSH into the QEMU VM
make view-logs    # View enclave logs in real-time
```

## Architecture

```
Host (your machine)          QEMU VM (simulated enclave)
+-----------------+          +-------------------------+
| vsock proxy     | <------> | coordinator binary      |
| (port 9000)     |  vsock   | (musl static, no libc)  |
+-----------------+          +-------------------------+
        |
+-----------------+
| LocalStack KMS  |
| (mock AWS)      |
+-----------------+
```

## Limitations

- The simulation provides a **dummy NSM device** — attestation documents are mock, not verifiable against the real AWS root of trust
- The coordinator binary is currently a **library crate** — a proper `main.rs` with vsock listener and attestation will be added in a future phase
- See `co-jolt-coordinator/DEVELOPMENT.md` for the full E2EE / attestation roadmap
