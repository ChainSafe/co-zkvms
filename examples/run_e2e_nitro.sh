#!/usr/bin/env bash
set -euo pipefail

# ── Nitro Enclave E2E Test ──────────────────────────────────────────────────
#
# Runs coordinator inside a real AWS Nitro Enclave, bridged via host_proxy.
# Workers and client run on the host.
#
# Env vars:
#   DEBUG=1          — run enclave with --debug-mode (console access, alters PCRs)
#   ENCLAVE_CPUS     — CPUs for enclave (default: 2)
#   ENCLAVE_MEM_MB   — Memory in MB for enclave (default: 4096)
#   VSOCK_PORT       — vsock/host_proxy port (default: 9000)
#   NETWORK_FORKS    — worker network forks (default: 4)
#   RAYON_THREADS    — worker rayon threads (default: 4)
#
# Prerequisites:
#   - Nitro-enabled EC2 instance with sudo nitro-cli and docker
#   - Dory SRS file (dory_urs_*.urs) in repo root (generated on first worker run)
#   - EIF pre-built: make -C co-jolt-coordinator/enclave build-eif
#   - host_proxy pre-built: make -C co-jolt-coordinator/enclave build-host-proxy
#
# Usage:
#   DEBUG=1 bash examples/run_e2e_nitro.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"

export RUSTFLAGS="${RUSTFLAGS:--A warnings}"

# ── Configuration ────────────────────────────────────────────────────────────

DEBUG=${DEBUG:-0}
ENCLAVE_CPUS=${ENCLAVE_CPUS:-2}
ENCLAVE_MEM_MB=${ENCLAVE_MEM_MB:-4096}
VSOCK_PORT=${VSOCK_PORT:-9000}
# Must match ENCLAVE_CPUS — twist_sumcheck_switch_index depends on thread count
RAYON_THREADS=${RAYON_THREADS:-$ENCLAVE_CPUS}
MPC_QUIC_CONN_LANES=${MPC_QUIC_CONN_LANES:-$RAYON_THREADS}
NETWORK_FORKS=${NETWORK_FORKS:-$RAYON_THREADS}
NUM_ITERS=${NUM_ITERS:-10}
TRACY_ALLOC=${TRACY_ALLOC:-0}
TRACY_CAPTURE=${TRACY_CAPTURE:-0}
JEMALLOC_PRESET=${JEMALLOC_PRESET:-default}
EXTRA_FEATURES=${EXTRA_FEATURES:-}
TRACE_SUFFIX="${NUM_ITERS}_${RAYON_THREADS}T_${MPC_QUIC_CONN_LANES}L_${NETWORK_FORKS}F"

ARTIFACT_DIR=${ARTIFACT_DIR:-"$REPO_DIR/.artifacts"}
TRACE_DIR=${TRACE_DIR:-"$REPO_DIR/.traces"}
PREPROC_DIR=${PREPROC_DIR:-"$REPO_DIR/co-jolt2/.preprocessing"}

# Ports — PORT_OFFSET shifts all port families for concurrent worktree runs
PORT_OFFSET=${PORT_OFFSET:-0}
INTER_PARTY_BASE_PORT=${INTER_PARTY_BASE_PORT:-$((10000 + PORT_OFFSET))}
USER_LISTEN_BASE_PORT=${USER_LISTEN_BASE_PORT:-$((30000 + PORT_OFFSET))}
TRACY_BASE_PORT=${TRACY_BASE_PORT:-$((8086 + PORT_OFFSET))}

ENCLAVE_DIR="$REPO_DIR/co-jolt-coordinator/enclave"

# ── Prereq checks ───────────────────────────────────────────────────────────

if [ ! -e /dev/nitro_enclaves ]; then
  echo "ERROR: /dev/nitro_enclaves not found — not a Nitro-enabled instance."
  exit 1
fi
command -v nitro-cli >/dev/null || { echo "ERROR: nitro-cli not found"; exit 1; }

if [ ! -f "$ENCLAVE_DIR/coordinator.eif" ]; then
  echo "ERROR: coordinator.eif not found."
  echo "  Build it: make -C co-jolt-coordinator/enclave build-eif"
  exit 1
fi
if [ ! -f "$ENCLAVE_DIR/host_proxy" ]; then
  echo "ERROR: host_proxy not found."
  echo "  Build it: make -C co-jolt-coordinator/enclave build-host-proxy"
  exit 1
fi

mkdir -p "$ARTIFACT_DIR" "$TRACE_DIR"

CO_JOLT2_FEATURES="test-utils"
if [ "$TRACY_ALLOC" = "1" ]; then
  CO_JOLT2_FEATURES="$CO_JOLT2_FEATURES,tracy-mem,jemalloc-stats"
fi
if [ -n "$EXTRA_FEATURES" ]; then
  CO_JOLT2_FEATURES="$CO_JOLT2_FEATURES,$EXTRA_FEATURES"
fi

setup_jemalloc_preset "$JEMALLOC_PRESET"

echo "=== Nitro Enclave E2E Test ==="
echo "  debug=$DEBUG cpus=$ENCLAVE_CPUS mem=${ENCLAVE_MEM_MB}MB vsock_port=$VSOCK_PORT"

# ── 1. Build host binaries (workers, client, gen_configs) ───────────────────

echo "Building host binaries..."

cd "$REPO_DIR"

cargo build --release \
  -p co-jolt2 --bin worker --features "$CO_JOLT2_FEATURES"

cargo build --release \
  -p mpc-net --bin gen_configs

cargo build --release \
  --manifest-path "$REPO_DIR/examples/sha2-chain/Cargo.toml" \
  --target-dir "$REPO_DIR/target"

echo "Build complete."

# ── 2. Generate configs ──────────────────────────────────────────────────────

rm -f "$ARTIFACT_DIR"/config_*.toml "$ARTIFACT_DIR"/*.der

"$REPO_DIR/target/release/gen_configs" \
  -n 1 \
  -o "$ARTIFACT_DIR" \
  -c "$ARTIFACT_DIR" \
  -k "$ARTIFACT_DIR" \
  --inter-party-base-port "$INTER_PARTY_BASE_PORT" \
  --coordinator-port "$VSOCK_PORT" \
  --user-listen-base-port "$USER_LISTEN_BASE_PORT" \
  --coordinator-protocol tls \
  --coordinator-addr "localhost:$VSOCK_PORT"

# ── 3. Launch Nitro enclave ─────────────────────────────────────────────────

# Terminate any existing enclave
sudo nitro-cli terminate-enclave --all 2>/dev/null || true

ENCLAVE_FLAGS=(
  --cpu-count "$ENCLAVE_CPUS"
  --memory "$ENCLAVE_MEM_MB"
  --eif-path "$ENCLAVE_DIR/coordinator.eif"
)
if [ "$DEBUG" = "1" ]; then
  ENCLAVE_FLAGS+=(--debug-mode)
fi

sudo nitro-cli run-enclave "${ENCLAVE_FLAGS[@]}"

enclave_cid=$(sudo nitro-cli describe-enclaves | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['EnclaveCID'])")
enclave_id=$(sudo nitro-cli describe-enclaves | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['EnclaveID'])")

# ── 4. Start host_proxy ─────────────────────────────────────────────────────

VSOCK_CID="$enclave_cid" VSOCK_PORT="$VSOCK_PORT" TCP_LISTEN_ADDR="0.0.0.0:$VSOCK_PORT" \
  "$ENCLAVE_DIR/host_proxy" &
host_proxy_pid=$!

for i in $(seq 1 20); do
  if lsof -i ":$VSOCK_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

# Give enclave time to initialize vsock listener
sleep 2

# ── 5. Launch 3 workers ─────────────────────────────────────────────────────

worker_pids=()
for p in 0 1 2; do
  NUM_ITERS="$NUM_ITERS" MPC_QUIC_CONN_LANES="$MPC_QUIC_CONN_LANES" NETWORK_FORKS="$NETWORK_FORKS" TRACY=1 TRACY_PORT=$((TRACY_BASE_PORT + p)) \
  "$REPO_DIR/target/release/worker" \
    -c "$ARTIFACT_DIR/config_worker0_${p}.toml" \
    -t "$TRACE_DIR" \
    --network-forks "$NETWORK_FORKS" \
    --rayon-threads "$RAYON_THREADS" \
    -p "$PREPROC_DIR" &
  worker_pids+=($!)
done

capture_pids=()
if [ "$TRACY_CAPTURE" = "1" ]; then
  TRACY_CAPTURE_BIN=${TRACY_CAPTURE_BIN:-$(command -v tracy-capture 2>/dev/null || echo tracy-capture)}
  for p in 0 1 2; do
    "$TRACY_CAPTURE_BIN" \
      -f \
      -o "$TRACE_DIR/worker${p}_${TRACE_SUFFIX}.tracy" \
      -a 127.0.0.1 \
      -p $((TRACY_BASE_PORT + p)) >/dev/null 2>&1 &
    capture_pids+=($!)
  done
fi

# ── Cleanup trap ─────────────────────────────────────────────────────────────

cleanup() {
  # Send SIGINT to tracy-capture first (graceful flush), then kill workers
  if [ ${#capture_pids[@]} -gt 0 ]; then
    kill -INT "${capture_pids[@]}" 2>/dev/null || true
  fi
  kill "$host_proxy_pid" "${worker_pids[@]}" 2>/dev/null || true
  if [ ${#capture_pids[@]} -gt 0 ]; then
    wait "${capture_pids[@]}" 2>/dev/null || true
  fi
  wait "$host_proxy_pid" "${worker_pids[@]}" 2>/dev/null || true
  sudo nitro-cli terminate-enclave --all 2>/dev/null || true
}
trap cleanup EXIT

# ── 6. Wait for workers to bind, then run client ────────────────────────────

sleep 3

WORKER_ADDRS="127.0.0.1:${USER_LISTEN_BASE_PORT}"
WORKER_ADDRS="${WORKER_ADDRS},127.0.0.1:$((USER_LISTEN_BASE_PORT + 1))"
WORKER_ADDRS="${WORKER_ADDRS},127.0.0.1:$((USER_LISTEN_BASE_PORT + 2))"

echo "Running sha2-chain client (workers=$WORKER_ADDRS)..."

"$REPO_DIR/target/release/sha2-chain" \
  --config-path "$ARTIFACT_DIR/config_delegator.toml" \
  --num-iters "$NUM_ITERS"

echo ""
echo "=== Nitro Enclave E2E Test PASSED ==="
