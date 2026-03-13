#!/usr/bin/env bash
set -euo pipefail

# ── E2E Test ─────────────────────────────────────────────────────────────────
#
# Runs 5 processes: 1 coordinator + 3 workers + 1 client.
#
# Transport (set via TRANSPORT env var):
#   quic  — coordinator uses QUIC transport (default)
#   tls   — coordinator uses TLS-over-TCP (emulated TEE, mimics vsock+TLS)
#
# Usage:
#   TRANSPORT=quic bash examples/run_e2e.sh
#   TRANSPORT=tls  bash examples/run_e2e.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"

export RUSTFLAGS="${RUSTFLAGS:--A warnings}"

TRANSPORT=${TRANSPORT:-tls}
ARTIFACT_DIR=${ARTIFACT_DIR:-"$REPO_DIR/.artifacts"}
TRACE_DIR=${TRACE_DIR:-"$REPO_DIR/.traces"}
PREPROC_DIR=${PREPROC_DIR:-"$REPO_DIR/co-jolt2/.preprocessing"}
NETWORK_FORKS=${NETWORK_FORKS:-4}
RAYON_THREADS=${RAYON_THREADS:-4}
NUM_ITERS=${NUM_ITERS:-10}
TRACY_BASE_PORT=${TRACY_BASE_PORT:-8086}
TRACY_ALLOC=${TRACY_ALLOC:-0}
TRACY_CAPTURE=${TRACY_CAPTURE:-0}
JEMALLOC_PRESET=${JEMALLOC_PRESET:-default}
EXTRA_FEATURES=${EXTRA_FEATURES:-}

# Ports
USER_LISTEN_BASE_PORT=${USER_LISTEN_BASE_PORT:-30000}

mkdir -p "$ARTIFACT_DIR" "$TRACE_DIR"

CO_JOLT2_FEATURES="test-utils"
if [ "$TRACY_ALLOC" = "1" ]; then
  CO_JOLT2_FEATURES="$CO_JOLT2_FEATURES,tracy-mem,jemalloc-stats"
fi
if [ -n "$EXTRA_FEATURES" ]; then
  CO_JOLT2_FEATURES="$CO_JOLT2_FEATURES,$EXTRA_FEATURES"
fi

setup_jemalloc_preset "$JEMALLOC_PRESET"

echo "=== E2E Test (transport=$TRANSPORT) ==="

# ── 1. Build binaries ────────────────────────────────────────────────────────

echo "Building binaries..."

cd "$REPO_DIR"

cargo build --release \
  -p co-jolt-coordinator --bin coordinator --features test-utils

cargo build --release \
  -p co-jolt2 --bin worker --features "$CO_JOLT2_FEATURES"

cargo build --release \
  -p mpc-net --bin gen_configs

cargo build --release \
  --manifest-path "$REPO_DIR/examples/sha2-chain/Cargo.toml" \
  --target-dir "$REPO_DIR/target"

echo "Build complete."

# ── 2. Generate configs ──────────────────────────────────────────────────────

# Regenerate configs every time (cheap, ensures consistency with TRANSPORT)
rm -f "$ARTIFACT_DIR"/config_*.toml "$ARTIFACT_DIR"/*.der

"$REPO_DIR/target/release/gen_configs" \
  -n 1 \
  -o "$ARTIFACT_DIR" \
  -c "$ARTIFACT_DIR" \
  -k "$ARTIFACT_DIR" \
  --user-listen-base-port "$USER_LISTEN_BASE_PORT" \
  --coordinator-protocol "$TRANSPORT"

# ── 3. Launch coordinator ────────────────────────────────────────────────────

NUM_ITERS="$NUM_ITERS" TRACY=1 TRACY_PORT=$((TRACY_BASE_PORT - 1)) \
"$REPO_DIR/target/release/coordinator" \
  --config-file "$ARTIFACT_DIR/config_coordinator.toml" \
  --transport "$TRANSPORT" \
  -t "$TRACE_DIR" \
  --rayon-threads "$RAYON_THREADS" &
coordinator_pid=$!

# In TLS mode, wait for the coordinator to bind before starting workers
if [ "$TRANSPORT" = "tls" ]; then
  for i in $(seq 1 20); do
    if lsof -i :20000 -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi

# ── 4. Launch 3 workers ─────────────────────────────────────────────────────

worker_pids=()
for p in 0 1 2; do
  NUM_ITERS="$NUM_ITERS" TRACY=1 TRACY_PORT=$((TRACY_BASE_PORT + p)) \
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
      -o "$TRACE_DIR/worker${p}.tracy" \
      -a 127.0.0.1 \
      -p $((TRACY_BASE_PORT + p)) >/dev/null 2>&1 &
    capture_pids+=($!)
  done
  "$TRACY_CAPTURE_BIN" \
    -f \
    -o "$TRACE_DIR/coordinator.tracy" \
    -a 127.0.0.1 \
    -p $((TRACY_BASE_PORT - 1)) >/dev/null 2>&1 &
  capture_pids+=($!)
fi

# ── Cleanup trap ─────────────────────────────────────────────────────────────

cleanup() {
  # Send SIGINT to tracy-capture first (graceful flush), then kill workers
  if [ ${#capture_pids[@]} -gt 0 ]; then
    kill -INT "${capture_pids[@]}" 2>/dev/null || true
  fi
  local pids=("$coordinator_pid" "${worker_pids[@]}")
  kill "${pids[@]}" 2>/dev/null || true
  if [ ${#capture_pids[@]} -gt 0 ]; then
    wait "${capture_pids[@]}" 2>/dev/null || true
  fi
  wait "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT

# ── 5. Wait for workers to bind, then run client ────────────────────────────

# Give workers time to bind their user-listen ports
sleep 3

WORKER_ADDRS="127.0.0.1:${USER_LISTEN_BASE_PORT}"
WORKER_ADDRS="${WORKER_ADDRS},127.0.0.1:$((USER_LISTEN_BASE_PORT + 1))"
WORKER_ADDRS="${WORKER_ADDRS},127.0.0.1:$((USER_LISTEN_BASE_PORT + 2))"

echo "Running sha2-chain client (workers=$WORKER_ADDRS)..."

"$REPO_DIR/target/release/sha2-chain" \
  --config-path "$ARTIFACT_DIR/config_delegator.toml" \
  --num-iters "$NUM_ITERS"

echo ""
echo "=== E2E Test PASSED (transport=$TRANSPORT) ==="

# Workers and coordinator are long-lived; kill them via trap
