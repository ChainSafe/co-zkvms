#!/usr/bin/env bash
set -euo pipefail

# ── TEE E2E Test ─────────────────────────────────────────────────────────────
#
# Runs 5 processes: 1 coordinator + 3 workers + 1 client.
#
# Modes (set via MODE env var):
#   quic  — coordinator uses QUIC transport (normal OS process)
#   tls   — coordinator uses TLS-over-TCP (emulated TEE, mimics vsock+TLS)
#
# Usage:
#   MODE=quic bash co-jolt2/examples/run_tee_e2e.sh
#   MODE=tls  bash co-jolt2/examples/run_tee_e2e.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CO_JOLT2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$CO_JOLT2_DIR/.." && pwd)"

export RUSTFLAGS="${RUSTFLAGS:--A warnings}"

MODE=${MODE:-quic}
ARTIFACT_DIR=${ARTIFACT_DIR:-"$CO_JOLT2_DIR/.tee-artifacts"}
TRACE_DIR=${TRACE_DIR:-"$CO_JOLT2_DIR/.tee-traces"}
PREPROC_DIR=${PREPROC_DIR:-"$CO_JOLT2_DIR/.tee-preprocessing"}
NETWORK_FORKS=${NETWORK_FORKS:-4}
RAYON_THREADS=${RAYON_THREADS:-4}

# Ports
USER_LISTEN_BASE_PORT=${USER_LISTEN_BASE_PORT:-30000}

mkdir -p "$ARTIFACT_DIR" "$TRACE_DIR"

echo "=== TEE E2E Test (mode=$MODE) ==="

# ── 1. Build binaries ────────────────────────────────────────────────────────

echo "Building binaries..."

cd "$REPO_DIR"

cargo build --release \
  -p co-jolt-coordinator --bin coordinator \
  -p co-jolt2 --bin worker \
  -p co-jolt2 --bin client \
  -p mpc-net --bin gen_configs

echo "Build complete."

# ── 2. Generate configs ──────────────────────────────────────────────────────

# Regenerate configs every time (cheap, ensures consistency with MODE)
rm -f "$ARTIFACT_DIR"/config_*.toml "$ARTIFACT_DIR"/*.der

"$REPO_DIR/target/release/gen_configs" \
  -n 1 \
  -o "$ARTIFACT_DIR" \
  -c "$ARTIFACT_DIR" \
  -k "$ARTIFACT_DIR" \
  --user-listen-base-port "$USER_LISTEN_BASE_PORT" \
  --coordinator-protocol "$MODE"

echo "Configs generated in $ARTIFACT_DIR (coordinator-protocol=$MODE)"

# ── 3. Launch coordinator ────────────────────────────────────────────────────

echo "Starting coordinator (transport=$MODE)..."

"$REPO_DIR/target/release/coordinator" \
  --config-file "$ARTIFACT_DIR/config_coordinator.toml" \
  --transport "$MODE" \
  --rayon-threads "$RAYON_THREADS" &
coordinator_pid=$!
echo "  coordinator PID=$coordinator_pid"

# In TLS mode, wait for the coordinator to bind before starting workers
if [ "$MODE" = "tls" ]; then
  echo "Waiting for coordinator to bind on port 20000..."
  for i in $(seq 1 20); do
    if lsof -i :20000 -sTCP:LISTEN >/dev/null 2>&1; then
      echo "  coordinator is listening"
      break
    fi
    sleep 0.5
  done
fi

# ── 4. Launch 3 workers ─────────────────────────────────────────────────────

worker_pids=()
for p in 0 1 2; do
  echo "Starting worker $p..."
  "$REPO_DIR/target/release/worker" \
    -c "$ARTIFACT_DIR/config_worker0_${p}.toml" \
    -t "$TRACE_DIR" \
    --network-forks "$NETWORK_FORKS" \
    --rayon-threads "$RAYON_THREADS" \
    -p "$PREPROC_DIR" &
  worker_pids+=($!)
  echo "  worker $p PID=$!"
done

# ── Cleanup trap ─────────────────────────────────────────────────────────────

cleanup() {
  echo "Cleaning up..."
  local pids=("$coordinator_pid" "${worker_pids[@]}")
  kill "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT

# ── 5. Wait for workers to bind, then run client ────────────────────────────

# Give workers time to bind their user-listen ports
echo "Waiting for workers to bind..."
sleep 3

WORKER_ADDRS="127.0.0.1:${USER_LISTEN_BASE_PORT}"
WORKER_ADDRS="${WORKER_ADDRS},127.0.0.1:$((USER_LISTEN_BASE_PORT + 1))"
WORKER_ADDRS="${WORKER_ADDRS},127.0.0.1:$((USER_LISTEN_BASE_PORT + 2))"

echo "Running client (workers=$WORKER_ADDRS)..."

"$REPO_DIR/target/release/client" \
  -w "$WORKER_ADDRS" \
  -t "$TRACE_DIR"

echo ""
echo "=== TEE E2E Test PASSED (mode=$MODE) ==="

# Workers and coordinator are long-lived; kill them via trap
