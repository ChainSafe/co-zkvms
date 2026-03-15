#!/usr/bin/env bash
set -euo pipefail

# ── Deploy MPC Infrastructure ────────────────────────────────────────────────
#
# Builds and launches 1 coordinator + 3 workers in stand-by mode.
# They loop internally, waiting for client delegate requests.
# Run any example client separately (e.g., sha2-chain, zkemail).
#
# Usage:
#   bash examples/deploy.sh
#   # Then in another terminal:
#   ./target/release/zkemail --config-path .artifacts/config_delegator.toml ...
#
# Env vars (same as run_e2e.sh):
#   TRANSPORT        quic | tls (default: tls)
#   RAYON_THREADS    (default: 4)
#   PORT_OFFSET      shift all ports for concurrent runs
#   TRACY_CAPTURE    1 to start tracy-capture sessions
#   REUSE_PREPROC    1 to reuse preprocessing across requests
#   See below for full list.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common.sh"

export RUSTFLAGS="${RUSTFLAGS:--A warnings}"

TRANSPORT=${TRANSPORT:-tls}
ARTIFACT_DIR=${ARTIFACT_DIR:-"$REPO_DIR/.artifacts"}
TRACE_DIR=${TRACE_DIR:-"$REPO_DIR/.traces"}
PREPROC_DIR=${PREPROC_DIR:-"$REPO_DIR/co-jolt2/.preprocessing"}
RAYON_THREADS=${RAYON_THREADS:-4}
MPC_QUIC_CONN_LANES=${MPC_QUIC_CONN_LANES:-$RAYON_THREADS}
NETWORK_FORKS=${NETWORK_FORKS:-$RAYON_THREADS}
TRACY_ALLOC=${TRACY_ALLOC:-0}
TRACY_CAPTURE=${TRACY_CAPTURE:-0}
JEMALLOC_PRESET=${JEMALLOC_PRESET:-default}
EXTRA_FEATURES=${EXTRA_FEATURES:-}
TRACE_SUFFIX="${RAYON_THREADS}T_${MPC_QUIC_CONN_LANES}L_${NETWORK_FORKS}F"

# Ports — PORT_OFFSET shifts all port families for concurrent runs
PORT_OFFSET=${PORT_OFFSET:-0}
INTER_PARTY_BASE_PORT=${INTER_PARTY_BASE_PORT:-$((10000 + PORT_OFFSET))}
COORDINATOR_PORT=${COORDINATOR_PORT:-$((20000 + PORT_OFFSET))}
USER_LISTEN_BASE_PORT=${USER_LISTEN_BASE_PORT:-$((30000 + PORT_OFFSET))}
TRACY_BASE_PORT=${TRACY_BASE_PORT:-$((8086 + PORT_OFFSET))}

mkdir -p "$ARTIFACT_DIR" "$TRACE_DIR"

CO_JOLT2_FEATURES="test-utils"
if [ "$TRACY_ALLOC" = "1" ]; then
  CO_JOLT2_FEATURES="$CO_JOLT2_FEATURES,tracy-mem,jemalloc-stats"
fi
if [ -n "$EXTRA_FEATURES" ]; then
  CO_JOLT2_FEATURES="$CO_JOLT2_FEATURES,$EXTRA_FEATURES"
fi

REUSE_PREPROC=${REUSE_PREPROC:-0}
if [ "$REUSE_PREPROC" = "1" ]; then
  CO_JOLT2_FEATURES="$CO_JOLT2_FEATURES,reuse-preproc"
fi

setup_jemalloc_preset "$JEMALLOC_PRESET"

echo "=== Deploy MPC Infrastructure (transport=$TRANSPORT) ==="

# ── 1. Build binaries ────────────────────────────────────────────────────────

echo "Building binaries..."

cd "$REPO_DIR"

cargo build --release \
  -p co-jolt-coordinator --bin coordinator --features test-utils

cargo build --release \
  -p co-jolt2 --bin worker --features "$CO_JOLT2_FEATURES"

cargo build --release \
  -p mpc-net --bin gen_configs

# ── 2. Generate configs ──────────────────────────────────────────────────────

rm -f "$ARTIFACT_DIR"/config_*.toml "$ARTIFACT_DIR"/*.der

"$REPO_DIR/target/release/gen_configs" \
  -n 1 \
  -o "$ARTIFACT_DIR" \
  -c "$ARTIFACT_DIR" \
  -k "$ARTIFACT_DIR" \
  --inter-party-base-port "$INTER_PARTY_BASE_PORT" \
  --coordinator-port "$COORDINATOR_PORT" \
  --user-listen-base-port "$USER_LISTEN_BASE_PORT" \
  --coordinator-protocol "$TRANSPORT"

# ── 3. Launch coordinator ────────────────────────────────────────────────────

MPC_QUIC_CONN_LANES="$MPC_QUIC_CONN_LANES" NETWORK_FORKS="$NETWORK_FORKS" TRACY=1 TRACY_PORT=$((TRACY_BASE_PORT - 1)) \
"$REPO_DIR/target/release/coordinator" \
  --config-file "$ARTIFACT_DIR/config_coordinator.toml" \
  --transport "$TRANSPORT" \
  -t "$TRACE_DIR" \
  --rayon-threads "$RAYON_THREADS" &
coordinator_pid=$!

# In TLS mode, wait for the coordinator to bind before starting workers
if [ "$TRANSPORT" = "tls" ]; then
  for i in $(seq 1 20); do
    if lsof -i :"${COORDINATOR_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
fi

# ── 4. Launch 3 workers ─────────────────────────────────────────────────────

worker_pids=()
for p in 0 1 2; do
  MPC_QUIC_CONN_LANES="$MPC_QUIC_CONN_LANES" NETWORK_FORKS="$NETWORK_FORKS" TRACY=1 TRACY_PORT=$((TRACY_BASE_PORT + p)) \
  "$REPO_DIR/target/release/worker" \
    -c "$ARTIFACT_DIR/config_worker0_${p}.toml" \
    -t "$TRACE_DIR" \
    --network-forks "$NETWORK_FORKS" \
    --rayon-threads "$RAYON_THREADS" \
    -p "$PREPROC_DIR" &
  worker_pids+=($!)
done

# ── 5. Tracy captures ───────────────────────────────────────────────────────

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
  "$TRACY_CAPTURE_BIN" \
    -f \
    -o "$TRACE_DIR/coordinator_${TRACE_SUFFIX}.tracy" \
    -a 127.0.0.1 \
    -p $((TRACY_BASE_PORT - 1)) >/dev/null 2>&1 &
  capture_pids+=($!)
fi

# ── Cleanup trap ─────────────────────────────────────────────────────────────

cleanup() {
  echo ""
  echo "Shutting down..."
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

# ── 6. Wait for workers to be ready ─────────────────────────────────────────

sleep 3

WORKER_ADDRS="127.0.0.1:${USER_LISTEN_BASE_PORT}"
WORKER_ADDRS="${WORKER_ADDRS},127.0.0.1:$((USER_LISTEN_BASE_PORT + 1))"
WORKER_ADDRS="${WORKER_ADDRS},127.0.0.1:$((USER_LISTEN_BASE_PORT + 2))"

echo ""
echo "=== Infrastructure Ready ==="
echo "  Config:  $ARTIFACT_DIR/config_delegator.toml"
echo "  Workers: $WORKER_ADDRS"
echo ""
echo "Run a client in another terminal, e.g.:"
echo "  ./target/release/sha2-chain --config-path $ARTIFACT_DIR/config_delegator.toml --num-iters 10"
echo "  ./target/release/zkemail --config-path $ARTIFACT_DIR/config_delegator.toml --email-path ... --from-domain ..."
echo ""
echo "Press Ctrl-C to shut down."

# Stay alive — wait for any background process to exit (or Ctrl-C)
wait
