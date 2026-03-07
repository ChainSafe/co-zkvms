#!/usr/bin/env bash
set -euo pipefail

export RUSTFLAGS="${RUSTFLAGS:--A warnings}"

NUM_ITERS=${NUM_ITERS:-1}
TRACE_DIR=${TRACE_DIR:-./.traces}
ARTIFACT_DIR=.artifacts
# Base port for Tracy profiling. Worker p listens on TRACY_BASE_PORT + p.
# Connect with: tracy-capture -a 127.0.0.1 -p <port>
#   worker 0 → 8086, worker 1 → 8087, worker 2 → 8088
TRACY_BASE_PORT=${TRACY_BASE_PORT:-8086}
# When set, preprocessing is saved to (and loaded from) this directory.
PREPROC_DIR=${PREPROC_DIR:-./.preprocessing}
# When set to 1, builds with the `reuse-preproc` feature so that mmap'd
# backing files are NOT zeroed on read and can be loaded multiple times.
REUSE_PREPROC=${REUSE_PREPROC:-0}
# When set to 1, only runs preprocessing and exits.
PREPROC_ONLY=${PREPROC_ONLY:-0}
# When set to 1, builds with Tracy allocation tracking and jemalloc plots.
TRACY_ALLOC=${TRACY_ALLOC:-0}
# When set to 1, runs `tracy-capture` and writes `worker{0,1,2}.tracy` into TRACE_DIR.
TRACY_CAPTURE=${TRACY_CAPTURE:-0}
# When set to 1, writes tracy-capture logs to TRACE_DIR (otherwise suppresses them).
TRACY_CAPTURE_LOG=${TRACY_CAPTURE_LOG:-0}
# Optional: jemalloc tuning preset (Linux most useful). Applied via MALLOC_CONF unless MALLOC_CONF is already set.
# Values:
#   default     - do nothing
#   return_os   - prefer returning memory to OS (may reduce RSS retention, may cost perf)
#   aggressive  - aggressively purge/decay (diagnostic; higher perf/noise risk)
#   narenas1    - use a single arena (diagnostic; can reduce retention with many threads)
JEMALLOC_PRESET=${JEMALLOC_PRESET:-default}
# Optional: override rayon thread count used by the example (passed as CLI arg).
RAYON_THREADS=${RAYON_THREADS:-}
# Optional: run multiple proofs in the same worker process (requires `reuse-preproc`).
REPEAT_PROOFS=${REPEAT_PROOFS:-1}

mkdir -p "$ARTIFACT_DIR"
mkdir -p "$TRACE_DIR"

FEATURES="test-utils"
if [ "$REUSE_PREPROC" = "1" ]; then
  FEATURES="test-utils,reuse-preproc"
fi
if [ "$TRACY_ALLOC" = "1" ]; then
  FEATURES="$FEATURES,tracy-mem,jemalloc-stats"
fi

MALLOC_CONF_EFFECTIVE=${MALLOC_CONF:-}
if [ -z "${MALLOC_CONF_EFFECTIVE}" ] && [ "$JEMALLOC_PRESET" != "default" ]; then
  case "$JEMALLOC_PRESET" in
    return_os)
      MALLOC_CONF_EFFECTIVE="background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000,retain:false"
      ;;
    aggressive)
      MALLOC_CONF_EFFECTIVE="background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0,retain:false"
      ;;
    narenas1)
      MALLOC_CONF_EFFECTIVE="background_thread:true,narenas:1,percpu_arena:disabled,dirty_decay_ms:1000,muzzy_decay_ms:1000,retain:false"
      ;;
    *)
      echo "Unknown JEMALLOC_PRESET=$JEMALLOC_PRESET (expected default|return_os|aggressive|narenas1)" >&2
      exit 1
      ;;
  esac
fi
if [ -n "${MALLOC_CONF_EFFECTIVE}" ]; then
  export MALLOC_CONF="${MALLOC_CONF_EFFECTIVE}"
fi

PROOF_ARGS=()
if [ -n "$RAYON_THREADS" ]; then
  PROOF_ARGS+=(--rayon-threads "$RAYON_THREADS")
fi
if [ -n "${NETWORK_FORKS:-}" ]; then
  PROOF_ARGS+=(--network-forks "$NETWORK_FORKS")
fi
if [ "$REPEAT_PROOFS" -gt 1 ]; then
  PROOF_ARGS+=(--repeat-proofs "$REPEAT_PROOFS")
fi

TIME_CMD=()
TIME_RSS_PATTERN="Maximum resident set size"
if /usr/bin/time -v true >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -v --)
elif /usr/bin/time -l true >/dev/null 2>&1; then
  TIME_CMD=(/usr/bin/time -l)
  TIME_RSS_PATTERN="maximum resident set size"
elif command -v gtime >/dev/null 2>&1 && gtime -v true >/dev/null 2>&1; then
  TIME_CMD=(gtime -v --)
else
  echo "warning: no verbose time command found; Max RSS will not be reported" >&2
fi

# Build the example binary (release mode)
# Note: Guest ELF is auto-compiled by Program::build() on first run
cargo build --example rep3_jolt --release --features "$FEATURES"

# Build gen_configs
cd ../mpc-net
cargo build --bin gen_configs --release
cd ../co-jolt2

# Generate network configs (1 worker per party)
# Configs, certs, and keys all go into ARTIFACT_DIR.
# The -c flag sets both where DER files are written AND the paths embedded in TOMLs.
../target/release/gen_configs \
  -n 1 \
  -o "$ARTIFACT_DIR" \
  -c "$ARTIFACT_DIR" \
  -k "$ARTIFACT_DIR"

# # Export RUST_LOG=trace for chrome tracing
# export RUST_LOG=trace

# Optionally pass --preproc-dir to workers.  Each worker stores its data in
# <PREPROC_DIR>/party_<id>/ so files from different parties don't collide.
PREPROC_ARGS=()
if [ -n "$PREPROC_DIR" ]; then
  mkdir -p "$PREPROC_DIR"
  PREPROC_ARGS=(--preproc-dir "$PREPROC_DIR")
fi
if [ "$PREPROC_ONLY" = "1" ]; then
  PREPROC_ARGS+=(--preprocess-only true)
fi

# Launch coordinator
../target/release/examples/rep3_jolt \
  -c "$ARTIFACT_DIR/config_coordinator.toml" \
  -t "$TRACE_DIR" -n "$NUM_ITERS" \
  ${PREPROC_ARGS[@]+"${PREPROC_ARGS[@]}"} \
  ${PROOF_ARGS[@]+"${PROOF_ARGS[@]}"} &
coordinator_pid=$!

capture_pids=()
if [ "$TRACY_CAPTURE" = "1" ]; then
  # Prefer brew-installed tracy-capture (0.13.1, protocol 76) over any system one.
  TRACY_CAPTURE_BIN=${TRACY_CAPTURE_BIN:-$(command -v tracy-capture 2>/dev/null || echo tracy-capture)}
  for p in 0 1 2; do
    capture_log="$TRACE_DIR/tracy/tracy-capture-worker${p}.log"
    if [ "$TRACY_CAPTURE_LOG" = "1" ]; then
      "$TRACY_CAPTURE_BIN" \
        -f \
        -o "$TRACE_DIR/worker${p}_$NUM_ITERS.tracy" \
        -a 127.0.0.1 \
        -p $((TRACY_BASE_PORT + p)) >"$capture_log" 2>&1 &
    else
      "$TRACY_CAPTURE_BIN" \
        -f \
        -o "$TRACE_DIR/worker${p}_$NUM_ITERS.tracy" \
        -a 127.0.0.1 \
        -p $((TRACY_BASE_PORT + p)) >/dev/null 2>&1 &
    fi
    capture_pids+=($!)
  done
fi

# Launch 3 workers (party 0, 1, 2) with Tracy on separate ports.
# Each runs in a subshell so we can capture /usr/bin/time -v stderr and extract Max RSS.
worker_pids=()
for p in 0 1 2; do
  (
    tmpfile=$(mktemp)
    if [ ${#TIME_CMD[@]} -gt 0 ]; then
      TRACY=1 TRACY_PORT=$((TRACY_BASE_PORT + p)) \
        "${TIME_CMD[@]}" ../target/release/examples/rep3_jolt \
          -c "$ARTIFACT_DIR/config_worker0_${p}.toml" \
          -t "$TRACE_DIR" -n "$NUM_ITERS" \
          ${PREPROC_ARGS[@]+"${PREPROC_ARGS[@]}"} \
          ${PROOF_ARGS[@]+"${PROOF_ARGS[@]}"} 2>"$tmpfile"
      maxrss_line=$(grep -i "$TIME_RSS_PATTERN" "$tmpfile" | tail -n 1 || true)
      case "$maxrss_line" in
        Maximum*)
          maxrss=$(printf '%s\n' "$maxrss_line" | awk '{print $NF}')
          ;;
        *)
          maxrss=$(printf '%s\n' "$maxrss_line" | awk '{print $1}')
          ;;
      esac
      if [ -n "$maxrss" ]; then
        echo "worker${p}: Max RSS = ${maxrss} kB"
      fi
    else
      TRACY=1 TRACY_PORT=$((TRACY_BASE_PORT + p)) \
        ../target/release/examples/rep3_jolt \
        -c "$ARTIFACT_DIR/config_worker0_${p}.toml" \
        -t "$TRACE_DIR" -n "$NUM_ITERS" \
        ${PREPROC_ARGS[@]+"${PREPROC_ARGS[@]}"} \
        ${PROOF_ARGS[@]+"${PROOF_ARGS[@]}"} 2>"$tmpfile"
    fi
    if [ "$JEMALLOC_PRESET" != "default" ]; then
      echo "worker${p}: JEMALLOC_PRESET=$JEMALLOC_PRESET"
    fi
    rm -f "$tmpfile"
  ) &
  worker_pids+=($!)
done

cleanup_children() {
  kill "$coordinator_pid" "${worker_pids[@]}" "${capture_pids[@]}" 2>/dev/null || true
  wait "$coordinator_pid" "${worker_pids[@]}" "${capture_pids[@]}" 2>/dev/null || true
}

trap cleanup_children EXIT

for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    cleanup_children
    exit 1
  fi
done

if ! wait "$coordinator_pid"; then
  cleanup_children
  exit 1
fi

for pid in "${capture_pids[@]}"; do
  wait "$pid"
done

trap - EXIT
echo "Traces written to $TRACE_DIR"
if [ -n "$PREPROC_DIR" ]; then
  echo "Preprocessing data in $PREPROC_DIR/{party_0,party_1,party_2}/"
  echo "(reuse-preproc: $REUSE_PREPROC)"
fi
