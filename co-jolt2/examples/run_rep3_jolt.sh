#!/usr/bin/env bash
set -euo pipefail

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

mkdir -p "$ARTIFACT_DIR"
mkdir -p "$TRACE_DIR"

FEATURES="test-utils"
if [ "$REUSE_PREPROC" = "1" ]; then
  FEATURES="test-utils,reuse-preproc"
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

# Launch coordinator
../target/release/examples/rep3_jolt \
  -c "$ARTIFACT_DIR/config_coordinator.toml" \
  -t "$TRACE_DIR" -n "$NUM_ITERS" &

# Launch 3 workers (party 0, 1, 2) with Tracy on separate ports.
for p in 0 1 2; do
  TRACY=1 TRACY_PORT=$((TRACY_BASE_PORT + p)) \
    ../target/release/examples/rep3_jolt \
      -c "$ARTIFACT_DIR/config_worker0_${p}.toml" \
      -t "$TRACE_DIR" -n "$NUM_ITERS" \
      "${PREPROC_ARGS[@]}" &
done

wait
echo "Traces written to $TRACE_DIR"
if [ -n "$PREPROC_DIR" ]; then
  echo "Preprocessing data in $PREPROC_DIR/{party_0,party_1,party_2}/"
  echo "(reuse-preproc: $REUSE_PREPROC)"
fi
