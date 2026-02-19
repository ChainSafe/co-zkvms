#!/usr/bin/env bash
set -euo pipefail

NUM_ITERS=${NUM_ITERS:-1}
TRACE_DIR=${TRACE_DIR:-./.traces}
ARTIFACT_DIR=.artifacts

mkdir -p "$ARTIFACT_DIR"
mkdir -p "$TRACE_DIR"

# Build the example binary (release mode)
# Note: Guest ELF is auto-compiled by Program::build() on first run
cargo build --example rep3_jolt --release --features test-utils

# Build gen_configs
cd ../mpc-net
cargo build --bin gen_configs --release
cd ../co-jolt2

# Generate network configs (1 worker per party)
# Configs, certs, and keys all go into .artifacts/
../target/release/gen_configs \
  -n 1 \
  -o "$ARTIFACT_DIR" \
  -c "$ARTIFACT_DIR" \
  -k "$ARTIFACT_DIR"

# Fix cert/key paths in generated configs: data/ -> .artifacts/
for f in "$ARTIFACT_DIR"/config_*.toml; do
  sed -i '' 's|"data/|".artifacts/|g' "$f"
done

# # Export RUST_LOG=trace for chrome tracing
# export RUST_LOG=trace

# Launch coordinator
../target/release/examples/rep3_jolt \
  -c "$ARTIFACT_DIR/config_coordinator.toml" \
  -t "$TRACE_DIR" -n "$NUM_ITERS" &

# Launch 3 workers (party 0, 1, 2)
for p in 0 1 2; do
  ../target/release/examples/rep3_jolt \
    -c "$ARTIFACT_DIR/config_worker0_${p}.toml" \
    -t "$TRACE_DIR" -n "$NUM_ITERS" &
done

wait
echo "Traces written to $TRACE_DIR"
