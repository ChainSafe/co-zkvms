#!/usr/bin/env bash
set -euo pipefail

# number of workers $NUM_WORKERS_PER_PARTY

mkdir -p data
cargo build --example rep3_jolt --release

cd ../mpc-net
# cargo build --bin gen_cert --release
cargo build --bin gen_configs --release
cd ../co-jolt
# [[ -f "data/cert_coordinator.der" ]] || ../target/release/gen_cert -k data/key_coordinator.der -c data/cert_coordinator.der -s localhost -s ip6-localhost -s 127.0.0.1 -s coordinator

# [[ -f "data/key0.der" ]] || ../target/release/gen_cert -k data/key0.der -c data/cert0.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party0
# [[ -f "data/key1.der" ]] || ../target/release/gen_cert -k data/key1.der -c data/cert1.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party1
# [[ -f "data/key2.der" ]] || ../target/release/gen_cert -k data/key2.der -c data/cert2.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party2


../target/release/gen_configs -n $NUM_WORKERS_PER_PARTY

# launch coordinator
../target/release/examples/rep3_jolt -c examples/config_coordinator.toml &

# launch workers
for w in $(seq 0 $((NUM_WORKERS_PER_PARTY-1))); do
    for p in 0 1 2; do
        ../target/release/examples/rep3_jolt -c examples/config_worker${w}_${p}.toml &
    done
done

wait
