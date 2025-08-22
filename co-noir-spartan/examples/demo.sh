#!/bin/bash
cd ./noir-r1cs/noir-examples/poseidon-rounds && nargo compile && nargo check && cd ../../..
mkdir -p ./artifacts/poseidon-rounds-10

cargo build --release --bin co-spartan --bin noir-r1cs

mkdir -p data
cd ../mpc-net
cargo build --bin gen_cert --release
cd ../co-noir-spartan
[[ -f "data/cert_coordinator.der" ]] || ../target/release/gen_cert -k data/key_coordinator.der -c data/cert_coordinator.der -s localhost -s ip6-localhost -s 127.0.0.1 -s coordinator

[[ -f "data/key0.der" ]] || ../target/release/gen_cert -k data/key0.der -c data/cert0.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party0
[[ -f "data/key1.der" ]] || ../target/release/gen_cert -k data/key1.der -c data/cert1.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party1
[[ -f "data/key2.der" ]] || ../target/release/gen_cert -k data/key2.der -c data/cert2.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party2

echo "Compiling Noir proof scheme..."
../target/release/noir-r1cs prepare ./noir-r1cs/noir-examples/poseidon-rounds/target/basic.json -o ./artifacts/poseidon-rounds-10/noir_proof_scheme.json

echo "Generating keys..."
../target/release/co-spartan setup --r1cs-noir-scheme-path ./artifacts/poseidon-rounds-10/noir_proof_scheme.json --artifacts-dir ./artifacts/poseidon-rounds-10 \
    --log-num-workers-per-party 1 

echo "Running coordinator and workers..."

export ARTIFACTS_DIR=./artifacts/poseidon-rounds-10
export LOG_NUM_WORKERS_PER_PARTY=1
export R1CS_NOIR_SCHEME_PATH=./artifacts/poseidon-rounds-10/noir_proof_scheme.json
export R1CS_INPUT_PATH=./noir-r1cs/noir-examples/poseidon-rounds/Prover.toml

../target/release/co-spartan work -c examples/config_coordinator.toml &
../target/release/co-spartan work -c examples/config_worker_0_1.toml &
../target/release/co-spartan work -c examples/config_worker_0_2.toml &
../target/release/co-spartan work -c examples/config_worker_0_3.toml &
../target/release/co-spartan work -c examples/config_worker_1_1.toml &
../target/release/co-spartan work -c examples/config_worker_1_2.toml &
../target/release/co-spartan work -c examples/config_worker_1_3.toml 
