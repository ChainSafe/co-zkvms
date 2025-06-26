#!/bin/bash
cd ./noir-examples/poseidon-rounds && nargo compile && nargo check && cd ../..
cargo run --bin noir-r1cs -- circuit_stats ./noir-examples/poseidon-rounds/target/basic.json
cargo run --bin noir-r1cs -- prepare ./noir-examples/poseidon-rounds/target/basic.json -o ./artifacts/noir_proof_scheme.json

cargo run --bin co-spartan -- setup --r1cs-noir-instance-path ./artifacts/noir_proof_scheme.json --r1cs-input-path ./noir-examples/poseidon-rounds/Prover.toml --log-num-workers-per-party 3 --log-num-public-workers 3 --key-out ./artifacts

# RUST_BACKTRACE=1 cargo run --bin co-spartan -- work --log-num-workers-per-party 1 --log-num-public-workers 3 --key-file ./artifacts/inst_3_0_1/test --worker-id 1 --local 0
# RUST_BACKTRACE=1 cargo run --bin co-spartan -- work --log-num-workers-per-party 1 --log-num-public-workers 3 --key-file ./artifacts/inst_3_0_1/test --worker-id 1 --local 1
mpirun -np 7 cargo run --bin co-spartan -- work --log-num-workers-per-party 1 --log-num-public-workers 2 --key-file ./artifacts/inst_3_1_2/test --worker-id 1 --local 0
