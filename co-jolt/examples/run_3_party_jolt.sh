#!/usr/bin/env bash
mkdir -p data
cargo build --example rep3_jolt --release

cd ../mpc-net
cargo build --bin gen_cert --release
cd ../co-jolt
[[ -f "data/cert_coordinator.der" ]] || ../target/release/gen_cert -k data/key_coordinator.der -c data/cert_coordinator.der -s ec2-18-221-225-225.us-east-2.compute.amazonaws.com -s coordinator

[[ -f "data/key0.der" ]] || ../target/release/gen_cert -k data/key0.der -c data/cert0.der -s ec2-18-222-169-228.us-east-2.compute.amazonaws.com -s ip6-localhost -s 127.0.0.1 -s party0
[[ -f "data/key1.der" ]] || ../target/release/gen_cert -k data/key1.der -c data/cert1.der -s ec2-3-15-170-134.us-east-2.compute.amazonaws.com -s ip6-localhost -s 127.0.0.1 -s party1
[[ -f "data/key2.der" ]] || ../target/release/gen_cert -k data/key2.der -c data/cert2.der -s ec2-18-221-34-219.us-east-2.compute.amazonaws.com -s ip6-localhost -s 127.0.0.1 -s party2

# ../target/release/examples/rep3_jolt -c examples/config_coordinator.toml &
# ../target/release/examples/rep3_jolt -c examples/config_party1.toml &
# ../target/release/examples/rep3_jolt -c examples/config_party2.toml &
# ../target/release/examples/rep3_jolt -c examples/config_party3.toml 
