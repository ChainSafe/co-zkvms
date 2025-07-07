#!/usr/bin/env bash
mkdir -p data
cargo build --example three_party --release
# [[ -f "data/key0.der" ]] || cargo run --bin gen_cert -- -k data/key0.der -c data/cert0.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party0
# [[ -f "data/key1.der" ]] || cargo run --bin gen_cert -- -k data/key1.der -c data/cert1.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party1
# [[ -f "data/key2.der" ]] || cargo run --bin gen_cert -- -k data/key2.der -c data/cert2.der -s localhost -s ip6-localhost -s 127.0.0.1 -s party2

log_num_inputs=${1:-6}

cargo run --example three_party --release -- -c examples/config_party1.toml --log-num-inputs $log_num_inputs &
cargo run --example three_party --release -- -c examples/config_party2.toml --log-num-inputs $log_num_inputs &
cargo run --example three_party --release -- -c examples/config_party3.toml --log-num-inputs $log_num_inputs 
