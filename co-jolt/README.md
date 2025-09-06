# Co-Jolt zkVM

Private collaborative zkVM proof delegation based on [Jolt](https://github.com/a16z/jolt).

See the design document: [Private Collaborative zkVM](https://hackmd.io/@timofey/SJ7OU5dugg)

## Requirements

- Rust `nightly`
- Misc packages: `build-essential libssl-dev pkg-config`

## Demo (SHA2 chain)

```bash
export NUM_ITERATIONS=10
export RUST_LOG=trace
bash ./examples/run_3_party_jolt.sh
```

## Benchmarks

`tracing_chrome` trace logs are stored in `traces/` directory and can be viewed in `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev/).

### Setup
- 3 parties (1 worker per party) — m7i.2xlarge, 8 vCPUs 32 GiB RAM
- coordinator — m7i.2xlarge, 8 vCPUs 32 GiB RAM
- program: [SHA2 chain (100-300 iterations)](https://github.com/nulltea/co-zkvms/blob/dc2b785b3d3a4873f36215e6e272665199bba885/co-jolt/examples/sha3-chain/guest/src/lib.rs#L6)

### Prover — Per module / Core operations

<img width="4899" height="1424" alt="image" src="https://github.com/user-attachments/assets/be8ad8e7-9604-4bb9-8442-e5b66862f370" />

### Witness generation

<img width="4645" height="1329" alt="image" src="https://github.com/user-attachments/assets/6642ea08-5f1d-4b73-b0f7-066831f17f1f" />


## Acknowledgements

This ongoing work builds up on the following projects:

- `co-jolt` introduces MPC proving into `jolt-core` from [a16z/jolt](https://github.com/a16z/jolt) rev: `42de0ca1f581dd212dda7ff44feee806556531d2`
- `mpc-net` and `mpc-core` crates are based on [TaceoLabs/co-snarks](https://github.com/TaceoLabs/co-snarks)

## Known issues

- Delegator acts as coordinator (with logarithmic communication overhead). See [coordinator role delegation](https://hackmd.io/@timofey/SJ7OU5dugg#Coordinator-role-delegation).
- Limited scaling. Currently, only the primary sumcheck for instruction lookups supports parallelization via worker subnets.
- See [other pending optimizations and features](https://hackmd.io/@timofey/SJ7OU5dugg#Known-issues-Optimizations-and-Features)
