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

## Acknowledgements

This ongoing work builds up on the following projects:

- `co-jolt` introduces MPC proving into `jolt-core` from [a16z/jolt](https://github.com/a16z/jolt) rev: `42de0ca1f581dd212dda7ff44feee806556531d2`
- `mpc-net` and `mpc-core` crates are based on [TaceoLabs/co-snarks](https://github.com/TaceoLabs/co-snarks)

## Known issues

- Delegator acts as coordinator (with logarithmic communication overhead). See [coordinator role delegation](https://hackmd.io/@timofey/SJ7OU5dugg#Coordinator-role-delegation).
- Limited private shared witness generation. Currently, the coordinator generates a witness secret share and sends it to the workers. See [witness extension in MPC](https://hackmd.io/@timofey/SJ7OU5dugg#Witness-extension-in-MPC)
- Limited scaling. Currently, only the primary sumcheck for instruction lookups supports parallelization via worker subnets.
- See [other pending optimizations and features](https://hackmd.io/@timofey/SJ7OU5dugg#Known-issues-Optimizations-and-Features)
