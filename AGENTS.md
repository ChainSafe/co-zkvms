# co-jolt2

Read first: `co-jolt2/PROJECT.md`

## Hard rule
Run all Rust commands with:
- `RUSTFLAGS="-A warnings"`

## Quick checks
- Integration test:
  - `RUSTFLAGS="-A warnings" cargo test -p co-jolt2 --test dag_correct --features test-utils -- --nocapture`
- Bench:
  - `cd co-jolt2 && REUSE_PREPROC=1 NUM_ITERS=1 bash examples/run_rep3_jolt.sh`
