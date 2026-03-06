# co-jolt2 (Claude Code)

## Context
- Read: `co-jolt2/PROJECT.md`

## Build / Test rule
- Always run with: `RUSTFLAGS="-A warnings"`

## Commands
- Integration test:
  - `RUSTFLAGS="-A warnings" cargo test -p co-jolt2 --test dag_correct --features test-utils -- --nocapture`
- Bench:
  - `cd co-jolt2 && REUSE_PREPROC=1 NUM_ITERS=1 bash examples/run_rep3_jolt.sh`
