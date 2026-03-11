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

## Code organization rules (within each file)
1. Core logic types
2. Core logic impls
3. Core logic funcs
4. Helper types
5. Helper impls
6. Helper funcs

## Performance rules
- Avoid clone(), to_vec(), collect() (including implicit collects) on hot paths unless unavoidable.
  Prefer: borrowing, iter_mut, chunks_exact(_mut), split_at_mut, SmallVec, preallocation with with_capacity, extend_from_slice, and iterator fusion.
- Avoid repeated full passes over large vectors:
  combine passes; compute multiple derived arrays in one traversal.
- Prefer par_iter / into_par_iter for large independent work:
  only if thread-safe + no ordering side effects.
- Avoid extra buffering between stages unless it reduces comms or enables parallelism.
