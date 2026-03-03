# Rep3 Jolt DAG stage2 prover refactor report

Date: 2026-03-03

Scope:
- `co-jolt2/src/zkvm/dag/*`
- stage2 prover files under `co-jolt2/src/zkvm/*/` (notably Spartan, RAM, instruction lookups)

Goals:
1) Match vanilla semantics where reasonable
2) Simplify structure
3) Reduce MPC overhead + allocations (fewer rounds/bytes, lower CPU, lower peak memory)

This change set is intended to preserve external behavior and determinism (same outputs for same inputs).

## Major structural changes

### Lookups stage2
- `Rep3LookupsDagWorker` now shares `one_hot_polys` and derived `G` across stage2 instances via `Arc`:
  - avoids deep cloning large vectors/polynomials when instantiating multiple sumchecks.
- Replaced a per-dimension scan to compute `g_c` with a fused pass that computes all `g_c[i][cycle]` in a single traversal, then converts to k-space.

### RAM stage2
- `Rep3RamDagWorker` now stores `val_init` / `val_final` as `Rep3DensePolynomial` (Arc-backed) rather than raw `Vec` coefficients.
  - stage2 workers consume these cheap clones rather than cloning large vectors.
- `Rep3OutputSumcheckWorker::new` now takes `Rep3DensePolynomial` directly.
- Read/write-checking checkpoint computation now fills a flattened checkpoints buffer in one pass (no intermediate `Vec<Vec<...>>`).

### Sumcheck message generation
- For common degree-3 paths:
  - Added a fixed-size `[...; 3]` evaluation path in `MixedPolynomial`.
  - Spartan inner sumcheck uses it when `max_degree <= 3` (falls back to the existing `Vec` path otherwise).
- Output-check sumcheck message generation avoids per-index heap allocations by computing base evals into fixed-size arrays and then extending.

### Minor build-ups
- DAG stage2 instance vectors avoid `chain(...).collect()` in favor of preallocated `Vec::with_capacity(...)` + `push/extend`.

## Perf wins (allocs/comms/CPU)

Allocations/copies removed or reduced:
- Avoid deep clones when wiring stage2 lookup workers (`Arc` for `one_hot_polys` and `G`).
- Avoid cloning large `val_init` / `val_final` vectors into RAM output-check stage2 workers (Arc-backed dense polynomials).
- Removed intermediate checkpoint/delta buffers in RAM read/write-checking (one streaming pass into a flattened buffer).
- Reduced repeated small heap allocations in tight per-round sumcheck message codepaths (`Vec` → fixed-size arrays).

Comms/MPC overhead changes:
- RAM output-check now uses `rep3_arithmetic::sub_shared_by_public` for shared-minus-public subtraction rather than promoting a public value to a trivial share and subtracting.
  - This keeps the operation explicit and avoids unnecessary representation changes (and reduces downstream risk of “public becomes rep3” propagation).

## `promote_to_trivial_share` red-flag inventory (stage2-adjacent)

Occurrences worth reviewing:
- `co-jolt2/src/zkvm/instruction_lookups/ra_virtual.rs` (multiple call sites)
- `co-jolt2/src/zkvm/instruction_lookups/read_raf_checking.rs` (local helper + claim pushes)
- `co-jolt2/src/zkvm/ram/val_evaluation.rs`
- `co-jolt2/src/zkvm/ram/read_write_checking.rs`
- `co-jolt2/src/zkvm/ram/output_check.rs` (still present for claims; subtraction site refactored to `sub_shared_by_public`)

Estimated cost impact (qualitative):
- Promoting public scalars into rep3 shares can inflate downstream computation if those values get multiplied/combined as shares rather than as public scalars (potentially turning what could be local field ops into MPC ops and/or extra communication).

Concrete fix direction:
- Keep values public as long as possible; only convert at protocol boundaries where a share is required.
- Prefer dedicated helpers for “shared op public” (`add_shared_by_public`, `sub_shared_by_public`, `mul_shared_by_public`) to prevent accidental “public → rep3” drift.
- For “claim vectors” that are logically public and only needed for transcript binding, consider retaining a parallel public representation and only sharing when required by the sumcheck interface.

## Bench results (e2e)

Command (run from `co-jolt2/`):
- `REUSE_PREPROC=1 NUM_ITERS=1 /usr/bin/time -l bash examples/run_rep3_jolt.sh`

Baseline (before refactor):
- 83.36s real, 3,499,360,256 bytes max RSS

After refactor (observed variance on laptop):
- 90.17s real, 3,598,434,304 bytes max RSS
- 93.16s real, 3,244,261,376 bytes max RSS

Notes:
- These timings include coordinator + 3 workers and can vary due to system load/thermal throttling and macOS `time -l` reporting across child processes.
- The changes are concentrated in stage2 (which is a small fraction of total proving time for this benchmark), so e2e “real” time may be dominated by unrelated phases.

## Semantic risks and validation

Risks:
- Using `Arc` changes ownership/lifetimes; incorrect sharing could lead to accidental mutation or dropped data at phase transitions.
- Switching degree-3 sumcheck message generation from `Vec` to fixed-size arrays must preserve coefficient order and transcript binding.
- Fusing lookup passes must preserve the same per-cycle/per-dimension semantics.

Validation performed:
- Correctness: `cargo test -p co-jolt2 --test dag_correct --features test-utils -- --nocapture` (passed; 1 test).
- e2e bench executed multiple times with `REUSE_PREPROC=1` to ensure no obvious regression or nondeterminism.

