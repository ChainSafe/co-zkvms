# Rep3 Stage 3 Refactor Report (ReadRaf + RAM)

Date: 2026-03-03

## Goals / non-negotiables

- Preserve external behavior + determinism (same outputs for same inputs).
- Preserve public APIs unless a strictly better alternative is obvious and localized.
- Match vanilla semantics where reasonable (ordering, transcript challenges, instance wiring).
- Performance: fewer allocations/copies, lower CPU, lower peak RSS, avoid unnecessary MPC work/rounds.

## Invariants checklist (vanilla-aligned)

- Sumcheck instance ordering within stage3 is unchanged (matches vanilla wiring expectations).
- Transcript and network exchange order is unchanged (no reordering of `exchange`/`send_response` calls).
- Rayon parallelism is only used for commutative reductions (sums of additive shares / public scalars).

## Baseline measurements

Bench:
- Command: `REUSE_PREPROC=1 NUM_ITERS=1 /usr/bin/time -l bash examples/run_rep3_jolt.sh`
- Baseline peak RSS: `6666076160` bytes (~6.21 GiB)
- Baseline wall time: `208.72s` (includes rebuild + run)

## Structural changes

### `src/zkvm/instruction_lookups/read_raf_checking.rs`

- Removed per-bucket `Vec` allocations in `prover_msg_read_checking_inner`:
  - Prefix evaluations are computed into fixed-size stack arrays.
  - Suffix coefficient reads use fixed-size stack buffers (`[AdditiveShare; 8]`), avoiding `collect()`.
- Refactored `combine_shared` into:
  - `combine_shared_weights`: compute the per-suffix public weights once for a `(table, prefixes)` pair.
  - `dot_weights_suffixes`: single dot-product against suffix shares.
- Reduced calls to `LookupTables::combine` by reusing the prefix-derived weights across both left/right
  suffix vectors for the `c2` path.

### `src/zkvm/ram/mod.rs` + `src/zkvm/ram/output_check.rs`

- Represent `val_final` as `MixedPolynomial<F>` (backed by `Vec<Rep3Value<F>>`) so that:
  - known-public memory regions remain `Public(F)`,
  - secret regions remain `Shared(Rep3PrimeFieldShare<F>)`,
  - we avoid materializing a full K-length `Vec<Rep3PrimeFieldShare<F>>` for the final state.
- Dropped `StateManagerWorker.prover_state.final_memory_state.data` immediately after ring→field
  conversion using `mem::take` + `truncate`, reducing peak memory.
- Updated `Rep3OutputSumcheckWorker` to operate on mixed `val_final`:
  - purely public contributions are accumulated as trivial additive shares locally,
  - shared contributions keep the existing `mul_public` + public subtraction flow.

## Performance wins (expected)

- ReadRaf:
  - Eliminated inner-loop heap allocations (`collect()` on prefixes/suffixes) in the per-bucket path.
  - Reduced `LookupTables::combine` calls (reuse weights for `prefixes_c2` across left/right suffixes).
- RAM:
  - Removed `initial_memory_state.clone()` for building `final_memory_field`.
  - Removed the K-length `final_memory_field: Vec<Rep3PrimeFieldShare<F>>` materialization entirely.
  - Removed `[..dram_convert_len].to_vec()` for DRAM ring shares (no extra copy).
  - Freed `final_memory_state.data` immediately after conversion.
- MPC/comms:
  - No new MPC rounds introduced.
  - Avoided unnecessary “public treated as shared” arithmetic in OutputSumcheck when `val_final` is public.

## `promote_to_trivial_share` audit (stage3 + adjacent)

### Category 1: public opening-claims shipped as secret shares (DEFERRED)

These are tracked as a deferred optimization in `TODO.md`:
- ReadRaf: table flags + `raf_flag_claim` returned as Rep3 shares.
- Registers/RAM: `wa_claim` returned as Rep3 shares.
- Opening proof plumbing: some public claims promoted before sending to coordinator.

Downstream impact:
- Increases bandwidth (shares vs scalars).
- Makes it easier to accidentally route a logically-public value into share-based arithmetic, which can
  inflate use of networked multiplication paths.

### Category 2: “representation unification” (more severe)

Observed patterns:
- Per-cycle witness conversion in `src/zkvm/dag/witness.rs` promotes public PC/imm/advice words into
  trivial Rep3 shares early.

Current impact:
- Forces downstream to treat these values as “shared”, even though the verifier knows them.
- Encourages `Shared × Shared -> Additive` arithmetic, and later resharing / network multiplication
  patterns where a `Shared × Public` fast path would suffice.

Missed optimization opportunities:
- Keep these values public longer (or in an integer domain) so:
  - multiplications remain `mul_public` (local) rather than `mul`/`mul_vec` (network),
  - intermediate results don’t enter the additive-share domain unnecessarily.
- Future idea (added to `TODO.md`): make `Rep3Value` support integers to avoid public→`F` casts and reduce
  premature promotion cascades.

## Semantic risks / validation

Risks:
- ReadRaf: weight computation refactor must preserve exact combine semantics and deterministic ordering.
- RAM: mixed `val_final` must match the old dense-shared `val_final` at all indices (including partial
  advice words and output overlays).

Validation:
- Run: `cargo test -p co-jolt2 --test dag_correct --features test-utils -- --nocapture`
- Run: bench with `/usr/bin/time -l` and compare peak RSS + runtime to baseline.

## Post-refactor measurements

- Test: `cargo test -p co-jolt2 --test dag_correct --features test-utils -- --nocapture` ✅
- Bench: `REUSE_PREPROC=1 NUM_ITERS=1 /usr/bin/time -l bash examples/run_rep3_jolt.sh`
  - Cold run (includes `cargo build --release`):
    - `136.22 real`, `493.65 user`, `48.59 sys`
    - Max RSS: `6880641024` bytes (~6.41 GiB)
  - Warm run (after artifacts already built; build step ~no-op):
    - `46.24 real`, `184.49 user`, `37.00 sys`
    - Max RSS: `1976156160` bytes (~1.84 GiB)
