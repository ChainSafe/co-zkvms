# co-jolt2 Deferred Optimizations

## Critical
VirtualAdvice - public `advice`!

ring_to_field_b2a_many
would it be better to pass batch: EdaBitsBatch<T, F> as owned instead of ref. Doesn't make sence to pass it as ref anyway

## Missing networked par_chunks
- `co-jolt2/src/zkvm/witness.rs` `for poly in polynomials` 
- `co-jolt2/src/zkvm/suffixes/future.rs` `ring_to_field_b2a_many`, `bit_inject_field_many`

## Memory optimizations
- ManuallyDrop in Dory::commit_rep3 — safe?

## Batch condensation + cache_phase `mul_vec` (saves 6 communication rounds)

**File**: `src/zkvm/instruction_lookups/read_raf_checking.rs`

In `cache_phase`, `condensation_mul_vec` and the `ra_acc * v_shifted` mul_vec are currently
two separate calls. Since they operate on independent data within the same phase, they could
be merged into a single `mul_vec` call, saving 1 communication round per phase × 6 phases
(phases 2-7) = 6 rounds total.

**Approach**: Concatenate inputs for both mul_vec calls, execute once, split the result.
`v_shifted` must be kept alive slightly longer (4 MB for 65536-element table).

**Complexity**: Moderate refactor — requires threading `v_shifted` through the condensation path.

## (Spartan/R1CS) Reduce `Rep3Value` branching overhead in hot loops

`Rep3Value`-based arithmetic is convenient for keeping public values public, but it can add
nontrivial branching/method-call overhead when used per-term inside tight inner loops (e.g.
`compute_claimed_witness_evals_rep3` and related Spartan/R1CS streaming).

**Follow-up idea**: keep the “public vs shared” split at the algorithm level (two accumulators),
or introduce a specialized, branch-minimized fast path for the common cases (Public/Public,
Shared/Public) while still avoiding `promote_to_trivial_share` cascades.

## Stage 3 (and close) public opening-claims sent as secret shares

Some openings/claims that are logically PUBLIC are currently transmitted and stored as Rep3 shares
via `promote_to_trivial_share`. This increases bandwidth and can cause accidental downstream
“share × share” arithmetic paths (e.g., via `AdditiveShare` conversions) when the claim is later
combined with shared values.

**Known occurrences (not exhaustive):**
- `src/zkvm/instruction_lookups/read_raf_checking.rs`: table-flag claims and `raf_flag_claim` returned as Rep3 shares.
- `src/zkvm/registers/val_evaluation.rs`: `wa_claim` returned as a Rep3 share.
- `src/zkvm/ram/val_evaluation.rs` and `src/zkvm/ram/output_check.rs`: `wa_claim` returned as a Rep3 share.
- `src/poly/opening_proof.rs`: some public claims promoted to Rep3 shares before sending to the coordinator.

**Goal**: keep these as public scalars end-to-end and only secret-share them when strictly required by
a downstream protocol interface.

## Risky optimizations (may hurt memory)

### Increase RA `wr_tile` default (stage4)
**File**: `src/subprotocols/mles_product_sum.rs:124-128`

`compute_mles_product_16_rep3` tiles the wr dimension with `RA_MLES_WR_TILE=32`. Each tile does
3 sequential reshares (levels 1→2→3, data-dependent — cannot be batched). Increasing `wr_tile`
reduces total reshare count from `3 * ceil(n_wr/32)` to `3 * ceil(n_wr/wr_tile)`.

Proposed: increase default to 256. Memory cost: level1 buffer grows to `256 * n_wl * 24 * sizeof(F)`.

**Risk**: Larger tile = larger intermediate buffers. For big instances could spike RSS.

### Remove B2A outer chunking loop (witness phases)
**File**: `src/zkvm/witness.rs:89-94`

`fill_field_from_operands_sparse_u64` processes jobs in sequential chunks of `B2A_CHUNK=8192`,
each calling `par_chunks_preproc`. The outer loop serializes work that could be a single call.

Proposed: remove the `for chunk in jobs.chunks(chunk_size)` loop, pass all jobs at once.

**Risk**: Single large batch means all preprocessing material consumed at once — higher peak memory.

### Parallelize ReadRaf condensation/cache_phase `mul_vec` across forks (stage3)
**Files**: `src/zkvm/instruction_lookups/read_raf_checking.rs:560,1237`

`init_phase` condensation and `cache_phase` each do `mul_vec` on up to ~500K elements using only
the main fork. Could use `par_chunks` to split across forks.

**Risk**: Duplicates intermediate vectors across forks. Memory scales with fork count.

### Async cache_phase || init_phase pipeline (stage3)
**File**: `src/zkvm/instruction_lookups/read_raf_checking.rs:1482-1490`

`cache_phase(i)` and `init_phase(i+1)` run sequentially at each phase boundary. They operate on
different data and could run concurrently on separate forks.

Note: plan.md's claim about overlapping init_phase with sumcheck rounds is wrong — bind triggers
at round end and the next round immediately needs the initialized data. The real opportunity is
concurrent cache_phase + init_phase within the same bind call.

**Risk**: Doubles memory during phase transition (both old and new phase data live simultaneously).
Requires splitting `&mut self` on `ReadRafProverState`. High complexity.

## Make `Rep3Value` support integers (avoid public → F casts)

Several witness/VM paths first interpret public data as integers (PC, immediates, advice words, RAM
words), then convert to field elements (`F`) and immediately promote to trivial shares. This:
- adds conversion overhead,
- encourages downstream code to treat the value as “shared” and accidentally use networked `mul_vec`
  or “share × share” multiplication paths even when one operand is public.

**Idea**: extend `Rep3Value` (or a sibling type) with integer variants (e.g. `u64`, `i128`), and add
fast-path arithmetic/conversion routines so public values can stay in the integer domain longer.
