# co-jolt2 Deferred Optimizations

ring_to_field_b2a_many
would it be better to pass batch: EdaBitsBatch<T, F> as owned instead of ref. Doesn't make sence to pass it as ref anyway

## Missing networked par_chunks
- `co-jolt2/src/zkvm/witness.rs` `for poly in polynomials` 
- `co-jolt2/src/zkvm/suffixes/future.rs` `ring_to_field_b2a_many`, `bit_inject_field_many`

## Missing misc optimizations
- Use FWHT in `one_hot::select_public_table_at_masked_index`
- init_operandQ_polys: Do shared indexing & unmask on rings (ring ehat16) -> b2a

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

## Make `Rep3Value` support integers (avoid public → F casts)

Several witness/VM paths first interpret public data as integers (PC, immediates, advice words, RAM
words), then convert to field elements (`F`) and immediately promote to trivial shares. This:
- adds conversion overhead,
- encourages downstream code to treat the value as “shared” and accidentally use networked `mul_vec`
  or “share × share” multiplication paths even when one operand is public.

**Idea**: extend `Rep3Value` (or a sibling type) with integer variants (e.g. `u64`, `i128`), and add
fast-path arithmetic/conversion routines so public values can stay in the integer domain longer.
