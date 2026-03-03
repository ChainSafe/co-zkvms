# co-jolt2 Deferred Optimizations

## Operand Q

Do shared indexing & unmask on rings (ring ehat16) -> b2a

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
