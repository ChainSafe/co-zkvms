# co-jolt2 Deferred Optimizations

## Batch condensation + cache_phase `mul_vec` (saves 6 communication rounds)

**File**: `src/zkvm/instruction_lookups/read_raf_checking.rs`

In `cache_phase`, `condensation_mul_vec` and the `ra_acc * v_shifted` mul_vec are currently
two separate calls. Since they operate on independent data within the same phase, they could
be merged into a single `mul_vec` call, saving 1 communication round per phase × 6 phases
(phases 2-7) = 6 rounds total.

**Approach**: Concatenate inputs for both mul_vec calls, execute once, split the result.
`v_shifted` must be kept alive slightly longer (4 MB for 65536-element table).

**Complexity**: Moderate refactor — requires threading `v_shifted` through the condensation path.

## Reduce `Rep3Value` branching overhead in hot loops

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

## Optimize `new_shifted()` (R>1 rotation path)

**Files**: `src/poly/ra_poly.rs`, `src/poly/one_hot_polynomial.rs`

When `RAND_OHV_ROTATIONS > 1`, `new_shifted()` eagerly materializes an n-length dense polynomial
(`RoundN`) because different cycles map to different rotation slots, preventing the lazy Round1
table-lookup representation.

### Problem
- Allocates n × 64 bytes (two field elements per `Rep3PrimeFieldShare`) per polynomial.
- Multiple one-hot polynomials live simultaneously during `reduce_and_prove`, spiking RSS.
- Dense representation has worse cache behavior than Round1's 1-byte-index + L1-cached table.

### Optimization ideas

1. **Lazy multi-slot Round1**: Extend `Rep3RaPolynomialRound1` to store `Vec<Vec<Rep3PrimeFieldShare<F>>>`
   (per-slot shifted tables) + `rotation_slot_by_row`. `get_bound_coeff(j)` looks up
   `shifted_tables[slot][index]` — still O(1) with L1-cached tables (R×K elements, e.g. 16×16=256
   field elements = 16 KB for R=16). Avoids n-length dense materialization entirely. Round1→Round2
   bind precomputes f_0/f_1 per slot (O(R×K) work). Defers RoundN materialization to round 3 at
   n/8 length.

2. **Streaming materialization in Round3→RoundN**: Instead of materializing the full n-length dense
   polynomial in `new_shifted()`, let Round1→Round2→Round3 handle the first 3 cycle-variable rounds
   (as in approach 1), then materialize at Round3→RoundN when the polynomial is n/8 length. This
   is 8× less memory than upfront materialization.

3. **Sparse representation**: Since many cycles are `None` (padding), store only active-cycle
   coefficients in a compact representation. During sumcheck, `get_bound_coeff` returns zero for
   padding cycles without touching memory. Benefit depends on sparsity ratio.

**Recommended**: Approach 1 (lazy multi-slot Round1) — minimal code change, eliminates the
n-length allocation entirely, and naturally falls back to existing Round1→Round2→Round3→RoundN
progression.

---

## Risky optimizations (may hurt memory)

### Increase RA `wr_tile` default (stage4)
**File**: `src/subprotocols/mles_product_sum.rs:124-128`

`compute_mles_product_16_rep3` tiles the wr dimension with `RA_MLES_WR_TILE=32`. Each tile does
3 sequential reshares (levels 1→2→3, data-dependent — cannot be batched). Increasing `wr_tile`
reduces total reshare count from `3 * ceil(n_wr/32)` to `3 * ceil(n_wr/wr_tile)`.

Proposed: increase default to 256. Memory cost: level1 buffer grows to `256 * n_wl * 24 * sizeof(F)`.

**Risk**: Larger tile = larger intermediate buffers. For big instances could spike RSS.

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
