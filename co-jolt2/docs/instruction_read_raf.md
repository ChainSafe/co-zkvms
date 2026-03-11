# Instruction Read RAF

## Intro

### Purpose

Describe the design of the instruction lookup read/RAF subsystem.

### Motivation

Instruction lookups are one of the main places where Jolt trades full obliviousness for structured lookup arguments. The MPC port keeps that structure, so the important questions are which indices remain shared, which helper data is opened, and what leakage budget is being accepted.

## Background

### Key definitions

- RAF: the read-argument framework used for instruction lookup checks.
- RandOHV: random one-hot vector used to mask lookup indices.
- Masked index: `open(k XOR r)` for a secret index `k` and mask `r`.

### References

- `papers/co-zkvms.md`, instruction lookup and witness-extension sections
- Jolt book: `src/how/twist-shout.md`
- Useful code anchors:
  - `co-jolt2/src/poly/one_hot_polynomial.rs`
  - `co-jolt2/src/poly/ra_poly.rs`
  - `co-jolt2/src/zkvm/suffixes/`

## Design

### Role

RAF proves that lookup reads are consistent with the intended instruction tables without materializing a fully secret dense table access.

### Assumptions

- Table identity and lookup family are public.
- Secret lookup indices may be transformed into masked public values if the masking discipline is acceptable.
- The same index ordering is used across witness generation, RAF, and final openings.

### Security invariants

- Any public shortcut must be justified by verifier knowledge or an explicit declassification rule.
- Masked index openings must use a masking discipline whose leakage is understood and accepted.
- Suffix and histogram helpers must consume the same lookup-index semantics as witness generation.

### Design choices

- Use RandOHV masking to avoid a fully oblivious dense lookup path.
- Keep suffix-table metadata public where it is table-structural rather than witness-derived.
- Reuse one-hot style representations across RAF and later openings.

### Tradeoffs

- The design is much cheaper than a fully oblivious lookup.
- It leaks more structure than an oblivious lookup, because masked indices are opened.

### Notable limitations

- The current one-hot path reuses one RandOHV mask across many rows of a polynomial, which leaks equality and XOR relations between masked indices.
- This is a masked-index-opening problem, not a missing mask-before-resharing-additive-shares problem.
- This is the main semi-honest leakage issue still confirmed in the current tree.

## Idea 1: Deferred Unmask — Keep Polynomial Masked Throughout Sumcheck

### Motivation

Currently, FWHT unmasking is done eagerly: before the sumcheck begins, each
masked histogram `H_masked[i]` (indexed by `c = k ⊕ r`) is transformed into
`H_true[i]` (indexed by `k`) via FWHT convolution with the one-hot mask vector.
With R rotation slots, this costs `O(R × M log M)` compute and `O(R × M)`
communication (reshare additive → Rep3 per slot).

The deferred-unmask idea observes that the sumcheck doesn't actually need the
unmasked polynomial until the very end. Instead of unmasking upfront, we run
the entire sumcheck on the *masked* polynomial and correct the result.

### Mathematical Basis

**MLE of a permuted function.** Let `f: {0,1}^n → F` be a function and let
`σ: {0,1}^n → {0,1}^n` be the XOR-by-`r` permutation: `σ(x) = x ⊕ r`. Define
`g = f ∘ σ⁻¹`, i.e., `g(x) = f(x ⊕ r)`. Then the multilinear extensions satisfy:

```
g̃(x₁, ..., xₙ) = f̃(x₁ ⊕ r₁, ..., xₙ ⊕ r₂)
```

where `xᵢ ⊕ rᵢ` means `(1 - rᵢ)·xᵢ + rᵢ·(1 - xᵢ)` — a conditional swap
(reflection) of `xᵢ` around `1/2` when `rᵢ = 1`.

**Sumcheck round structure.** At round `i`, the prover computes:

```
gᵢ(X) = Σ_{x_{i+1},...,x_n ∈ {0,1}} g̃(α₁,...,α_{i-1}, X, x_{i+1},...,xₙ) · eq(...)
```

The sum over free variables `x_{i+1}, ..., xₙ` is XOR-invariant: summing over all
binary assignments is the same regardless of whether each `xⱼ` is reflected. The
XOR-by-`r` mask on free variables disappears in the sum.

For the *bound* variable `X` at round `i`: `g̃` evaluates `X` at the masked
position `X ⊕ rᵢ`. When `rᵢ = 0`, no change. When `rᵢ = 1`, the evaluations
at 0 and 1 are swapped: `gᵢ(0) = fᵢ(1)` and `gᵢ(1) = fᵢ(0)`.

### Protocol

#### Sumcheck Rounds (No Change to Prover Logic)

Run the sumcheck exactly as today, but on the masked polynomial `H_masked`
(indexed by `c = k ⊕ r`) instead of `H_true`. The prover computes round
polynomials `gᵢ(X)` from `H_masked` using the standard algorithm.

At each round `i`, after computing `gᵢ(X)`:
- If mask bit `rᵢ = 0`: send `gᵢ(X)` as-is.
- If mask bit `rᵢ = 1`: send `gᵢ(X)` with evaluations at 0 and 1 swapped.

Since `rᵢ` is a secret bit (Rep3-shared), the conditional swap is done via
MPC: `output(t) = (1 - rᵢ) · gᵢ(t) + rᵢ · gᵢ(1 - t)` for `t ∈ {0, 1, 2, ...}`.

For a degree-d round polynomial represented by evaluations at `{0, 1, ..., d}`,
the swap operates on the evaluation vector. Let `g = [g(0), g(1), g(2), ..., g(d)]`.
The swapped vector `g' = [g(1), g(0), g'(2), ..., g'(d)]` where `g'(t)` is the
polynomial that passes through `(0, g(1)), (1, g(0))` and the remaining points
are derived from the Lagrange interpolation adjustment.

More precisely, define the "reflected" polynomial `g*(X) = g(1 - X)`. Then
the conditional swap outputs `(1 - r) · g(X) + r · g*(X)`. Since `g*` is also
degree-d, computing `g*(t)` for `t ∈ {0, ..., d}` is:
- `g*(0) = g(1)`, `g*(1) = g(0)`, `g*(2) = g(-1)`, `g*(3) = g(-2)`, etc.
- For `t ≥ 2`, `g*(t) = g(1-t)` which requires evaluating `g` at negative
  integers. This is a Lagrange interpolation from the `d+1` known points.

The conditional swap is 1 MPC multiplication (by secret bit `rᵢ`) per evaluation
point, i.e., `O(d)` per round. With `d = 3` (ReadRaf degree), this is 4
multiplications per round.

#### Final Evaluation

After all `n = log₂(M)` rounds, the verifier has bound all variables to challenges
`α₁, ..., αₙ`. The claimed evaluation is `g̃(α₁, ..., αₙ)`, which equals
`f̃(α₁ ⊕ r₁, ..., αₙ ⊕ rₙ)`.

The opening proof must evaluate the committed polynomial at the "unmasked" point
`(α₁ ⊕ r₁, ..., αₙ ⊕ rₙ)`. Since `r` is shared, this point is computed in
MPC as `βᵢ = (1 - rᵢ) · αᵢ + rᵢ · (1 - αᵢ)` — one multiplication per variable.

The Dory opening proof then evaluates at `(β₁, ..., βₙ)`.

### Cost Analysis

| Operation | Current (R slots) | Deferred Unmask |
|-----------|-------------------|-----------------|
| FWHT unmask (compute) | O(R × M log M) | 0 |
| Reshare histograms (comms) | O(R × M) | 0 |
| Sumcheck round correction | 0 | O(d × n) muls |
| Opening point correction | 0 | O(n) muls |
| Ehat16 tensor product | O(R × M) interactive | O(M) (1 slot) |
| Total per histogram | O(R × M log M) | O(M + d × n) |

For the 4-phase case: M = 65536, n = 16, d = 3, R = 16:
- Current: `16 × 65536 × 16 ≈ 16M` operations per histogram.
- Deferred: `65536 + 3 × 16 = 65584` operations per histogram.
- **~256× reduction** in the unmask-related cost.

Even for 8-phase: M = 256, n = 8, d = 3, R = 16:
- Current: `16 × 256 × 8 ≈ 32K` operations.
- Deferred: `256 + 24 = 280` operations.
- **~115× reduction**.

### Implications for Histogram Construction

With deferred unmask, each histogram remains in the masked domain throughout the
sumcheck. This means:

1. **No per-slot separation needed**: Since we never FWHT-unmask, we don't need
   separate histograms per rotation slot. All cycles can use a single mask (R = 1),
   and the unmask happens implicitly via the per-round conditional swap.

2. **R = 1 suffices**: The deferred approach makes rotation unnecessary for the
   sumcheck path. However, rotation may still be desired for the commitment path
   (Dory `commit_rows`) or for other consumers of `c[j]` values. If commitment
   is the only remaining consumer, the security argument shifts to the commitment
   context.

3. **Histogram is dense Rep3**: The masked histogram `H_masked` is built from
   public `c[j]` indices and secret coefficients, producing a dense Rep3 vector
   of length M. No reshare step is needed — it's already in Rep3 form if the
   coefficients are Rep3.

### Interaction with Ehat16 Tensor Product

Currently, `init_phase` builds `ehat16_by_slot[slot][M]` for each rotation slot
via a tree tensor product of per-chunk `e_field` vectors. Each level of the tree
uses interactive `mul_vec` (Rep3 × Rep3 multiplication).

With deferred unmask and R = 1, only one `ehat16` vector is needed. But more
importantly, `ehat16` is no longer consumed per-round during the sumcheck — it's
only needed at the very end for the opening proof (to compute the unmasked
evaluation point). The tensor product can therefore be deferred or eliminated
if the opening proof can work directly with per-chunk `e_field` vectors.

### Open Questions

1. **Round polynomial reflection**: Computing `g*(t) = g(1-t)` for `t ≥ 2`
   requires evaluating the round polynomial at negative integers. With degree 3,
   we need `g(-1)` and `g(-2)`. These can be computed from the 4 known evaluations
   `g(0), g(1), g(2), g(3)` via Lagrange interpolation. This is a fixed-cost
   local computation (no MPC needed), followed by the conditional swap MPC mul.

2. **Batching across histograms**: Multiple suffix histograms are processed per
   phase. The deferred-unmask correction (conditional swap) uses the SAME mask
   bits `rᵢ` for all histograms in a phase (since they share the same one-hot
   polynomial). The swap multiplications can be batched.

3. **Compatibility with condensation**: The ReadRaf sumcheck uses "condensation"
   where multiple shifted tables are combined. Currently, `shifted_tables_from_public_table`
   produces R shifted tables (one per rotation slot). With deferred unmask, only
   one table is needed (the masked one), simplifying condensation.

4. **Verifier changes**: The verifier needs to know that the opening proof
   evaluation point is `(α ⊕ r)` rather than `α`. Since `r` is secret-shared,
   the corrected point is computed in MPC and only the final opening claim is
   sent to the verifier. No change to verifier logic if the opening proof
   already handles MPC-computed evaluation points.
