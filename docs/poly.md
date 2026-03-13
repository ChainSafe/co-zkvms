# Polynomials

## Intro

### Purpose

Describe the polynomial representations used by the MPC prover and why more than one representation exists.

### Motivation

Different parts of the prover want different things: dense evaluation, sparse access, mixed public/shared semantics, or cheap opening reduction. The polynomial layer exists to preserve those properties without flattening everything into one opaque format.

## Background

### Key definitions

- `Rep3MultilinearPolynomial`: top-level public/shared split.
- Dense polynomial: field-share MLE for ordinary witness data.
- Mixed polynomial: explicit mixture of public, shared, and additive terms.
- One-hot polynomial: sparse RAF/read-argument representation using RandOHV masking.
- RLC polynomial: opening-reduction wrapper.

### References

- `papers/co-zkvms.md`, witness extension and lookup sections
- Jolt book: `src/how/twist-shout.md`, `src/how/optimizations/batched-openings.md`
- Useful code anchors:
  - `co-jolt2/src/poly/multilinear_polynomial.rs`
  - `co-jolt2/src/poly/one_hot_polynomial.rs`
  - `co-jolt2/src/poly/rlc_polynomial.rs`

## Design

### `Rep3MultilinearPolynomial`

- Role:
  - explicit ledger of whether a polynomial is public or shared
- Invariant:
  - “public” means public to the prover/coordinator, not merely cheap to derive
- Tradeoff:
  - more variants, but much better auditability of visibility boundaries

### Dense and mixed polynomials

- Role:
  - dense polynomials carry ordinary shared witness data
  - mixed polynomials preserve the fact that some terms are public and should stay public
- Design choice:
  - mixed polynomials avoid promoting genuinely public data into trivial secret shares everywhere
- Limitation:
  - they require the caller to preserve domain and visibility information carefully

### One-hot polynomials

- Role:
  - encode lookup/read-address structure without materializing a full dense secret index table
- Design choice:
  - use a shared RandOHV vector plus opened masked indices
- Tradeoff:
  - much cheaper than fully oblivious dense access
  - but it exposes masked indices to all workers
- Limitation:
  - the current implementation reuses one RandOHV mask across many rows in the same polynomial, leaking equality and XOR structure across those rows

### RLC polynomials

- Role:
  - collapse many opening claims into a single reduction-friendly representation
- Design choice:
  - keep one-hot terms lazy and fold dense/public parts eagerly
- Limitation:
  - correctness depends entirely on matching accumulator ordering and commitment ordering
