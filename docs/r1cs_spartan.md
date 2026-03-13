# R1CS / Spartan

## Intro

### Purpose

Describe the Spartan part of the prover and why it stays mixed public/shared.

### Motivation

Spartan is where the witness becomes algebraic constraints. The design problem is not only to prove the right relation, but to keep public structure public and secret witness data shared instead of flattening them into one domain.

## Background

### Key definitions

- Stage 1: outer Spartan sumcheck over interleaved constraint structure.
- Stage 2: inner sumcheck over the reduced witness view.
- `Product`: the witness-product column required by the vanilla Jolt relation.
- Mixed witness: a row containing both public and shared entries.

### References

- `papers/co-zkvms.md`, R1CS and distributed proving sections
- Jolt book: `src/how/appendix/sumcheck.md`
- Useful code anchors:
  - `co-jolt2/src/poly/spartan_interleaved_poly.rs`
  - `co-jolt2/src/zkvm/r1cs/inputs.rs`
  - `co-jolt2/src/zkvm/spartan/inner.rs`

## Design

### Role

Spartan proves the uniform constraint system used by Jolt while preserving the existing public/shared witness split.

### Assumptions

- Vanilla Jolt’s `Product` column semantics remain the source of truth.
- Structural flags and addresses are public.
- Witness-derived values remain shared until an explicit opening stage.

### Security invariants

- `Product` must match vanilla semantics for every row, not just multiply-like instructions.
- Claimed witness evaluations reused later must correspond to the same witness view Spartan proved.
- Public columns stay public end-to-end; MPC does not retroactively hide them.

### Design choices

- Use mixed polynomials instead of trivially sharing all public columns.
- Keep stage separation aligned with vanilla Jolt’s sumcheck structure.
- Reuse cached cycle witness values instead of recomputing row semantics in later stages.

### Tradeoffs

- Mixed witnesses preserve visibility and reduce unnecessary MPC work.
- They also make stage coupling stricter: a visibility mismatch in witness generation can break both correctness and the audit model.

### Notable limitations

- The current design assumes the same row ordering and binding order as vanilla Jolt.
- This path does not add zero-knowledge masking beyond vanilla Jolt’s structure.
