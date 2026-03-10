# Dory Commitment

## Intro

### Purpose

Describe the commitment layer used for shared witness polynomials.

### Motivation

The commitment layer is where shared witness state first becomes part of the public proof transcript. The main design requirement is therefore not just efficiency, but preserving the public/shared boundary until a value is meant to become public.

## Background

### Key definitions

- Dory: the multilinear PCS used by Jolt here.
- `MaybeShared<T>`: distinguishes public values from worker commitment shares.
- Shared polynomial: committed by workers and combined by the coordinator.
- Public polynomial: committed once in vanilla form.

### References

- `papers/co-zkvms.md`, polynomial commitment and distributed proving sections
- Jolt book: `src/how/dory.md`, `src/how/architecture/opening-proof.md`
- Useful code anchors:
  - `co-jolt2/src/poly/commitment/dory.rs`
  - `co-jolt-coordinator/src/poly/commitment/dory.rs`

## Design

### Role

Dory is the bridge between shared witness polynomials and the public proof transcript.

### Assumptions

- Commitments are binding, not hiding.
- Commitment ordering must match the rest of the prover exactly.
- Public/shared classification is fixed before commitment starts.

### Security invariants

- Shared witness data must go through the Rep3 commitment path, never through vanilla `PCS::commit` on reconstructed plaintext.
- Commitment shares and opening hints must preserve the same ordering on workers and coordinator.
- Any wrongly-public polynomial becomes a public proof object immediately.

### Design choices

- Public polynomials use the vanilla Dory path.
- Shared polynomials use worker-local commitment shares plus coordinator recombination.
- The coordinator owns transcript binding and opening-proof assembly.

### Tradeoffs

- This keeps compatibility with vanilla Jolt proof structure.
- It also means Dory inherits vanilla Jolt’s non-hiding behavior.

### Notable limitations

- The PCS is not verifier-zero-knowledge by default.
- Leakage in upstream polynomial representations, especially the current one-hot masking scheme, carries into commitments and openings.
