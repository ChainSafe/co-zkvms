# Ring MSM

## Intro

### Purpose

Describe the ring-MSM design used to avoid converting general shared ring scalars into field scalars for the naive MSM path.

### Motivation

The commitment path wants fast MSM on public bases with secret shared coefficients. Direct full scalar conversion is expensive and leaks the wrong information if done carelessly. The ring-MSM design exists to separate the cheap local part from the carry-correction part that must remain secret.

## Background

### Key definitions

- Naive MSM share: local MSM over lifted ring-share limbs.
- Carry correction: secret correction term removing the `2^32` carry ambiguity.
- Bit-times-public-point preprocessing: correlated randomness for masked bit openings.

### References

- `papers/co-zkvms.md`, commitment/distributed proving sections
- Useful code anchor:
  - `co-jolt2/src/poly/commitment/dory.rs`

## Design

### Role

Ring MSM lets the prover reuse small shared ring limbs in the commitment path while correcting the hidden carry term in MPC.

### Assumptions

- Bases are public.
- Secret coefficients are bounded so the carry term is small and structured.
- The correlated randomness for masked bit openings is one-time use.

### Security invariants

- Carry bits are never opened directly.
- Only masked bits of the form `b XOR r` are opened, with fresh independent masks.
- The correction term must stay secret-shared until it is folded into the final group share.

### Design choices

- Split the problem into:
  - cheap local MSM on lifted limbs
  - secret MPC carry extraction and correction
- Use bit-times-public-point preprocessing instead of opening carry structure.

### Tradeoffs

- Better performance than full scalar conversion.
- More preprocessing and more protocol complexity around correction terms.

### Notable limitations

- This path depends on correct bounded-scalar assumptions.
- Reusing the correlated bit masks would break privacy.
- The current note is design-level; malicious robustness would require authenticated preprocessing.
