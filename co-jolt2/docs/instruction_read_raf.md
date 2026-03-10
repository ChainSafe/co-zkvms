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
- This is the main semi-honest leakage issue still confirmed in the current tree.
