# Remaining Public Prover Instances

## Intro

### Purpose

Record which prover-visible objects are intentionally public and why.

### Motivation

This file is a design ledger, not an implementation shortcut list. If a value is public, there should be a verifier-model reason or an explicit declassification rule.

## Background

### Key definitions

- Public by verifier knowledge: the verifier is expected to know it already.
- Intentional declassification: not inherently verifier-known, but intentionally exposed by the current design.

### References

- `papers/co-zkvms.md`
- `co-jolt2/docs/witness_generation.md`

## Design

### Public by verifier knowledge

- bytecode and opcode flags
- register indices and immediates
- PC-derived program position
- RAM addresses and ordering/timestamp structure
- public inputs, outputs, `panic`, and `memory_layout`
- public sumcheck instances inherited from vanilla Jolt

### Intentional declassifications

- `VirtualPow2` and `VirtualShiftRightBitmask` expose `rs1`
- `VirtualSRL` and `VirtualSRA` expose `rs2`
- immediate variants expose the immediate-derived right operand

These exceptions are narrow and should not be generalized to ordinary register values.

### Still security-relevant even though opened

- `Rep3OneHotPolynomial.masked_indices_c`

They are opened in masked form, but mask reuse still leaks cross-row structure.

### Notable limitations

- This ledger is only useful if new public prover objects are added deliberately and documented here.
