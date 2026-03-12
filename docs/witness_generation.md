# Witness Generation

## Intro

### Purpose

Describe how `co-jolt2` turns a vanilla Jolt execution trace into shared witness data for MPC proving.

### Motivation

Witness generation is the main privacy boundary in this system. It decides which execution data stays shared, which data is treated as public, and where ring-domain values cross into field-domain proving.

## Background

### Key definitions

- Trace domain: Rep3 binary/ring shares used for traced operands and early lookup work.
- Proving domain: Rep3 field shares used by Spartan, Dory, and opening proofs.
- `Rep3ProgramIOInput`: worker-side container for shared advice and public program IO metadata.

### References

- `papers/co-zkvms.md`, sections on witness extension in MPC
- `papers/B2A.pdf`, `papers/maestro.pdf`
- Jolt book: `src/how/twist-shout.md`, `src/how/architecture/ram.md`
- Useful code anchors:
  - `co-jolt2/src/host/program.rs`
  - `co-jolt2/src/host/jolt_device.rs`
  - `co-jolt2/src/zkvm/witness.rs`

## Design

### Role

Witness generation starts from a vanilla Jolt trace on the coordinator, secret-shares the witness-relevant parts, and then extends that shared state into the polynomial view expected by the prover.

### Assumptions

- Program code and instruction encoding are public.
- Register values, RAM values, and advice are the primary secret witness material.
- The coordinator may see the cleartext trace in the current model.

### Visibility policy

Public by design:

- bytecode, opcode flags, PC-derived position data
- register indices and immediates
- RAM addresses and ordering/timestamp structure
- public inputs, outputs, `panic`, and `memory_layout`

Shared by design:

- register values
- RAM values
- lookup inputs and outputs derived from secret state
- trusted advice
- untrusted advice

### Security invariants

- Advice must remain shared on worker paths from input through commitment and opening.
- Ring-to-field conversion is a security boundary; secret values stay shared across it.
- Public/shared classification at trace sharing must match later proving assumptions.
- Values are public because the verifier model knows them, not because a helper API is easier to call on public data.
- Additive field shares are valid internal results of shared-by-shared MPC work, but they are not fresh replicated shares.
  - If later code reshapes them back into Rep3 state or reuses them across another non-linear MPC step, the additive value must already have been produced under the standard masked-additive discipline from `mpc-core`.
  - This rule is separate from masked-index openings such as `open(k XOR r)`.

### Design choices

- Hybrid tracing:
  - vanilla Jolt produces the execution trace
  - `co-jolt2` shares only the witness-sensitive parts
- Two-domain witness path:
  - ring/binary shares for lookup-friendly tracing
  - field shares for proof systems and commitments
- Shared advice path:
  - workers receive `Rep3ProgramIOInput`, not plaintext `JoltDevice`
  - advice is packed from shares and converted to shared field coefficients

### Intentional declassification

The virtual shift/pow helper family is a narrow exception to the default “register values are shared” rule.

- `VirtualPow2` and `VirtualShiftRightBitmask` use public `rs1`.
  - In those helper instructions, `rs1` is the shift amount by design.
- `VirtualSRL` and `VirtualSRA` use public `rs2`.
  - In those helper instructions, `rs2` is the derived public bitmask.
- Immediate variants use a public right operand because the immediate is already public.

### Tradeoffs

- The hybrid design reduces implementation churn relative to a fully secret-shared tracer.
- Public address/order structure keeps RAM proofs closer to vanilla Jolt, but does not hide access patterns.
- The virtual helper declassification keeps the lookup path simple at the cost of exposing those helper operands.

### Notable limitations

- The coordinator still starts from a cleartext trace.
- RAM access patterns are public.
- `Rep3OneHotPolynomial` uses opened masked indices; this is not the same as keeping lookup indices fully secret.
