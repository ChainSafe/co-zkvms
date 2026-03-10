# MPC Topology

## Intro

### Purpose

Describe the proving split between the coordinator and the three Rep3 workers.

### Motivation

`co-jolt2` keeps Fiat-Shamir and final proof assembly in one place while pushing witness-dependent arithmetic into MPC workers. This keeps the vanilla Jolt proof shape mostly intact and narrows the MPC surface to witness generation, commitments, and sumcheck messages.

## Background

### Key definitions

- Coordinator: owns the transcript, combines shares, and assembles the public proof.
- Worker: holds shared witness state and performs MPC algebra.
- Rep3: 3-party replicated secret sharing with explicit `open` and `reshare` boundaries.

### References

- `papers/co-zkvms.md`, sections on distributed proving and witness extension in MPC
- Jolt book: `src/how/appendix/sumcheck.md`, `src/how/architecture/opening-proof.md`
- Useful code anchors:
  - `co-jolt2/src/zkvm/dag/worker.rs`
  - `co-jolt-coordinator/src/zkvm/dag/coordinator.rs`

## Design

### Coordinator role

- Owns the transcript and all Fiat-Shamir challenges.
- Receives commitment shares, round-message shares, and opening-proof shares.
- Reconstructs only values that are intended to become public at that stage.

This design prevents workers from independently deriving challenges from local views.

### Worker role

- Owns shared trace, shared memory, shared advice, and preprocessing state.
- Builds witness polynomials, commits shared polynomials, and answers coordinator-driven sumcheck rounds.
- Never produces the final proof object directly.

### Security invariants

- Transcript ownership is coordinator-only.
- Worker send/receive order is protocol state; changing it changes the proof.
- Only verifier-known or intentionally declassified values may become public coordinator inputs.
- Any worker value sent in the clear must be justified by an explicit open boundary.

### Design choices

- Coordinator-centric transcript:
  - simpler integration with vanilla Jolt
  - easier proof assembly
  - less MPC state exposed to transcript logic
- Worker-centric witness algebra:
  - keeps secret data inside Rep3 types until explicit open/combine points
  - lets commitment and sumcheck code reuse MPC primitives directly

### Tradeoffs

- Simpler than a fully distributed transcript, but the coordinator is a trusted orchestration point.
- The architecture supports private witness handling at the worker layer, not private trace generation on the coordinator.

### Notable limitations

- The current design assumes a coordinator that can start from a cleartext trace.
- Coordinator failure or protocol desynchronization invalidates the run immediately.
- This is semi-honest MPC architecture; malicious consistency checks are mostly out of scope.
