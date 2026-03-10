# Audit

## Overview

- Reviewed:
  - `co-jolt2`
  - `co-jolt-coordinator`
  - `mpc-core`
- Threat model:
  - 3-party Rep3 workers
  - semi-honest worker adversaries
  - malicious-only concerns called out separately when relevant
- High-level conclusion:
  - The plaintext worker-advice leak is fixed in the current tree: workers now consume `Rep3ProgramIOInput`, and trusted/untrusted advice stay shared through worker polynomial construction and untrusted-advice commitment aggregation.
  - I found 1 confirmed semi-honest leakage issue, 1 accepted declassification note, and 1 proof-system note.

## Audit basis

- Jolt / co-zkVM design references:
  - `co-jolt2/PROJECT.md`
  - `papers/co-zkvms.md`
  - `papers/DFS.pdf`
  - `papers/maestro.pdf`
  - `papers/B2A.pdf`
  - `co-jolt2/docs/ring_msm.md`
- MPC security ground truth:
  - `/Users/timofey/repos/examples/co-snarks/mpc-core`
- Comparison rule:
  - for `mpc-core` itself and for `co-jolt2` code that depends on Rep3 semantics, I treat the analogous `co-snarks/mpc-core` logic as the masking/open/reshare reference
  - local differences are classified as safe equivalents, documentation-only differences, open questions, or confirmed findings

## Conclusion by audit area

### Leakage

- Confirmed finding:
  - `Rep3OneHotPolynomial` reuses one RandOHV mask across many opened masked indices, leaking equality and XOR structure across rows.
- Accepted design note:
  - the virtual shift/pow helper family intentionally exposes one helper operand.
- No additional concrete leakage issue found in the reviewed worker advice path after the plaintext-advice fix.

### Missing masking / incorrect reshare

- Confirmed issue:
  - the one-hot / RAF masking scheme is too weak because one mask is reused across many openings.
- Reviewed with no additional concrete finding:
  - `mpc-core` Rep3 arithmetic and binary multiplication paths mask local products before `reshare`, matching the `co-snarks/mpc-core` reference.
  - `mpc-core` `open` paths follow the same reference sequencing and do not skip a required masking step.
  - `mpc-core` `rep3_ring/gadgets/{lut,ohv}` matches the reference masking discipline:
    - fresh RandOHV masks for per-lookup masked opens
    - masked local field terms before resharing in LUT write/update paths
  - `co-jolt2` custom advice and RAM initialization paths keep advice shared through `binary_ring_to_field_many` and do not reconstruct to plaintext on workers.
  - `co-jolt2` Dory commitment sharing and coordinator recombination do not bypass the shared commitment path.

### MPC correctness/security bugs

- No additional confirmed share-domain or preprocessing-reuse bug found in the reviewed code.
- Reviewed with no concrete finding:
  - Rep3 field/ring conversion boundaries used by advice packing, RAM init, suffix helpers, and Dory inputs
  - additive-to-replicated resharing in custom batched product helpers
  - coordinator/worker commitment-share aggregation and opening ordering
- Architectural limitation, not a confirmed semi-honest bug:
  - witness generation still starts from a cleartext coordinator trace.

### SNARK / proof-system hazards

- Note:
  - the final proof path is not verifier-zero-knowledge by default.

### `mpc-core`

- Reviewed against `/Users/timofey/repos/examples/co-snarks/mpc-core`:
  - `rep3/{arithmetic,binary,conversion,network}`
  - `rep3_ring/{binary,casts,conversion}`
  - `rep3_ring/gadgets/{lut,ohv}`
  - preprocessing / reuse assumptions
- Conclusion:
  - the reviewed local `mpc-core` masking/open/reshare discipline is materially aligned with the `co-snarks` reference
  - the main security gap in the current tree is in `co-jolt2`’s higher-level one-hot masking design, not in a weakened `mpc-core` primitive
  - reviewed differences from the reference were API or helper-surface differences, not weaker MPC semantics

## Findings

#### [Medium] `Rep3OneHotPolynomial` reuses one RandOHV mask across many secret indices
- **Component:** `co-jolt2/src/poly/one_hot_polynomial.rs`, `co-jolt2/src/poly/ra_poly.rs`
- **Condition:** any `Rep3OneHotPolynomial` built from multiple active secret indices
- **Issue:** `from_indices` samples one secret mask `r` and opens `c[j] = k(j) XOR r` for every active row of the polynomial. That makes the opened masked indices linkable across rows.
- **Why it matters:** workers learn equality and pairwise-XOR relations between secret indices. If any one true index becomes known, the shared mask is recovered and all masked indices in that polynomial are revealed.
- **Evidence:** `Rep3OneHotPolynomial::from_indices` samples one `r_share`, converts one RandOHV, and opens all `masked_indices_c`; `papers/co-zkvms.md` explicitly warns that reusing one RandOHV mask across rows leaks cross-row structure; compared with `/Users/timofey/repos/examples/co-snarks/mpc-core/src/protocols/rep3_ring/gadgets/lut.rs`, the reference LUT path samples a fresh RandOHV per lookup instead of reusing one mask across a full polynomial.
- **Fix:** use fresh RandOHV masks per access or per short epoch, or replace the opened-index scheme with an oblivious-access primitive that does not reveal masked indices at all.
- **Confidence:** high

#### [Note] Virtual shift/pow helper operands are intentionally declassified
- **Component:** `co-jolt2/src/host/program.rs`, `co-jolt2/src/zkvm/instruction/virtual_pow2.rs`, `co-jolt2/src/zkvm/instruction/virtual_shift_right_bitmask.rs`, `co-jolt2/src/zkvm/instruction/virtual_srl.rs`, `co-jolt2/src/zkvm/instruction/virtual_sra.rs`
- **Condition:** traces containing `VirtualPow2`, `VirtualShiftRightBitmask`, `VirtualSRL`, or `VirtualSRA`
- **Issue:** these helpers intentionally keep one operand public on the worker path.
- **Why it matters:** this is a real declassification boundary and must stay documented; it is not the default rule for register operands.
- **Evidence:** `co-jolt2/src/host/program.rs` keeps `rs1` public for `VirtualPow2` / `VirtualShiftRightBitmask` and `rs2` public for `VirtualSRL` / `VirtualSRA`; upstream Jolt uses those operands as the shift amount or derived bitmask; this matches `co-jolt2/docs/witness_generation.md`.
- **Fix:** none if this v1 declassification rule is accepted; otherwise these helpers need a different MPC design.
- **Confidence:** high

#### [Note] The current proof path is not zero-knowledge against the verifier
- **Component:** overall Jolt/Dory proof system
- **Condition:** if verifier-facing confidentiality is part of the deployment goal
- **Issue:** the current PCS and sumcheck path match vanilla Jolt: Dory commitments are not hiding by default, and the proof path does not add sumcheck masking.
- **Why it matters:** MPC protects worker-side witness handling, but the final proof itself is not a ZK proof in the usual verifier-facing sense.
- **Evidence:** `co-jolt2/docs/dory.md` and `papers/co-zkvms.md` both describe the current PCS layer as binding rather than hiding.
- **Fix:** add masking polynomials plus a hiding PCS, or wrap the proof in a zkSNARK.
- **Confidence:** high

## Reviewed components with no concrete finding

- `mpc-core` `rep3` arithmetic/binary/open/reshare paths reviewed against the `co-snarks` reference in `src/protocols/rep3/{arithmetic,binary,network}.rs`; no weakened masking or reshare sequencing found.
- `mpc-core` `rep3_ring` LUT/OHV paths reviewed against `/Users/timofey/repos/examples/co-snarks/mpc-core/src/protocols/rep3_ring/gadgets/{lut,ohv}.rs`; local masking and resharing discipline matches the reference pattern.
- `mpc-core` ring/field conversion paths reviewed in `src/protocols/rep3_ring/{casts,conversion}.rs`; no concrete boundary bug found in the reviewed usage.
- `mpc-core` preprocessing pool machinery reviewed in `src/protocols/rep3_ring/preprocessing/{dabits,edabits}.rs`; `reuse-preproc` remains benchmark-only and unsafe for production reuse, but not a hidden default.
- `co-jolt2` worker advice path reviewed in `src/host/jolt_device.rs`, `src/zkvm/mod.rs`, `src/zkvm/dag/{state_manager,worker}.rs`, and `src/zkvm/ram/mod.rs`; workers now consume `Rep3ProgramIOInput`, and trusted/untrusted advice remain shared through worker polynomial construction.
- `co-jolt-coordinator` untrusted-advice commitment aggregation reviewed in `src/zkvm/dag/coordinator.rs`; the coordinator combines commitment shares instead of relying on duplicated plaintext commitments.
- `co-jolt2` Spartan stage-1/inner-sumcheck plumbing reviewed in `src/poly/spartan_interleaved_poly.rs`, `src/zkvm/r1cs/inputs.rs`, and `src/zkvm/spartan/{worker,inner}.rs`; no concrete share-domain or masking bug found.
- `co-jolt2` opening-accumulator / RLC reduction plumbing reviewed in `src/poly/opening_proof.rs` and `src/poly/rlc_polynomial.rs`; no concrete share-combination bug found.

## Open questions / needs manual confirmation

- Current witness generation still starts from a cleartext trace on the coordinator/delegator path. That is compatible with the coordinator=delegator model, but not with private shared-state witness generation; confirm whether that stronger model is expected for this codebase.
