# Rep3 BlindFold Integration

This note documents the BlindFold design implemented in `co-jolt2`.

It only covers what is specific to the DAG MPC integration. For background on BlindFold itself, see:

- [upstream BlindFold note](/Users/timofey/repos/jolt/book/src/how/blindfold.md)
- [local Dory MPC note](/Users/timofey/repos/co-zkvms/co-jolt2/docs/dory.md)

## Summary

BlindFold is integrated against the `co-jolt2` DAG proving pipeline.

The implemented DAG BlindFold checkpoints are:

1. stage1 Spartan outer
2. stage2 DAG batch
3. stage3 DAG batch
4. stage4 DAG batch
5. stage5 opening binding

This is the canonical BlindFold shape for `co-jolt2`.

There is no DAG uni-skip BlindFold stage. The DAG prover and verifier reconstruct BlindFold using the actual DAG checkpoints above.

## Coordinator-Proved BlindFold

BlindFold is proved by the coordinator only.

This follows directly from the existing MPC split:

- workers hold secret-shared witness polynomials
- workers compute MPC round evaluations and opening-proof shares
- the coordinator owns the Fiat-Shamir transcript
- the coordinator already reconstructs the public round polynomials and stage claims needed for the final proof

BlindFold needs exactly the data that already converges at the coordinator:

- reconstructed round polynomial coefficients for stages 1-4
- Pedersen blindings for the committed ZK sumcheck rounds
- hidden output-claim values and their row commitments
- stage5 opening-binding witness data, including the reduced hidden claims, `joint_claim`, `constraint_coeffs`, and `y_blinding`

So BlindFold is naturally coordinator-local. Moving it to workers would not improve privacy. It would only duplicate witness assembly and require extra traffic for data the coordinator already reconstructs.

## Motivation

The goal is to preserve the existing MPC architecture:

- workers do MPC witness computation
- coordinator owns the transcript and assembles the public proof

BlindFold fits that split cleanly. The coordinator already reconstructs the public prover messages that BlindFold needs as witness inputs, so the smallest and cleanest integration is:

- workers keep proving the underlying DAG protocol
- coordinator builds the final BlindFold witness locally
- verifier checks the final BlindFold proof as part of DAG verification

## Trust Model

This does not introduce a new trusted party assumption.

The coordinator is already the party that:

- combines worker commitments
- derives Fiat-Shamir challenges
- combines opening shares
- assembles the final proof object

BlindFold adds one more coordinator-local witness:

- the hidden witness proving consistency of the committed ZK sumcheck messages and the final stage5 opening-binding relation

That does not weaken verifier zero-knowledge because the verifier still sees only the public proof objects.

It also does not weaken worker hiding because workers do not receive any new reconstructed witness values back from the coordinator.

## What Changes For Workers

Very little changes at the design level.

Workers still:

- compute MPC sumcheck round evaluations
- answer coordinator challenges
- contribute stage5 opening-reduction data
- contribute Dory proof shares

Workers do not:

- build BlindFold stage configs
- reconstruct round polynomial coefficient vectors
- derive BlindFold witnesses
- run the final BlindFold proof

So BlindFold does not introduce a new worker protocol phase. It is a coordinator-side extension of the existing DAG proof assembly flow.

## DAG-Native Stage5

The important DAG-specific design choice is stage5.

Stage5 is the final BlindFold opening-binding checkpoint for the DAG pipeline. Its role is analogous to upstream vanilla stage8: it is the final hidden PCS-binding stage after the earlier proof checkpoints have accumulated opening claims.

For `co-jolt2`, stage5 is defined over the interface exposed by the DAG opening-reduction pipeline:

- BlindFold stage5 uses the hidden reduced opening claims produced by opening reduction
- those claims are addressed as `OpeningId::ReducedOpeningClaim(..)`
- the final linear stage5 relation is built over those reduced claims with `constraint_coeffs`
- that relation is bound to the final hidden `joint_claim`
- `y_com` is the public PCS evaluation commitment bound into the transcript
- `y_blinding` is the hidden blinding witness used for the stage5 extra constraint

This is intentional. It is the final BlindFold stage for the DAG pipeline as implemented in `co-jolt2`.

## Relation To Vanilla

The upstream and DAG protocols play the same role at the final BlindFold opening-binding stage:

- upstream vanilla uses its final BlindFold stage for the hidden opening-binding relation of the vanilla prover pipeline
- `co-jolt2` uses DAG stage5 for the hidden opening-binding relation of the DAG prover pipeline

What differs is the exact stage interface:

- upstream writes the final BlindFold relation over the hidden openings exposed by the upstream opening path
- `co-jolt2` writes the final BlindFold relation over the hidden reduced claims exposed by the DAG opening-reduction path

So the design is analogous to vanilla, but specialized to the DAG protocol shape and stage boundaries rather than upstream's exact stage structure.

## Security Properties

### Verifier Zero-Knowledge

Verifier-ZK is preserved because the verifier sees:

- Pedersen commitments for ZK sumcheck rounds, not clear coefficients
- output-claim row commitments, not clear hidden claims
- `y_com`, not the clear hidden stage5 joint evaluation
- the final BlindFold proof

The coordinator knowing the BlindFold witness is not a verifier-ZK problem. That witness is never sent to the verifier.

### Worker Hiding

Worker hiding is preserved because workers do not learn any new reconstructed values. In particular, workers do not receive:

- reconstructed round polynomial coefficients
- hidden output claims
- stage5 reduced hidden claims
- `y_blinding`
- BlindFold witness data

BlindFold remains entirely coordinator-local on the proving side.

### Soundness Structure

The end-to-end DAG proof soundness is split across the same protocol components already present in the DAG proof:

- stages 1-4 verify the DAG sumcheck checkpoints
- stage5 opening reduction and Dory verification bind the hidden PCS opening relation
- BlindFold proves that the hidden committed ZK messages and the final stage5 relation are consistent

In particular, stage5 BlindFold is stated over the reduced opening claims produced by opening reduction. This means BlindFold is adapted to the actual DAG stage5 interface rather than re-expressing the stage in terms of upstream vanilla's exact hidden variables.

That is a protocol-shape choice, not an extra trust assumption.

## Worker / Coordinator Boundary

The intended division of responsibility is:

- workers: MPC witness computation and proof-share generation
- coordinator: transcript ownership, proof assembly, BlindFold witness construction, BlindFold proving
- verifier: DAG-stage reconstruction and BlindFold verification

This keeps BlindFold aligned with the rest of the `co-jolt2` architecture: MPC work stays on workers, while final proof assembly and transcript-bound ZK proving stay on the coordinator.

## Final Design

The implemented BlindFold design for `co-jolt2` is:

- DAG-native rather than upstream-stage-native
- coordinator-proved
- transparent to workers at the protocol-design level
- verifier-zero-knowledge preserving
- compatible with the existing Dory-based stage5 opening-binding flow

The final BlindFold stage for DAG is stage5 over reduced opening claims. That is the intended `co-jolt2` design.
