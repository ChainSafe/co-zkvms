# Rep3 Dory ZK Notes

This note only covers what `co-jolt2` adds on top of upstream vanilla Dory.

Upstream already gives us:
- verifier-ZK Dory commitments and openings behind `zk`
- `y_com` / `y_blinding`
- transcript binding to the hidden evaluation instead of the raw opening

What upstream does not have is our MPC split:
- workers hold additive shares of witness polynomials
- the coordinator assembles the public transcript
- reconstructed opening hints must not leak back to workers

That is the gap this design closes.

## Goals

We need Dory to hide two different things:

- from the verifier:
  - the final opening value must stay hidden behind upstream ZK Dory
  - public commitments in the transcript must be blinded
- from workers:
  - raw reconstructed row commitments used as Dory opening hints must never be sent back to them

## Commitment Blinding

The shared-commit path is still additive on workers:

1. Each worker computes its raw Dory commitment share and raw row-hint share.
2. The coordinator combines commitment shares into the raw public Dory commitment.
3. Before appending that commitment to the transcript, the coordinator samples a fresh Dory blinding scalar `r_d1`.
4. It blinds the commitment exactly at the public Dory layer, by adding the upstream `HT` blinding term.
5. The blinded commitment becomes the canonical public commitment stored in coordinator state and bound into the transcript.

The coordinator also stores every per-commitment `r_d1`. During stage 5 it derives the joint opening commitment blinding with the same RLC coefficients used to build the joint commitment. That derived blinding is passed into the upstream-style ZK Dory prover, so the final proof is consistent with the blinded transcript commitment instead of the raw combined one.

This is the verifier-facing hiding story. It mirrors upstream semantics, but the blinding is sampled by the coordinator because the public transcript is assembled there.

## Masked Public Rows

Vanilla Dory has no worker boundary, so it never has to solve this problem.

In MPC, workers need a public `v1`-like object to keep the existing `prove_rep3` flow cheap. But if the coordinator reconstructs raw row commitments and sends them back, workers learn the opening hint.

We keep the public-row structure but change what “public” means:

1. Workers send additive shares of the joint opening row commitments to the coordinator.
2. The coordinator reconstructs the raw rows.
3. It samples one mask scalar per row and forms masked rows:
   - `R'_i = R_i + H1 * m_i`
4. It sends only the masked rows back to workers.
5. It also sends replicated shares of the same mask scalars, so workers can help correct the few blocked masked terms without learning the masks.

As a result:
- workers still run almost the same Dory prover structure as before
- workers never see the raw reconstructed rows
- the verifier still sees a standard upstream-style ZK Dory proof

## Which Terms Need Coordinator Correction

Most masked-row effects are linear and cheap to correct on the coordinator after combining worker outputs.

These public-row terms are corrected directly:
- `e1`
- `d1_left`
- `d1_right`
- `e1_plus`
- `e1_minus`

The coordinator subtracts the known mask contribution from the worker-combined public term. Those corrections are simple because the row masks only interact with public generators or public folded scalars.

Three terms are different:
- `vmv.c`
- `second.c_plus`
- `second.c_minus`

These contain a contraction of hidden row masks with worker-secret state. Workers can compute the masked term, but they cannot locally remove the mask without learning it.

That is why we added the Phase 2 MPC corrections:
- for `vmv.c`: workers compute a correction share for the hidden-mask dot shared-scalar term
- for `second.c_plus` / `second.c_minus`: workers compute correction shares for the hidden-mask MSM against folded shared `v2`

The coordinator combines those correction shares and removes the hidden mask contribution from the masked GT term. No raw rows are revealed.

## Why The First ZK Implementation Was Bad

The first masked-row version was correct, but it pushed too much work onto the coordinator.

After introducing masked rows, the coordinator:
- reconstructed raw rows
- reconstructed `v2`
- recomputed VMV locally
- recomputed first-round local terms
- recomputed second-round local terms
- only then derived transcript challenges and sent them back

That made the coordinator the critical path for every Dory round. Workers mostly waited.

Measured on the `NUM_ITERS=20` Tracy repro:
- coordinator `Dory::coordinate_prove`: about `14.88s`
- worker0 `Dory::prove`: about `14.45s`
- worker0 `wait_beta + wait_alpha`: about `7.00s`

The regression was not worker arithmetic. It was coordinator-local recomputation blocking challenge broadcasts.

## Optimizations

### 1. Hybrid Rollback

We first rolled all non-blocked work back to workers.

Workers again compute and send:
- `d2`
- `d2_left`, `d2_right`
- `e1`
- `e1_beta`, `e2_beta`
- `e1_plus`, `e1_minus`
- `e2_plus`, `e2_minus`

The coordinator keeps only:
- combining worker outputs
- correcting the linear masked-row terms
- recomputing the genuinely blocked GT terms

Result:
- coordinator `Dory::coordinate_prove`: about `12.01s`
- worker0 `wait_beta + wait_alpha`: about `4.21s`

### 2. Blocked-GT Prep Cleanup

The remaining hotspot was coordinator recomputation of `c_plus` and `c_minus`.

We then:
- split blocked GT work into explicit kernels
- shared affine preparation between `c_plus` and `c_minus`
- removed redundant normalization and temporary vector rebuilding

Result:
- coordinator `Dory::coordinate_prove`: about `10.88s`
- coordinator `second_round_local_recompute`: about `1.30s`
- worker0 `wait_alpha`: about `1.34s`

At that point the remaining cost was the blocked GT algebra itself, not preparation.

### 3. Phase 2 MPC Correction

Next we stopped recomputing the blocked GT terms on the coordinator.

Instead:
- workers compute masked GT terms as before
- workers also compute small correction shares using replicated shares of the row masks
- coordinator combines those corrections and unmasks `c`, `c_plus`, `c_minus`

This removed the last round-by-round coordinator bottleneck.

Result:
- coordinator `Dory::coordinate_prove`: about `9.88s`
- worker0 `Dory::prove`: about `9.11s`
- worker0 `wait_alpha`: about `0.19s`

### 4. Init Cleanup

After Phase 2, the coordinator no longer needed the initial unfolded `v2`.

So we removed it from `init_receive_reconstruct`:
- workers no longer send initial `v2` in the init payload
- coordinator init now only receives row shares, reconstructs raw rows, masks them, and sends masked rows back
- the final folded `v2` is still sent at the end for the scalar-product proof

Result from the same repro:
- coordinator `Dory::coordinate_prove`: about `7.92s`
- coordinator `init_receive_reconstruct`: about `1.56s`
- worker `Dory::prove`: about `7.51s`

Init now breaks down roughly as:
- `init_receive_rows`: `1.06s`
- `init_mask_rows`: `496ms`
- `init_send_masked_rows`: `<1ms`

## Current Shape

The current MPC Dory flow is:

1. Workers produce additive commitment shares and opening-hint row shares.
2. Coordinator blinds each public Dory commitment before transcript append and stores the commitment blinding.
3. During stage 5, workers send row shares for the joint opening polynomial.
4. Coordinator reconstructs raw rows, samples row masks, and sends masked rows plus replicated mask shares.
5. Workers run the normal Dory reduction on masked public rows.
6. Coordinator:
   - combines worker outputs
   - corrects the linear masked terms directly
   - corrects `c`, `c_plus`, `c_minus` via the small MPC correction path
7. Coordinator derives the joint commitment blinding from the commitment RLC and feeds it into the upstream ZK Dory prover state.
8. Final proof uses upstream ZK Dory objects (`y_com`, sigma proofs, scalar-product proof), while `y_blinding` stays coordinator-local for future BlindFold work.

## Important Limits

- This is still the single-worker-subnet Dory path.
- `y_blinding` is carried through the API but is not serialized yet; BlindFold is still deferred.
- The masking here protects raw reconstructed opening hints from workers. It does not try to change upstream polynomial representations themselves.
