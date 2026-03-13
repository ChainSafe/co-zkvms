#[cfg(feature = "ring-msm")]
use crate::poly::compact_polynomial::Rep3CompactPolynomial;
use crate::poly::{Rep3DensePolynomial, Rep3MultilinearPolynomial, Rep3SharedPoly};
use crate::utils::types::MaybeShared;
use ark_ec::bn::BnConfig as ArkBnConfig;
use ark_ec::pairing::{MillerLoopOutput, Pairing as ArkPairing, PairingOutput};
use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, CyclotomicMultSubgroup, Field, One, PrimeField};
use ark_std::Zero;
use dory::primitives::{arithmetic::PairingCurve, poly::compute_left_right_vectors};
use jolt_core::ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use jolt_core::jolt_optimizations;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
#[cfg(feature = "ring-msm")]
use mpc_core::protocols::rep3;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
#[cfg(feature = "ring-msm")]
use mpc_core::protocols::rep3_ring;
#[cfg(feature = "ring-msm")]
use mpc_core::protocols::rep3_ring::conversion as ring_conv;
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
#[cfg(feature = "ring-msm")]
use mpc_core::protocols::rep3_ring::ring::bit::Bit;
#[cfg(feature = "ring-msm")]
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
#[cfg(feature = "ring-msm")]
use mpc_core::protocols::rep3_ring::ring::u34::U34;
#[cfg(feature = "ring-msm")]
use mpc_core::protocols::rep3_ring::ring::u66::U66;
#[cfg(feature = "ring-msm")]
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rayon::prelude::*;
use std::borrow::Borrow;

// Re-export vanilla Jolt Dory types (wrappers, globals, commitment scheme, proof types, ...)
pub use jolt_core::poly::commitment::dory::*;

use super::Rep3CommitmentScheme;

type DoryOpenParams = (usize, usize, usize, Vec<Fr>, Vec<Fr>);
type DoryMaskedRowsRequest = (Vec<G1Affine>, Vec<Rep3PrimeFieldShare<Fr>>);
type DoryVmvShareMsg = ((Fq12, Fq12), Option<G1Affine>, Fr);
type DoryFirstReducePublicMsg = (Option<Fq12>, Option<Fq12>, Option<G1Affine>, Option<G2Affine>);
type DoryFirstReduceShareMsg = ((Fq12, Fq12), DoryFirstReducePublicMsg);
type DorySecondReducePublicMsg = (Option<G1Affine>, Option<G1Affine>);
type DorySecondReduceShareMsg = (((Fq12, Fq12), (G2Affine, G2Affine)), DorySecondReducePublicMsg, (G2Affine, G2Affine));
type DoryInitShareMsg = (usize, Vec<G1Affine>);
#[cfg(all(feature = "ring-msm", feature = "rv64"))]
type DoryCarryRing = U66;
#[cfg(all(feature = "ring-msm", not(feature = "rv64")))]
type DoryCarryRing = U34;

// =============================================================================
// Rep3CommitmentScheme implementation
// =============================================================================

impl<ProofTranscript: Transcript> Rep3CommitmentScheme<Fr, ProofTranscript> for DoryCommitmentScheme {
    fn commit_rep3<N: Rep3NetworkWorker>(
        poly: &Rep3MultilinearPolynomial<Fr>,
        setup: &Self::ProverSetup,
        commit_to_public: bool,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<Fr>,
    ) -> eyre::Result<(MaybeShared<Self::Commitment>, MaybeShared<Self::OpeningProofHint>)> {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => {
                if commit_to_public {
                    let (c, hint) = commit_public(poly, setup);
                    Ok((MaybeShared::Public(Some(c)), MaybeShared::Public(Some(hint))))
                } else {
                    Ok((MaybeShared::Public(None), MaybeShared::Public(None)))
                }
            }
            Rep3MultilinearPolynomial::Shared(shared_poly) => {
                commit_shared::<ProofTranscript, N>(shared_poly, setup, io_ctx, preproc)
            }
        }
    }

    #[tracing::instrument(skip_all, name = "Dory::batch_commit")]
    fn batch_commit_rep3<U, N>(
        polys: &[U],
        setup: &Self::ProverSetup,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<Fr>,
    ) -> eyre::Result<Vec<(MaybeShared<Self::Commitment>, MaybeShared<Self::OpeningProofHint>)>>
    where
        U: Borrow<Rep3MultilinearPolynomial<Fr>> + Sync,
        N: Rep3NetworkWorker,
    {
        let party_id = io_ctx.party_id();

        // Distribute public poly commits across workers: public poly i is
        // committed by worker (i % 3). This balances work evenly.
        let per_poly_commit_public: Vec<bool> = polys
            .par_iter()
            .enumerate()
            .map(|(i, p)| {
                if matches!(p.borrow(), Rep3MultilinearPolynomial::Public(_)) {
                    let should_commit = i % 3 == party_id as usize;
                    should_commit
                } else {
                    false // not applicable for shared polys
                }
            })
            .collect();

        #[cfg(feature = "ring-msm")]
        {
            // Partition into U64Scalars (need io_ctx/preproc) vs local-only.
            let mut ring_idxs = Vec::new();
            let mut local_idxs = Vec::new();
            for (i, p) in polys.iter().enumerate() {
                if matches!(p.borrow(), Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(_) | Rep3SharedPoly::IRingScalars(_))) {
                    ring_idxs.push(i);
                } else {
                    local_idxs.push(i);
                }
            }

            type CommitResult = eyre::Result<(MaybeShared<DoryCommitment>, MaybeShared<DoryOpeningProofHint>)>;

            // rayon::join: local polys in parallel (branch A), U64Scalars sequentially (branch B).
            let (local_results, ring_results): (Vec<CommitResult>, Vec<CommitResult>) = rayon::join(
                || {
                    local_idxs
                        .par_iter()
                        .map(|&i| {
                            commit_local_rep3::<ProofTranscript>(polys[i].borrow(), setup, per_poly_commit_public[i])
                        })
                        .collect()
                },
                || {
                    ring_idxs
                        .iter()
                        .map(|&i| {
                            <Self as Rep3CommitmentScheme<Fr, ProofTranscript>>::commit_rep3(
                                polys[i].borrow(),
                                setup,
                                per_poly_commit_public[i],
                                io_ctx,
                                preproc,
                            )
                        })
                        .collect()
                },
            );

            // Merge results back into original order.
            let mut out: Vec<Option<(MaybeShared<DoryCommitment>, MaybeShared<DoryOpeningProofHint>)>> =
                (0..polys.len()).map(|_| None).collect();
            for (&i, r) in local_idxs.iter().zip(local_results) {
                out[i] = Some(r?);
            }
            for (&i, r) in ring_idxs.iter().zip(ring_results) {
                out[i] = Some(r?);
            }
            Ok(out.into_iter().map(|o| o.unwrap()).collect())
        }

        #[cfg(not(feature = "ring-msm"))]
        {
            let _ = (io_ctx, preproc); // suppress unused warnings
            polys
                .par_iter()
                .zip(per_poly_commit_public.par_iter())
                .map(|(p, &commit_public)| commit_local_rep3::<ProofTranscript>(p.borrow(), setup, commit_public))
                .collect()
        }
    }

    #[tracing::instrument(skip_all, name = "Dory::prove")]
    fn prove_rep3<Network>(
        poly: &Rep3MultilinearPolynomial<Fr>,
        setup: &Self::ProverSetup,
        opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        opening_hint: Option<Self::OpeningProofHint>,
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker,
    {
        if network.is_distributed() {
            return Err(eyre::eyre!("Dory opening proof: distributed subnets unsupported (single-worker mode only)"));
        }
        let party_id = network.party_id();

        let ((sigma, num_vars, nu, l_vec, r_vec), g1_all, g2_all, g1_affine_all, g2_affine_all) = {
            let _span = tracing::info_span!("precompute").entered();
            let params = compute_open_params(poly, opening_point);

            // Zero-copy generator slices (JoltGroupWrapper is #[repr(transparent)])
            let g1_all = setup_g1_projective(setup);
            let g2_all = setup_g2_projective(setup);

            let g1_affine_all = G1Projective::normalize_batch(&g1_all[..(1 << params.0)]);
            let g2_affine_all = G2Projective::normalize_batch(&g2_all[..(1 << params.0)]);
            (params, g1_all, g2_all, g1_affine_all, g2_affine_all)
        };

        // 1) Compute row commitment shares — dispatch based on variant + hint
        let num_rows_target = 1usize << nu;
        let num_columns = 1usize << sigma;
        let row_commit_shares: Vec<G1Projective> = {
            let _span = tracing::trace_span!("row_commits").entered();
            if let Some(hint) = opening_hint {
                // Pre-combined hint: use directly (already the correct additive shares)
                let mut rows: Vec<G1Projective> = hint.iter().map(|h| h.0).collect();
                rows.truncate(num_rows_target);
                rows.resize(num_rows_target, G1Projective::zero());
                rows
            } else {
                let g1_col_affine = &g1_affine_all[..num_columns];
                match poly {
                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(dense)) => {
                        compute_row_commitment_shares_a(dense, setup, nu)
                    }
                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(_) | Rep3SharedPoly::IRingScalars(_)) => {
                        return Err(eyre::eyre!(
                            "Dory prove_rep3: ring scalars require an opening_hint (networked recompute unsupported)"
                        ));
                    }
                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(one_hot)) => {
                        one_hot.commit_rows::<G1Projective>(g1_col_affine)?
                    }
                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RLC(rlc)) => {
                        rlc.commit_rows::<G1Projective>(g1_col_affine)?
                    }
                    Rep3MultilinearPolynomial::Public(_) => {
                        return Err(eyre::eyre!("prove_rep3 does not handle public polynomials"));
                    }
                }
            }
        };

        // 2) compute v_vec share — dispatch based on variant
        let v_vec_share: Vec<Fr> = match poly {
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(dense)) => {
                let mut v = vec![<Fr as ark_ff::Zero>::zero(); num_columns];
                let (global_offset, local_coeffs) = rep3_local_coeffs_a(dense);
                for (k, coeff) in local_coeffs.iter().enumerate() {
                    let idx = global_offset + k;
                    let row = idx / num_columns;
                    let col = idx % num_columns;
                    if row < l_vec.len() {
                        v[col] += *coeff * l_vec[row];
                    }
                }
                v
            }
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RLC(rlc)) => rlc.compute_v_vec_share(&l_vec),
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(one_hot)) => {
                let mut v = vec![<Fr as ark_ff::Zero>::zero(); num_columns];
                one_hot.compute_v_vec_share(Fr::from(1u64), &l_vec, &mut v);
                v
            }
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(_) | Rep3SharedPoly::IRingScalars(_)) => {
                return Err(eyre::eyre!(
                    "Dory prove_rep3: ring scalars unsupported (field-domain v_vec computation missing)"
                ));
            }
            Rep3MultilinearPolynomial::Public(_) => {
                return Err(eyre::eyre!("prove_rep3 does not handle public polynomials"));
            }
        };
        let row_commit_shares_affine: Vec<G1Affine> = G1Projective::normalize_batch(&row_commit_shares);
        let mut v2_share: Vec<G2Projective> = {
            let _span = tracing::trace_span!("v2_init").entered();
            fixed_base_vector_msm_g2(setup, &v_vec_share)
        };
        network.send_response((num_vars, row_commit_shares_affine))?;

        // 3) receive masked row commitments from coordinator
        let (row_commitments_affine, mut row_mask_shares): DoryMaskedRowsRequest = {
            let _span = tracing::trace_span!("receive_masked_rows").entered();
            network.receive_request()?
        };
        let mut padded_row_commitments_affine = row_commitments_affine.clone();
        if nu < sigma {
            padded_row_commitments_affine.resize(1 << sigma, G1Affine::zero());
        }
        debug_assert_eq!(row_mask_shares.len(), padded_row_commitments_affine.len());

        // c_share + d2_share: MSMs + pairings for VMV message
        let ((c_share, d2_share), public_e1, blocked_vmv_correction_share) = {
            let _span = tracing::trace_span!("vmv_message").entered();
            let ((c_and_d2, public_e1), blocked_vmv_correction_share) = rayon::join(
                || {
                    rayon::join(
                        || {
                            let t_vec_v = msm_g1_affine(&padded_row_commitments_affine, &v_vec_share);
                            let g_fin_affine = setup.g2_vec[0].0.into_affine();
                            let c_share = Bn254::pairing(t_vec_v, g_fin_affine).0;

                            let gamma1_v = msm_g1_affine(&g1_affine_all, &v_vec_share);
                            let d2_share = Bn254::pairing(gamma1_v, g_fin_affine).0;
                            (c_share, d2_share)
                        },
                        || compute_public_vmv_e1(party_id, &row_commitments_affine, &l_vec),
                    )
                },
                || {
                    let _span = tracing::trace_span!("blocked_vmv_c_local").entered();
                    local_masked_scalar_inner_product(network, &row_mask_shares, &v_vec_share)
                },
            );
            (c_and_d2, public_e1, blocked_vmv_correction_share?)
        };

        network.send_response(((c_share, d2_share), public_e1, blocked_vmv_correction_share))?;

        // s1 = R, s2 = L (matches dory vmv_state_to_dory_prover_state)
        let mut s1 = r_vec;
        let mut s2 = l_vec;
        if nu < sigma {
            s1.resize(1 << sigma, Fr::zero());
            s2.resize(1 << sigma, Fr::zero());
        }

        let mut v1_pub: Vec<G1Projective> = padded_row_commitments_affine.iter().map(|a| a.into_group()).collect();
        let mut curr_rounds = sigma;

        let _loop_span = tracing::info_span!("reduction_loop").entered();
        while curr_rounds > 0 {
            let n2 = 1usize << (curr_rounds - 1);

            // First message: d2_left/d2_right pairings (parallel)
            let (d2_share_round, public_first_round) = {
                let _span = tracing::trace_span!("d2_pairing", n2).entered();
                let v2_affine_pre = G2Projective::normalize_batch(&v2_share);
                rayon::join(
                    || {
                        let g1_prime_aff = &g1_affine_all[..n2];
                        let (v2_l_aff, v2_r_aff) = v2_affine_pre.split_at(n2);
                        rayon::join(
                            || multi_pairing_both_affine(g1_prime_aff, v2_l_aff),
                            || multi_pairing_both_affine(g1_prime_aff, v2_r_aff),
                        )
                    },
                    || {
                        compute_public_first_reduce_terms(
                            party_id,
                            setup,
                            &v1_pub,
                            &g1_affine_all,
                            &g2_affine_all,
                            &s1,
                            &s2,
                            curr_rounds,
                        )
                    },
                )
            };

            network.send_response((d2_share_round, public_first_round))?;

            // Receive beta challenge from coordinator
            let (beta, beta_inv): (Fr, Fr) = {
                let _span = tracing::trace_span!("wait_beta", n2).entered();
                network.receive_request()?
            };

            // Update v1/v2 with generators
            {
                let _span = tracing::trace_span!("v1v2_update", n2).entered();
                jolt_optimizations::vector_add_scalar_mul_g1_online(&mut v1_pub, &g1_all[..(1 << curr_rounds)], beta);

                if party_id == PartyID::ID0 {
                    jolt_optimizations::vector_add_scalar_mul_g2_online(
                        &mut v2_share,
                        &g2_all[..(1 << curr_rounds)],
                        beta_inv,
                    );
                }
            }

            // Second message: c_plus/c_minus pairings + e2 MSMs (all 4 in parallel)
            let (share_second_round, public_second_round, blocked_second_corrections) = {
                let _span = tracing::trace_span!("second_msg", n2).entered();
                let v1_affine_post = G1Projective::normalize_batch(&v1_pub);
                let v2_affine_post = G2Projective::normalize_batch(&v2_share);
                let replicated_prev_v2_affine = {
                    let _span = tracing::trace_span!("blocked_second_v2_exchange", n2).entered();
                    network.reshare_many(&v2_affine_post)?
                };
                let (s1_l, s1_r) = s1.split_at(n2);
                let (((c_plus_share_round, c_minus_share_round), (e2_plus_share, e2_minus_share)), public_second_round) =
                    rayon::join(
                        || {
                            rayon::join(
                                || {
                                    let (v1_l_aff, v1_r_aff) = v1_affine_post.split_at(n2);
                                    let (v2_l_aff, v2_r_aff) = v2_affine_post.split_at(n2);
                                    rayon::join(
                                        || multi_pairing_both_affine(v1_l_aff, v2_r_aff),
                                        || multi_pairing_both_affine(v1_r_aff, v2_l_aff),
                                    )
                                },
                                || {
                                    let (v2_l_aff, v2_r_aff) = v2_affine_post.split_at(n2);
                                    rayon::join(
                                        || msm_g2_affine(v2_r_aff, s1_l).into_affine(),
                                        || msm_g2_affine(v2_l_aff, s1_r).into_affine(),
                                    )
                                },
                            )
                        },
                        || compute_public_second_reduce_terms(party_id, &v1_affine_post, &s2, n2),
                    );
                let blocked_second_corrections = {
                    let (mask_left, mask_right) = row_mask_shares.split_at(n2);
                    let (v2_l_aff, v2_r_aff) = v2_affine_post.split_at(n2);
                    let (prev_v2_l_aff, prev_v2_r_aff) = replicated_prev_v2_affine.split_at(n2);
                    rayon::join(
                        || {
                            let _span = tracing::trace_span!("blocked_second_c_plus_local").entered();
                            accumulate_masked_g2_msm(mask_left, v2_r_aff, prev_v2_r_aff)
                        },
                        || {
                            let _span = tracing::trace_span!("blocked_second_c_minus_local").entered();
                            accumulate_masked_g2_msm(mask_right, v2_l_aff, prev_v2_l_aff)
                        },
                    )
                };
                (
                    ((c_plus_share_round, c_minus_share_round), (e2_plus_share, e2_minus_share)),
                    public_second_round,
                    blocked_second_corrections,
                )
            };

            network.send_response((
                share_second_round,
                public_second_round,
                (blocked_second_corrections.0.into_affine(), blocked_second_corrections.1.into_affine()),
            ))?;

            // Receive alpha challenge from coordinator
            let (alpha, alpha_inv): (Fr, Fr) = {
                let _span = tracing::trace_span!("wait_alpha", n2).entered();
                network.receive_request()?
            };

            // Fold v1, v2, s1, s2 — in-place GLV-accelerated for group elements
            {
                let _span = tracing::trace_span!("fold", n2).entered();

                // v1[i] = alpha * v1_l[i] + v1_r[i] (in-place, GLV 2D Shamir)
                let (v1_l_mut, v1_r_ref) = v1_pub.split_at_mut(n2);
                jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(v1_l_mut, alpha, v1_r_ref);
                v1_pub.truncate(n2);

                // v2[i] = alpha_inv * v2_l[i] + v2_r[i] (in-place, GLV 4D Shamir)
                let (v2_l_mut, v2_r_ref) = v2_share.split_at_mut(n2);
                jolt_optimizations::vector_scalar_mul_add_gamma_g2_online(v2_l_mut, alpha_inv, v2_r_ref);
                v2_share.truncate(n2);

                // s1, s2 scalar folds (no GLV needed for field elements)
                let (s1_l, s1_r) = s1.split_at(n2);
                let (s2_l, s2_r) = s2.split_at(n2);
                let (s1_next, s2_next): (Vec<Fr>, Vec<Fr>) =
                    (0..n2).into_par_iter().map(|i| (s1_l[i] * alpha + s1_r[i], s2_l[i] * alpha_inv + s2_r[i])).unzip();
                s1 = s1_next;
                s2 = s2_next;
                fold_mask_shares(&mut row_mask_shares, alpha, n2);
            }

            curr_rounds -= 1;
        }
        drop(_loop_span);

        // Final: send v2 final share to coordinator for scalar product message
        debug_assert_eq!(v2_share.len(), 1);
        network.send_response(v2_share[0].into_affine())?;

        Ok(())
    }

    fn combine_hints_rep3(
        hints: Vec<MaybeShared<Self::OpeningProofHint>>,
        coeffs: &[Fr],
        party_id: PartyID,
    ) -> Self::OpeningProofHint {
        debug_assert_eq!(hints.len(), coeffs.len());
        let num_rows = DoryGlobals::get_max_num_rows();

        // Mirror vanilla combine_hints pattern: Horner-style accumulation using
        // the fused GLV-accelerated `v[i] = scalar * v[i] + gamma[i]`.
        let mut rlc_hint = vec![ArkG1(G1Projective::zero()); num_rows];
        for (coeff, hint) in coeffs.iter().zip(hints.into_iter()) {
            // Determine the effective hint for this polynomial.
            // Public(None) → skip (this worker didn't commit this public poly).
            // Public(Some(h)) → this worker committed it; add its hint.
            let mut effective_hint = match hint {
                MaybeShared::Shared(h) => h.into_rows(),
                MaybeShared::Public(Some(h)) => h.into_rows(),
                _ => continue,
            };

            effective_hint.resize(num_rows, ArkG1(G1Projective::zero()));

            // Safety: JoltGroupWrapper<G1Projective> is #[repr(transparent)]
            let row_commitments: &mut [G1Projective] = unsafe {
                std::slice::from_raw_parts_mut(effective_hint.as_mut_ptr() as *mut G1Projective, effective_hint.len())
            };
            let rlc_row_commitments: &[G1Projective] =
                unsafe { std::slice::from_raw_parts(rlc_hint.as_ptr() as *const G1Projective, rlc_hint.len()) };

            // v[i] = coeff * v[i] + accumulated[i]
            jolt_core::jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(
                row_commitments,
                *coeff,
                rlc_row_commitments,
            );

            let _ = std::mem::replace(&mut rlc_hint, effective_hint);
        }

        DoryOpeningProofHint::new(rlc_hint)
    }
}

fn owns_public_vmv_e1(party_id: PartyID) -> bool {
    party_id == PartyID::ID0
}

fn owns_first_reduce_d1_left(party_id: PartyID) -> bool {
    party_id == PartyID::ID0
}

fn owns_first_reduce_d1_right(party_id: PartyID) -> bool {
    party_id == PartyID::ID1
}

fn owns_first_reduce_e_betas(party_id: PartyID) -> bool {
    party_id == PartyID::ID2
}

fn owns_second_reduce_e1_plus(party_id: PartyID) -> bool {
    party_id == PartyID::ID0
}

fn owns_second_reduce_e1_minus(party_id: PartyID) -> bool {
    party_id == PartyID::ID1
}

#[inline]
fn compute_nu(num_vars: usize, sigma: usize) -> usize {
    num_vars.checked_sub(sigma).expect("Dory opening point must have at least sigma coordinates")
}

fn fold_mask_shares(mask_shares: &mut Vec<Rep3PrimeFieldShare<Fr>>, alpha: Fr, n2: usize) {
    let (left, right) = mask_shares.split_at(n2);
    let next: Vec<_> = (0..n2)
        .into_par_iter()
        .map(|i| Rep3PrimeFieldShare::new(left[i].a * alpha + right[i].a, left[i].b * alpha + right[i].b))
        .collect();
    *mask_shares = next;
}

fn local_masked_scalar_inner_product<N: Rep3NetworkWorker>(
    network: &mut N,
    mask_shares: &[Rep3PrimeFieldShare<Fr>],
    additive_values: &[Fr],
) -> eyre::Result<Fr> {
    let replicated_prev_values = {
        let _span = tracing::trace_span!("blocked_vmv_c_exchange").entered();
        network.reshare_many(additive_values)?
    };
    Ok(mask_shares
        .par_iter()
        .zip(additive_values.par_iter())
        .zip(replicated_prev_values.par_iter())
        .map(|((mask_share, self_share), prev_share)| {
            (mask_share.a * *self_share) + (mask_share.a * *prev_share) + (mask_share.b * *self_share)
        })
        .reduce(Fr::zero, |acc, value| acc + value))
}

fn accumulate_masked_g2_msm(
    mask_shares: &[Rep3PrimeFieldShare<Fr>],
    additive_points_affine: &[G2Affine],
    replicated_prev_points: &[G2Affine],
) -> G2Projective {
    let self_scalars: Vec<Fr> = mask_shares.iter().map(|mask_share| mask_share.a + mask_share.b).collect();
    let prev_scalars: Vec<Fr> = mask_shares.iter().map(|mask_share| mask_share.a).collect();
    let (self_term, prev_term) = rayon::join(
        || msm_g2_affine(additive_points_affine, &self_scalars),
        || msm_g2_affine(replicated_prev_points, &prev_scalars),
    );
    self_term + prev_term
}

fn compute_open_params(
    poly: &Rep3MultilinearPolynomial<Fr>,
    opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
) -> DoryOpenParams {
    let sigma = DoryGlobals::get_num_columns().log_2();
    let num_vars = poly.get_num_vars();
    let nu = compute_nu(num_vars, sigma);

    // Dory uses opposite endian-ness to Jolt.
    let point_wrapped: Vec<_> = opening_point
        .iter()
        .rev()
        .map(|c| {
            let c_fr: Fr = (*c).into();
            jolt_to_ark(&c_fr)
        })
        .collect();
    let (l_vec_w, r_vec_w) = compute_left_right_vectors(&point_wrapped, nu, sigma);
    let l_vec: Vec<Fr> = l_vec_w.into_iter().map(|x| ark_to_jolt(&x)).collect();
    let r_vec = r_vec_w.into_iter().map(|x| ark_to_jolt(&x)).collect();

    (sigma, num_vars, nu, l_vec, r_vec)
}

fn compute_public_vmv_e1(party_id: PartyID, row_commitments_affine: &[G1Affine], l_vec: &[Fr]) -> Option<G1Affine> {
    owns_public_vmv_e1(party_id).then(|| msm_g1_affine(row_commitments_affine, l_vec).into_affine())
}

fn compute_public_first_reduce_terms(
    party_id: PartyID,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    v1_pub: &[G1Projective],
    g1_affine_all: &[G1Affine],
    g2_affine_all: &[G2Affine],
    s1: &[Fr],
    s2: &[Fr],
    curr_nu: usize,
) -> DoryFirstReducePublicMsg {
    let n2 = 1usize << (curr_nu - 1);
    match party_id {
        party_id if owns_first_reduce_d1_left(party_id) => {
            let d1_left = multi_pairing_setup_g2_cached_affine(
                &G1Projective::normalize_batch(&v1_pub[..n2]),
                setup,
                g2_affine_all,
            );
            (Some(d1_left), None, None, None)
        }
        party_id if owns_first_reduce_d1_right(party_id) => {
            let d1_right = multi_pairing_setup_g2_cached_affine(
                &G1Projective::normalize_batch(&v1_pub[n2..(1 << curr_nu)]),
                setup,
                g2_affine_all,
            );
            (None, Some(d1_right), None, None)
        }
        party_id if owns_first_reduce_e_betas(party_id) => {
            let (e1_beta, e2_beta) = rayon::join(
                || msm_g1_affine(&g1_affine_all[..(1 << curr_nu)], s2).into_affine(),
                || msm_g2_affine(&g2_affine_all[..(1 << curr_nu)], s1).into_affine(),
            );
            (None, None, Some(e1_beta), Some(e2_beta))
        }
        _ => unreachable!(),
    }
}

fn compute_public_second_reduce_terms(
    party_id: PartyID,
    v1_affine_post: &[G1Affine],
    s2: &[Fr],
    n2: usize,
) -> DorySecondReducePublicMsg {
    let (v1_l_aff, v1_r_aff) = v1_affine_post.split_at(n2);
    let (s2_l, s2_r) = s2.split_at(n2);
    match party_id {
        party_id if owns_second_reduce_e1_plus(party_id) => (Some(msm_g1_affine(v1_l_aff, s2_r).into_affine()), None),
        party_id if owns_second_reduce_e1_minus(party_id) => (None, Some(msm_g1_affine(v1_r_aff, s2_l).into_affine())),
        _ => (None, None),
    }
}

// =============================================================================
// Helpers (Rep3)
// =============================================================================

/// Commit a local-only (non-RingScalars) polynomial. No io_ctx/preproc needed.
/// Used by the parallel branch of `batch_commit_rep3`.
pub fn commit_local_rep3<ProofTranscript: Transcript>(
    poly: &Rep3MultilinearPolynomial<Fr>,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    commit_to_public: bool,
) -> eyre::Result<(
    MaybeShared<<DoryCommitmentScheme as CommitmentScheme>::Commitment>,
    MaybeShared<<DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint>,
)> {
    match poly {
        Rep3MultilinearPolynomial::Public(poly) => {
            if commit_to_public {
                let _span = tracing::trace_span!("commit_public").entered();
                let (c, hint) = commit_public(poly, setup);
                Ok((MaybeShared::Public(Some(c)), MaybeShared::Public(Some(hint))))
            } else {
                Ok((MaybeShared::Public(None), MaybeShared::Public(None)))
            }
        }
        Rep3MultilinearPolynomial::Shared(shared_poly) => {
            assert!(
                !matches!(shared_poly, Rep3SharedPoly::RingScalars(_) | Rep3SharedPoly::IRingScalars(_)),
                "commit_local_rep3 called on ring scalars poly; use commit_rep3 instead"
            );
            let sigma = DoryGlobals::get_num_columns().log_2();
            let num_columns = DoryGlobals::get_num_columns();
            let (num_vars, row_commitments_share) = match shared_poly {
                Rep3SharedPoly::Dense(poly) => {
                    let nu = compute_nu(poly.get_num_vars(), sigma);
                    (poly.get_num_vars(), compute_row_commitment_shares_a(poly, setup, nu))
                }
                Rep3SharedPoly::OneHot(poly) => {
                    let g1_proj = &setup_g1_projective(setup)[..num_columns];
                    let bases = G1Projective::normalize_batch(g1_proj);
                    let rows = poly.commit_rows::<G1Projective>(&bases).expect("OneHot commit_rows preconditions met");
                    (poly.get_num_vars(), rows)
                }
                Rep3SharedPoly::RingScalars(_) | Rep3SharedPoly::IRingScalars(_) => unreachable!(),
                Rep3SharedPoly::RLC(_) => {
                    unreachable!("RLC polynomials should not be committed directly")
                }
            };
            rows_to_commitment(row_commitments_share, num_vars, sigma, setup)
        }
    }
}

/// Row-commit + pairing for a shared polynomial.
/// Dense/OneHot are committed locally; RingScalars (ring-msm) dispatches to ring MSM.
#[tracing::instrument(skip_all)]
fn commit_shared<ProofTranscript: Transcript, N: Rep3NetworkWorker>(
    shared_poly: &Rep3SharedPoly<Fr>,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    _io_ctx: &mut IoContextPool<N>,
    _preproc: &mut PreprocessingPool<Fr>,
) -> eyre::Result<(
    MaybeShared<<DoryCommitmentScheme as CommitmentScheme>::Commitment>,
    MaybeShared<<DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint>,
)> {
    let sigma = DoryGlobals::get_num_columns().log_2();
    let num_columns = DoryGlobals::get_num_columns();

    let (num_vars, row_commitments_share) = match shared_poly {
        Rep3SharedPoly::Dense(poly) => {
            let nu = compute_nu(poly.get_num_vars(), sigma);
            (poly.get_num_vars(), compute_row_commitment_shares_a(poly, setup, nu))
        }
        Rep3SharedPoly::OneHot(poly) => {
            let g1_proj = &setup_g1_projective(setup)[..num_columns];
            let bases = G1Projective::normalize_batch(g1_proj);
            let rows = poly.commit_rows::<G1Projective>(&bases).expect("OneHot commit_rows preconditions met");
            (poly.get_num_vars(), rows)
        }
        #[cfg(feature = "ring-msm")]
        Rep3SharedPoly::RingScalars(poly_ring) => {
            let nu = compute_nu(poly_ring.get_num_vars(), sigma);
            let rows = compute_row_commitment_shares_ring(poly_ring, setup, nu, _io_ctx, _preproc)?;
            (poly_ring.get_num_vars(), rows)
        }
        #[cfg(feature = "ring-msm")]
        Rep3SharedPoly::IRingScalars(poly_inc) => {
            let nu = compute_nu(poly_inc.get_num_vars(), sigma);
            let rows = compute_row_commitment_shares_iring(poly_inc, setup, nu, _io_ctx, _preproc)?;
            (poly_inc.get_num_vars(), rows)
        }
        #[cfg(not(feature = "ring-msm"))]
        Rep3SharedPoly::RingScalars(_) | Rep3SharedPoly::IRingScalars(_) => unreachable!(),
        Rep3SharedPoly::RLC(_) => {
            unreachable!("RLC polynomials should not be committed directly")
        }
    };

    rows_to_commitment(row_commitments_share, num_vars, sigma, setup)
}

/// Shared helper: pad row commitments, compute pairing, return commitment + hint.
#[tracing::instrument(skip_all, level = "trace")]
fn rows_to_commitment(
    row_commitments_share: Vec<G1Projective>,
    num_vars: usize,
    sigma: usize,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
) -> eyre::Result<(
    MaybeShared<<DoryCommitmentScheme as CommitmentScheme>::Commitment>,
    MaybeShared<<DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint>,
)> {
    let _span = tracing::trace_span!("combine_rows").entered();
    let nu = compute_nu(num_vars, sigma);
    let num_rows_target = 1usize << nu;

    let mut row_commitments = row_commitments_share;
    row_commitments.resize(num_rows_target, G1Projective::zero());
    let row_commitments_wrapped: Vec<ArkG1> = row_commitments.into_iter().map(ArkG1).collect();

    let _pairing_span = tracing::trace_span!("multi_pairing").entered();
    let commitment_share = <BN254 as PairingCurve>::multi_pair_g2_setup(
        &row_commitments_wrapped,
        &setup.g2_vec[..row_commitments_wrapped.len()],
    );
    drop(_pairing_span);

    let hint_share = DoryOpeningProofHint::new(row_commitments_wrapped);

    Ok((MaybeShared::Shared(DoryCommitment(commitment_share)), MaybeShared::Shared(hint_share)))
}

/// Zero-copy view of `setup.core.g1_vec` as `&[G1Projective]`.
/// Safety: `JoltGroupWrapper<G1Projective>` is `#[repr(transparent)]`.
pub fn setup_g1_projective(setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup) -> &[G1Projective] {
    unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr() as *const G1Projective, setup.g1_vec.len()) }
}

/// Zero-copy view of `setup.core.g2_vec` as `&[G2Projective]`.
/// Safety: `JoltGroupWrapper<G2Projective>` is `#[repr(transparent)]`.
pub fn setup_g2_projective(setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup) -> &[G2Projective] {
    unsafe { std::slice::from_raw_parts(setup.g2_vec.as_ptr() as *const G2Projective, setup.g2_vec.len()) }
}

#[cfg(feature = "ring-msm")]
/// Precompute the full Q-point array for daPoint preprocessing of U64Scalars wrap correction.
///
/// Returns `2 * num_coeffs` points ordered to match consumption in
/// `compute_row_commitment_shares_ring`: for each row, [q0_segment, q1_segment].
///
/// Q points: `q0[c] = 2^XLEN * g1_vec[c]`, `q1[c] = 2 * q0[c]` for c in 0..num_columns.
#[tracing::instrument(skip_all)]
pub fn precompute_dapoint_qs(
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    num_coeffs: usize,
    num_columns: usize,
) -> Vec<G1Projective> {
    use jolt_common::constants::XLEN;

    let g1_proj = &setup_g1_projective(setup)[..num_columns];

    let q0_cols: Vec<G1Projective> = g1_proj
        .iter()
        .map(|b| {
            let mut p = *b;
            for _ in 0..XLEN {
                p.double_in_place();
            }
            p
        })
        .collect();
    let q1_cols: Vec<G1Projective> = q0_cols.iter().map(|p| *p + *p).collect();

    let mut all_q = Vec::with_capacity(2 * num_coeffs);
    let num_full_rows = num_coeffs / num_columns;
    let remainder = num_coeffs % num_columns;
    for _ in 0..num_full_rows {
        all_q.extend_from_slice(&q0_cols);
        all_q.extend_from_slice(&q1_cols);
    }
    if remainder > 0 {
        all_q.extend_from_slice(&q0_cols[..remainder]);
        all_q.extend_from_slice(&q1_cols[..remainder]);
    }
    all_q
}

/// Precompute the daPoint q-values for IRingScalars (biased inc) wrap correction.
///
/// IRingScalars uses u64 scalars regardless of XLEN, so the doublings are always
/// 64 (not XLEN). Q points: `q0[c] = 2^64 * g1_vec[c]`, `q1[c] = 2 * q0[c]`.
///
/// Returns `2 * num_coeffs` points ordered to match consumption in the ring-MSM
/// commit: for each row, [q0_segment, q1_segment].
#[cfg(feature = "ring-msm")]
#[tracing::instrument(skip_all)]
pub fn precompute_dapoint_qs_iring(
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    num_coeffs: usize,
    num_columns: usize,
) -> Vec<G1Projective> {
    let g1_proj = &setup_g1_projective(setup)[..num_columns];

    let q0_cols: Vec<G1Projective> = g1_proj
        .iter()
        .map(|b| {
            let mut p = *b;
            for _ in 0..64 {
                p.double_in_place();
            }
            p
        })
        .collect();
    let q1_cols: Vec<G1Projective> = q0_cols.iter().map(|p| *p + *p).collect();

    let mut all_q = Vec::with_capacity(2 * num_coeffs);
    let num_full_rows = num_coeffs / num_columns;
    let remainder = num_coeffs % num_columns;
    for _ in 0..num_full_rows {
        all_q.extend_from_slice(&q0_cols);
        all_q.extend_from_slice(&q1_cols);
    }
    if remainder > 0 {
        all_q.extend_from_slice(&q0_cols[..remainder]);
        all_q.extend_from_slice(&q1_cols[..remainder]);
    }
    all_q
}

fn rep3_local_coeffs_a(poly: &Rep3DensePolynomial<Fr>) -> (usize, Vec<Fr>) {
    let coeffs_ref = poly.coeffs_ref();
    let local = coeffs_ref.iter().map(|s| s.a).collect::<Vec<Fr>>();
    let global_offset = poly.global_chunk_range.map(|(s, _)| s).unwrap_or(0);
    (global_offset, local)
}

#[tracing::instrument(skip_all, level = "trace")]
fn commit_public(
    poly: &jolt_core::poly::multilinear_polynomial::MultilinearPolynomial<Fr>,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
) -> (DoryCommitment, DoryOpeningProofHint) {
    DoryCommitmentScheme::commit(poly, setup)
}

#[tracing::instrument(skip_all, name = "dense::commit_rows", level = "trace")]
fn compute_row_commitment_shares_a(
    poly: &Rep3DensePolynomial<Fr>,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    nu: usize,
) -> Vec<G1Projective> {
    let sigma = DoryGlobals::get_num_columns().log_2();
    let num_columns = 1usize << sigma;
    let num_rows_target = 1usize << nu;

    let g1_proj = &setup_g1_projective(setup)[..num_columns];
    let bases = G1Projective::normalize_batch(g1_proj);

    let (global_offset, local_coeffs) = rep3_local_coeffs_a(poly);
    let mut row_commitments = vec![G1Projective::zero(); num_rows_target];

    // local coeffs correspond to contiguous global indices [global_offset, global_offset + local_len)
    let local_len = local_coeffs.len();
    let start = global_offset;
    let end = global_offset + local_len;

    if local_len == 0 {
        return row_commitments;
    }

    let first_row = start / num_columns;
    let last_row = (end - 1) / num_columns;

    for row in first_row..=last_row {
        let row_start = row * num_columns;
        let row_end = row_start + num_columns;
        let seg_start = start.max(row_start);
        let seg_end = end.min(row_end);
        let seg_len = seg_end - seg_start;
        if seg_len == 0 {
            continue;
        }
        let col_start = seg_start - row_start;
        let local_start = seg_start - start;

        let scalars = &local_coeffs[local_start..local_start + seg_len];
        let msm: G1Projective = ArkVariableBaseMSM::msm(&bases[col_start..col_start + seg_len], scalars)
            .expect("row segment MSM should succeed");
        if row < row_commitments.len() {
            row_commitments[row] += msm;
        }
    }

    row_commitments
}

#[cfg(feature = "ring-msm")]
/// Compute row commitment shares for a U64Scalars polynomial.
///
/// Public coefficients (NoOp padding, immediates) skip ring B2A, wrap extraction,
/// and daPoint correction — only shared coefficients consume MPC preprocessing.
#[tracing::instrument(skip_all, name = "dense::commit_rows_ring", level = "trace")]
fn compute_row_commitment_shares_ring<N: Rep3NetworkWorker>(
    poly: &Rep3CompactPolynomial,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    nu: usize,
    io_ctx: &mut IoContextPool<N>,
    preproc: &mut PreprocessingPool<Fr>,
) -> eyre::Result<Vec<G1Projective>> {
    use crate::zkvm::instruction::types::rep3_operand::Rep3Operand;
    use jolt_common::constants::{XlenInt, XLEN};
    use mpc_core::protocols::rep3_ring::casts::downcast;

    let sigma = DoryGlobals::get_num_columns().log_2();
    let num_columns = 1usize << sigma;
    let num_rows_target = 1usize << nu;

    let g1_proj = &setup_g1_projective(setup)[..num_columns];
    let bases_aff = G1Projective::normalize_batch(g1_proj);

    let n = poly.coeffs.len();
    if n == 0 {
        return Ok(vec![G1Projective::zero(); num_rows_target]);
    }

    let party_id = io_ctx.main().id;
    let mut io = io_ctx.main();

    // Partition coefficients: extract shared-only binary + arithmetic shares,
    // and build a position map from global index → shared index.
    let mut shared_pos_map: Vec<Option<usize>> = vec![None; n];
    let mut shared_bins: Vec<Rep3RingShare<XlenInt>> = Vec::new();
    let mut shared_ariths: Vec<Rep3RingShare<XlenInt>> = Vec::new();
    for (i, coeff) in poly.coeffs.iter().enumerate() {
        if let Rep3Operand::Shared { binary, arithmetic, .. } = coeff {
            shared_pos_map[i] = Some(shared_bins.len());
            shared_bins.push(*binary);
            shared_ariths.push(downcast(arithmetic.unwrap()));
        }
    }
    let num_shared = shared_bins.len();

    // Ring B2A + wrap extraction only for shared coefficients.
    let (m0_bin, m1_bin) = if num_shared > 0 {
        // Zero-extend shared xlen value → Dory carry ring.
        let arith_ext: Vec<Rep3RingShare<DoryCarryRing>> = shared_ariths
            .iter()
            .map(|s| Rep3RingShare {
                a: RingElement(DoryCarryRing::try_from(s.a.0 as u128).expect("xlen share fits in Dory carry ring")),
                b: RingElement(DoryCarryRing::try_from(s.b.0 as u128).expect("xlen share fits in Dory carry ring")),
            })
            .collect();
        let bin_ext: Vec<Rep3RingShare<DoryCarryRing>> = shared_bins
            .iter()
            .map(|s| Rep3RingShare {
                a: RingElement(DoryCarryRing::try_from(s.a.0 as u128).expect("xlen share fits in Dory carry ring")),
                b: RingElement(DoryCarryRing::try_from(s.b.0 as u128).expect("xlen share fits in Dory carry ring")),
            })
            .collect();

        // Ring B2A via edaBits Π₂ — 2 rounds
        let ring_edabits = preproc.take_ring_edabits_dory(num_shared)?;
        let val_arith: Vec<Rep3RingShare<DoryCarryRing>> =
            rep3_ring::conversion::b2a_preproc_many(&bin_ext, &ring_edabits, &mut io)?;
        let diff: Vec<Rep3RingShare<DoryCarryRing>> =
            arith_ext.iter().zip(val_arith.iter()).map(|(a, v)| *a - *v).collect();

        // Extract m bits via DaBit mask+open (1 round)
        let wrap_masks = preproc.take_wrap_masks(num_shared)?;
        let (m0, m1) = rep3_ring::wrap_mask::extract_wrap_m2_from_diff_many(&diff, &wrap_masks, &mut io)?;
        (m0, m1)
    } else {
        (Vec::new(), Vec::new())
    };

    // Precompute q0/q1 for all columns: q0[c] = 2^XLEN*Γ1[c], q1[c] = 2*q0[c].
    let mut q0_cols: Vec<G1Projective> = Vec::with_capacity(num_columns);
    for b in g1_proj.iter() {
        let mut p = *b;
        for _ in 0..XLEN {
            p.double_in_place();
        }
        q0_cols.push(p);
    }
    let q1_cols: Vec<G1Projective> = q0_cols.iter().map(|p| *p + *p).collect();

    // MSM + correction, computed per row segment.
    let mut row_commitments = vec![G1Projective::zero(); num_rows_target];
    let first_row = 0;
    let last_row = (n - 1) / num_columns;

    for row in first_row..=last_row {
        let row_start = row * num_columns;
        let row_end = row_start + num_columns;
        let seg_end = n.min(row_end);
        let seg_len = seg_end - row_start;
        if seg_len == 0 {
            continue;
        }
        let local_start = row_start;

        // Build MSM scalars: shared → a-limb, public → trivial share for ID0.
        //
        // Public values (signed immediates) must be cast through XlenInt to match
        // the field-share representation: `F::from_i128(v as XlenInt as i128)`.
        // This wraps negative values to their unsigned XlenInt representation
        // (e.g. -5i128 → 4294967291u32 → 4294967291u64), which fits in u64
        // and matches the vanilla Jolt field encoding.
        let scalars_u64: Vec<u64> = poly.coeffs[local_start..local_start + seg_len]
            .iter()
            .map(|op| match op {
                Rep3Operand::Shared { arithmetic, .. } => {
                    let arith_xlen: Rep3RingShare<XlenInt> = downcast(arithmetic.unwrap());
                    arith_xlen.a.0 as u64
                }
                Rep3Operand::Public(v) => {
                    if party_id == PartyID::ID0 {
                        // Cast through XlenInt to match vanilla: `v as XlenInt as u64`
                        (*v as XlenInt) as u64
                    } else {
                        0
                    }
                }
            })
            .collect();
        let msm: G1Projective = ArkVariableBaseMSM::msm_u64(&bases_aff[..seg_len], &scalars_u64, false);

        // daPoint correction only for shared coefficients in this segment.
        // Take daPoints for ALL positions in the segment (matching the offline
        // precompute_dapoint_qs order: all m0/q0 first, then all m1/q1),
        // then select only the shared-position entries for the dot product.
        let batch = preproc.take_dapoints(2 * seg_len)?;
        let mut bits_all: Vec<Rep3RingShare<Bit>> = Vec::new();
        let mut q_all: Vec<G1Projective> = Vec::new();
        let mut dp_selected: Vec<usize> = Vec::new();
        for seg_i in 0..seg_len {
            if let Some(shared_idx) = shared_pos_map[local_start + seg_i] {
                bits_all.push(m0_bin[shared_idx]);
                q_all.push(q0_cols[seg_i]);
                dp_selected.push(seg_i);
            }
        }
        for seg_i in 0..seg_len {
            if let Some(shared_idx) = shared_pos_map[local_start + seg_i] {
                bits_all.push(m1_bin[shared_idx]);
                q_all.push(q1_cols[seg_i]);
                dp_selected.push(seg_len + seg_i);
            }
        }

        if !bits_all.is_empty() {
            let filtered_batch = batch.select(&dp_selected);
            let corr_add = rep3::pointshare::dot_product_dapoints(&bits_all, &q_all, &filtered_batch, &mut io)?;
            if row < row_commitments.len() {
                row_commitments[row] += msm - corr_add;
            }
        } else if row < row_commitments.len() {
            row_commitments[row] += msm;
        }
    }

    Ok(row_commitments)
}

#[cfg(feature = "ring-msm")]
/// Compute row commitment shares for an IRingScalars polynomial (biased inc, u64 scalars).
///
/// All coefficients are Shared (biased_inc = post - pre + 2^XLEN, always non-negative).
/// Uses U66 carry ring for wrap correction, 64-bit q doublings, and per-row bias correction
/// to account for the public 2^XLEN bias added to each scalar.
///
/// After MSM + wrap correction, each row subtracts: `2^XLEN * Σ bases[col_in_row]`.
#[tracing::instrument(skip_all, name = "dense::commit_rows_iring", level = "trace")]
fn compute_row_commitment_shares_iring<N: Rep3NetworkWorker>(
    poly: &Rep3CompactPolynomial,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    nu: usize,
    io_ctx: &mut IoContextPool<N>,
    preproc: &mut PreprocessingPool<Fr>,
) -> eyre::Result<Vec<G1Projective>> {
    use crate::zkvm::instruction::types::rep3_operand::Rep3Operand;
    use jolt_common::constants::XLEN;

    let sigma = DoryGlobals::get_num_columns().log_2();
    let num_columns = 1usize << sigma;
    let num_rows_target = 1usize << nu;

    let g1_proj = &setup_g1_projective(setup)[..num_columns];
    let bases_aff = G1Projective::normalize_batch(g1_proj);

    let n = poly.coeffs.len();
    if n == 0 {
        return Ok(vec![G1Projective::zero(); num_rows_target]);
    }

    let party_id = io_ctx.main().id;
    let mut io = io_ctx.main();

    // Extract arithmetic u64 shares from all coefficients.
    // IRingScalars has no Public operands — all are Shared with arithmetic = Some(u64).
    let ariths_u64: Vec<Rep3RingShare<u64>> = poly
        .coeffs
        .iter()
        .map(|op| match op {
            Rep3Operand::Shared { arithmetic, .. } => {
                let wide = arithmetic.expect("IRingScalars: missing arithmetic share");
                // ArithmeticWideInt is u64 for rv32, downcast is identity
                Rep3RingShare { a: RingElement(wide.a.0 as u64), b: RingElement(wide.b.0 as u64) }
            }
            Rep3Operand::Public(_) => {
                // Padding zeros: biased_inc = 2^XLEN (public), stored as trivial share in Phase 1.
                // The arithmetic share for these was set to a trivial share of 2^XLEN.
                unreachable!("IRingScalars should not contain Public operands")
            }
        })
        .collect();

    // A2B: arithmetic u64 → binary u64 (1 comm round)
    let bins_u64: Vec<Rep3RingShare<u64>> = ring_conv::a2b_many(&ariths_u64, &mut io)?;

    // Zero-extend to U66 carry ring for B2A + wrap extraction.
    let arith_ext: Vec<Rep3RingShare<U66>> = ariths_u64
        .iter()
        .map(|s| Rep3RingShare {
            a: RingElement(U66::try_from(s.a.0 as u128).expect("u64 share fits in U66")),
            b: RingElement(U66::try_from(s.b.0 as u128).expect("u64 share fits in U66")),
        })
        .collect();
    let bin_ext: Vec<Rep3RingShare<U66>> = bins_u64
        .iter()
        .map(|s| Rep3RingShare {
            a: RingElement(U66::try_from(s.a.0 as u128).expect("u64 share fits in U66")),
            b: RingElement(U66::try_from(s.b.0 as u128).expect("u64 share fits in U66")),
        })
        .collect();

    // Ring B2A via edaBits Π₂ in U66 — 2 rounds
    let ring_edabits = preproc.take_ring_edabits::<U66>(n)?;
    let val_arith: Vec<Rep3RingShare<U66>> =
        rep3_ring::conversion::b2a_preproc_many(&bin_ext, &ring_edabits, &mut io)?;
    let diff: Vec<Rep3RingShare<U66>> =
        arith_ext.iter().zip(val_arith.iter()).map(|(a, v)| *a - *v).collect();

    // Extract m bits via DaBit mask+open (1 round)
    let wrap_masks = preproc.take_wrap_masks_iring(n)?;
    let (m0_bin, m1_bin) = rep3_ring::wrap_mask::extract_wrap_m2_from_diff_many(&diff, &wrap_masks, &mut io)?;

    // Precompute q0/q1 for all columns: q0[c] = 2^64 * Γ1[c], q1[c] = 2 * q0[c].
    let mut q0_cols: Vec<G1Projective> = Vec::with_capacity(num_columns);
    for b in g1_proj.iter() {
        let mut p = *b;
        for _ in 0..64 {
            p.double_in_place();
        }
        q0_cols.push(p);
    }
    let q1_cols: Vec<G1Projective> = q0_cols.iter().map(|p| *p + *p).collect();

    // Precompute bias correction base per column: bias_base[c] = 2^XLEN * Γ1[c].
    // Only one party subtracts (rep3 additive sharing of group elements).
    let bias_bases: Vec<G1Projective> = g1_proj
        .iter()
        .map(|b| {
            let mut p = *b;
            for _ in 0..XLEN {
                p.double_in_place();
            }
            p
        })
        .collect();

    // MSM + wrap correction + bias correction, per row segment.
    let mut row_commitments = vec![G1Projective::zero(); num_rows_target];
    let last_row = (n - 1) / num_columns;

    for row in 0..=last_row {
        let row_start = row * num_columns;
        let seg_end = n.min(row_start + num_columns);
        let seg_len = seg_end - row_start;
        if seg_len == 0 {
            continue;
        }

        // Build MSM scalars: u64 a-limb of arithmetic share.
        let scalars_u64: Vec<u64> = ariths_u64[row_start..row_start + seg_len]
            .iter()
            .map(|s| s.a.0)
            .collect();
        let msm: G1Projective = ArkVariableBaseMSM::msm_u64(&bases_aff[..seg_len], &scalars_u64, false);

        // daPoint wrap correction — all positions are shared (no filtering needed).
        let batch = preproc.take_dapoints_iring(2 * seg_len)?;
        let mut bits_all: Vec<Rep3RingShare<Bit>> = Vec::with_capacity(2 * seg_len);
        let mut q_all: Vec<G1Projective> = Vec::with_capacity(2 * seg_len);
        for seg_i in 0..seg_len {
            bits_all.push(m0_bin[row_start + seg_i]);
            q_all.push(q0_cols[seg_i]);
        }
        for seg_i in 0..seg_len {
            bits_all.push(m1_bin[row_start + seg_i]);
            q_all.push(q1_cols[seg_i]);
        }
        let corr_add = rep3::pointshare::dot_product_dapoints(&bits_all, &q_all, &batch, &mut io)?;

        // Bias correction: subtract 2^XLEN * Σ bases[col] for this row.
        // Each scalar encodes `true_val + 2^XLEN`. The MSM committed `Σ (true_val + 2^XLEN) * base[col]`.
        // We want `Σ true_val * base[col]`, so subtract `2^XLEN * Σ base[col]`.
        // Only one party does this (the other parties' shares don't include the public bias).
        let bias_correction: G1Projective = if party_id == PartyID::ID0 {
            bias_bases[..seg_len].iter().copied().sum()
        } else {
            G1Projective::zero()
        };

        if row < row_commitments.len() {
            row_commitments[row] += msm - corr_add - bias_correction;
        }
    }

    Ok(row_commitments)
}

/// MSM with projective bases (normalizes to affine internally).
pub fn msm_g1(bases: &[G1Projective], scalars: &[Fr]) -> G1Projective {
    let bases_aff = G1Projective::normalize_batch(bases);
    ArkVariableBaseMSM::msm(&bases_aff, scalars).expect("msm should succeed")
}

/// MSM with pre-computed affine bases (avoids redundant normalization).
pub fn msm_g1_affine(bases_aff: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    ArkVariableBaseMSM::msm(&bases_aff[..scalars.len()], scalars).expect("msm should succeed")
}

fn msm_g2(bases: &[G2Projective], scalars: &[Fr]) -> G2Projective {
    let bases_aff = G2Projective::normalize_batch(bases);
    ArkVariableBaseMSM::msm(&bases_aff, scalars).expect("msm should succeed")
}

fn fixed_base_vector_msm_g2(
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    scalars: &[Fr],
) -> Vec<G2Projective> {
    if scalars.is_empty() {
        return vec![];
    }

    let g_fin = setup.g2_vec[0].0;
    scalars.par_iter().map(|&scalar| jolt_optimizations::glv_four_scalar_mul_online(scalar, &[g_fin])[0]).collect()
}

pub fn msm_g2_affine(bases_aff: &[G2Affine], scalars: &[Fr]) -> G2Projective {
    ArkVariableBaseMSM::msm(&bases_aff[..scalars.len()], scalars).expect("msm should succeed")
}

type Bn254EllCoeff = (jolt_core::ark_bn254::Fq2, jolt_core::ark_bn254::Fq2, jolt_core::ark_bn254::Fq2);

fn bn254_ell(f: &mut Fq12, coeffs: &Bn254EllCoeff, p: &G1Affine) {
    let (mut c0, mut c1, mut c2) = *coeffs;
    // BN254 has D-twist.
    c0.mul_assign_by_fp(&p.y);
    c1.mul_assign_by_fp(&p.x);
    f.mul_by_034(&c0, &c1, &c2);
}

fn bn254_miller_loop_from_cached_g2_chunk(ps_aff: &[G1Affine], qs: &[G2Affine]) -> Fq12 {
    debug_assert_eq!(ps_aff.len(), qs.len());
    Bn254::multi_pairing(ps_aff, qs).0
}

fn multi_pairing(ps: &[G1Projective], qs: &[G2Projective]) -> Fq12 {
    let ps_aff = G1Projective::normalize_batch(ps);
    let qs_aff = G2Projective::normalize_batch(qs);
    Bn254::multi_pairing(ps_aff, qs_aff).0
}

fn multi_pairing_both_affine(ps_aff: &[G1Affine], qs_aff: &[G2Affine]) -> Fq12 {
    let n = ps_aff.len().min(qs_aff.len());
    Bn254::multi_pairing(&ps_aff[..n], &qs_aff[..n]).0
}

fn multi_pairing_setup_g2_cached_affine(
    ps_aff: &[G1Affine],
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    g2_affine_all: &[G2Affine],
) -> Fq12 {
    Bn254::multi_pairing(ps_aff, &g2_affine_all[..ps_aff.len()]).0
}

/// Pairing with pre-computed G1 affine bases.
fn multi_pairing_g1_affine(ps_aff: &[G1Affine], qs: &[G2Projective]) -> Fq12 {
    let qs_aff = G2Projective::normalize_batch(qs);
    Bn254::multi_pairing(&ps_aff[..qs_aff.len()], qs_aff).0
}

/// Pairing with pre-computed G2 affine bases.
pub fn multi_pairing_g2_affine(ps: &[G1Projective], qs_aff: &[G2Affine]) -> Fq12 {
    let ps_aff = G1Projective::normalize_batch(ps);
    let n = ps_aff.len();
    Bn254::multi_pairing(ps_aff, &qs_aff[..n]).0
}

#[cfg(any(test, feature = "test-utils"))]
pub mod test_support {
    use super::*;

    static DORY_GUARD: std::sync::OnceLock<DoryGlobals> = std::sync::OnceLock::new();

    pub fn init_dory_globals(k: usize, t: usize) {
        let _ = DORY_GUARD.get_or_init(|| DoryGlobals::initialize(k, t));
        assert_eq!(DoryGlobals::get_T(), t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;
    use itertools::Itertools;
    use jolt_core::field::JoltField;
    use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
    use jolt_core::poly::multilinear_polynomial::PolynomialEvaluation;
    use jolt_core::poly::one_hot_polynomial::OneHotPolynomial as VanillaOneHotPolynomial;
    use jolt_core::transcripts::Blake2bTranscript;
    use mpc_core::protocols::rep3;
    use mpc_core::protocols::rep3::test_utils::run_rep3_local_test_with_coordinator;
    use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
    use mpc_core::protocols::rep3_ring;
    use mpc_core::protocols::rep3_ring::conversion as ring_conv;
    use mpc_core::protocols::rep3_ring::ring::bit::Bit;
    use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
    use mpc_core::protocols::rep3_ring::Rep3RingShare;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;
    use std::sync::Arc;

    fn share_poly_rep3(coeffs: &[Fr], rng: &mut (impl rand::Rng + rand::CryptoRng)) -> [Rep3DensePolynomial<Fr>; 3] {
        let mut party_coeffs: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<Fr>>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(coeffs.len()));

        for &c in coeffs {
            let shares = rep3::share_field_element(c, rng);
            party_coeffs[0].push(shares[0]);
            party_coeffs[1].push(shares[1]);
            party_coeffs[2].push(shares[2]);
        }

        std::array::from_fn(|pid| Rep3DensePolynomial::new(party_coeffs[pid].clone()))
    }

    fn unwrap_shared_hint(hint: MaybeShared<DoryOpeningProofHint>) -> DoryOpeningProofHint {
        match hint {
            MaybeShared::Shared(hint) => hint,
            _ => panic!("expected shared hint"),
        }
    }

    #[test]
    fn dory_opening_public_owners_correct() {
        assert!(owns_public_vmv_e1(PartyID::ID0));
        assert!(!owns_public_vmv_e1(PartyID::ID1));
        assert!(!owns_public_vmv_e1(PartyID::ID2));

        assert!(owns_first_reduce_d1_left(PartyID::ID0));
        assert!(!owns_first_reduce_d1_left(PartyID::ID1));
        assert!(!owns_first_reduce_d1_left(PartyID::ID2));

        assert!(owns_first_reduce_d1_right(PartyID::ID1));
        assert!(!owns_first_reduce_d1_right(PartyID::ID0));
        assert!(!owns_first_reduce_d1_right(PartyID::ID2));

        assert!(owns_first_reduce_e_betas(PartyID::ID2));
        assert!(!owns_first_reduce_e_betas(PartyID::ID0));
        assert!(!owns_first_reduce_e_betas(PartyID::ID1));

        assert!(owns_second_reduce_e1_plus(PartyID::ID0));
        assert!(!owns_second_reduce_e1_plus(PartyID::ID1));
        assert!(!owns_second_reduce_e1_plus(PartyID::ID2));

        assert!(owns_second_reduce_e1_minus(PartyID::ID1));
        assert!(!owns_second_reduce_e1_minus(PartyID::ID0));
        assert!(!owns_second_reduce_e1_minus(PartyID::ID2));
    }

    #[test]
    fn dory_opening_dense_correct() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let sigma = DoryGlobals::get_num_columns().log_2();
        let num_vars = sigma;
        let len = 1usize << num_vars;

        let coeffs = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let public_poly = MultilinearPolynomial::from(coeffs.clone());
        let opening_point: Vec<<Fr as JoltField>::Challenge> =
            (0..num_vars).map(|_| <Fr as JoltField>::Challenge::random(&mut rng)).collect::<Vec<_>>();
        let claim = public_poly.evaluate(&opening_point);

        let setup = Arc::new(<DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars)));
        let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&setup);
        let (commitment, row_commitments) = <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);

        let mut direct_transcript = Blake2bTranscript::new(b"dory_open_dense_direct");
        let (direct_proof, _direct_y_blinding) = <DoryCommitmentScheme as CommitmentScheme>::prove(
            &setup,
            &public_poly,
            &opening_point,
            Some(row_commitments),
            &mut direct_transcript,
        );
        let mut direct_verify_transcript = Blake2bTranscript::new(b"dory_open_dense_direct");
        <DoryCommitmentScheme as CommitmentScheme>::verify(
            &direct_proof,
            &verifier_setup,
            &mut direct_verify_transcript,
            &opening_point,
            &claim,
            &commitment,
        )
        .expect("direct dory opening should verify");

        let shared_polys = share_poly_rep3(&coeffs, &mut rng);
        let worker_inputs: [Rep3MultilinearPolynomial<Fr>; 3] =
            std::array::from_fn(|pid| Rep3MultilinearPolynomial::shared(shared_polys[pid].clone()));

        let worker_setup = Arc::clone(&setup);
        let worker_point = opening_point.clone();
        let coord_setup = Arc::clone(&setup);
        let coord_point = opening_point.clone();
        let coord_claim = claim;
        let coord_commitment = commitment.clone();

        let (_worker_out, proof) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| worker_inputs[party_idx].clone(),
            || (),
            move |poly, mut io_ctx| {
                <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::prove_rep3(
                    &poly,
                    &worker_setup,
                    &worker_point,
                    None,
                    io_ctx.network(),
                )?;
                Ok(())
            },
            move |(), net| {
                let mut transcript = Blake2bTranscript::new(b"dory_open_dense");
                <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                    Fr,
                    Blake2bTranscript,
                >>::coordinate_prove(
                    &coord_setup,
                    &mut transcript,
                    net,
                    &coord_point,
                    &coord_claim,
                    &coord_commitment,
                    None,
                )
                .map(|(proof, _y_blinding)| proof)
            },
        );

        let mut verify_transcript = Blake2bTranscript::new(b"dory_open_dense");
        <DoryCommitmentScheme as CommitmentScheme>::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &opening_point,
            &claim,
            &commitment,
        )
        .expect("dense dory opening should verify");
    }

    #[test]
    fn dory_opening_rlc_hint_correct() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let sigma = DoryGlobals::get_num_columns().log_2();
        let num_vars = sigma;
        let len = 1usize << num_vars;

        let coeffs_a = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let coeffs_b = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let rlc_coeffs = vec![Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let joint_coeffs = coeffs_a
            .iter()
            .zip(coeffs_b.iter())
            .map(|(a, b)| rlc_coeffs[0] * *a + rlc_coeffs[1] * *b)
            .collect::<Vec<_>>();
        let public_joint_poly = MultilinearPolynomial::from(joint_coeffs);

        let opening_point: Vec<<Fr as JoltField>::Challenge> =
            (0..num_vars).map(|_| <Fr as JoltField>::Challenge::random(&mut rng)).collect::<Vec<_>>();
        let claim = public_joint_poly.evaluate(&opening_point);

        let setup = Arc::new(<DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars)));
        let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&setup);
        let (commitment, _) = <DoryCommitmentScheme as CommitmentScheme>::commit(&public_joint_poly, &setup);

        let shared_polys_a = share_poly_rep3(&coeffs_a, &mut rng);
        let shared_polys_b = share_poly_rep3(&coeffs_b, &mut rng);
        let worker_inputs: [(Rep3DensePolynomial<Fr>, Rep3DensePolynomial<Fr>); 3] =
            std::array::from_fn(|pid| (shared_polys_a[pid].clone(), shared_polys_b[pid].clone()));

        let worker_setup = Arc::clone(&setup);
        let worker_point = opening_point.clone();
        let worker_rlc_coeffs = rlc_coeffs.clone();
        let coord_setup = Arc::clone(&setup);
        let coord_point = opening_point.clone();
        let coord_claim = claim;
        let coord_commitment = commitment.clone();

        let (_worker_out, proof) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| worker_inputs[party_idx].clone(),
            || (),
            move |(dense_a, dense_b), mut io_ctx| {
                let poly_a = Rep3MultilinearPolynomial::shared(dense_a);
                let poly_b = Rep3MultilinearPolynomial::shared(dense_b);
                let (_, hint_a) = commit_local_rep3::<Blake2bTranscript>(&poly_a, &worker_setup, false)?;
                let (_, hint_b) = commit_local_rep3::<Blake2bTranscript>(&poly_b, &worker_setup, false)?;
                let combined_hint =
                    <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::combine_hints_rep3(
                        vec![hint_a, hint_b],
                        &worker_rlc_coeffs,
                        io_ctx.party_id(),
                    );

                let joint_rlc = crate::poly::rlc_polynomial::Rep3RLCPolynomial::linear_combination(
                    vec![Arc::new(poly_a), Arc::new(poly_b)],
                    &worker_rlc_coeffs,
                    io_ctx.party_id(),
                );
                let joint_poly = Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RLC(joint_rlc));

                <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::prove_rep3(
                    &joint_poly,
                    &worker_setup,
                    &worker_point,
                    Some(combined_hint),
                    io_ctx.network(),
                )?;
                Ok(())
            },
            move |(), net| {
                let mut transcript = Blake2bTranscript::new(b"dory_open_rlc");
                <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                    Fr,
                    Blake2bTranscript,
                >>::coordinate_prove(
                    &coord_setup,
                    &mut transcript,
                    net,
                    &coord_point,
                    &coord_claim,
                    &coord_commitment,
                    None,
                )
                .map(|(proof, _y_blinding)| proof)
            },
        );

        let mut verify_transcript = Blake2bTranscript::new(b"dory_open_rlc");
        <DoryCommitmentScheme as CommitmentScheme>::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &opening_point,
            &claim,
            &commitment,
        )
        .expect("RLC dory opening should verify");
    }

    #[test]
    fn dory_commit_hint_correct() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        // Use the same DoryGlobals sizing as the zkVM witness tests (T=512) to
        // avoid global re-initialization conflicts within the test binary.
        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let sigma = DoryGlobals::get_num_columns().log_2();
        let num_vars = sigma;
        let num_rows = DoryGlobals::get_max_num_rows();

        let len = 1usize << num_vars;
        let coeffs = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        // Dory's URS for `max_log_n` generates `sqrt(2^max_log_n)` generators in each of G1/G2.
        // Vanilla `DoryCommitmentScheme::commit` requires `2^sigma` columns, so we need
        // `2^sigma <= sqrt(2^max_log_n)` => `max_log_n >= 2*sigma`.
        let setup =
            std::sync::Arc::new(<DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars)));

        // Vanilla Dory commit on public polynomial.
        let public_poly = MultilinearPolynomial::from(coeffs.clone());
        let (vanilla_commitment, mut vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);
        vanilla_hint.resize(num_rows, JoltGroupWrapper(G1Projective::zero()));

        // Rep3 commit shares (each party commits to its local share.a coefficients).
        let shared_polys = share_poly_rep3(&coeffs, &mut rng);
        let (comm_0, hint_0) = commit_local_rep3::<Blake2bTranscript>(
            &Rep3MultilinearPolynomial::shared(shared_polys[0].clone()),
            &setup,
            false,
        )
        .unwrap();
        let (comm_1, hint_1) = commit_local_rep3::<Blake2bTranscript>(
            &Rep3MultilinearPolynomial::shared(shared_polys[1].clone()),
            &setup,
            false,
        )
        .unwrap();
        let (comm_2, hint_2) = commit_local_rep3::<Blake2bTranscript>(
            &Rep3MultilinearPolynomial::shared(shared_polys[2].clone()),
            &setup,
            false,
        )
        .unwrap();

        let reconstructed_commitment =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_commitment_shares(&[&comm_0, &comm_1, &comm_2]);

        let reconstructed_hint =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_hint_shares(&[&hint_0, &hint_1, &hint_2]);

        assert_eq!(reconstructed_commitment, vanilla_commitment);
        assert_eq!(reconstructed_hint, vanilla_hint);
    }

    #[test]
    fn dory_one_hot_commit_hint_correct() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let sigma = DoryGlobals::get_num_columns().log_2();
        let num_rows = DoryGlobals::get_max_num_rows();
        let t = DoryGlobals::get_T();
        let k = 256usize;
        let num_vars = t.log_2() + k.log_2();

        let setup =
            std::sync::Arc::new(<DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars)));

        let nonzero_indices_plain: Vec<Option<u8>> =
            (0..t).map(|i| if i % 5 == 0 { None } else { Some((i % k) as u8) }).collect();
        let vanilla_poly = VanillaOneHotPolynomial::<Fr>::from_indices(nonzero_indices_plain.clone(), k);
        let (vanilla_commitment, mut vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&MultilinearPolynomial::OneHot(vanilla_poly), &setup);
        vanilla_hint.resize(num_rows, JoltGroupWrapper(G1Projective::zero()));

        let r_mask = 0x5au8;
        let masked_indices_c = std::sync::Arc::new(
            nonzero_indices_plain.iter().map(|opt| opt.map(|idx| idx ^ r_mask)).collect::<Vec<_>>(),
        );

        let mut e_field_party: [Vec<Rep3PrimeFieldShare<Fr>>; 3] = std::array::from_fn(|_| Vec::with_capacity(k));
        for i in 0..k {
            let bit = if i as u8 == r_mask { Fr::one() } else { Fr::zero() };
            let shares = rep3::share_field_element(bit, &mut rng);
            for pid in 0..3 {
                e_field_party[pid].push(shares[pid]);
            }
        }

        let rep3_polys: [Rep3MultilinearPolynomial<Fr>; 3] = std::array::from_fn(|pid| {
            let one_hot = crate::poly::one_hot_polynomial::Rep3OneHotPolynomial::from_parts(
                k,
                masked_indices_c.clone(),
                std::sync::Arc::new(e_field_party[pid].clone()),
            );
            Rep3MultilinearPolynomial::shared_one_hot(one_hot)
        });

        let (comm_0, hint_0) = commit_local_rep3::<Blake2bTranscript>(&rep3_polys[0], &setup, false).unwrap();
        let (comm_1, hint_1) = commit_local_rep3::<Blake2bTranscript>(&rep3_polys[1], &setup, false).unwrap();
        let (comm_2, hint_2) = commit_local_rep3::<Blake2bTranscript>(&rep3_polys[2], &setup, false).unwrap();

        let reconstructed_commitment =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_commitment_shares(&[&comm_0, &comm_1, &comm_2]);

        let reconstructed_hint =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_hint_shares(&[&hint_0, &hint_1, &hint_2]);

        assert_eq!(reconstructed_commitment, vanilla_commitment);
        assert_eq!(reconstructed_hint, vanilla_hint);
    }

    #[test]
    fn dory_public_gating_correct() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let sigma = DoryGlobals::get_num_columns().log_2();
        let num_vars = sigma;

        let len = 1usize << num_vars;
        let coeffs = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        let public_poly = MultilinearPolynomial::from(coeffs);
        let poly = Rep3MultilinearPolynomial::public(public_poly.clone());

        let (c0, h0) = commit_local_rep3::<Blake2bTranscript>(&poly, &setup, false).unwrap();
        assert!(matches!(c0, MaybeShared::Public(None)));
        assert!(matches!(h0, MaybeShared::Public(None)));

        let (c1, h1) = commit_local_rep3::<Blake2bTranscript>(&poly, &setup, true).unwrap();
        let (vanilla_commitment, mut vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);
        vanilla_hint.resize(DoryGlobals::get_max_num_rows(), JoltGroupWrapper(G1Projective::zero()));

        assert!(matches!(c1, MaybeShared::Public(Some(_))));
        assert!(matches!(h1, MaybeShared::Public(Some(_))));

        match (c1, h1) {
            (MaybeShared::Public(Some(c)), MaybeShared::Public(Some(h))) => {
                assert_eq!(c, vanilla_commitment);
                assert_eq!(h, vanilla_hint);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn dory_batch_eq_single() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let sigma = DoryGlobals::get_num_columns().log_2();
        let num_vars = sigma;

        let len = 1usize << num_vars;
        let coeffs_0 = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let coeffs_1 = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        let public_poly = Rep3MultilinearPolynomial::public(MultilinearPolynomial::from(coeffs_0));

        let shared_coeffs = coeffs_1.clone();
        let shared_polys = share_poly_rep3(&shared_coeffs, &mut rng);
        let shared_poly = Rep3MultilinearPolynomial::shared(shared_polys[0].clone());

        let polys = vec![public_poly, shared_poly];

        // For Dense/Public polys without U64Scalars, verify batch == single
        // via individual commit_local_rep3 calls.
        let batch: Vec<_> =
            polys.iter().map(|p| commit_local_rep3::<Blake2bTranscript>(p, &setup, true).unwrap()).collect();

        let single_0 = commit_local_rep3::<Blake2bTranscript>(&polys[0], &setup, true).unwrap();
        let single_1 = commit_local_rep3::<Blake2bTranscript>(&polys[1], &setup, true).unwrap();

        fn assert_commit_and_hint_eq(
            a: &(MaybeShared<DoryCommitment>, MaybeShared<DoryOpeningProofHint>),
            b: &(MaybeShared<DoryCommitment>, MaybeShared<DoryOpeningProofHint>),
        ) {
            match (&a.0, &b.0) {
                (MaybeShared::Public(Some(ca)), MaybeShared::Public(Some(cb))) => {
                    assert_eq!(ca, cb)
                }
                (MaybeShared::Public(None), MaybeShared::Public(None)) => {}
                (MaybeShared::Shared(ca), MaybeShared::Shared(cb)) => assert_eq!(ca, cb),
                _ => panic!("commitment mismatch"),
            }

            match (&a.1, &b.1) {
                (MaybeShared::Public(Some(ha)), MaybeShared::Public(Some(hb))) => {
                    assert_eq!(ha, hb)
                }
                (MaybeShared::Public(None), MaybeShared::Public(None)) => {}
                (MaybeShared::Shared(ha), MaybeShared::Shared(hb)) => assert_eq!(ha, hb),
                _ => panic!("hint mismatch"),
            }
        }

        assert_commit_and_hint_eq(&batch[0], &single_0);
        assert_commit_and_hint_eq(&batch[1], &single_1);
    }

    #[cfg(feature = "ring-msm")]
    #[test]
    fn dory_u64_scalars_commit_correct() {
        use jolt_common::constants::{ArithmeticWideInt, XlenInt};

        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let num_columns = DoryGlobals::get_num_columns();
        let sigma = num_columns.log_2();
        let num_vars = sigma;
        let num_rows = DoryGlobals::get_max_num_rows();

        let len = 1usize << num_vars;
        let values: Vec<u64> = (0..len).map(|_| rng.r#gen::<XlenInt>() as u64).collect();
        let coeffs_fr: Vec<Fr> = values.iter().copied().map(Fr::from).collect();

        let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        let public_poly = MultilinearPolynomial::from(coeffs_fr.clone());
        let (vanilla_commitment, mut vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);
        vanilla_hint.resize(num_rows, JoltGroupWrapper(G1Projective::zero()));

        // Share each value in both arithmetic (ArithmeticWideInt) and XOR (XlenInt) ring forms.
        let all_arith_shares: Vec<_> = values
            .iter()
            .map(|&v| rep3_ring::share_ring_element(RingElement(v as ArithmeticWideInt), &mut rng))
            .collect();
        let all_bin_shares: Vec<_> =
            values.iter().map(|&v| rep3_ring::share_ring_element_binary(RingElement(v as XlenInt), &mut rng)).collect();

        let polys_by_party: [Rep3MultilinearPolynomial<Fr>; 3] = std::array::from_fn(|pid| {
            let shares: Vec<Rep3RingShare<ArithmeticWideInt>> = all_arith_shares.iter().map(|s| s[pid]).collect();
            let shares_bin: Vec<Rep3RingShare<XlenInt>> = all_bin_shares.iter().map(|s| s[pid]).collect();
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(Rep3CompactPolynomial::from_shares(
                shares, shares_bin,
            )))
        });

        let (results, _) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| polys_by_party[party_idx].clone(),
            || (),
            move |poly, mut io_ctx| {
                use mpc_core::protocols::rep3_ring::edabits;

                let pool_dir = std::env::temp_dir().join(format!("co-jolt2-dory-test-{}", io_ctx.party_idx()));
                let mut preproc =
                    edabits::preprocess_pool::<Fr, _>(&pool_dir, [0, 0, 0, 0, 0], 0, len, len, 0, 0, &mut io_ctx)?;

                // daPoints for Dory wrap correction (depend on SRS)
                let qs = precompute_dapoint_qs(&setup, len, num_columns);
                let lazy_dp = rep3_ring::daPoint::random_dapoints(&qs, &mut io_ctx)?;
                preproc.set_dapoints(lazy_dp);

                let polys = vec![&poly];
                let out = <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::batch_commit_rep3(
                    &polys,
                    &setup,
                    &mut io_ctx,
                    &mut preproc,
                )?;
                Ok(out[0].clone())
            },
            |(), _net| Ok(()),
        );

        let (c0, h0) = results[0].clone();
        let (c1, h1) = results[1].clone();
        let (c2, h2) = results[2].clone();

        let reconstructed_commitment =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_commitment_shares(&[&c0, &c1, &c2]);

        let reconstructed_hint =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_hint_shares(&[&h0, &h1, &h2]);

        assert_eq!(reconstructed_commitment, vanilla_commitment);
        assert_eq!(reconstructed_hint, vanilla_hint);
    }

    /// Same as `dory_u64_scalars_commit_correct` but with mixed Public + Shared operands.
    /// Exercises the daPoint Q-value alignment when public positions are skipped.
    #[cfg(feature = "ring-msm")]
    #[test]
    fn dory_u64_scalars_mixed_public_shared_commit_correct() {
        use crate::zkvm::instruction::types::rep3_operand::Rep3Operand;
        use jolt_common::constants::{ArithmeticWideInt, XlenInt};
        use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;

        let mut rng = ChaCha12Rng::seed_from_u64(0);

        crate::poly::commitment::dory::test_support::init_dory_globals(256, 512);
        let num_columns = DoryGlobals::get_num_columns();
        let sigma = num_columns.log_2();
        let num_vars = sigma;
        let num_rows = DoryGlobals::get_max_num_rows();

        let len = 1usize << num_vars;
        // Every other value is public (even indices), the rest are shared.
        let values: Vec<u64> = (0..len).map(|_| rng.r#gen::<XlenInt>() as u64).collect();
        let is_public: Vec<bool> = (0..len).map(|i| i % 2 == 0).collect();
        let coeffs_fr: Vec<Fr> = values.iter().copied().map(Fr::from).collect();

        let setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        let public_poly = MultilinearPolynomial::from(coeffs_fr.clone());
        let (vanilla_commitment, mut vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);
        vanilla_hint.resize(num_rows, JoltGroupWrapper(G1Projective::zero()));

        // Share shared values in both arithmetic and XOR ring forms.
        let all_arith_shares: Vec<Option<[Rep3RingShare<ArithmeticWideInt>; 3]>> = values
            .iter()
            .zip(is_public.iter())
            .map(|(&v, &pub_)| {
                if pub_ {
                    None
                } else {
                    Some(rep3_ring::share_ring_element(RingElement(v as ArithmeticWideInt), &mut rng))
                }
            })
            .collect();
        let all_bin_shares: Vec<Option<[Rep3RingShare<XlenInt>; 3]>> = values
            .iter()
            .zip(is_public.iter())
            .map(|(&v, &pub_)| {
                if pub_ {
                    None
                } else {
                    Some(rep3_ring::share_ring_element_binary(RingElement(v as XlenInt), &mut rng))
                }
            })
            .collect();

        let polys_by_party: [Rep3MultilinearPolynomial<Fr>; 3] = std::array::from_fn(|pid| {
            let operands: Vec<Rep3Operand> = (0..len)
                .map(|i| {
                    if is_public[i] {
                        Rep3Operand::Public(values[i] as i128)
                    } else {
                        let arith = all_arith_shares[i].as_ref().unwrap()[pid];
                        let bin = all_bin_shares[i].as_ref().unwrap()[pid];
                        Rep3Operand::Shared { binary: bin, arithmetic: Some(arith), public: None }
                    }
                })
                .collect();
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RingScalars(Rep3CompactPolynomial::from_operands(
                operands,
            )))
        });

        let (results, _) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| polys_by_party[party_idx].clone(),
            || (),
            move |poly, mut io_ctx| {
                use mpc_core::protocols::rep3_ring::edabits;

                let pool_dir = std::env::temp_dir().join(format!("co-jolt2-dory-test2-{}", io_ctx.party_idx()));
                let mut preproc =
                    edabits::preprocess_pool::<Fr, _>(&pool_dir, [0, 0, 0, 0, 0], 0, len, len, 0, 0, &mut io_ctx)?;

                // daPoints (depend on SRS)
                let qs = precompute_dapoint_qs(&setup, len, num_columns);
                let lazy_dp = rep3_ring::daPoint::random_dapoints(&qs, &mut io_ctx)?;
                preproc.set_dapoints(lazy_dp);

                let polys = vec![&poly];
                let out = <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::batch_commit_rep3(
                    &polys,
                    &setup,
                    &mut io_ctx,
                    &mut preproc,
                )?;
                Ok(out[0].clone())
            },
            |(), _net| Ok(()),
        );

        let (c0, h0) = results[0].clone();
        let (c1, h1) = results[1].clone();
        let (c2, h2) = results[2].clone();

        let reconstructed_commitment =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_commitment_shares(&[&c0, &c1, &c2]);

        let reconstructed_hint =
            <DoryCommitmentScheme as co_jolt_coordinator::poly::commitment::Rep3CommitmentScheme<
                Fr,
                Blake2bTranscript,
            >>::combine_hint_shares(&[&h0, &h1, &h2]);

        assert_eq!(reconstructed_commitment, vanilla_commitment);
        assert_eq!(reconstructed_hint, vanilla_hint);
    }

    /// Verify MPC wrap correction for arithmetic u32 shares embedded into Fr for MSM.
    /// Uses both arithmetic and binary u32 shares: B2A + subtract → open wrap count m → public correction.
    #[cfg(feature = "ring-msm")]
    #[test]
    fn ring_shared_msm_correctness() {
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        let n = 64;

        // Random G1 bases
        let bases_proj: Vec<G1Projective> = (0..n).map(|_| G1Projective::rand(&mut rng)).collect();
        let bases_aff = G1Projective::normalize_batch(&bases_proj);

        // Small random coefficients
        let values: Vec<u32> = (0..n).map(|_| rng.r#gen()).collect();

        // True MSM
        let scalars_fr: Vec<Fr> = values.iter().map(|&v| Fr::from(v as u32)).collect();
        let true_msm: G1Projective = ArkVariableBaseMSM::msm(&bases_aff, &scalars_fr).expect("true MSM should succeed");

        // --- MPC correction path ---
        // Works with normal arithmetic u32 shares (no range-bound trick).
        // Uses both arithmetic and binary u32 shares of the same values.
        // Computes wrap count m via B2A + subtract + open, then corrects publicly.
        let all_arith_shares: Vec<_> =
            values.iter().map(|&v| rep3_ring::share_ring_element(RingElement(v), &mut rng)).collect();
        let all_bin_shares: Vec<_> =
            values.iter().map(|&v| rep3_ring::share_ring_element_binary(RingElement(v), &mut rng)).collect();

        // Naive sum with arithmetic shares is also wrong
        let naive_arith_msms: [G1Projective; 3] = std::array::from_fn(|pid| {
            let scalars: Vec<Fr> = all_arith_shares.iter().map(|s| Fr::from(s[pid].a.0)).collect();
            ArkVariableBaseMSM::msm(&bases_aff, &scalars).unwrap()
        });
        let naive_arith_sum = naive_arith_msms[0] + naive_arith_msms[1] + naive_arith_msms[2];
        assert_ne!(naive_arith_sum, true_msm, "naive arithmetic sum should differ from true MSM");

        let (mpc_results, _) = run_rep3_local_test_with_coordinator(
            0,
            |party_idx| {
                let my_arith: Vec<Rep3RingShare<u32>> = all_arith_shares.iter().map(|s| s[party_idx]).collect();
                let my_bin: Vec<Rep3RingShare<u32>> = all_bin_shares.iter().map(|s| s[party_idx]).collect();
                (my_arith, my_bin, bases_aff.clone(), bases_proj.clone())
            },
            || (),
            |(arith_u32, bin_u32, bases_aff, bases_proj), mut io_ctx| {
                // Step 1: Zero-extend arithmetic u32 → u64 (LOCAL)
                let arith_ext: Vec<Rep3RingShare<u64>> = arith_u32
                    .iter()
                    .map(|s| Rep3RingShare { a: RingElement(s.a.0 as u64), b: RingElement(s.b.0 as u64) })
                    .collect();

                // Step 2: Zero-extend binary u32 → u64 (LOCAL)
                let bin_ext: Vec<Rep3RingShare<u64>> = bin_u32
                    .iter()
                    .map(|s| Rep3RingShare { a: RingElement(s.a.0 as u64), b: RingElement(s.b.0 as u64) })
                    .collect();

                // Step 3: B2A on binary_ext → arithmetic u64 shares of val (COMM)
                let val_arith: Vec<Rep3RingShare<u64>> = ring_conv::b2a_many(&bin_ext, io_ctx.main())?;

                // Step 4: Subtract → [m * 2^32] in Z_{2^64} (LOCAL)
                let diff: Vec<Rep3RingShare<u64>> =
                    arith_ext.iter().zip(val_arith.iter()).map(|(a, v)| *a - *v).collect();

                // Step 5: Convert diff to binary and extract m bits WITHOUT opening (COMM + LOCAL)
                // diff = m * 2^32, with m ∈ {0,1,2}. We'll represent m as two shared bits.
                let diff_bin: Vec<Rep3RingShare<u64>> = ring_conv::a2b_many(&diff, io_ctx.main())?;
                let m_bin_u64: Vec<Rep3RingShare<u64>> = diff_bin.iter().map(|d| d >> 32).collect();
                let m0_bin: Vec<Rep3RingShare<Bit>> = m_bin_u64.iter().map(|m| m.get_bit(0)).collect();
                let m1_bin: Vec<Rep3RingShare<Bit>> = m_bin_u64.iter().map(|m| m.get_bit(1)).collect();

                // Step 6: Per-party MSM with u32 scalars (cheap 32-bit MSM)
                let scalars: Vec<Fr> = arith_u32.iter().map(|s| Fr::from(s.a.0)).collect();
                let party_msm: G1Projective = ArkVariableBaseMSM::msm(&bases_aff, &scalars).unwrap();

                // Step 7: Secure correction using bit × public-point (offline+online).
                // Compute public points Q0=2^32*base and Q1=2^33*base; then add shared
                // corrections m0*Q0 + m1*Q1 (additive shares) and subtract from party_msm.
                let two_pow_32 = Fr::from(1u64 << 32);
                let q0: Vec<G1Projective> = bases_proj.iter().map(|b| *b * two_pow_32).collect();
                let q1: Vec<G1Projective> = q0.iter().map(|p| *p + *p).collect();
                let mut q_all: Vec<G1Projective> = Vec::with_capacity(2 * q0.len());
                q_all.extend(q0.iter().copied());
                q_all.extend(q1.iter().copied());

                // Offline: generate daPoints
                let mut lazy_dapoints = rep3_ring::daPoint::random_dapoints(&q_all, &mut io_ctx)?;
                let batch = lazy_dapoints.take_batch(q_all.len())?;

                let mut bits_all: Vec<Rep3RingShare<Bit>> = Vec::with_capacity(2 * m0_bin.len());
                bits_all.extend(m0_bin.iter().copied());
                bits_all.extend(m1_bin.iter().copied());

                // Online: dot product
                let total_corr_add = rep3::pointshare::dot_product_dapoints(&bits_all, &q_all, &batch, io_ctx.main())?;

                Ok(party_msm - total_corr_add)
            },
            |(), _net| Ok(()),
        );

        // Sum of per-party corrected MSMs = true MSM
        let mpc_sum = mpc_results[0] + mpc_results[1] + mpc_results[2];
        assert_eq!(mpc_sum, true_msm, "MPC-corrected MSM must equal true MSM");
    }
}
