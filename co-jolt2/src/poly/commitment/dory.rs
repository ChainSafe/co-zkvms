use crate::poly::{Rep3DensePolynomial, Rep3MultilinearPolynomial, Rep3SharedPoly};
use crate::utils::types::MaybeShared;
use ark_ec::pairing::Pairing as ArkPairing;
use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::One;
use ark_std::Zero;
use dory::{DoryProofBuilder, ProofBuilder};
use jolt_core::ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use mpc_core::protocols::rep3::network::{Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;
use std::borrow::Borrow;

// Re-export vanilla Jolt Dory types (wrappers, globals, commitment scheme, proof types, ...)
pub use jolt_core::poly::commitment::dory::*;

use super::Rep3CommitmentScheme;

type DoryTranscriptRef<'a, T> = JoltToDoryTranscriptRef<'a, Fr, T>;

type DoryProofBuilderRef<'a, T> = DoryProofBuilder<
    JoltG1Wrapper,
    JoltG2Wrapper,
    JoltGTBn254,
    JoltFieldWrapper<Fr>,
    DoryTranscriptRef<'a, T>,
>;

// =============================================================================
// Rep3CommitmentScheme implementation
// =============================================================================

impl<ProofTranscript: Transcript> Rep3CommitmentScheme<Fr, ProofTranscript>
    for DoryCommitmentScheme
{
    fn commit_rep3(
        poly: &Rep3MultilinearPolynomial<Fr>,
        setup: &Self::ProverSetup,
        commit_to_public: bool,
    ) -> (
        MaybeShared<Self::Commitment>,
        MaybeShared<Self::OpeningProofHint>,
    ) {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => {
                if commit_to_public {
                    let (c, hint) = <Self as CommitmentScheme>::commit(poly, setup);
                    (
                        MaybeShared::Public(Some(c)),
                        MaybeShared::Public(Some(hint)),
                    )
                } else {
                    (MaybeShared::Public(None), MaybeShared::Public(None))
                }
            }
            Rep3MultilinearPolynomial::Shared(shared_poly) => {
                let (num_vars, row_commitments_share) = match shared_poly {
                    Rep3SharedPoly::Dense(poly) => {
                        let nu = dory::vmv::compute_nu(
                            poly.get_num_vars(),
                            DoryGlobals::get_num_columns().log_2(),
                        );
                        (
                            poly.get_num_vars(),
                            compute_row_commitment_shares_a(poly, setup, nu),
                        )
                    }
                    Rep3SharedPoly::OneHot(poly) => {
                        let num_columns = DoryGlobals::get_num_columns();
                        let bases: Vec<G1Affine> = setup.core.g1_vec[..num_columns]
                            .iter()
                            .map(|g| g.0.into_affine())
                            .collect();
                        let rows = poly
                            .commit_rows::<G1Projective>(&bases)
                            .expect("OneHot commit_rows preconditions met");
                        (poly.get_num_vars(), rows)
                    }
                };

                let sigma = DoryGlobals::get_num_columns().log_2();
                let nu = dory::vmv::compute_nu(num_vars, sigma);
                let num_rows_target = 1usize << nu;

                let mut row_commitments = row_commitments_share;
                row_commitments.resize(num_rows_target, G1Projective::zero());

                let row_commitments_aff: Vec<G1Affine> =
                    row_commitments.iter().map(|p| p.into_affine()).collect();

                let g2: Vec<G2Affine> = setup.core.g2_vec[..row_commitments_aff.len()]
                    .iter()
                    .map(|g| g.0.into_affine())
                    .collect();

                let commitment_share = Bn254::multi_pairing(row_commitments_aff, g2);

                let hint_share: Vec<JoltG1Wrapper> =
                    row_commitments.into_iter().map(JoltGroupWrapper).collect();

                (
                    MaybeShared::Shared(DoryCommitment(commitment_share.into())),
                    MaybeShared::Shared(hint_share),
                )
            }
        }
    }

    fn batch_commit_rep3<U>(
        polys: &[U],
        setup: &Self::ProverSetup,
        commit_to_public: bool,
    ) -> Vec<(
        MaybeShared<Self::Commitment>,
        MaybeShared<Self::OpeningProofHint>,
    )>
    where
        U: Borrow<Rep3MultilinearPolynomial<Fr>> + Sync,
    {
        polys
            .par_iter()
            .map(|p| {
                <Self as Rep3CommitmentScheme<Fr, ProofTranscript>>::commit_rep3(
                    p.borrow(),
                    setup,
                    commit_to_public,
                )
            })
            .collect()
    }

    fn prove_rep3<Network>(
        poly: &Rep3DensePolynomial<Fr>,
        setup: &Self::ProverSetup,
        opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker,
    {
        if network.is_distributed() {
            return Err(eyre::eyre!(
                "Dory opening proof: distributed subnets unsupported (single-worker mode only)"
            ));
        }

        let sigma = DoryGlobals::get_num_columns().log_2();
        let num_vars = poly.get_num_vars();
        let nu = dory::vmv::compute_nu(num_vars, sigma);

        // Dory uses opposite endian-ness to Jolt
        let point_rev: Vec<Fr> = opening_point.iter().rev().map(|c| (*c).into()).collect();
        let point_wrapped: Vec<JoltFieldWrapper<Fr>> =
            point_rev.iter().copied().map(JoltFieldWrapper).collect();

        let (l_vec_w, r_vec_w) = dory::compute_left_right_vec(&point_wrapped, sigma, nu);
        let l_vec: Vec<Fr> = l_vec_w.iter().map(|x| x.0).collect();
        let r_vec: Vec<Fr> = r_vec_w.iter().map(|x| x.0).collect();

        // 1) send (num_vars, row_commit_shares) as tuple
        let row_commit_shares_affine = compute_row_commitment_shares_a(poly, setup, nu)
            .into_iter()
            .map(|p| p.into_affine())
            .collect::<Vec<G1Affine>>();

        network.send_response((num_vars, row_commit_shares_affine))?;

        // 2) receive reconstructed row commitments from coordinator
        let row_commitments_affine: Vec<G1Affine> = network.receive_request()?;
        let row_commitments: Vec<G1Projective> = row_commitments_affine
            .iter()
            .map(|a| a.into_group())
            .collect();

        // 3) compute v_vec share and v2 share
        let num_columns = 1usize << sigma;
        let mut v_vec_share = vec![<Fr as ark_ff::Zero>::zero(); num_columns];
        let (global_offset, local_coeffs) = rep3_local_coeffs_a(poly);

        for (k, coeff) in local_coeffs.iter().enumerate() {
            let idx = global_offset + k;
            let row = idx / num_columns;
            let col = idx % num_columns;
            if row < l_vec.len() {
                v_vec_share[col] += *coeff * l_vec[row];
            }
        }

        // c_share = e(msm(row_commitments, v_vec_share), g_fin)
        let t_vec_v = msm_g1(&row_commitments, &v_vec_share);
        let c_share = Bn254::pairing(t_vec_v, setup.core.g_fin.0.into_affine()).0;

        // d2_share = e(msm(g1_vec[..2^nu], v_vec_share), g_fin)
        // Dory expects MSM over Γ1[nu] (length 2^nu) with v_vec of length 2^sigma. With fixed sigma,
        // we have nu >= sigma; we simply pad v_vec with zeros to length 2^nu.
        let gamma1: Vec<G1Projective> =
            setup.core.g1_vec[..(1 << nu)].iter().map(|g| g.0).collect();
        let mut v_vec_padded = v_vec_share.clone();
        v_vec_padded.resize(1 << nu, <Fr as ark_ff::Zero>::zero());
        let gamma1_v = msm_g1(&gamma1, &v_vec_padded);
        let d2_share = Bn254::pairing(gamma1_v, setup.core.g_fin.0.into_affine()).0;

        network.send_response((c_share, d2_share))?;

        // v2_share[j] = g_fin * v_vec_share[j]
        let g_fin = setup.core.g_fin.0;
        let mut v2_share: Vec<G2Projective> = v_vec_share.iter().map(|s| g_fin * s).collect();

        // s1 = R, s2 = L (matches dory vmv_state_to_dory_prover_state)
        let mut s1 = r_vec;
        let mut s2 = l_vec;

        let mut v1_pub: Vec<G1Projective> = row_commitments;
        let mut curr_nu = nu;

        while curr_nu > 0 {
            let n2 = 1usize << (curr_nu - 1);

            // First message: only witness-dependent shares (d2_left/right in GT)
            let g1_prime: Vec<G1Projective> = setup.core.g1_vec[..n2].iter().map(|g| g.0).collect();
            let (d2_left_share_round, d2_right_share_round) = {
                let (v2_l, v2_r) = v2_share.split_at(n2);
                (
                    multi_pairing(&g1_prime, v2_l),
                    multi_pairing(&g1_prime, v2_r),
                )
            };

            network.send_response((d2_left_share_round, d2_right_share_round))?;

            // Receive beta challenge from coordinator
            let (beta, beta_inv): (Fr, Fr) = network.receive_request()?;

            // Update v1 public: v1 += beta * Gamma1[curr_nu]
            let gamma1_full: Vec<G1Projective> = setup.core.g1_vec[..(1 << curr_nu)]
                .iter()
                .map(|g| g.0)
                .collect();
            v1_pub
                .par_iter_mut()
                .zip(gamma1_full.par_iter())
                .for_each(|(v1i, gi)| *v1i += *gi * beta);

            // Update v2 share: v2 += beta_inv * Gamma2 (public term: only party 0 adds)
            if network.party_id() == PartyID::ID0 {
                let gamma2_full: Vec<G2Projective> = setup.core.g2_vec[..(1 << curr_nu)]
                    .iter()
                    .map(|g| g.0)
                    .collect();
                v2_share
                    .par_iter_mut()
                    .zip(gamma2_full.par_iter())
                    .for_each(|(v2i, gi)| *v2i += *gi * beta_inv);
            }

            // Second message: witness-dependent (c_plus/c_minus in GT, e2_plus/e2_minus in G2)
            let (v1_l, v1_r) = v1_pub.split_at(n2);
            let (v2_l, v2_r) = v2_share.split_at(n2);
            let (s1_l, s1_r) = s1.split_at(n2);
            let (_s2_l, _s2_r) = s2.split_at(n2);

            let c_plus_share_round = multi_pairing(v1_l, v2_r);
            let c_minus_share_round = multi_pairing(v1_r, v2_l);

            let e2_plus_share = msm_g2(v2_r, s1_l).into_affine();
            let e2_minus_share = msm_g2(v2_l, s1_r).into_affine();

            network.send_response((
                (c_plus_share_round, c_minus_share_round),
                (e2_plus_share, e2_minus_share),
            ))?;

            // Receive alpha challenge from coordinator
            let (alpha, alpha_inv): (Fr, Fr) = network.receive_request()?;

            // Fold v1 public: v1 = alpha*v1L + v1R (truncate)
            let mut v1_next = Vec::with_capacity(n2);
            for i in 0..n2 {
                v1_next.push(v1_l[i] * alpha + v1_r[i]);
            }
            v1_pub = v1_next;

            // Fold v2 share: v2 = alpha_inv*v2L + v2R
            let mut v2_next = Vec::with_capacity(n2);
            for i in 0..n2 {
                v2_next.push(v2_l[i] * alpha_inv + v2_r[i]);
            }
            v2_share = v2_next;

            // Fold s1,s2
            let (s1_l, s1_r) = s1.split_at(n2);
            let (s2_l, s2_r) = s2.split_at(n2);
            let mut s1_next = Vec::with_capacity(n2);
            let mut s2_next = Vec::with_capacity(n2);
            for i in 0..n2 {
                s1_next.push(s1_l[i] * alpha + s1_r[i]);
                s2_next.push(s2_l[i] * alpha_inv + s2_r[i]);
            }
            s1 = s1_next;
            s2 = s2_next;

            curr_nu -= 1;
        }

        // Final: send v2 final share to coordinator for scalar product message
        debug_assert_eq!(v2_share.len(), 1);
        network.send_response(v2_share[0].into_affine())?;

        Ok(())
    }

    fn coordinate_prove<Network>(
        setup: &Self::ProverSetup,
        transcript: &mut ProofTranscript,
        network: &mut Network,
        opening_point: &[<Fr as jolt_core::field::JoltField>::Challenge],
        claimed_opening: &Fr,
        commitment: &Self::Commitment,
    ) -> eyre::Result<Self::Proof>
    where
        Network: Rep3NetworkCoordinator,
    {
        if network.is_distributed() {
            return Err(eyre::eyre!(
                "Dory opening proof: distributed subnets unsupported (single-worker mode only)"
            ));
        }

        let sigma = DoryGlobals::get_num_columns().log_2();

        // Init: receive (num_vars, row_commit_shares) from parties
        let init_msgs: Vec<(usize, Vec<G1Affine>)> = network.receive_responses()?;
        let num_vars = init_msgs[0].0;
        let nu = dory::vmv::compute_nu(num_vars, sigma);

        // Reconstruct row commitments: sum in G1
        let rows_len = init_msgs[0].1.len();
        let mut row_commitments = vec![G1Projective::zero(); rows_len];
        for (_nv, shares) in init_msgs.iter() {
            debug_assert_eq!(*_nv, num_vars);
            for (acc, s) in row_commitments.iter_mut().zip(shares.iter()) {
                *acc += s.into_group();
            }
        }
        let row_commitments_affine = row_commitments
            .iter()
            .map(|p| p.into_affine())
            .collect::<Vec<_>>();

        network.broadcast_request(row_commitments_affine)?;

        // Compute L,R and public parts
        let point_rev: Vec<Fr> = opening_point.iter().rev().map(|&x| x.into()).collect();
        let point_wrapped: Vec<JoltFieldWrapper<Fr>> =
            point_rev.iter().copied().map(JoltFieldWrapper).collect();

        let (l_vec_w, r_vec_w) = dory::compute_left_right_vec(&point_wrapped, sigma, nu);
        let l_vec: Vec<Fr> = l_vec_w.iter().map(|x| x.0).collect();
        let r_vec: Vec<Fr> = r_vec_w.iter().map(|x| x.0).collect();

        // Build a Dory proof builder backed by the shared transcript
        let dory_transcript: DoryTranscriptRef<'_, ProofTranscript> =
            JoltToDoryTranscriptRef::<Fr, ProofTranscript>::new(transcript);
        let mut builder: DoryProofBuilderRef<'_, ProofTranscript> =
            DoryProofBuilder::new(dory_transcript);

        // Receive VMV shares and reconstruct c,d2. Compute e1 public and append VMV message.
        let vmv_shares: Vec<(Fq12, Fq12)> = network.receive_responses()?;
        let mut c = Fq12::one();
        let mut d2 = Fq12::one();
        for (cs, d2s) in vmv_shares {
            c *= cs;
            d2 *= d2s;
        }

        let e1 = msm_g1(&row_commitments, &l_vec);
        let vmv_message = dory::messages::VMVMessage::<JoltG1Wrapper, JoltGTBn254> {
            c: JoltGTWrapper::<Bn254>(c),
            d2: JoltGTWrapper::<Bn254>(d2),
            e1: JoltGroupWrapper(e1),
        };
        builder = builder.append_vmv_message(vmv_message);

        // Initialize prover-side public state for coordinator computations
        let mut v1_pub = row_commitments;
        let mut s1 = r_vec;
        let mut s2 = l_vec;

        let mut curr_nu = nu;
        while curr_nu > 0 {
            let n2 = 1usize << (curr_nu - 1);

            // First message reconstruction (witness-dependent pieces only)
            let d2_lr_shares: Vec<(Fq12, Fq12)> = network.receive_responses()?;
            let mut d2_left = Fq12::one();
            let mut d2_right = Fq12::one();
            for (l, r) in d2_lr_shares {
                d2_left *= l;
                d2_right *= r;
            }

            // Public-only terms (computed by coordinator)
            let g2_prime: Vec<G2Projective> = setup.core.g2_vec[..n2].iter().map(|g| g.0).collect();
            let (d1_left, d1_right) = {
                let (v1_l, v1_r) = v1_pub.split_at(n2);
                (
                    multi_pairing(v1_l, &g2_prime),
                    multi_pairing(v1_r, &g2_prime),
                )
            };

            let g1_full: Vec<G1Projective> = setup.core.g1_vec[..(1 << curr_nu)]
                .iter()
                .map(|g| g.0)
                .collect();
            let g2_full: Vec<G2Projective> = setup.core.g2_vec[..(1 << curr_nu)]
                .iter()
                .map(|g| g.0)
                .collect();

            let e1_beta = msm_g1(&g1_full, &s2);
            let e2_beta = msm_g2(&g2_full, &s1);

            let first_msg =
                dory::messages::FirstReduceMessage::<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254> {
                    d1_left: JoltGTWrapper::<Bn254>(d1_left),
                    d1_right: JoltGTWrapper::<Bn254>(d1_right),
                    d2_left: JoltGTWrapper::<Bn254>(d2_left),
                    d2_right: JoltGTWrapper::<Bn254>(d2_right),
                    e1_beta: JoltGroupWrapper(e1_beta),
                    e2_beta: JoltGroupWrapper(e2_beta),
                };

            let (beta_chal, b2) = builder.append_first_reduce_message(first_msg);
            builder = b2;
            let beta = beta_chal.beta.0;
            let beta_inv = beta_chal.beta_inverse.0;
            network.broadcast_request((beta, beta_inv))?;

            // Update v1_pub for next computations
            let gamma1_full: Vec<G1Projective> = setup.core.g1_vec[..(1 << curr_nu)]
                .iter()
                .map(|g| g.0)
                .collect();
            v1_pub
                .par_iter_mut()
                .zip(gamma1_full.par_iter())
                .for_each(|(v1i, gi)| *v1i += *gi * beta);

            // Second message reconstruction
            let second_msgs: Vec<((Fq12, Fq12), (G2Affine, G2Affine))> =
                network.receive_responses()?;

            let mut c_plus = Fq12::one();
            let mut c_minus = Fq12::one();
            let mut e2_plus = G2Projective::zero();
            let mut e2_minus = G2Projective::zero();
            for ((cp, cm), (e2p, e2m)) in second_msgs {
                c_plus *= cp;
                c_minus *= cm;
                e2_plus += e2p.into_group();
                e2_minus += e2m.into_group();
            }

            let (s2_l, s2_r) = s2.split_at(n2);
            let (v1_l, v1_r) = v1_pub.split_at(n2);
            let e1_plus = msm_g1(v1_l, s2_r);
            let e1_minus = msm_g1(v1_r, s2_l);

            let second_msg =
                dory::messages::SecondReduceMessage::<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254> {
                    c_plus: JoltGTWrapper::<Bn254>(c_plus),
                    c_minus: JoltGTWrapper::<Bn254>(c_minus),
                    e1_plus: JoltGroupWrapper(e1_plus),
                    e1_minus: JoltGroupWrapper(e1_minus),
                    e2_plus: JoltGroupWrapper(e2_plus),
                    e2_minus: JoltGroupWrapper(e2_minus),
                };

            let (alpha_chal, b3) = builder.append_second_reduce_message(second_msg);
            builder = b3;
            let alpha = alpha_chal.alpha.0;
            let alpha_inv = alpha_chal.alpha_inverse.0;
            network.broadcast_request((alpha, alpha_inv))?;

            // Fold v1 and s1,s2 for next round (public)
            let (v1_l, v1_r) = v1_pub.split_at(n2);
            let mut v1_next = Vec::with_capacity(n2);
            for i in 0..n2 {
                v1_next.push(v1_l[i] * alpha + v1_r[i]);
            }
            v1_pub = v1_next;

            let (s1_l, s1_r) = s1.split_at(n2);
            let mut s1_next = Vec::with_capacity(n2);
            let mut s2_next = Vec::with_capacity(n2);
            for i in 0..n2 {
                s1_next.push(s1_l[i] * alpha + s1_r[i]);
                s2_next.push(s2_l[i] * alpha_inv + s2_r[i]);
            }
            s1 = s1_next;
            s2 = s2_next;

            curr_nu -= 1;
        }

        // Derive fold-scalars + scalar-product challenges (for transcript sync)
        let (gamma_chal, b4) = builder.challenge_fold_scalars();
        let (_d_chal, b5): (
            dory::ScalarProductChallenge<JoltFieldWrapper<Fr>>,
            DoryProofBuilderRef<'_, ProofTranscript>,
        ) = <DoryProofBuilderRef<'_, ProofTranscript> as ProofBuilder>::challenge_scalar_product_scalars(b4);
        builder = b5;

        // Receive v2 final share from parties and reconstruct v2
        let v2_shares: Vec<G2Affine> = network.receive_responses()?;
        let mut v2 = G2Projective::zero();
        for s in v2_shares {
            v2 += s.into_group();
        }

        // Compute final scalar product message using fold-scalars challenge
        debug_assert_eq!(v1_pub.len(), 1);
        debug_assert_eq!(s1.len(), 1);
        debug_assert_eq!(s2.len(), 1);

        let gamma = gamma_chal.gamma.0;
        let gamma_inv = gamma_chal.gamma_inverse.0;

        let gamma_s1 = gamma * s1[0];
        let e1_final = v1_pub[0] + setup.core.h1.0 * gamma_s1;

        let gamma_inv_s2 = gamma_inv * s2[0];
        let e2_final = v2 + setup.core.h2.0 * gamma_inv_s2;

        let final_msg = dory::messages::ScalarProductMessage::<JoltG1Wrapper, JoltG2Wrapper> {
            e1: JoltGroupWrapper(e1_final),
            e2: JoltGroupWrapper(e2_final),
        };
        builder = builder.append_scalar_product_message(final_msg, None, None);

        let _ = (claimed_opening, commitment);
        Ok(DoryProofData {
            sigma,
            dory_proof_data: builder.build(),
        })
    }

    fn combine_commitment_shares(
        commitments: &[&MaybeShared<Self::Commitment>],
    ) -> Self::Commitment {
        let public = commitments
            .iter()
            .find(|c| matches!(c, MaybeShared::Public(Some(_))));
        match public {
            Some(MaybeShared::Public(Some(c))) => c.clone(),
            None => {
                // All Public(None) → skipped polynomial, return default commitment
                if commitments
                    .iter()
                    .all(|c| matches!(c, MaybeShared::Public(None)))
                {
                    return DoryCommitment::default();
                }
                // Otherwise all must be Shared
                let mut acc = JoltGTWrapper::<Bn254>(Fq12::one());
                for c in commitments {
                    match c {
                        MaybeShared::Shared(c) => {
                            // group law in GT is multiplication
                            acc.0 *= (c.0).0;
                        }
                        _ => unreachable!(),
                    }
                }
                DoryCommitment(acc)
            }
            _ => unreachable!(),
        }
    }

    fn combine_hint_shares(
        hints: &[&MaybeShared<Self::OpeningProofHint>],
    ) -> Self::OpeningProofHint {
        let public = hints
            .iter()
            .find(|h| matches!(h, MaybeShared::Public(Some(_))));
        match public {
            Some(MaybeShared::Public(Some(h))) => h.clone(),
            None => {
                let num_rows = DoryGlobals::get_max_num_rows();
                let mut acc = vec![JoltGroupWrapper(G1Projective::zero()); num_rows];

                for h in hints {
                    match h {
                        MaybeShared::Shared(hint_share) => {
                            for (i, row) in hint_share.iter().enumerate() {
                                if i >= num_rows {
                                    break;
                                }
                                acc[i].0 += row.0;
                            }
                        }
                        MaybeShared::Public(None) => {}
                        _ => unreachable!(),
                    }
                }

                acc
            }
            _ => unreachable!(),
        }
    }
}

// =============================================================================
// Helpers (Rep3)
// =============================================================================

fn rep3_local_coeffs_a(poly: &Rep3DensePolynomial<Fr>) -> (usize, Vec<Fr>) {
    let coeffs_ref = poly.coeffs_ref();
    let local = coeffs_ref.iter().map(|s| s.a).collect::<Vec<Fr>>();
    let global_offset = poly.global_chunk_range.map(|(s, _)| s).unwrap_or(0);
    (global_offset, local)
}

fn compute_row_commitment_shares_a(
    poly: &Rep3DensePolynomial<Fr>,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    nu: usize,
) -> Vec<G1Projective> {
    let sigma = DoryGlobals::get_num_columns().log_2();
    let num_columns = 1usize << sigma;
    let num_rows_target = 1usize << nu;

    let bases = setup.core.g1_vec[..num_columns]
        .iter()
        .map(|g| g.0.into_affine())
        .collect::<Vec<G1Affine>>();

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
        let msm: G1Projective =
            ArkVariableBaseMSM::msm(&bases[col_start..col_start + seg_len], scalars)
                .expect("row segment MSM should succeed");
        if row < row_commitments.len() {
            row_commitments[row] += msm;
        }
    }

    row_commitments
}

fn msm_g1(bases: &[G1Projective], scalars: &[Fr]) -> G1Projective {
    let bases_aff = G1Projective::normalize_batch(bases);
    ArkVariableBaseMSM::msm(&bases_aff, scalars).expect("msm should succeed")
}

fn msm_g2(bases: &[G2Projective], scalars: &[Fr]) -> G2Projective {
    let bases_aff = G2Projective::normalize_batch(bases);
    ArkVariableBaseMSM::msm(&bases_aff, scalars).expect("msm should succeed")
}

fn multi_pairing(ps: &[G1Projective], qs: &[G2Projective]) -> Fq12 {
    let ps_aff = G1Projective::normalize_batch(ps);
    let qs_aff = G2Projective::normalize_batch(qs);
    Bn254::multi_pairing(ps_aff, qs_aff).0
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    static DORY_GUARD: std::sync::OnceLock<DoryGlobals> = std::sync::OnceLock::new();

    pub(crate) fn init_dory_globals(k: usize, t: usize) {
        let _ = DORY_GUARD.get_or_init(|| DoryGlobals::initialize(k, t));
        assert_eq!(DoryGlobals::get_T(), t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;
    use ark_std::UniformRand;
    use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
    use jolt_core::transcripts::Blake2bTranscript;
    use mpc_core::protocols::rep3::arithmetic::generate_shares_rep3;

    fn share_poly_rep3(coeffs: &[Fr], rng: &mut impl rand::Rng) -> [Rep3DensePolynomial<Fr>; 3] {
        let mut party_coeffs: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<Fr>>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(coeffs.len()));

        for &c in coeffs {
            let shares = generate_shares_rep3(c, rng);
            party_coeffs[0].push(shares[0]);
            party_coeffs[1].push(shares[1]);
            party_coeffs[2].push(shares[2]);
        }

        std::array::from_fn(|pid| Rep3DensePolynomial::new(party_coeffs[pid].clone()))
    }

    #[test]
    fn dory_commit_hint_correct() {
        let mut rng = test_rng();

        let num_vars = 6;
        // Match the OneHotPolynomial tests' default sizing.
        crate::poly::commitment::dory::test_support::init_dory_globals(256, 32);
        let sigma = DoryGlobals::get_num_columns().log_2();
        let num_rows = DoryGlobals::get_max_num_rows();

        let len = 1usize << num_vars;
        let coeffs = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        // Dory's URS for `max_log_n` generates `sqrt(2^max_log_n)` generators in each of G1/G2.
        // Vanilla `DoryCommitmentScheme::commit` requires `2^sigma` columns, so we need
        // `2^sigma <= sqrt(2^max_log_n)` => `max_log_n >= 2*sigma`.
        let setup =
            <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        // Vanilla Dory commit on public polynomial.
        let public_poly = MultilinearPolynomial::from(coeffs.clone());
        let (vanilla_commitment, mut vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);
        vanilla_hint.resize(num_rows, JoltGroupWrapper(G1Projective::zero()));

        // Rep3 commit shares (each party commits to its local share.a coefficients).
        let shared_polys = share_poly_rep3(&coeffs, &mut rng);
        let (comm_0, hint_0) =
            <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::commit_rep3(
                &Rep3MultilinearPolynomial::shared(shared_polys[0].clone()),
                &setup,
                false,
            );
        let (comm_1, hint_1) =
            <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::commit_rep3(
                &Rep3MultilinearPolynomial::shared(shared_polys[1].clone()),
                &setup,
                false,
            );
        let (comm_2, hint_2) =
            <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::commit_rep3(
                &Rep3MultilinearPolynomial::shared(shared_polys[2].clone()),
                &setup,
                false,
            );

        let reconstructed_commitment = <DoryCommitmentScheme as Rep3CommitmentScheme<
            Fr,
            Blake2bTranscript,
        >>::combine_commitment_shares(&[
            &comm_0, &comm_1, &comm_2,
        ]);

        let reconstructed_hint = <DoryCommitmentScheme as Rep3CommitmentScheme<
            Fr,
            Blake2bTranscript,
        >>::combine_hint_shares(&[&hint_0, &hint_1, &hint_2]);

        assert_eq!(reconstructed_commitment, vanilla_commitment);
        assert_eq!(reconstructed_hint, vanilla_hint);
    }

    #[test]
    fn dory_public_gating_correct() {
        let mut rng = test_rng();

        let num_vars = 6;
        crate::poly::commitment::dory::test_support::init_dory_globals(256, 32);
        let sigma = DoryGlobals::get_num_columns().log_2();

        let len = 1usize << num_vars;
        let coeffs = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        let setup =
            <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        let public_poly = MultilinearPolynomial::from(coeffs);
        let poly = Rep3MultilinearPolynomial::public(public_poly.clone());

        let (c0, h0) =
            <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::commit_rep3(
                &poly, &setup, false,
            );
        assert!(matches!(c0, MaybeShared::Public(None)));
        assert!(matches!(h0, MaybeShared::Public(None)));

        let (c1, h1) =
            <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::commit_rep3(
                &poly, &setup, true,
            );
        let (vanilla_commitment, vanilla_hint) =
            <DoryCommitmentScheme as CommitmentScheme>::commit(&public_poly, &setup);

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
        let mut rng = test_rng();

        let num_vars = 6;
        crate::poly::commitment::dory::test_support::init_dory_globals(256, 32);
        let sigma = DoryGlobals::get_num_columns().log_2();

        let len = 1usize << num_vars;
        let coeffs_0 = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let coeffs_1 = (0..len).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        let setup =
            <DoryCommitmentScheme as CommitmentScheme>::setup_prover((2 * sigma).max(num_vars));

        let public_poly = Rep3MultilinearPolynomial::public(MultilinearPolynomial::from(coeffs_0));

        let shared_coeffs = coeffs_1.clone();
        let shared_polys = share_poly_rep3(&shared_coeffs, &mut rng);
        let shared_poly = Rep3MultilinearPolynomial::shared(shared_polys[0].clone());

        let polys = vec![public_poly, shared_poly];

        let batch = <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::batch_commit_rep3(
            &polys,
            &setup,
            true,
        );

        let single_0 =
            <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::commit_rep3(
                &polys[0], &setup, true,
            );
        let single_1 =
            <DoryCommitmentScheme as Rep3CommitmentScheme<Fr, Blake2bTranscript>>::commit_rep3(
                &polys[1], &setup, true,
            );

        fn assert_commit_and_hint_eq(
            a: &(MaybeShared<DoryCommitment>, MaybeShared<Vec<JoltG1Wrapper>>),
            b: &(MaybeShared<DoryCommitment>, MaybeShared<Vec<JoltG1Wrapper>>),
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
}
