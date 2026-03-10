use ark_ec::pairing::Pairing as ArkPairing;
use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{One, Zero};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{
    DoryCommitment, DoryCommitmentScheme, DoryProofData, JoltFieldWrapper, JoltG1Wrapper,
    JoltG2Wrapper, JoltGTBn254, JoltGTWrapper, JoltGroupWrapper, JoltToDoryTranscriptRef,
};
use mpc_core::MaybeShared;
use dory::{DoryProofBuilder, ProofBuilder};
use jolt_core::ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use jolt_core::jolt_optimizations;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use rayon::prelude::*;

use crate::poly::commitment::Rep3CommitmentScheme;

type DoryTranscriptRef<'a, T> = JoltToDoryTranscriptRef<'a, Fr, T>;
type DoryProofBuilderRef<'a, T> = DoryProofBuilder<
    JoltG1Wrapper,
    JoltG2Wrapper,
    JoltGTBn254,
    JoltFieldWrapper<Fr>,
    DoryTranscriptRef<'a, T>,
>;

impl<ProofTranscript> Rep3CommitmentScheme<Fr, ProofTranscript> for DoryCommitmentScheme
where
    ProofTranscript: Transcript,
{
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
        let init_msgs: Vec<(usize, Vec<G1Affine>)> = network.receive_responses()?;
        let num_vars = init_msgs[0].0;
        let nu = dory::vmv::compute_nu(num_vars, sigma);

        let g1_all = setup_g1_projective(setup);
        let g2_all = setup_g2_projective(setup);
        let g1_affine_all = G1Projective::normalize_batch(&g1_all[..(1 << nu)]);
        let g2_affine_all = G2Projective::normalize_batch(&g2_all[..(1 << nu)]);

        let rows_len = init_msgs[0].1.len();
        let mut row_commitments = vec![G1Projective::zero(); rows_len];
        for (_nv, shares) in &init_msgs {
            debug_assert_eq!(*_nv, num_vars);
            for (acc, s) in row_commitments.iter_mut().zip(shares.iter()) {
                *acc += s.into_group();
            }
        }
        let row_commitments_affine = G1Projective::normalize_batch(&row_commitments);
        network.broadcast_request(row_commitments_affine)?;

        let point_wrapped: Vec<JoltFieldWrapper<Fr>> = opening_point
            .iter()
            .rev()
            .map(|&x| JoltFieldWrapper(x.into()))
            .collect();
        let (l_vec_w, r_vec_w) = dory::compute_left_right_vec(&point_wrapped, sigma, nu);
        let l_vec: Vec<Fr> = l_vec_w.iter().map(|x| x.0).collect();
        let r_vec: Vec<Fr> = r_vec_w.iter().map(|x| x.0).collect();

        let dory_transcript: DoryTranscriptRef<'_, ProofTranscript> =
            JoltToDoryTranscriptRef::<Fr, ProofTranscript>::new(transcript);
        let mut builder: DoryProofBuilderRef<'_, ProofTranscript> =
            DoryProofBuilder::new(dory_transcript);

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

        let mut v1_pub = row_commitments;
        let mut s1 = r_vec;
        let mut s2 = l_vec;

        let mut curr_nu = nu;
        while curr_nu > 0 {
            let n2 = 1usize << (curr_nu - 1);

            let d2_lr_shares: Vec<(Fq12, Fq12)> = network.receive_responses()?;
            let mut d2_left = Fq12::one();
            let mut d2_right = Fq12::one();
            for (l, r) in d2_lr_shares {
                d2_left *= l;
                d2_right *= r;
            }

            let g2_prime_aff = &g2_affine_all[..n2];
            let ((d1_left, d1_right), (e1_beta, e2_beta)) = {
                let (v1_l, v1_r) = v1_pub.split_at(n2);
                let g1_aff_nu = &g1_affine_all[..(1 << curr_nu)];
                let g2_aff_nu = &g2_affine_all[..(1 << curr_nu)];
                rayon::join(
                    || {
                        rayon::join(
                            || multi_pairing_g2_affine(v1_l, g2_prime_aff),
                            || multi_pairing_g2_affine(v1_r, g2_prime_aff),
                        )
                    },
                    || {
                        rayon::join(
                            || msm_g1_affine(g1_aff_nu, &s2),
                            || msm_g2_affine(g2_aff_nu, &s1),
                        )
                    },
                )
            };

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

            jolt_optimizations::vector_add_scalar_mul_g1_online(
                &mut v1_pub,
                &g1_all[..(1 << curr_nu)],
                beta,
            );

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
            let (e1_plus, e1_minus) = rayon::join(|| msm_g1(v1_l, s2_r), || msm_g1(v1_r, s2_l));

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

            let (v1_l_mut, v1_r_ref) = v1_pub.split_at_mut(n2);
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(v1_l_mut, alpha, v1_r_ref);
            v1_pub.truncate(n2);

            let (s1_l, s1_r) = s1.split_at(n2);
            let (s1_next, s2_next): (Vec<Fr>, Vec<Fr>) = (0..n2)
                .into_par_iter()
                .map(|i| (s1_l[i] * alpha + s1_r[i], s2_l[i] * alpha_inv + s2_r[i]))
                .unzip();
            s1 = s1_next;
            s2 = s2_next;
            curr_nu -= 1;
        }

        let (gamma_chal, b4) = builder.challenge_fold_scalars();
        let (_d_chal, b5) = <DoryProofBuilderRef<'_, ProofTranscript> as ProofBuilder>::challenge_scalar_product_scalars(b4);
        builder = b5;

        let v2_shares: Vec<G2Affine> = network.receive_responses()?;
        let mut v2 = G2Projective::zero();
        for s in v2_shares {
            v2 += s.into_group();
        }

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
                if commitments
                    .iter()
                    .all(|c| matches!(c, MaybeShared::Public(None)))
                {
                    return DoryCommitment::default();
                }
                let mut acc = JoltGTWrapper::<Bn254>(Fq12::one());
                for c in commitments {
                    match c {
                        MaybeShared::Shared(c) => acc.0 *= (c.0).0,
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
// Dory helper functions (shared between coordinator and workers)
// =============================================================================

/// Zero-copy view of `setup.core.g1_vec` as `&[G1Projective]`.
/// Safety: `JoltGroupWrapper<G1Projective>` is `#[repr(transparent)]`.
pub fn setup_g1_projective(
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
) -> &[G1Projective] {
    unsafe {
        std::slice::from_raw_parts(
            setup.core.g1_vec.as_ptr() as *const G1Projective,
            setup.core.g1_vec.len(),
        )
    }
}

/// Zero-copy view of `setup.core.g2_vec` as `&[G2Projective]`.
/// Safety: `JoltGroupWrapper<G2Projective>` is `#[repr(transparent)]`.
pub fn setup_g2_projective(
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
) -> &[G2Projective] {
    unsafe {
        std::slice::from_raw_parts(
            setup.core.g2_vec.as_ptr() as *const G2Projective,
            setup.core.g2_vec.len(),
        )
    }
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

/// MSM with pre-computed affine bases (avoids redundant normalization).
pub fn msm_g2_affine(bases_aff: &[G2Affine], scalars: &[Fr]) -> G2Projective {
    ArkVariableBaseMSM::msm(&bases_aff[..scalars.len()], scalars).expect("msm should succeed")
}

/// Pairing with pre-computed G2 affine bases.
pub fn multi_pairing_g2_affine(ps: &[G1Projective], qs_aff: &[G2Affine]) -> Fq12 {
    let ps_aff = G1Projective::normalize_batch(ps);
    let n = ps_aff.len();
    <Bn254 as ArkPairing>::multi_pairing(ps_aff, &qs_aff[..n]).0
}
