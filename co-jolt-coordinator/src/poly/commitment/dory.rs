use ark_ec::pairing::Pairing as ArkPairing;
use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{One, Zero};
use dory::{DoryProofBuilder, ProofBuilder};
use jolt_core::ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use jolt_core::jolt_optimizations;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::commitment::dory::{
    DoryCommitment, DoryCommitmentScheme, DoryProofData, JoltFieldWrapper, JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254,
    JoltGTWrapper, JoltGroupWrapper, JoltToDoryTranscriptRef,
};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::protocols::rep3::PartyID;
use mpc_core::MaybeShared;
use rayon::prelude::*;
use tracing::info_span;

use crate::poly::commitment::Rep3CommitmentScheme;

type DoryTranscriptRef<'a, T> = JoltToDoryTranscriptRef<'a, Fr, T>;
type DoryProofBuilderRef<'a, T> =
    DoryProofBuilder<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254, JoltFieldWrapper<Fr>, DoryTranscriptRef<'a, T>>;
type DoryVmvShareMsg = ((Fq12, Fq12), Option<G1Affine>);
type DoryFirstReducePublicMsg = (Option<Fq12>, Option<Fq12>, Option<G1Affine>, Option<G2Affine>);
type DoryFirstReduceShareMsg = ((Fq12, Fq12), DoryFirstReducePublicMsg);
type DorySecondReducePublicMsg = (Option<G1Affine>, Option<G1Affine>);
type DorySecondReduceShareMsg = (((Fq12, Fq12), (G2Affine, G2Affine)), DorySecondReducePublicMsg);

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
            return Err(eyre::eyre!("Dory opening proof: distributed subnets unsupported (single-worker mode only)"));
        }

        let sigma = DoryGlobals::get_num_columns().log_2();
        let g1_all = setup_g1_projective(setup);
        let (mut row_commitments, nu, l_vec, r_vec) = {
            let init_msgs: Vec<(usize, Vec<G1Affine>)> = network.receive_responses()?;
            let num_vars = init_msgs[0].0;
            let nu = dory::vmv::compute_nu(num_vars, sigma);

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

            let point_wrapped: Vec<JoltFieldWrapper<Fr>> =
                opening_point.iter().rev().map(|&x| JoltFieldWrapper(x.into())).collect();
            let (l_vec_w, r_vec_w) = dory::compute_left_right_vec(&point_wrapped, sigma, nu);
            let l_vec: Vec<Fr> = l_vec_w.iter().map(|x| x.0).collect();
            let r_vec: Vec<Fr> = r_vec_w.iter().map(|x| x.0).collect();
            (row_commitments, nu, l_vec, r_vec)
        };

        let dory_transcript: DoryTranscriptRef<'_, ProofTranscript> =
            JoltToDoryTranscriptRef::<Fr, ProofTranscript>::new(transcript);
        let mut builder: DoryProofBuilderRef<'_, ProofTranscript> = DoryProofBuilder::new(dory_transcript);

        let vmv_shares: Vec<DoryVmvShareMsg> = network.receive_responses()?;
        let ((c, d2), e1) = combine_vmv_shares(&vmv_shares)?;

        let vmv_message = dory::messages::VMVMessage::<JoltG1Wrapper, JoltGTBn254> {
            c: JoltGTWrapper::<Bn254>(c),
            d2: JoltGTWrapper::<Bn254>(d2),
            e1: JoltGroupWrapper(e1.into_group()),
        };
        builder = builder.append_vmv_message(vmv_message);

        let mut v1_pub = row_commitments;
        let mut s1 = r_vec;
        let mut s2 = l_vec;

        let _dory_loop = info_span!("dory_reduction_loop", nu).entered();
        let mut curr_nu = nu;
        while curr_nu > 0 {
            let n2 = 1usize << (curr_nu - 1);

            let first_msgs: Vec<DoryFirstReduceShareMsg> = network.receive_responses()?;
            let ((d2_left, d2_right), (d1_left, d1_right, e1_beta, e2_beta)) =
                combine_first_reduce_shares(&first_msgs)?;

            let first_msg = dory::messages::FirstReduceMessage::<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254> {
                d1_left: JoltGTWrapper::<Bn254>(d1_left),
                d1_right: JoltGTWrapper::<Bn254>(d1_right),
                d2_left: JoltGTWrapper::<Bn254>(d2_left),
                d2_right: JoltGTWrapper::<Bn254>(d2_right),
                e1_beta: JoltGroupWrapper(e1_beta.into_group()),
                e2_beta: JoltGroupWrapper(e2_beta.into_group()),
            };

            let (beta_chal, b2) = builder.append_first_reduce_message(first_msg);
            builder = b2;
            let beta = beta_chal.beta.0;
            let beta_inv = beta_chal.beta_inverse.0;
            network.broadcast_request((beta, beta_inv))?;

            jolt_optimizations::vector_add_scalar_mul_g1_online(&mut v1_pub, &g1_all[..(1 << curr_nu)], beta);

            let second_msgs: Vec<DorySecondReduceShareMsg> = network.receive_responses()?;
            let ((c_plus, c_minus), (e2_plus, e2_minus), (e1_plus, e1_minus)) =
                combine_second_reduce_shares(&second_msgs)?;

            let second_msg = dory::messages::SecondReduceMessage::<JoltG1Wrapper, JoltG2Wrapper, JoltGTBn254> {
                c_plus: JoltGTWrapper::<Bn254>(c_plus),
                c_minus: JoltGTWrapper::<Bn254>(c_minus),
                e1_plus: JoltGroupWrapper(e1_plus.into_group()),
                e1_minus: JoltGroupWrapper(e1_minus.into_group()),
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
            let (s2_l, s2_r) = s2.split_at(n2);
            let (s1_next, s2_next): (Vec<Fr>, Vec<Fr>) =
                (0..n2).into_par_iter().map(|i| (s1_l[i] * alpha + s1_r[i], s2_l[i] * alpha_inv + s2_r[i])).unzip();
            s1 = s1_next;
            s2 = s2_next;
            curr_nu -= 1;
        }
        drop(_dory_loop);

        let _dory_final = info_span!("dory_final_scalar_product").entered();
        let (gamma_chal, b4) = builder.challenge_fold_scalars();
        let (_d_chal, b5) =
            <DoryProofBuilderRef<'_, ProofTranscript> as ProofBuilder>::challenge_scalar_product_scalars(b4);
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

        drop(_dory_final);

        let _ = (claimed_opening, commitment);
        Ok(DoryProofData { sigma, dory_proof_data: builder.build() })
    }

    fn combine_commitment_shares(commitments: &[&MaybeShared<Self::Commitment>]) -> Self::Commitment {
        let public = commitments.iter().find(|c| matches!(c, MaybeShared::Public(Some(_))));
        match public {
            Some(MaybeShared::Public(Some(c))) => c.clone(),
            None => {
                if commitments.iter().all(|c| matches!(c, MaybeShared::Public(None))) {
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

    fn combine_hint_shares(hints: &[&MaybeShared<Self::OpeningProofHint>]) -> Self::OpeningProofHint {
        let public = hints.iter().find(|h| matches!(h, MaybeShared::Public(Some(_))));
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

fn take_owned_public_term<T: Clone>(values: &[Option<T>], owner: PartyID, label: &str) -> eyre::Result<T> {
    let owner_idx = usize::from(owner);
    eyre::ensure!(
        values.get(owner_idx).and_then(|v| v.as_ref()).is_some(),
        "{label}: owner {owner_idx} did not provide the public term"
    );
    debug_assert!(values.iter().enumerate().all(|(idx, value)| idx == owner_idx || value.is_none()));
    Ok(values[owner_idx].clone().unwrap())
}

fn combine_vmv_shares(msgs: &[DoryVmvShareMsg]) -> eyre::Result<((Fq12, Fq12), G1Affine)> {
    let mut c = Fq12::one();
    let mut d2 = Fq12::one();
    let mut e1_vals = Vec::with_capacity(msgs.len());
    for ((cs, d2s), e1) in msgs {
        c *= *cs;
        d2 *= *d2s;
        e1_vals.push(*e1);
    }
    let e1 = take_owned_public_term(&e1_vals, PartyID::ID0, "vmv.e1")?;
    Ok(((c, d2), e1))
}

fn combine_first_reduce_shares(
    msgs: &[DoryFirstReduceShareMsg],
) -> eyre::Result<((Fq12, Fq12), (Fq12, Fq12, G1Affine, G2Affine))> {
    let mut d2_left = Fq12::one();
    let mut d2_right = Fq12::one();
    let mut d1_left_vals = Vec::with_capacity(msgs.len());
    let mut d1_right_vals = Vec::with_capacity(msgs.len());
    let mut e1_beta_vals = Vec::with_capacity(msgs.len());
    let mut e2_beta_vals = Vec::with_capacity(msgs.len());

    for ((d2l, d2r), (d1_left, d1_right, e1_beta, e2_beta)) in msgs {
        d2_left *= *d2l;
        d2_right *= *d2r;
        d1_left_vals.push(*d1_left);
        d1_right_vals.push(*d1_right);
        e1_beta_vals.push(*e1_beta);
        e2_beta_vals.push(*e2_beta);
    }

    let d1_left = take_owned_public_term(&d1_left_vals, PartyID::ID0, "first.d1_left")?;
    let d1_right = take_owned_public_term(&d1_right_vals, PartyID::ID1, "first.d1_right")?;
    let e1_beta = take_owned_public_term(&e1_beta_vals, PartyID::ID2, "first.e1_beta")?;
    let e2_beta = take_owned_public_term(&e2_beta_vals, PartyID::ID2, "first.e2_beta")?;

    Ok(((d2_left, d2_right), (d1_left, d1_right, e1_beta, e2_beta)))
}

fn combine_second_reduce_shares(
    msgs: &[DorySecondReduceShareMsg],
) -> eyre::Result<((Fq12, Fq12), (G2Projective, G2Projective), (G1Affine, G1Affine))> {
    let mut c_plus = Fq12::one();
    let mut c_minus = Fq12::one();
    let mut e2_plus = G2Projective::zero();
    let mut e2_minus = G2Projective::zero();
    let mut e1_plus_vals = Vec::with_capacity(msgs.len());
    let mut e1_minus_vals = Vec::with_capacity(msgs.len());

    for (((cp, cm), (e2p, e2m)), (e1_plus, e1_minus)) in msgs {
        c_plus *= *cp;
        c_minus *= *cm;
        e2_plus += e2p.into_group();
        e2_minus += e2m.into_group();
        e1_plus_vals.push(*e1_plus);
        e1_minus_vals.push(*e1_minus);
    }

    let e1_plus = take_owned_public_term(&e1_plus_vals, PartyID::ID0, "second.e1_plus")?;
    let e1_minus = take_owned_public_term(&e1_minus_vals, PartyID::ID1, "second.e1_minus")?;

    Ok(((c_plus, c_minus), (e2_plus, e2_minus), (e1_plus, e1_minus)))
}

// =============================================================================
// Dory helper functions (shared between coordinator and workers)
// =============================================================================

/// Zero-copy view of `setup.core.g1_vec` as `&[G1Projective]`.
/// Safety: `JoltGroupWrapper<G1Projective>` is `#[repr(transparent)]`.
pub fn setup_g1_projective(setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup) -> &[G1Projective] {
    unsafe { std::slice::from_raw_parts(setup.core.g1_vec.as_ptr() as *const G1Projective, setup.core.g1_vec.len()) }
}

/// Zero-copy view of `setup.core.g2_vec` as `&[G2Projective]`.
/// Safety: `JoltGroupWrapper<G2Projective>` is `#[repr(transparent)]`.
pub fn setup_g2_projective(setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup) -> &[G2Projective] {
    unsafe { std::slice::from_raw_parts(setup.core.g2_vec.as_ptr() as *const G2Projective, setup.core.g2_vec.len()) }
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
