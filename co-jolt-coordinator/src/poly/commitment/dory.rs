use ark_ec::pairing::Pairing as ArkPairing;
use ark_ec::scalar_mul::variable_base::VariableBaseMSM as ArkVariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{Field, One, Zero};
#[cfg(feature = "zk")]
use ark_std::UniformRand;
use dory::messages::{FirstReduceMessage, ScalarProductMessage, SecondReduceMessage, VMVMessage};
#[cfg(feature = "zk")]
use dory::messages::ScalarProductProof;
#[cfg(feature = "zk")]
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::poly::compute_left_right_vectors;
use dory::primitives::transcript::Transcript as DoryTranscript;
#[cfg(feature = "zk")]
use dory::reduce_and_fold::{generate_sigma1_proof, generate_sigma2_proof};
use jolt_core::ark_bn254::{Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use jolt_core::jolt_optimizations;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::commitment::dory::{
    ark_to_jolt, jolt_to_ark, ArkDoryProof, ArkG1, ArkG2, ArkGT, DoryCommitment,
    DoryCommitmentScheme, DoryProofData, DoryOpeningProofHint, JoltToDoryTranscript,
};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::protocols::rep3::PartyID;
use mpc_core::MaybeShared;
use rayon::prelude::*;

use crate::poly::commitment::Rep3CommitmentScheme;

type DoryVmvShareMsg = ((Fq12, Fq12), Option<G1Affine>);
type DoryFirstReducePublicMsg = (Option<Fq12>, Option<Fq12>, Option<G1Affine>, Option<G2Affine>);
type DoryFirstReduceShareMsg = ((Fq12, Fq12), DoryFirstReducePublicMsg);
type DorySecondReducePublicMsg = (Option<G1Affine>, Option<G1Affine>);
type DorySecondReduceShareMsg = (((Fq12, Fq12), (G2Affine, G2Affine)), DorySecondReducePublicMsg);

#[inline]
fn compute_nu(num_vars: usize, sigma: usize) -> usize {
    num_vars
        .checked_sub(sigma)
        .expect("Dory opening point must have at least sigma coordinates")
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy)]
struct ZkRoundBlinds {
    d1: [Fr; 2],
    d2: [Fr; 2],
    c: [Fr; 2],
    e1: [Fr; 2],
    e2: [Fr; 2],
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy)]
struct ZkAccumBlinds {
    c: Fr,
    d1: Fr,
    d2: Fr,
    e1: Fr,
    e2: Fr,
}

#[cfg(feature = "zk")]
fn sample_fr() -> Fr {
    let mut rng = ark_std::rand::thread_rng();
    Fr::rand(&mut rng)
}

#[cfg(feature = "zk")]
fn mask_gt(base: Fq12, blind: Fr, setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup) -> ArkGT {
    let ark_blind = jolt_to_ark(&blind);
    ArkGT((ArkGT(base) + setup.ht.scale(&ark_blind)).0)
}

#[cfg(feature = "zk")]
fn mask_g1(
    base: G1Projective,
    blind: Fr,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
) -> ArkG1 {
    ArkG1(base + setup.h1.0 * blind)
}

#[cfg(feature = "zk")]
fn mask_g2(
    base: G2Projective,
    blind: Fr,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
) -> ArkG2 {
    ArkG2(base + setup.h2.0 * blind)
}

#[cfg(feature = "zk")]
fn sample_round_blinds() -> ZkRoundBlinds {
    ZkRoundBlinds {
        d1: [sample_fr(), sample_fr()],
        d2: [sample_fr(), sample_fr()],
        c: [sample_fr(), sample_fr()],
        e1: [sample_fr(), sample_fr()],
        e2: [sample_fr(), sample_fr()],
    }
}

#[cfg(feature = "zk")]
fn scalar_product_proof<ProofTranscript: Transcript>(
    transcript: &mut JoltToDoryTranscript<'_, ProofTranscript>,
    setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup,
    v1: G1Projective,
    v2: G2Projective,
    blinds: ZkAccumBlinds,
) -> (ScalarProductProof<ArkG1, ArkG2, jolt_core::poly::commitment::dory::ArkFr, ArkGT>, Fr) {
    let sd1 = sample_fr();
    let sd2 = sample_fr();
    let d1 = setup.g1_vec[0].0 * sd1;
    let d2 = setup.g2_vec[0].0 * sd2;

    let rp1 = sample_fr();
    let rp2 = sample_fr();
    let rq = sample_fr();
    let rr = sample_fr();

    let p1 = ArkGT(Bn254::pairing(d1, setup.g2_vec[0].0).0) + setup.ht.scale(&jolt_to_ark(&rp1));
    let p2 = ArkGT(Bn254::pairing(setup.g1_vec[0].0, d2).0) + setup.ht.scale(&jolt_to_ark(&rp2));
    let q = ArkGT(Bn254::pairing(d1, v2).0)
        + ArkGT(Bn254::pairing(v1, d2).0)
        + setup.ht.scale(&jolt_to_ark(&rq));
    let r = ArkGT(Bn254::pairing(d1, d2).0) + setup.ht.scale(&jolt_to_ark(&rr));

    for (label, value) in [
        (b"sigma_p1" as &[u8], &p1),
        (b"sigma_p2", &p2),
        (b"sigma_q", &q),
        (b"sigma_r", &r),
    ] {
        transcript.append_serde(label, value);
    }
    let sigma_c_ark = transcript.challenge_scalar(b"sigma_c");
    let sigma_c = ark_to_jolt(&sigma_c_ark);

    (
        ScalarProductProof {
            p1,
            p2,
            q,
            r,
            e1: ArkG1(d1 + v1 * sigma_c),
            e2: ArkG2(d2 + v2 * sigma_c),
            r1: jolt_to_ark(&(rp1 + sigma_c * blinds.d1)),
            r2: jolt_to_ark(&(rp2 + sigma_c * blinds.d2)),
            r3: jolt_to_ark(&(rr + sigma_c * rq + sigma_c.square() * blinds.c)),
        },
        sigma_c,
    )
}

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
    ) -> eyre::Result<(Self::Proof, Option<Fr>)>
    where
        Network: Rep3NetworkCoordinator,
    {
        if network.is_distributed() {
            return Err(eyre::eyre!("Dory opening proof: distributed subnets unsupported (single-worker mode only)"));
        }

        let sigma = DoryGlobals::get_num_columns().log_2();
        let g1_all = setup_g1_projective(setup);
        let (row_commitments, nu, l_vec, r_vec) = {
            let init_msgs: Vec<(usize, Vec<G1Affine>)> = network.receive_responses()?;
            let num_vars = init_msgs[0].0;
            let nu = compute_nu(num_vars, sigma);

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

            let point_wrapped: Vec<_> = opening_point
                .iter()
                .rev()
                .map(|&x| {
                    let x_fr: Fr = x.into();
                    jolt_to_ark(&x_fr)
                })
                .collect();
            let (l_vec_w, r_vec_w) = compute_left_right_vectors(&point_wrapped, nu, sigma);
            let l_vec: Vec<Fr> = l_vec_w.iter().map(ark_to_jolt).collect();
            let r_vec: Vec<Fr> = r_vec_w.iter().map(ark_to_jolt).collect();
            (row_commitments, nu, l_vec, r_vec)
        };

        let mut dory_transcript = JoltToDoryTranscript::new(transcript);

        let vmv_shares: Vec<DoryVmvShareMsg> = network.receive_responses()?;
        let ((c, d2), e1) = combine_vmv_shares(&vmv_shares)?;

        #[cfg(feature = "zk")]
        let mut zk_blinds = ZkAccumBlinds {
            c: sample_fr(),
            d1: Fr::zero(),
            d2: sample_fr(),
            e1: sample_fr(),
            e2: sample_fr(),
        };

        #[cfg(feature = "zk")]
        let vmv_message = VMVMessage::<ArkG1, ArkGT> {
            c: mask_gt(c, zk_blinds.c, setup),
            d2: mask_gt(d2, zk_blinds.d2, setup),
            e1: mask_g1(e1.into_group(), zk_blinds.e1, setup),
        };
        #[cfg(not(feature = "zk"))]
        let vmv_message = VMVMessage::<ArkG1, ArkGT> {
            c: ArkGT(c),
            d2: ArkGT(d2),
            e1: ArkG1(e1.into_group()),
        };
        dory_transcript.append_serde(b"vmv_c", &vmv_message.c);
        dory_transcript.append_serde(b"vmv_d2", &vmv_message.d2);
        dory_transcript.append_serde(b"vmv_e1", &vmv_message.e1);

        #[cfg(feature = "zk")]
        let (zk_e2, zk_y_com, zk_sigma1, zk_sigma2, y_blinding) = {
            let r_y = sample_fr();
            let e2 = mask_g2(setup.g2_vec[0].0 * *claimed_opening, zk_blinds.e2, setup);
            let y_com = ArkG1(setup.g1_vec[0].0 * *claimed_opening + setup.h1.0 * r_y);
            dory_transcript.append_serde(b"vmv_e2", &e2);
            dory_transcript.append_serde(b"vmv_y_com", &y_com);
            let sigma1 = generate_sigma1_proof::<jolt_core::poly::commitment::dory::BN254, _>(
                &jolt_to_ark(claimed_opening),
                &jolt_to_ark(&zk_blinds.e2),
                &jolt_to_ark(&r_y),
                setup,
                &mut dory_transcript,
            );
            let sigma2 = generate_sigma2_proof::<jolt_core::poly::commitment::dory::BN254, _>(
                &jolt_to_ark(&zk_blinds.e1),
                &jolt_to_ark(&-zk_blinds.d2),
                setup,
                &mut dory_transcript,
            );
            (Some(e2), Some(y_com), Some(sigma1), Some(sigma2), Some(r_y))
        };
        #[cfg(not(feature = "zk"))]
        let (zk_e2, zk_y_com, zk_sigma1, zk_sigma2, y_blinding) = (None, None, None, None, None);

        let mut padded_row_commitments = row_commitments;
        if nu < sigma {
            padded_row_commitments.resize(1 << sigma, G1Projective::zero());
        }
        let mut v1_pub: Vec<G1Projective> = padded_row_commitments;
        let mut s1: Vec<Fr> = r_vec;
        let mut s2: Vec<Fr> = l_vec;
        if nu < sigma {
            s1.resize(1 << sigma, Fr::zero());
            s2.resize(1 << sigma, Fr::zero());
        }
        let mut first_messages = Vec::with_capacity(sigma);
        let mut second_messages = Vec::with_capacity(sigma);

        let mut curr_rounds = sigma;
        while curr_rounds > 0 {
            let n2 = 1usize << (curr_rounds - 1);

            let first_msgs: Vec<DoryFirstReduceShareMsg> = network.receive_responses()?;
            let ((d2_left, d2_right), (d1_left, d1_right, e1_beta, e2_beta)) =
                combine_first_reduce_shares(&first_msgs)?;

            #[cfg(feature = "zk")]
            let round_blinds = sample_round_blinds();
            #[cfg(feature = "zk")]
            let first_msg = FirstReduceMessage::<ArkG1, ArkG2, ArkGT> {
                d1_left: mask_gt(d1_left, round_blinds.d1[0], setup),
                d1_right: mask_gt(d1_right, round_blinds.d1[1], setup),
                d2_left: mask_gt(d2_left, round_blinds.d2[0], setup),
                d2_right: mask_gt(d2_right, round_blinds.d2[1], setup),
                e1_beta: ArkG1(e1_beta.into_group()),
                e2_beta: ArkG2(e2_beta.into_group()),
            };
            #[cfg(not(feature = "zk"))]
            let first_msg = FirstReduceMessage::<ArkG1, ArkG2, ArkGT> {
                d1_left: ArkGT(d1_left),
                d1_right: ArkGT(d1_right),
                d2_left: ArkGT(d2_left),
                d2_right: ArkGT(d2_right),
                e1_beta: ArkG1(e1_beta.into_group()),
                e2_beta: ArkG2(e2_beta.into_group()),
            };
            dory_transcript.append_serde(b"d1_left", &first_msg.d1_left);
            dory_transcript.append_serde(b"d1_right", &first_msg.d1_right);
            dory_transcript.append_serde(b"d2_left", &first_msg.d2_left);
            dory_transcript.append_serde(b"d2_right", &first_msg.d2_right);
            dory_transcript.append_serde(b"e1_beta", &first_msg.e1_beta);
            dory_transcript.append_serde(b"e2_beta", &first_msg.e2_beta);
            let beta_ark = dory_transcript.challenge_scalar(b"beta");
            let beta = ark_to_jolt(&beta_ark);
            let beta_inv = beta.inverse().expect("beta must be invertible");
            network.broadcast_request((beta, beta_inv))?;
            first_messages.push(first_msg);

            #[cfg(feature = "zk")]
            {
                zk_blinds.c += zk_blinds.d2 * beta + zk_blinds.d1 * beta_inv;
            }

            jolt_optimizations::vector_add_scalar_mul_g1_online(&mut v1_pub, &g1_all[..(1 << curr_rounds)], beta);

            let second_msgs: Vec<DorySecondReduceShareMsg> = network.receive_responses()?;
            let ((c_plus, c_minus), (e2_plus, e2_minus), (e1_plus, e1_minus)) =
                combine_second_reduce_shares(&second_msgs)?;

            #[cfg(feature = "zk")]
            let second_msg = SecondReduceMessage::<ArkG1, ArkG2, ArkGT> {
                c_plus: mask_gt(c_plus, round_blinds.c[0], setup),
                c_minus: mask_gt(c_minus, round_blinds.c[1], setup),
                e1_plus: mask_g1(e1_plus.into_group(), round_blinds.e1[0], setup),
                e1_minus: mask_g1(e1_minus.into_group(), round_blinds.e1[1], setup),
                e2_plus: mask_g2(e2_plus, round_blinds.e2[0], setup),
                e2_minus: mask_g2(e2_minus, round_blinds.e2[1], setup),
            };
            #[cfg(not(feature = "zk"))]
            let second_msg = SecondReduceMessage::<ArkG1, ArkG2, ArkGT> {
                c_plus: ArkGT(c_plus),
                c_minus: ArkGT(c_minus),
                e1_plus: ArkG1(e1_plus.into_group()),
                e1_minus: ArkG1(e1_minus.into_group()),
                e2_plus: ArkG2(e2_plus),
                e2_minus: ArkG2(e2_minus),
            };
            dory_transcript.append_serde(b"c_plus", &second_msg.c_plus);
            dory_transcript.append_serde(b"c_minus", &second_msg.c_minus);
            dory_transcript.append_serde(b"e1_plus", &second_msg.e1_plus);
            dory_transcript.append_serde(b"e1_minus", &second_msg.e1_minus);
            dory_transcript.append_serde(b"e2_plus", &second_msg.e2_plus);
            dory_transcript.append_serde(b"e2_minus", &second_msg.e2_minus);
            let alpha_ark = dory_transcript.challenge_scalar(b"alpha");
            let alpha = ark_to_jolt(&alpha_ark);
            let alpha_inv = alpha.inverse().expect("alpha must be invertible");
            network.broadcast_request((alpha, alpha_inv))?;
            second_messages.push(second_msg);

            #[cfg(feature = "zk")]
            {
                zk_blinds.c += round_blinds.c[0] * alpha + round_blinds.c[1] * alpha_inv;
                zk_blinds.d1 = round_blinds.d1[0] * alpha + round_blinds.d1[1];
                zk_blinds.d2 = round_blinds.d2[0] * alpha_inv + round_blinds.d2[1];
                zk_blinds.e1 += round_blinds.e1[0] * alpha + round_blinds.e1[1] * alpha_inv;
                zk_blinds.e2 += round_blinds.e2[0] * alpha + round_blinds.e2[1] * alpha_inv;
            }

            let (v1_l_mut, v1_r_ref) = v1_pub.split_at_mut(n2);
            jolt_optimizations::vector_scalar_mul_add_gamma_g1_online(v1_l_mut, alpha, v1_r_ref);
            v1_pub.truncate(n2);

            let (s1_l, s1_r) = s1.split_at(n2);
            let (s2_l, s2_r) = s2.split_at(n2);
            let (s1_next, s2_next): (Vec<Fr>, Vec<Fr>) =
                (0..n2).into_par_iter().map(|i| (s1_l[i] * alpha + s1_r[i], s2_l[i] * alpha_inv + s2_r[i])).unzip();
            s1 = s1_next;
            s2 = s2_next;
            curr_rounds -= 1;
        }

        let gamma_ark = dory_transcript.challenge_scalar(b"gamma");
        let gamma = ark_to_jolt(&gamma_ark);
        let gamma_inv = gamma.inverse().expect("gamma must be invertible");

        let v2_shares: Vec<G2Affine> = network.receive_responses()?;
        let mut v2 = G2Projective::zero();
        for s in v2_shares {
            v2 += s.into_group();
        }

        #[cfg(feature = "zk")]
        let scalar_product_proof = {
            let (proof, _sigma_c) =
                scalar_product_proof(&mut dory_transcript, setup, v1_pub[0], v2, zk_blinds);
            Some(proof)
        };
        #[cfg(not(feature = "zk"))]
        let scalar_product_proof = None;

        let gamma_s1 = gamma * s1[0]
            + {
                #[cfg(feature = "zk")]
                {
                    sample_fr()
                }
                #[cfg(not(feature = "zk"))]
                {
                    Fr::zero()
                }
            };
        let e1_final = v1_pub[0] + setup.h1.0 * gamma_s1;

        let gamma_inv_s2 = gamma_inv * s2[0]
            + {
                #[cfg(feature = "zk")]
                {
                    sample_fr()
                }
                #[cfg(not(feature = "zk"))]
                {
                    Fr::zero()
                }
            };
        let e2_final = v2 + setup.h2.0 * gamma_inv_s2;

        let final_message = ScalarProductMessage::<ArkG1, ArkG2> {
            e1: ArkG1(e1_final),
            e2: ArkG2(e2_final),
        };
        dory_transcript.append_serde(b"final_e1", &final_message.e1);
        dory_transcript.append_serde(b"final_e2", &final_message.e2);
        let _d: jolt_core::poly::commitment::dory::ArkFr = dory_transcript.challenge_scalar(b"d");

        let _ = commitment;
        Ok((
            DoryProofData {
                sigma,
                dory_proof_data: ArkDoryProof {
                    vmv_message,
                    first_messages,
                    second_messages,
                    final_message,
                    nu,
                    sigma,
                    #[cfg(feature = "zk")]
                    e2: zk_e2,
                    #[cfg(feature = "zk")]
                    y_com: zk_y_com,
                    #[cfg(feature = "zk")]
                    sigma1_proof: zk_sigma1,
                    #[cfg(feature = "zk")]
                    sigma2_proof: zk_sigma2,
                    #[cfg(feature = "zk")]
                    scalar_product_proof,
                },
            },
            y_blinding,
        ))
    }

    fn combine_commitment_shares(commitments: &[&MaybeShared<Self::Commitment>]) -> Self::Commitment {
        let public = commitments.iter().find(|c| matches!(c, MaybeShared::Public(Some(_))));
        match public {
            Some(MaybeShared::Public(Some(c))) => c.clone(),
            None => {
                if commitments.iter().all(|c| matches!(c, MaybeShared::Public(None))) {
                    return DoryCommitment::default();
                }
                let mut acc = Fq12::one();
                for c in commitments {
                    match c {
                        MaybeShared::Shared(c) => acc *= c.0.0,
                        _ => unreachable!(),
                    }
                }
                DoryCommitment(ArkGT(acc))
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
                let mut acc = vec![G1Projective::zero(); num_rows];
                for h in hints {
                    match h {
                        MaybeShared::Shared(hint_share) => {
                            for (i, row) in hint_share.iter().enumerate() {
                                if i >= num_rows {
                                    break;
                                }
                                acc[i] += row.0;
                            }
                        }
                        MaybeShared::Public(None) => {}
                        _ => unreachable!(),
                    }
                }
                DoryOpeningProofHint::new(acc.into_iter().map(ArkG1).collect())
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
    unsafe { std::slice::from_raw_parts(setup.g1_vec.as_ptr() as *const G1Projective, setup.g1_vec.len()) }
}

/// Zero-copy view of `setup.core.g2_vec` as `&[G2Projective]`.
/// Safety: `JoltGroupWrapper<G2Projective>` is `#[repr(transparent)]`.
pub fn setup_g2_projective(setup: &<DoryCommitmentScheme as CommitmentScheme>::ProverSetup) -> &[G2Projective] {
    unsafe { std::slice::from_raw_parts(setup.g2_vec.as_ptr() as *const G2Projective, setup.g2_vec.len()) }
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
