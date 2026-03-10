use ark_bn254::{Bn254, Fr};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::Zero;
use ark_poly_commit::multilinear_pc::data_structures::Proof;
use co_jolt2::poly::commitment::pst13::{PST13Commitment, PST13};
use co_jolt2::utils::types::MaybeShared;
use jolt_core::field::JoltField;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

use crate::poly::commitment::Rep3CoordinatorCommitmentScheme;

impl<ProofTranscript> Rep3CoordinatorCommitmentScheme<Fr, ProofTranscript> for PST13<Bn254>
where
    ProofTranscript: Transcript,
{
    fn coordinate_prove<Network>(
        _setup: &Self::ProverSetup,
        _transcript: &mut ProofTranscript,
        network: &mut Network,
        _opening_point: &[<Fr as JoltField>::Challenge],
        _claimed_opening: &Fr,
        _commitment: &Self::Commitment,
    ) -> eyre::Result<Self::Proof>
    where
        Network: Rep3NetworkCoordinator,
    {
        let proofs = if network.is_distributed() {
            let subnet_proofs = network
                .receive_responses_from_subnets::<Vec<<Bn254 as Pairing>::G1Affine>>()?
                .into_iter()
                .map(|shares| {
                    let [pf0, pf1, pf2]: [Vec<_>; 3] = shares.try_into().unwrap();
                    itertools::multizip((pf0, pf1, pf2))
                        .map(|(a, b, c)| (a + b + c).into_affine())
                        .collect::<Vec<<Bn254 as Pairing>::G1Affine>>()
                })
                .collect::<Vec<_>>();

            let mut proofs = subnet_proofs.into_iter();
            let mut combined = proofs.next().unwrap();
            for next in proofs {
                for (acc, share) in combined.iter_mut().zip(next) {
                    *acc = (*acc + share).into_affine();
                }
            }
            combined
        } else {
            let [pf0, pf1, pf2]: [Vec<<Bn254 as Pairing>::G1Affine>; 3] =
                network.receive_responses()?.try_into().unwrap();

            itertools::multizip((pf0, pf1, pf2))
                .map(|(a, b, c)| (a + b + c).into_affine())
                .collect::<Vec<_>>()
        };

        Ok(Proof { proofs })
    }

    fn combine_commitment_shares(
        commitments: &[&MaybeShared<Self::Commitment>],
    ) -> Self::Commitment {
        let public = commitments
            .iter()
            .find(|c| matches!(c, MaybeShared::Public(Some(_))));
        let (g_product, nv) = match public {
            Some(MaybeShared::Public(Some(commitment))) => (commitment.g_product, commitment.nv),
            None => {
                let mut g_product = <Bn254 as Pairing>::G1::zero();
                let mut nv = None;
                for commitment in commitments {
                    match commitment {
                        MaybeShared::Shared(commitment) => {
                            g_product += commitment.g_product;
                            match nv {
                                Some(nv) => assert_eq!(nv, commitment.nv),
                                None => nv = Some(commitment.nv),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                (g_product.into_affine(), nv.unwrap())
            }
            _ => unreachable!(),
        };
        PST13Commitment { nv, g_product }
    }

    fn combine_hint_shares(
        _hints: &[&MaybeShared<Self::OpeningProofHint>],
    ) -> Self::OpeningProofHint {
    }
}
