use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::{commitment_scheme::CommitmentScheme, mock::MockProof},
        multilinear_polynomial::PolynomialEvaluation,
    },
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, Transcript},
    },
};
use mpc_core::protocols::rep3::{
    network::{Rep3NetworkCoordinator, Rep3NetworkWorker},
    PartyID,
};
use std::{borrow::Borrow, marker::PhantomData};

pub use jolt_core::poly::commitment::commitment_scheme;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::Rep3CommitmentScheme;
use crate::poly::{combine_poly_shares_rep3, Rep3DensePolynomial, Rep3MultilinearPolynomial};
use crate::utils::element::MaybeShared;

#[derive(Clone)]
pub struct MockCommitScheme<F: JoltField, ProofTranscript: Transcript> {
    _marker: PhantomData<(F, ProofTranscript)>,
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Default, Debug, PartialEq)]
pub struct MockCommitment<F: JoltField> {
    poly: Rep3MultilinearPolynomial<F>,
}

impl<F: JoltField, ProofTranscript: Transcript> Rep3CommitmentScheme<F, ProofTranscript>
    for MockCommitScheme<F, ProofTranscript>
{
    fn combine_commitment_shares(
        commitments: &[&MaybeShared<MockCommitment<F>>],
    ) -> MockCommitment<F> {
        let public = commitments
            .iter()
            .find(|c| matches!(c, MaybeShared::Public(Some(_))));
        match public {
            Some(MaybeShared::Public(Some(commitment))) => commitment.clone(),
            None => {
                let poly_shares = commitments
                    .iter()
                    .map(|c| match c {
                        MaybeShared::Shared(c) => c.poly.as_shared().clone(),
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                let poly = combine_poly_shares_rep3(poly_shares);
                MockCommitment {
                    poly: Rep3MultilinearPolynomial::public(MultilinearPolynomial::LargeScalars(
                        poly,
                    )),
                }
            }
            _ => unreachable!(),
        }
    }

    fn coordinate_prove<Network>(network: &mut Network) -> eyre::Result<MockProof<F>>
    where
        Network: Rep3NetworkCoordinator,
    {
        let pf = network.receive_response(PartyID::ID0, 0)?;

        Ok(pf)
    }

    fn prove_rep3<Network>(
        poly: &Rep3DensePolynomial<F>,
        setup: &Self::Setup,
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker,
    {
        if network.party_id() == PartyID::ID0 {
            network.send_response(MockProof {
                opening_point: opening_point.to_owned(),
            })?
        }

        Ok(())
    }

    fn commit_rep3(
        poly: &Rep3MultilinearPolynomial<F>,
        setup: &Self::Setup,
        commit_to_public: bool,
    ) -> MaybeShared<Self::Commitment> {
        match poly {
            Rep3MultilinearPolynomial::Public { poly, .. } => {
                if commit_to_public {
                    let commitment =
                        <Self as CommitmentScheme<ProofTranscript>>::commit(poly, setup);
                    MaybeShared::Public(Some(commitment))
                } else {
                    MaybeShared::Public(None)
                }
            }
            Rep3MultilinearPolynomial::Shared(poly) => MaybeShared::Shared(MockCommitment {
                poly: poly.clone().into(),
            }),
        }
    }

    fn batch_commit_rep3<U>(
        polys: &[U],
        setup: &Self::Setup,
        commit_to_public: bool,
    ) -> Vec<MaybeShared<Self::Commitment>>
    where
        U: Borrow<Rep3MultilinearPolynomial<F>> + Sync,
    {
        let commitments = polys
            .par_iter()
            .map(|poly| {
                let commitment = <Self as Rep3CommitmentScheme<F, ProofTranscript>>::commit_rep3(
                    poly.borrow(),
                    setup,
                    commit_to_public,
                );
                commitment
            })
            .collect::<Vec<_>>();

        commitments
    }
}

impl<F, ProofTranscript> CommitmentScheme<ProofTranscript> for MockCommitScheme<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    type Field = F;
    type Setup = ();
    type Commitment = MockCommitment<F>;
    type Proof = MockProof<F>;
    type BatchedProof = MockProof<F>;

    fn setup(_max_poly_len: usize) -> Self::Setup {}
    fn commit(poly: &MultilinearPolynomial<Self::Field>, _setup: &Self::Setup) -> Self::Commitment {
        MockCommitment {
            poly: poly.clone().into(),
        }
    }
    fn batch_commit<P>(polys: &[P], setup: &Self::Setup) -> Vec<Self::Commitment>
    where
        P: Borrow<MultilinearPolynomial<Self::Field>>,
    {
        polys
            .iter()
            .map(|poly| Self::commit(poly.borrow(), setup))
            .collect()
    }
    fn prove(
        _setup: &Self::Setup,
        _poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[Self::Field],
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        MockProof {
            opening_point: opening_point.to_owned(),
        }
    }

    fn combine_commitments(
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let polys: Vec<_> = commitments
            .iter()
            .map(|commitment| commitment.poly.as_public())
            .collect();

        MockCommitment {
            poly: MultilinearPolynomial::linear_combination(&polys, coeffs).into(),
        }
    }

    fn verify(
        proof: &Self::Proof,
        _setup: &Self::Setup,
        _transcript: &mut ProofTranscript,
        opening_point: &[Self::Field],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let evaluation = commitment.poly.as_public().evaluate(opening_point);
        assert_eq!(evaluation, *opening);
        assert_eq!(proof.opening_point, opening_point);
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"mock_commit"
    }

    fn srs_size(setup: &Self::Setup) -> usize {
        1 << 20
    }
}

impl<F: JoltField> AppendToTranscript for MockCommitment<F> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"mocker");
    }
}
