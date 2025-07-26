use crate::{
    poly::{Rep3DensePolynomial, Rep3MultilinearPolynomial},
    utils::element::MaybeShared,
};
use jolt_core::utils::transcript::KeccakTranscript;
use jolt_core::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    utils::transcript::Transcript,
};
use mpc_core::protocols::rep3::network::{Rep3NetworkCoordinator, Rep3NetworkWorker};
use std::borrow::Borrow;

pub use jolt_core::poly::commitment::commitment_scheme;
pub mod pst13;

pub trait Rep3CommitmentScheme<F: JoltField, ProofTranscript: Transcript = KeccakTranscript>:
    CommitmentScheme<ProofTranscript, Field = F>
{
    fn commit_rep3(
        poly: &Rep3MultilinearPolynomial<F>,
        setup: &Self::Setup,
        commit_to_public: bool,
    ) -> MaybeShared<Self::Commitment>
    where
        F: JoltField;

    fn batch_commit_rep3<U>(
        polys: &[U],
        setup: &Self::Setup,
        commit_to_public: bool,
    ) -> Vec<MaybeShared<Self::Commitment>>
    where
        U: Borrow<Rep3MultilinearPolynomial<F>> + Sync;

    fn coordinate_prove<Network>(network: &mut Network) -> eyre::Result<Self::Proof>
    where
        Network: Rep3NetworkCoordinator;

    fn prove_rep3<Network>(
        poly: &Rep3DensePolynomial<F>,
        ck: &Self::Setup,
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker;

    fn combine_commitment_shares(
        commitments: &[&MaybeShared<Self::Commitment>],
    ) -> Self::Commitment;
}
