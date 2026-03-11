use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::MaybeShared;

pub mod dory;

pub trait Rep3CommitmentScheme<F: JoltField, ProofTranscript: Transcript>:
    CommitmentScheme<Field = F>
{
    fn coordinate_prove<Network>(
        setup: &Self::ProverSetup,
        transcript: &mut ProofTranscript,
        network: &mut Network,
        opening_point: &[<F as jolt_core::field::JoltField>::Challenge],
        claimed_opening: &F,
        commitment: &Self::Commitment,
        commitment_blinding: Option<F>,
    ) -> eyre::Result<(Self::Proof, Option<F>)>
    where
        Network: Rep3NetworkCoordinator;

    fn blind_transcript_commitment(
        setup: &Self::ProverSetup,
        commitment: Self::Commitment,
    ) -> (Self::Commitment, Option<F>);

    fn combine_commitment_shares(
        commitments: &[&MaybeShared<Self::Commitment>],
    ) -> Self::Commitment;

    fn combine_hint_shares(
        hints: &[&MaybeShared<Self::OpeningProofHint>],
    ) -> Self::OpeningProofHint;
}
