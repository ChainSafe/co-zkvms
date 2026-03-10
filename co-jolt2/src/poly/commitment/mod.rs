use crate::poly::Rep3MultilinearPolynomial;
use crate::utils::types::MaybeShared;
use jolt_core::field::JoltField;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::IoContextPool;
use mpc_core::protocols::rep3::network::{Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use std::borrow::Borrow;

pub use jolt_core::poly::commitment::commitment_scheme;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;

pub mod dory;
pub use dory::*;

pub trait Rep3CommitmentScheme<F: JoltField, ProofTranscript: Transcript>:
    CommitmentScheme<Field = F>
{
    fn commit_rep3<N: Rep3NetworkWorker>(
        poly: &Rep3MultilinearPolynomial<F>,
        setup: &Self::ProverSetup,
        commit_to_public: bool,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<(
        MaybeShared<Self::Commitment>,
        MaybeShared<Self::OpeningProofHint>,
    )>;

    fn distributed_commit_rep3(
        _poly: &Rep3MultilinearPolynomial<F>,
        _setup: &Self::ProverSetup,
        _commit_to_public: bool,
    ) -> (
        MaybeShared<Self::Commitment>,
        MaybeShared<Self::OpeningProofHint>,
    ) {
        todo!("distributed commit not implemented for this PCS")
    }

    fn batch_commit_rep3<U, N>(
        polys: &[U],
        setup: &Self::ProverSetup,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
    ) -> eyre::Result<
        Vec<(
            MaybeShared<Self::Commitment>,
            MaybeShared<Self::OpeningProofHint>,
        )>,
    >
    where
        U: Borrow<Rep3MultilinearPolynomial<F>> + Sync,
        N: Rep3NetworkWorker;

    fn prove_rep3<Network>(
        poly: &Rep3MultilinearPolynomial<F>,
        setup: &Self::ProverSetup,
        opening_point: &[<F as jolt_core::field::JoltField>::Challenge],
        opening_hint: Option<Self::OpeningProofHint>,
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker;

    /// Homomorphically combine per-polynomial hint shares using public RLC coefficients.
    ///
    /// Each `MaybeShared::Shared(hint_share)` is this party's additive share of a polynomial's
    /// row commitments. Returns the combined hint share: `combined[row] = Σ coeff_i * hint_i[row]`.
    /// For `MaybeShared::Public(Some(hint))`, only party ID0 adds (trivial share promotion).
    fn combine_hints_rep3(
        hints: Vec<MaybeShared<Self::OpeningProofHint>>,
        coeffs: &[F],
        party_id: PartyID,
    ) -> Self::OpeningProofHint;

    fn concat_commitments(_a: &Self::Commitment, _b: &Self::Commitment) -> Self::Commitment {
        todo!("concat_commitments not implemented for this PCS")
    }
}
