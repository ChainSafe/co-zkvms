use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::Rep3NetworkWorker;

use crate::field::JoltField;
pub use crate::subprotocols::sumcheck::{
    BatchedSumcheckInstance, BatchedSumcheckWorkerInstance, PublicSumcheckInstance,
    PublicSumcheckInstanceWorker, Rep3SumcheckInstance, Rep3SumcheckInstanceWorker,
};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

// ---------------------------------------------------------------------------
// Staged sumcheck pipeline traits (per-subsystem interface)
// ---------------------------------------------------------------------------

/// Worker side of the staged sumcheck pipeline.
///
/// Each subsystem DAG node (e.g. `Rep3LookupsDagWorker`)
/// implements this trait to contribute sumcheck instances from shared polynomials.
pub trait SumcheckStagesWorker<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
{
    fn stage1_prove(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Result<(), eyre::Report> {
        Ok(())
    }

    fn stage2_instances(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F, N>> {
        vec![]
    }

    fn stage3_instances(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F, N>> {
        vec![]
    }

    fn stage4_instances(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F, N>> {
        vec![]
    }
}

/// Coordinator side of the staged sumcheck pipeline.
///
/// Each subsystem DAG node (e.g. `Rep3LookupsDag`)
/// implements this trait to drive sumcheck rounds via the Fiat-Shamir transcript.
pub trait SumcheckStagesCoordinator<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
>
{
    fn stage1_prove(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), eyre::Report> {
        Ok(())
    }

    fn stage2_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        vec![]
    }

    fn stage3_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        vec![]
    }

    fn stage4_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        vec![]
    }
}
