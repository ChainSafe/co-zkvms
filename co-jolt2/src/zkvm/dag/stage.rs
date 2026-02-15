use crate::field::JoltField;
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::Rep3NetworkWorker;

/// Worker side of the staged sumcheck pipeline.
///
/// Each subsystem DAG node (e.g. `SpartanDagWorker`, `RegistersDagWorker`)
/// implements this trait to contribute sumcheck instances from shared polynomials.
pub trait SumcheckStagesWorker<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
{
    fn stage1_prove(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS, N>,
    ) -> Result<(), eyre::Report> {
        Ok(())
    }

    // Placeholder return type — will be replaced with actual sumcheck instance
    // type (e.g. Vec<Box<dyn Rep3SumcheckInstance<F>>>) in Step 3.
    fn stage2_instances(&mut self, _sm: &mut StateManagerWorker<'_, F, PCS, N>) -> Vec<()> {
        vec![]
    }

    fn stage3_instances(&mut self, _sm: &mut StateManagerWorker<'_, F, PCS, N>) -> Vec<()> {
        vec![]
    }

    fn stage4_instances(&mut self, _sm: &mut StateManagerWorker<'_, F, PCS, N>) -> Vec<()> {
        vec![]
    }
}

/// Coordinator side of the staged sumcheck pipeline.
///
/// Each subsystem DAG node (e.g. `SpartanDagCoordinator`, `RegistersDagCoordinator`)
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

    // Placeholder return type — will be replaced with actual sumcheck instance
    // type in Step 3.
    fn stage2_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<()> {
        vec![]
    }

    fn stage3_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<()> {
        vec![]
    }

    fn stage4_instances(
        &mut self,
        _sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<()> {
        vec![]
    }
}
