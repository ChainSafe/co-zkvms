use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;

use jolt_core::field::JoltField;
use crate::zkvm::dag::stage::{BatchedSumcheckInstance, SumcheckStagesCoordinator};
use crate::zkvm::dag::state_manager::StateManager;

use self::read_write_checking::Rep3RegistersReadWriteChecking;
use self::val_evaluation::Rep3ValEvaluation;

pub mod read_write_checking;
pub mod val_evaluation;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3RegistersDag;

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>, N>
    SumcheckStagesCoordinator<F, ProofTranscript, PCS, N> for Rep3RegistersDag
where
    N: mpc_core::protocols::rep3::network::Rep3NetworkCoordinator,
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        let rwc = Rep3RegistersReadWriteChecking::new(sm);
        Ok(vec![BatchedSumcheckInstance::Secret(Box::new(rwc))])
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        let val_eval = Rep3ValEvaluation::new(sm);
        Ok(vec![BatchedSumcheckInstance::Secret(Box::new(val_eval))])
    }
}
