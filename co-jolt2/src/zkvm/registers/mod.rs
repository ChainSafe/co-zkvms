use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;

use crate::field::JoltField;
use crate::zkvm::dag::stage::{
    BatchedSumcheckInstance, BatchedSumcheckWorkerInstance, Rep3SumcheckInstance,
    Rep3SumcheckInstanceWorker, SumcheckStagesCoordinator, SumcheckStagesWorker,
};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

use self::read_write_checking::{
    Rep3RegistersReadWriteChecking, Rep3RegistersReadWriteCheckingWorker,
};
use self::val_evaluation::{Rep3ValEvaluation, Rep3ValEvaluationWorker};

pub mod read_write_checking;
pub mod val_evaluation;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3RegistersDagWorker<F: JoltField> {
    stage2: Option<(F, F)>,
    stage3: Option<F>,
}

impl<F: JoltField> Rep3RegistersDagWorker<F> {
    pub fn new() -> Self {
        Self {
            stage2: None,
            stage3: None,
        }
    }

    pub fn set_stage2_init(&mut self, gamma: F, input_claim: F) {
        self.stage2 = Some((gamma, input_claim));
    }

    pub fn set_stage3_init(&mut self, val_claim: F) {
        self.stage3 = Some(val_claim);
    }
}

impl<F: JoltField> Default for Rep3RegistersDagWorker<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> SumcheckStagesWorker<F, PCS>
    for Rep3RegistersDagWorker<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F>> {
        let (gamma, input_claim) = self
            .stage2
            .take()
            .expect("Rep3RegistersDagWorker stage2 init not set");
        let rwc = Rep3RegistersReadWriteCheckingWorker::new(sm, gamma, input_claim);
        vec![BatchedSumcheckWorkerInstance::Secret(Box::new(rwc))]
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F>> {
        let val_claim = self
            .stage3
            .take()
            .expect("Rep3RegistersDagWorker stage3 init not set");
        let val_eval = Rep3ValEvaluationWorker::new(sm, val_claim);
        vec![BatchedSumcheckWorkerInstance::Secret(Box::new(val_eval))]
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3RegistersDag;

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStagesCoordinator<F, ProofTranscript, PCS> for Rep3RegistersDag
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        let rwc = Rep3RegistersReadWriteChecking::new(sm);
        vec![BatchedSumcheckInstance::Secret(Box::new(rwc))]
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        let val_eval = Rep3ValEvaluation::new(sm);
        vec![BatchedSumcheckInstance::Secret(Box::new(val_eval))]
    }
}
