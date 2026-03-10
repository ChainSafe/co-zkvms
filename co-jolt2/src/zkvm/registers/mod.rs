use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;

use crate::zkvm::dag::stage::{BatchedSumcheckWorkerInstance, SumcheckStagesWorker};
use crate::zkvm::dag::state_manager::StateManagerWorker;
use jolt_core::field::JoltField;

use self::read_write_checking::Rep3RegistersReadWriteCheckingWorker;
use self::val_evaluation::Rep3ValEvaluationWorker;

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

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
    SumcheckStagesWorker<F, PCS, N> for Rep3RegistersDagWorker<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        let (gamma, input_claim) = self
            .stage2
            .take()
            .expect("Rep3RegistersDagWorker stage2 init not set");
        let rwc = Rep3RegistersReadWriteCheckingWorker::new(sm, gamma, input_claim);
        Ok(vec![BatchedSumcheckWorkerInstance::Secret(Box::new(rwc))])
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        let val_claim = self
            .stage3
            .take()
            .expect("Rep3RegistersDagWorker stage3 init not set");
        let val_eval = Rep3ValEvaluationWorker::new(sm, val_claim);
        Ok(vec![BatchedSumcheckWorkerInstance::Secret(Box::new(
            val_eval,
        ))])
    }
}

// ---------------------------------------------------------------------------
