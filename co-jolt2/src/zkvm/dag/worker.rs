use crate::field::JoltField;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use mpc_core::protocols::rep3::network::Rep3NetworkWorker;

/// Worker side of the MPC DAG prover.
///
/// Generates shared witness polynomials, commits shares, and participates
/// in distributed sumcheck rounds driven by the coordinator.
pub struct JoltDAGWorker;

impl JoltDAGWorker {
    #[allow(unused_variables)]
    pub fn prove<F, PCS, N>(mut state: StateManagerWorker<'_, F, PCS, N>) -> eyre::Result<()>
    where
        F: JoltField,
        PCS: CommitmentScheme<Field = F>,
        N: Rep3NetworkWorker,
    {
        // Step 2+: sync with coordinator
        // Step 2: generate & commit witness polynomials, send commitment shares
        // Stage 1: participate in Spartan outer sumcheck
        // Stage 2: contribute batched sumcheck instances
        // Stage 3: contribute batched sumcheck instances
        // Stage 4: contribute batched sumcheck instances
        // Stage 5: opening proof — accumulator.reduce_and_prove()
        todo!("implement worker prove flow")
    }
}
