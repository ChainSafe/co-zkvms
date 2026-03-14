use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::host::memory::Rep3Memory;
use crate::poly::multilinear_polynomial::Rep3MultilinearPolynomial;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::zkvm::dag::witness::Rep3CycleWitnesses;
use crate::zkvm::instruction::Rep3Cycle;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::zkvm::JoltProverPreprocessing;
use mpc_core::protocols::rep3::PartyID;
use mpc_core::MaybeShared;

// Re-export vanilla DAG types
pub use jolt_core::zkvm::dag::state_manager::{ProofData, ProofKeys, Proofs};

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct ProverStateWorker<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: Option<Vec<Rep3Cycle>>,
    pub final_memory_state: Rep3Memory,
    pub untrusted_advice_polynomial: Option<Rep3MultilinearPolynomial<F>>,
    pub untrusted_advice_hint: Option<MaybeShared<PCS::OpeningProofHint>>,
    pub trusted_advice_polynomial: Option<Rep3MultilinearPolynomial<F>>,
    /// Field-domain per-cycle cache for R1CS virtual inputs.
    pub cycle_witness: Rep3CycleWitnesses<F>,
}

pub struct StateManagerWorker<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub party_id: PartyID,
    pub commitments: Vec<PCS::Commitment>,
    pub untrusted_advice_commitment: Option<MaybeShared<PCS::Commitment>>,
    pub ram_K: usize,
    pub twist_sumcheck_switch_index: usize,
    pub program_io: Rep3ProgramIOInput,
    pub prover_state: ProverStateWorker<'a, F, PCS>,
    pub accumulator: Rep3OpeningAccumulatorWorker<F>,
}

impl<'a, F, PCS> StateManagerWorker<'a, F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn new(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        trace: Vec<Rep3Cycle>,
        program_io: Rep3ProgramIOInput,
        final_memory_state: Rep3Memory,
        party_id: PartyID,
        ram_K: usize,
    ) -> Self {
        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = if num_chunks > 0 { T / num_chunks } else { T };
        let twist_sumcheck_switch_index = if chunk_size > 0 { chunk_size.trailing_zeros() as usize } else { 0 };

        Self {
            party_id,
            commitments: vec![],
            untrusted_advice_commitment: None,
            ram_K,
            twist_sumcheck_switch_index,
            program_io,
            prover_state: ProverStateWorker {
                preprocessing,
                trace: Some(trace),
                final_memory_state,
                untrusted_advice_polynomial: None,
                untrusted_advice_hint: None,
                trusted_advice_polynomial: None,
                cycle_witness: Rep3CycleWitnesses::default(),
            },
            accumulator: Rep3OpeningAccumulatorWorker::new(party_id),
        }
    }

    pub fn get_prover_data(
        &self,
    ) -> (&'a JoltProverPreprocessing<F, PCS>, &[Rep3Cycle], &Rep3ProgramIOInput, &Rep3Memory) {
        (self.prover_state.preprocessing, self.trace_ref(), &self.program_io, &self.prover_state.final_memory_state)
    }

    pub fn get_cycle_witness(&self) -> &Rep3CycleWitnesses<F> {
        &self.prover_state.cycle_witness
    }

    pub fn trace_ref(&self) -> &[Rep3Cycle] {
        self.prover_state.trace.as_deref().expect("trace already dropped")
    }

    pub fn trace_mut(&mut self) -> &mut Vec<Rep3Cycle> {
        self.prover_state.trace.as_mut().expect("trace already dropped")
    }

    pub fn trace_len(&self) -> usize {
        self.trace_ref().len()
    }
}
