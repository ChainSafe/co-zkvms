use std::collections::BTreeMap;

use crate::field::JoltField;
use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::host::memory::Rep3Memory;
use crate::poly::multilinear_polynomial::Rep3MultilinearPolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::zkvm::dag::witness::Rep3CycleWitnesses;
use crate::zkvm::instruction::Rep3Cycle;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::{JoltProverPreprocessing, JoltVerifierPreprocessing};
use mpc_core::protocols::rep3::PartyID;
use tracer::JoltDevice;

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
    pub trusted_advice_polynomial: Option<Rep3MultilinearPolynomial<F>>,
    /// Field-domain per-cycle cache for R1CS virtual inputs.
    pub cycle_witness: Rep3CycleWitnesses<F>,
}

pub struct StateManagerWorker<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub party_id: PartyID,
    pub commitments: Vec<PCS::Commitment>,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub ram_K: usize,
    pub twist_sumcheck_switch_index: usize,
    pub program_io: JoltDevice,
    pub advice_shares: Option<Rep3ProgramIOInput>,
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
        program_io: JoltDevice,
        final_memory_state: Rep3Memory,
        party_id: PartyID,
        ram_K: usize,
        advice_shares: Option<Rep3ProgramIOInput>,
    ) -> Self {
        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = if num_chunks > 0 { T / num_chunks } else { T };
        let twist_sumcheck_switch_index = if chunk_size > 0 {
            chunk_size.trailing_zeros() as usize
        } else {
            0
        };

        Self {
            party_id,
            commitments: vec![],
            untrusted_advice_commitment: None,
            ram_K,
            twist_sumcheck_switch_index,
            program_io,
            advice_shares,
            prover_state: ProverStateWorker {
                preprocessing,
                trace: Some(trace),
                final_memory_state,
                untrusted_advice_polynomial: None,
                trusted_advice_polynomial: None,
                cycle_witness: Rep3CycleWitnesses::default(),
            },
            accumulator: Rep3OpeningAccumulatorWorker::new(party_id),
        }
    }

    pub fn get_prover_data(
        &self,
    ) -> (
        &'a JoltProverPreprocessing<F, PCS>,
        &[Rep3Cycle],
        &JoltDevice,
        &Rep3Memory,
    ) {
        (
            self.prover_state.preprocessing,
            self.trace_ref(),
            &self.program_io,
            &self.prover_state.final_memory_state,
        )
    }

    pub fn get_cycle_witness(&self) -> &Rep3CycleWitnesses<F> {
        &self.prover_state.cycle_witness
    }

    pub fn trace_ref(&self) -> &[Rep3Cycle] {
        self.prover_state
            .trace
            .as_deref()
            .expect("trace already dropped")
    }

    pub fn trace_mut(&mut self) -> &mut Vec<Rep3Cycle> {
        self.prover_state
            .trace
            .as_mut()
            .expect("trace already dropped")
    }

    pub fn trace_len(&self) -> usize {
        self.trace_ref().len()
    }
}

// ---------------------------------------------------------------------------
// Coordinator (no `Coordinator` suffix by convention)
// ---------------------------------------------------------------------------

pub struct StateManager<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
> {
    pub transcript: ProofTranscript,
    pub proofs: BTreeMap<ProofKeys, ProofData<F, PCS, ProofTranscript>>,
    pub commitments: Vec<PCS::Commitment>,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    pub ram_K: usize,
    pub twist_sumcheck_switch_index: usize,
    pub trace_length: usize,
    pub program_io: JoltDevice,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub pcs_setup: Option<&'a PCS::ProverSetup>,
    pub accumulator: Rep3OpeningAccumulator<F>,
}

impl<'a, F, ProofTranscript, PCS> StateManager<'a, F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn new(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        program_io: JoltDevice,
        ram_K: usize,
        twist_sumcheck_switch_index: usize,
    ) -> Self {
        Self {
            transcript: ProofTranscript::new(b"Jolt"),
            proofs: BTreeMap::new(),
            commitments: vec![],
            untrusted_advice_commitment: None,
            trusted_advice_commitment: None,
            ram_K,
            twist_sumcheck_switch_index,
            trace_length: 0,
            program_io,
            preprocessing,
            pcs_setup: None,
            accumulator: Rep3OpeningAccumulator::new(),
        }
    }

    pub fn with_pcs_setup(mut self, setup: &'a PCS::ProverSetup) -> Self {
        self.pcs_setup = Some(setup);
        self
    }

    pub fn fiat_shamir_preamble(&mut self, trace_length: usize) {
        self.transcript
            .append_u64(self.program_io.memory_layout.max_input_size);
        self.transcript
            .append_u64(self.program_io.memory_layout.max_output_size);
        self.transcript
            .append_u64(self.program_io.memory_layout.memory_size);
        self.transcript.append_bytes(&self.program_io.inputs);
        self.transcript.append_bytes(&self.program_io.outputs);
        self.transcript.append_u64(self.program_io.panic as u64);
        self.transcript.append_u64(self.ram_K as u64);
        self.transcript.append_u64(trace_length as u64);
    }
}
