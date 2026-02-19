use std::collections::BTreeMap;

use crate::field::JoltField;
use crate::host::memory::Rep3Memory;
use crate::poly::multilinear_polynomial::Rep3MultilinearPolynomial;
use crate::zkvm::instruction::Rep3Cycle;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::{JoltProverPreprocessing, JoltVerifierPreprocessing};
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use tracer::JoltDevice;

// Re-export vanilla DAG types
pub use jolt_core::zkvm::dag::state_manager::{ProofData, ProofKeys, Proofs};

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct ProverStateWorker<'a, F: JoltField, PCS: CommitmentScheme<Field = F>> {
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: Vec<Rep3Cycle>,
    pub final_memory_state: Rep3Memory,
    pub untrusted_advice_polynomial: Option<Rep3MultilinearPolynomial<F>>,
    pub trusted_advice_polynomial: Option<Rep3MultilinearPolynomial<F>>,
}

pub struct StateManagerWorker<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
> {
    pub io_ctx: IoContextPool<N>,
    pub commitments: Vec<PCS::Commitment>,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub ram_K: usize,
    pub twist_sumcheck_switch_index: usize,
    pub program_io: JoltDevice,
    pub prover_state: ProverStateWorker<'a, F, PCS>,
}

impl<'a, F, PCS, N> StateManagerWorker<'a, F, PCS, N>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    N: Rep3NetworkWorker,
{
    pub fn new(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        trace: Vec<Rep3Cycle>,
        program_io: JoltDevice,
        final_memory_state: Rep3Memory,
        io_ctx: IoContextPool<N>,
        ram_K: usize,
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
            io_ctx,
            commitments: vec![],
            untrusted_advice_commitment: None,
            ram_K,
            twist_sumcheck_switch_index,
            program_io,
            prover_state: ProverStateWorker {
                preprocessing,
                trace,
                final_memory_state,
                untrusted_advice_polynomial: None,
                trusted_advice_polynomial: None,
            },
        }
    }

    pub fn get_prover_data(
        &self,
    ) -> (
        &'a JoltProverPreprocessing<F, PCS>,
        &Vec<Rep3Cycle>,
        &JoltDevice,
        &Rep3Memory,
    ) {
        (
            self.prover_state.preprocessing,
            &self.prover_state.trace,
            &self.program_io,
            &self.prover_state.final_memory_state,
        )
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct StateManagerCoordinator<
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
    pub program_io: JoltDevice,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
}

impl<'a, F, ProofTranscript, PCS> StateManagerCoordinator<'a, F, ProofTranscript, PCS>
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
            program_io,
            preprocessing,
        }
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
