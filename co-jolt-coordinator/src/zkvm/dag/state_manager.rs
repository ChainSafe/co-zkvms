use std::collections::BTreeMap;

use jolt_core::curve::Bn254Curve;
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::subprotocols::blindfold::BlindFoldAccumulator;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::JoltVerifierPreprocessing;
use tracer::JoltDevice;

use crate::poly::opening_proof::Rep3OpeningAccumulator;

pub use jolt_core::zkvm::dag::state_manager::{ProofData, ProofKeys, Proofs};

pub struct StateManager<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>> {
    pub transcript: ProofTranscript,
    pub proofs: BTreeMap<ProofKeys, ProofData<F, Bn254Curve, PCS, ProofTranscript>>,
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
    pub stage5_y_blinding: Option<F>,
    #[cfg(feature = "zk")]
    pub blindfold_accumulator: BlindFoldAccumulator<F, Bn254Curve>,
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
            stage5_y_blinding: None,
            #[cfg(feature = "zk")]
            blindfold_accumulator: BlindFoldAccumulator::new(),
        }
    }

    pub fn with_pcs_setup(mut self, setup: &'a PCS::ProverSetup) -> Self {
        self.pcs_setup = Some(setup);
        self
    }

    pub fn fiat_shamir_preamble(&mut self, trace_length: usize) {
        self.transcript.append_u64(self.program_io.memory_layout.max_input_size);
        self.transcript.append_u64(self.program_io.memory_layout.max_output_size);
        self.transcript.append_u64(self.program_io.memory_layout.memory_size);
        self.transcript.append_bytes(&self.program_io.inputs);
        self.transcript.append_bytes(&self.program_io.outputs);
        self.transcript.append_u64(self.program_io.panic as u64);
        self.transcript.append_u64(self.ram_K as u64);
        self.transcript.append_u64(trace_length as u64);
    }
}
