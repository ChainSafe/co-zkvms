use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{
    OpeningPoint, ReducedOpeningProof, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::subprotocols::blindfold::BlindFoldProof;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::transcripts::Transcript;
use crate::zkvm::dag::proof_serialization::JoltProof;
use crate::zkvm::witness::VirtualPolynomial;
use crate::zkvm::JoltVerifierPreprocessing;
use num_derive::FromPrimitive;
use tracer::JoltDevice;

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, FromPrimitive)]
#[repr(u8)]
pub enum ProofKeys {
    Stage1Sumcheck,
    Stage2Sumcheck,
    Stage3Sumcheck,
    Stage4Sumcheck,
    ReducedOpeningProof,
    TrustedAdviceProof,
    UntrustedAdviceProof,
}

pub enum ProofData<F: JoltField, C: JoltCurve, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> {
    SumcheckProof(SumcheckInstanceProof<F, C, ProofTranscript>),
    ReducedOpeningProof(ReducedOpeningProof<F, C, PCS, ProofTranscript>),
    OpeningProof(PCS::Proof),
}

pub type Proofs<F, C, PCS, ProofTranscript> = BTreeMap<ProofKeys, ProofData<F, C, PCS, ProofTranscript>>;

// ---------------------------------------------------------------------------
// Vanilla verifier StateManager
// ---------------------------------------------------------------------------

pub struct StateManager<'a, F: JoltField, C: JoltCurve, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>> {
    pub transcript: Rc<RefCell<ProofTranscript>>,
    pub proofs: Rc<RefCell<Proofs<F, C, PCS, ProofTranscript>>>,
    pub commitments: Rc<RefCell<Vec<PCS::Commitment>>>,
    pub untrusted_advice_commitment: Option<PCS::Commitment>,
    pub trusted_advice_commitment: Option<PCS::Commitment>,
    #[cfg(feature = "zk")]
    pub blindfold_proof: Option<BlindFoldProof<F, C>>,
    pub ram_K: usize,
    pub twist_sumcheck_switch_index: usize,
    pub trace_length: usize,
    pub program_io: JoltDevice,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub(crate) accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
}

impl<'a, F, C, ProofTranscript, PCS> StateManager<'a, F, C, ProofTranscript, PCS>
where
    F: JoltField,
    C: JoltCurve,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn from_proof(
        proof: JoltProof<F, C, PCS, ProofTranscript>,
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        program_io: JoltDevice,
        ram_K: usize,
        twist_sumcheck_switch_index: usize,
    ) -> Self {
        #[cfg(feature = "zk")]
        let zk_mode = proof.blindfold_proof.is_some();
        #[cfg(not(feature = "zk"))]
        let zk_mode = false;

        let mut accumulator =
            if zk_mode { VerifierOpeningAccumulator::new_zk() } else { VerifierOpeningAccumulator::new() };
        // Seed any serialized openings that are present. In the full BlindFold path this can
        // legitimately be empty, but mixed clear/ZK staging still relies on these values.
        accumulator.prime_openings(proof.opening_claims.0.clone());

        Self {
            transcript: Rc::new(RefCell::new(ProofTranscript::new(b"Jolt"))),
            proofs: Rc::new(RefCell::new(proof.proofs)),
            commitments: Rc::new(RefCell::new(proof.commitments)),
            untrusted_advice_commitment: proof.untrusted_advice_commitment,
            trusted_advice_commitment: None,
            #[cfg(feature = "zk")]
            blindfold_proof: proof.blindfold_proof,
            ram_K,
            twist_sumcheck_switch_index,
            trace_length: proof.trace_length,
            program_io,
            preprocessing,
            accumulator: Rc::new(RefCell::new(accumulator)),
        }
    }

    pub fn fiat_shamir_preamble(&self) {
        let mut transcript = self.transcript.borrow_mut();
        transcript.append_u64(self.program_io.memory_layout.max_input_size);
        transcript.append_u64(self.program_io.memory_layout.max_output_size);
        transcript.append_u64(self.program_io.memory_layout.memory_size);
        transcript.append_bytes(&self.program_io.inputs);
        transcript.append_bytes(&self.program_io.outputs);
        transcript.append_u64(self.program_io.panic as u64);
        transcript.append_u64(self.ram_K as u64);
        transcript.append_u64(self.trace_length as u64);
    }

    pub fn get_verifier_data(&self) -> (&JoltVerifierPreprocessing<F, PCS>, &JoltDevice, usize) {
        (self.preprocessing, &self.program_io, self.trace_length)
    }

    pub fn get_verifier_accumulator(&self) -> Rc<RefCell<VerifierOpeningAccumulator<F>>> {
        self.accumulator.clone()
    }

    pub fn get_transcript(&self) -> Rc<RefCell<ProofTranscript>> {
        self.transcript.clone()
    }

    pub fn get_commitments(&self) -> Rc<RefCell<Vec<PCS::Commitment>>> {
        self.commitments.clone()
    }

    pub fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        self.accumulator.borrow().get_virtual_polynomial_opening(polynomial, sumcheck)
    }
}
