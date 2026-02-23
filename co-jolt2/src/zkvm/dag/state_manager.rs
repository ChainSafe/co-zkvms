use std::collections::BTreeMap;

use crate::field::JoltField;
use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::host::memory::Rep3Memory;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::multilinear_polynomial::Rep3MultilinearPolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::zkvm::instruction::Rep3Cycle;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction::{CircuitFlags, NUM_CIRCUIT_FLAGS};
use jolt_core::zkvm::{JoltProverPreprocessing, JoltVerifierPreprocessing};
use mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::Rep3RingShare;
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
    /// Field-domain per-cycle cache for R1CS virtual inputs.
    pub cycle_witness: Rep3CycleWitnesses<F>,
}

/// Field-domain per-cycle witness cache (struct-of-arrays).
///
/// This is the post-witness representation that lets us drop the ring-shared trace.
#[derive(Clone, Debug, Default)]
pub struct Rep3CycleWitnesses<F: JoltField> {
    pub pc: Vec<u64>,
    pub unexpanded_pc: Vec<u64>,
    pub imm: Vec<i128>,
    pub rd_addr: Vec<u8>,
    pub rs1_addr: Vec<u8>,
    pub rs2_addr: Vec<u8>,
    pub ram_addr: Vec<u64>,
    /// Bit i corresponds to `CircuitFlags as usize == i`.
    pub flags_bits: Vec<u32>,
    /// Advice payload (public for now); only meaningful when `Advice` flag is set.
    pub advice: Vec<u64>,

    /// Cached lookup output per cycle (field shares).
    pub lookup_output: Vec<Rep3PrimeFieldShare<F>>,

    pub rs1_value: Vec<Rep3PrimeFieldShare<F>>,
    pub rs2_value: Vec<Rep3PrimeFieldShare<F>>,
    pub rd_write_value: Vec<Rep3PrimeFieldShare<F>>,
    pub ram_read_value: Vec<Rep3PrimeFieldShare<F>>,
    pub ram_write_value: Vec<Rep3PrimeFieldShare<F>>,

    /// Full 128-bit lookup indices per cycle (ring-shared).
    /// Persisted from witness gen for use in ReadRaf suffix evaluation.
    pub lookup_indices: Vec<Rep3RingShare<u128>>,

    /// RdInc polynomial (post - pre for register writes). Stored as
    /// `Option<Rep3DensePolynomial>` so provers can `.take()` ownership.
    pub rd_inc: Option<Rep3DensePolynomial<F>>,
    /// RamInc polynomial (post - pre for RAM writes).
    pub ram_inc: Option<Rep3DensePolynomial<F>>,
}

impl<F: JoltField> Rep3CycleWitnesses<F> {
    pub fn len(&self) -> usize {
        self.pc.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pc.is_empty()
    }

    pub fn row(&self, t: usize) -> Rep3CycleWitnessRef<'_, F> {
        Rep3CycleWitnessRef { w: self, t }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Rep3CycleWitnessRef<'a, F: JoltField> {
    w: &'a Rep3CycleWitnesses<F>,
    t: usize,
}

impl<'a, F: JoltField> Rep3CycleWitnessRef<'a, F> {
    pub fn flags_bits(&self) -> u32 {
        self.w.flags_bits[self.t]
    }

    pub fn pc(&self) -> u64 {
        self.w.pc[self.t]
    }

    pub fn unexpanded_pc(&self) -> u64 {
        self.w.unexpanded_pc[self.t]
    }

    pub fn imm(&self) -> i128 {
        self.w.imm[self.t]
    }

    pub fn rd_addr(&self) -> u8 {
        self.w.rd_addr[self.t]
    }

    pub fn rs1_addr(&self) -> u8 {
        self.w.rs1_addr[self.t]
    }

    pub fn rs2_addr(&self) -> u8 {
        self.w.rs2_addr[self.t]
    }

    pub fn ram_addr(&self) -> u64 {
        self.w.ram_addr[self.t]
    }

    pub fn advice(&self) -> u64 {
        self.w.advice[self.t]
    }

    pub fn lookup_output(&self) -> Rep3PrimeFieldShare<F> {
        self.w.lookup_output[self.t]
    }

    pub fn rs1_value(&self) -> Rep3PrimeFieldShare<F> {
        self.w.rs1_value[self.t]
    }

    pub fn rs2_value(&self) -> Rep3PrimeFieldShare<F> {
        self.w.rs2_value[self.t]
    }

    pub fn rd_write_value(&self) -> Rep3PrimeFieldShare<F> {
        self.w.rd_write_value[self.t]
    }

    pub fn ram_read_value(&self) -> Rep3PrimeFieldShare<F> {
        self.w.ram_read_value[self.t]
    }

    pub fn ram_write_value(&self) -> Rep3PrimeFieldShare<F> {
        self.w.ram_write_value[self.t]
    }

    pub fn flag(&self, flag: CircuitFlags) -> bool {
        debug_assert!(NUM_CIRCUIT_FLAGS <= 32);
        let bit = 1u32 << (flag as usize);
        (self.w.flags_bits[self.t] & bit) != 0
    }

    pub fn next_is_noop(&self) -> bool {
        if self.t + 1 >= self.w.len() {
            // Vanilla `R1CSCycleInputs::from_trace` uses `false` for the last cycle.
            false
        } else {
            let bit = 1u32 << (CircuitFlags::IsNoop as usize);
            (self.w.flags_bits[self.t + 1] & bit) != 0
        }
    }

    pub fn next_pc(&self) -> u64 {
        if self.t + 1 >= self.w.len() {
            0
        } else {
            self.w.pc[self.t + 1]
        }
    }

    pub fn next_unexpanded_pc(&self) -> u64 {
        if self.t + 1 >= self.w.len() {
            0
        } else {
            self.w.unexpanded_pc[self.t + 1]
        }
    }

    pub fn should_jump(&self) -> bool {
        self.flag(CircuitFlags::Jump) && !self.next_is_noop()
    }

    /// Returns the left instruction input as a public field element.
    /// Only valid when the left operand is NOT `Rs1Value` (i.e., it's PC or zero).
    pub fn to_left_public_input(&self) -> F {
        if self.flag(CircuitFlags::LeftOperandIsPC) {
            F::from_u64(self.unexpanded_pc())
        } else {
            F::zero()
        }
    }

    /// Returns the right instruction input as a public field element.
    /// Only valid when the right operand is NOT `Rs2Value` (i.e., it's Imm or zero).
    pub fn to_right_public_input(&self) -> F {
        if self.flag(CircuitFlags::RightOperandIsImm) {
            F::from_i128(self.imm())
        } else {
            F::zero()
        }
    }

    /// Mirrors vanilla `LookupQuery::to_instruction_inputs`.
    /// Returns `(left_input, right_input)` as field shares.
    pub fn to_instruction_inputs(
        &self,
        party_id: PartyID,
    ) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>) {
        let left = if self.flag(CircuitFlags::LeftOperandIsRs1Value) {
            self.rs1_value()
        } else if self.flag(CircuitFlags::LeftOperandIsPC) {
            promote_to_trivial_share(party_id, F::from_u64(self.unexpanded_pc()))
        } else {
            Rep3PrimeFieldShare::zero_share()
        };
        let right = if self.flag(CircuitFlags::RightOperandIsRs2Value) {
            self.rs2_value()
        } else if self.flag(CircuitFlags::RightOperandIsImm) {
            promote_to_trivial_share(party_id, F::from_i128(self.imm()))
        } else {
            Rep3PrimeFieldShare::zero_share()
        };
        (left, right)
    }

    /// Mirrors vanilla `LookupQuery::to_lookup_operands`.
    /// Returns `(left_lookup, right_lookup)` as field shares.
    ///
    /// For Mul: `right_lookup = product = left * right`; since this requires
    /// shared multiplication (MPC communication), the caller must supply
    /// the pre-computed product.
    pub fn to_lookup_operands(
        &self,
        party_id: PartyID,
        product: Rep3PrimeFieldShare<F>,
    ) -> (Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>) {
        // Lookup operands use XLEN-bit (u64) semantics, even when the underlying
        // instruction input is an i128 immediate (see e.g. `ADDI::to_lookup_operands`).
        let left_u64 = if self.flag(CircuitFlags::LeftOperandIsRs1Value) {
            self.rs1_value()
        } else if self.flag(CircuitFlags::LeftOperandIsPC) {
            promote_to_trivial_share(party_id, F::from_u64(self.unexpanded_pc()))
        } else {
            Rep3PrimeFieldShare::zero_share()
        };

        let right_u64 = if self.flag(CircuitFlags::RightOperandIsRs2Value) {
            self.rs2_value()
        } else if self.flag(CircuitFlags::RightOperandIsImm) {
            // Two's-complement / truncation semantics: cast imm to u64.
            promote_to_trivial_share(party_id, F::from_u64(self.imm() as u64))
        } else {
            Rep3PrimeFieldShare::zero_share()
        };

        let zero = Rep3PrimeFieldShare::zero_share();

        if self.flag(CircuitFlags::AddOperands) {
            // (0, x + y_u64)
            (zero, left_u64 + right_u64)
        } else if self.flag(CircuitFlags::SubtractOperands) {
            // (0, x + (2^64 - y_u64)) == (0, x - y_u64 + 2^64)
            let two_pow_64 = promote_to_trivial_share(party_id, F::from_u128(1u128 << 64));
            (zero, left_u64 - right_u64 + two_pow_64)
        } else if self.flag(CircuitFlags::MultiplyOperands) {
            // (0, x * y_u64) as a u128 (represented in the field); provided by the caller.
            (zero, product)
        } else if self.flag(CircuitFlags::Advice) {
            (zero, promote_to_trivial_share(party_id, F::from_u64(self.advice())))
        } else {
            // Default: operands are the instruction inputs interpreted as (x, y_u64).
            (left_u64, right_u64)
        }
    }

    /// Mirrors vanilla `LookupQuery::to_lookup_output`.
    /// Returns the cached lookup output (already computed during witness gen).
    pub fn to_lookup_output(&self) -> Rep3PrimeFieldShare<F> {
        self.w.lookup_output[self.t]
    }
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
                trace,
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

    pub fn get_cycle_witness(&self) -> &Rep3CycleWitnesses<F> {
        &self.prover_state.cycle_witness
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
    pub trace_length: usize,
    pub program_io: JoltDevice,
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub accumulator: Rep3OpeningAccumulator<F>,
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
            trace_length: 0,
            program_io,
            preprocessing,
            accumulator: Rep3OpeningAccumulator::new(),
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
