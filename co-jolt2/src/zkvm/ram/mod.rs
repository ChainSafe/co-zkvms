use jolt2_common::constants::RAM_START_ADDRESS;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::ram::remap_address;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::{binary_ring_to_field_many, ring_to_field_a2b_many};
use mpc_core::protocols::rep3_ring::yao::ring_to_field_many;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::zkvm::dag::stage::{
    Rep3SumcheckInstance, Rep3SumcheckInstanceWorker, SumcheckStagesCoordinator,
    SumcheckStagesWorker,
};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

use self::output_check::{
    Rep3OutputSumcheck, Rep3OutputSumcheckWorker, Rep3ValFinalSumcheck, Rep3ValFinalSumcheckWorker,
};
use self::raf_evaluation::{Rep3RafEvaluation, Rep3RafEvaluationWorker};
use self::read_write_checking::{Rep3RamReadWriteChecking, Rep3RamReadWriteCheckingWorker};

pub mod output_check;
pub mod raf_evaluation;
pub mod read_write_checking;

pub(crate) fn build_initial_memory_state(
    ram_preprocessing: &jolt_core::zkvm::ram::RAMPreprocessing,
    program_io: &tracer::JoltDevice,
    K: usize,
) -> Vec<u64> {
    let memory_layout = &program_io.memory_layout;
    let mut initial_memory_state: Vec<u64> = vec![0; K];

    // Copy bytecode
    let mut index =
        remap_address(ram_preprocessing.min_bytecode_address, memory_layout).unwrap() as usize;
    for word in &ram_preprocessing.bytecode_words {
        initial_memory_state[index] = *word;
        index += 1;
    }

    // Copy trusted advice
    index = remap_address(memory_layout.trusted_advice_start, memory_layout).unwrap() as usize;
    for chunk in program_io.trusted_advice.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        initial_memory_state[index] = u64::from_le_bytes(word);
        index += 1;
    }

    // Copy untrusted advice
    index = remap_address(memory_layout.untrusted_advice_start, memory_layout).unwrap() as usize;
    for chunk in program_io.untrusted_advice.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        initial_memory_state[index] = u64::from_le_bytes(word);
        index += 1;
    }

    // Copy inputs
    index = remap_address(memory_layout.input_start, memory_layout).unwrap() as usize;
    for chunk in program_io.inputs.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        initial_memory_state[index] = u64::from_le_bytes(word);
        index += 1;
    }

    initial_memory_state
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3RamDagWorker<F: JoltField> {
    /// PUBLIC initial memory state (bytecode + inputs + advice)
    initial_memory_state: Vec<u64>,
    /// SHARED final memory state (converted from Rep3Memory ring shares → field shares)
    final_memory_field: Vec<Rep3PrimeFieldShare<F>>,
    stage2: Option<(F, F, Vec<F::Challenge>)>,
    stage3: Option<F>,
}

impl<F: JoltField> Rep3RamDagWorker<F> {
    /// Build initial memory (PUBLIC, same as vanilla) and convert final memory
    /// from `Rep3Memory` binary ring shares to `Rep3PrimeFieldShare<F>`.
    ///
    /// The ring→field conversion requires one round of MPC communication (a2b).
    pub fn new<PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>(
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<Self> {
        let ram_preprocessing = &sm.prover_state.preprocessing.shared.ram;
        let K = sm.ram_K;

        // --- Build initial_memory_state (PUBLIC, same logic as vanilla RamDag) ---
        let initial_memory_state = build_initial_memory_state(ram_preprocessing, &sm.program_io, K);
        let memory_layout = &sm.program_io.memory_layout;

        // --- Convert final memory: Rep3Memory (binary ring shares) → field shares ---
        // This requires MPC communication (a2b conversion).
        let party_id = sm.party_id;

        // Only convert the portion of DRAM that fits within the K-length address space.
        // The full Rep3Memory may be much larger (e.g. 16M words for 128MB emulator capacity)
        // but ram_K only covers actual memory accesses.
        let dram_start_index = remap_address(RAM_START_ADDRESS, memory_layout).unwrap() as usize;
        let dram_words_needed = K.saturating_sub(dram_start_index);
        let dram_words_available = sm.prover_state.final_memory_state.data.len();
        let dram_convert_len = dram_words_needed.min(dram_words_available);

        let final_memory_ring: Vec<_> =
            sm.prover_state.final_memory_state.data[..dram_convert_len].to_vec();

        let dram_field: Vec<Rep3PrimeFieldShare<F>> =
            binary_ring_to_field_many(&final_memory_ring, io_ctx.main())?;

        // Build full K-length vector: start from initial state (PUBLIC→trivial), overlay DRAM
        let mut final_memory_field: Vec<Rep3PrimeFieldShare<F>> = initial_memory_state
            .par_iter()
            .map(|&x| rep3_arith::promote_to_trivial_share(party_id, F::from_u64(x)))
            .collect();

        // Overlay DRAM region with SHARED final memory values
        for (i, share) in dram_field.into_iter().enumerate() {
            final_memory_field[dram_start_index + i] = share;
        }

        // Overlay outputs (PUBLIC) — the verifier knows the expected outputs
        let mut index = remap_address(memory_layout.output_start, memory_layout).unwrap() as usize;
        for chunk in sm.program_io.outputs.chunks(8) {
            let mut word = [0u8; 8];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u64::from_le_bytes(word);
            final_memory_field[index] =
                rep3_arith::promote_to_trivial_share(party_id, F::from_u64(word));
            index += 1;
        }

        // Copy panic bit to final state
        let panic_index = remap_address(memory_layout.panic, memory_layout).unwrap() as usize;
        final_memory_field[panic_index] =
            rep3_arith::promote_to_trivial_share(party_id, F::from_u64(sm.program_io.panic as u64));
        if !sm.program_io.panic {
            let termination_index =
                remap_address(memory_layout.termination, memory_layout).unwrap() as usize;
            final_memory_field[termination_index] =
                rep3_arith::promote_to_trivial_share(party_id, F::one());
        }

        Ok(Self {
            initial_memory_state,
            final_memory_field,
            stage2: None,
            stage3: None,
        })
    }

    pub fn set_stage2_init(&mut self, gamma: F, input_claim: F, r_address: Vec<F::Challenge>) {
        self.stage2 = Some((gamma, input_claim, r_address));
    }

    pub fn set_stage3_init(&mut self, input_claim: F) {
        self.stage3 = Some(input_claim);
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> SumcheckStagesWorker<F, PCS>
    for Rep3RamDagWorker<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<Box<dyn Rep3SumcheckInstanceWorker<F>>> {
        let (gamma, input_claim, r_address) = self
            .stage2
            .take()
            .expect("Rep3RamDagWorker stage2 init not set");
        let raf_evaluation = Rep3RafEvaluationWorker::new(sm);
        let read_write_checking =
            Rep3RamReadWriteCheckingWorker::new(&self.initial_memory_state, sm, gamma, input_claim);
        let output_check = Rep3OutputSumcheckWorker::new(
            self.initial_memory_state.clone(),
            self.final_memory_field.clone(),
            r_address,
            sm,
        );

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<Box<dyn Rep3SumcheckInstanceWorker<F>>> {
        let input_claim = self
            .stage3
            .take()
            .expect("Rep3RamDagWorker stage3 init not set");
        let val_final = Rep3ValFinalSumcheckWorker::new(sm, input_claim);
        // TODO: Add ValEvaluationSumcheck and HammingBooleanity when ported
        vec![Box::new(val_final)]
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3RamDag;

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStagesCoordinator<F, ProofTranscript, PCS> for Rep3RamDag
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> {
        let raf_evaluation = Rep3RafEvaluation::new(sm);
        let read_write_checking = Rep3RamReadWriteChecking::new(sm);
        let output_check = Rep3OutputSumcheck::new(sm);

        vec![
            Box::new(raf_evaluation),
            Box::new(read_write_checking),
            Box::new(output_check),
        ]
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> {
        let val_final = Rep3ValFinalSumcheck::new(sm);
        // TODO: Add ValEvaluationSumcheck and HammingBooleanity when ported
        vec![Box::new(val_final)]
    }
}
