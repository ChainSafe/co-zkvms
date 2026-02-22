use jolt2_common::constants::RAM_START_ADDRESS;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::ram::remap_address;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::binary_ring_to_field_many;
use mpc_core::protocols::rep3_ring::Rep3RingShare;

use crate::field::JoltField;
use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::zkvm::dag::stage::{
    BatchedSumcheckInstance, BatchedSumcheckWorkerInstance, Rep3SumcheckInstance,
    Rep3SumcheckInstanceWorker, SumcheckStagesCoordinator, SumcheckStagesWorker,
};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

use self::output_check::{
    Rep3OutputSumcheck, Rep3OutputSumcheckWorker, Rep3ValFinalSumcheck, Rep3ValFinalSumcheckWorker,
};
use self::raf_evaluation::{Rep3RafEvaluation, Rep3RafEvaluationWorker};
use self::read_write_checking::{Rep3RamReadWriteChecking, Rep3RamReadWriteCheckingWorker};

pub mod hamming_booleanity;
pub mod output_check;
pub mod raf_evaluation;
pub mod read_write_checking;

#[allow(dead_code)]
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

/// Build the initial memory state as secret-shared field elements.
///
/// Public regions (bytecode, inputs) use trivial shares. Advice regions
/// (trusted, untrusted) are packed from `Rep3RingShare<u8>` byte shares into
/// `Rep3RingShare<u64>` word shares, then converted to field shares via
/// `binary_ring_to_field_many` (one MPC round).
pub(crate) fn build_initial_memory_state_shared<F: JoltField, N: Rep3NetworkWorker>(
    ram_preprocessing: &jolt_core::zkvm::ram::RAMPreprocessing,
    program_io: &tracer::JoltDevice,
    advice: &Rep3ProgramIOInput,
    party_id: PartyID,
    K: usize,
    io_ctx: &mut IoContextPool<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let memory_layout = &program_io.memory_layout;
    let mut initial_memory_state: Vec<Rep3PrimeFieldShare<F>> =
        vec![Rep3PrimeFieldShare::zero_share(); K];

    // Copy bytecode (PUBLIC → trivial shares)
    let mut index =
        remap_address(ram_preprocessing.min_bytecode_address, memory_layout).unwrap() as usize;
    for word in &ram_preprocessing.bytecode_words {
        initial_memory_state[index] =
            rep3_arith::promote_to_trivial_share(party_id, F::from_u64(*word));
        index += 1;
    }

    // Pack and convert trusted advice (SHARED)
    let trusted_advice_start =
        remap_address(memory_layout.trusted_advice_start, memory_layout).unwrap() as usize;
    if !advice.trusted_advice.is_empty() {
        let trusted_words: Vec<Rep3RingShare<u64>> = advice
            .trusted_advice
            .chunks(8)
            .map(|chunk| Rep3RingShare::<u64>::from_le_bytes(chunk))
            .collect();
        let trusted_field: Vec<Rep3PrimeFieldShare<F>> =
            binary_ring_to_field_many(&trusted_words, io_ctx.main())?;
        for (i, share) in trusted_field.into_iter().enumerate() {
            initial_memory_state[trusted_advice_start + i] = share;
        }
    }

    // Pack and convert untrusted advice (SHARED)
    let untrusted_advice_start =
        remap_address(memory_layout.untrusted_advice_start, memory_layout).unwrap() as usize;
    if !advice.untrusted_advice.is_empty() {
        let untrusted_words: Vec<Rep3RingShare<u64>> = advice
            .untrusted_advice
            .chunks(8)
            .map(|chunk| Rep3RingShare::<u64>::from_le_bytes(chunk))
            .collect();
        let untrusted_field: Vec<Rep3PrimeFieldShare<F>> =
            binary_ring_to_field_many(&untrusted_words, io_ctx.main())?;
        for (i, share) in untrusted_field.into_iter().enumerate() {
            initial_memory_state[untrusted_advice_start + i] = share;
        }
    }

    // Copy inputs (PUBLIC → trivial shares)
    index = remap_address(memory_layout.input_start, memory_layout).unwrap() as usize;
    for chunk in program_io.inputs.chunks(8) {
        let mut word = [0u8; 8];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        initial_memory_state[index] =
            rep3_arith::promote_to_trivial_share(party_id, F::from_u64(u64::from_le_bytes(word)));
        index += 1;
    }

    Ok(initial_memory_state)
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3RamDagWorker<F: JoltField> {
    /// Initial memory state (bytecode/inputs as trivial shares, advice as real shares)
    initial_memory_state: Vec<Rep3PrimeFieldShare<F>>,
    /// SHARED final memory state (converted from Rep3Memory ring shares → field shares)
    final_memory_field: Vec<Rep3PrimeFieldShare<F>>,
    stage2: Option<(F, F, Vec<F::Challenge>)>,
    stage3: Option<F>,
}

impl<F: JoltField> Rep3RamDagWorker<F> {
    /// Build initial memory state (with shared advice) and convert final memory
    /// from `Rep3Memory` binary ring shares to `Rep3PrimeFieldShare<F>`.
    ///
    /// The ring→field conversion requires MPC communication rounds.
    pub fn new<PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>(
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<Self> {
        let ram_preprocessing = &sm.prover_state.preprocessing.shared.ram;
        let K = sm.ram_K;
        let memory_layout = &sm.program_io.memory_layout;
        let party_id = sm.party_id;

        // --- Build initial_memory_state with shared advice ---
        let advice = sm
            .advice_shares
            .as_ref()
            .expect("advice_shares must be set on StateManagerWorker for RAM");
        let initial_memory_state = build_initial_memory_state_shared(
            ram_preprocessing,
            &sm.program_io,
            advice,
            party_id,
            K,
            io_ctx,
        )?;

        // --- Convert final memory: Rep3Memory (binary ring shares) → field shares ---
        // Only convert the portion of DRAM that fits within the K-length address space.
        let dram_start_index = remap_address(RAM_START_ADDRESS, memory_layout).unwrap() as usize;
        let dram_words_needed = K.saturating_sub(dram_start_index);
        let dram_words_available = sm.prover_state.final_memory_state.data.len();
        let dram_convert_len = dram_words_needed.min(dram_words_available);

        let final_memory_ring: Vec<_> =
            sm.prover_state.final_memory_state.data[..dram_convert_len].to_vec();

        let dram_field: Vec<Rep3PrimeFieldShare<F>> =
            binary_ring_to_field_many(&final_memory_ring, io_ctx.main())?;

        // Start from initial state (already has shared advice), overlay DRAM
        let mut final_memory_field = initial_memory_state.clone();

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
    ) -> Vec<BatchedSumcheckWorkerInstance<F>> {
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
            BatchedSumcheckWorkerInstance::Secret(Box::new(raf_evaluation)),
            BatchedSumcheckWorkerInstance::Secret(Box::new(read_write_checking)),
            BatchedSumcheckWorkerInstance::Secret(Box::new(output_check)),
        ]
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F>> {
        let input_claim = self
            .stage3
            .take()
            .expect("Rep3RamDagWorker stage3 init not set");
        let val_final = Rep3ValFinalSumcheckWorker::new(sm, input_claim);
        // TODO: Add ValEvaluationSumcheck and HammingBooleanity when ported
        vec![BatchedSumcheckWorkerInstance::Secret(Box::new(val_final))]
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
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        let raf_evaluation = Rep3RafEvaluation::new(sm);
        let read_write_checking = Rep3RamReadWriteChecking::new(sm);
        let output_check = Rep3OutputSumcheck::new(sm);

        vec![
            BatchedSumcheckInstance::Secret(Box::new(raf_evaluation)),
            BatchedSumcheckInstance::Secret(Box::new(read_write_checking)),
            BatchedSumcheckInstance::Secret(Box::new(output_check)),
        ]
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        let val_final = Rep3ValFinalSumcheck::new(sm);
        // TODO: Add ValEvaluationSumcheck and HammingBooleanity when ported
        vec![BatchedSumcheckInstance::Secret(Box::new(val_final))]
    }
}
