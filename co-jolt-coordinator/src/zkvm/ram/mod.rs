use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt2_common::constants::{RAM_START_ADDRESS, RAM_WORD_SIZE};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::{compute_d_parameter, VirtualPolynomial, DTH_ROOT_OF_K};
use co_jolt2::zkvm::ram::RamStage4Init;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::binary_ring_to_field_many;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSlice;

use crate::field::JoltField;
use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::zkvm::dag::stage::{BatchedSumcheckInstance, SumcheckStagesCoordinator};
use crate::zkvm::dag::state_manager::StateManager;

use self::output_check::{Rep3OutputSumcheck, Rep3ValFinalSumcheck};
use self::raf_evaluation::Rep3RafEvaluation;
use self::read_write_checking::Rep3RamReadWriteChecking;

pub mod booleanity;
pub mod hamming_booleanity;
pub mod hamming_weight;
pub mod output_check;
pub mod ra_virtual;
pub mod raf_evaluation;
pub mod raf_evaluation_public;
pub mod read_write_checking;
pub mod val_evaluation;

#[allow(dead_code)]
pub fn build_initial_memory_state(
    ram_preprocessing: &jolt_core::zkvm::ram::RAMPreprocessing,
    program_io: &tracer::JoltDevice,
    K: usize,
) -> Vec<u64> {
    let memory_layout = &program_io.memory_layout;
    let ws = RAM_WORD_SIZE as usize;
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
    for chunk in program_io.trusted_advice.chunks(ws) {
        initial_memory_state[index] = jolt_core::zkvm::ram::bytes_to_ram_word(chunk);
        index += 1;
    }

    // Copy untrusted advice
    index = remap_address(memory_layout.untrusted_advice_start, memory_layout).unwrap() as usize;
    for chunk in program_io.untrusted_advice.chunks(ws) {
        initial_memory_state[index] = jolt_core::zkvm::ram::bytes_to_ram_word(chunk);
        index += 1;
    }

    // Copy inputs
    index = remap_address(memory_layout.input_start, memory_layout).unwrap() as usize;
    for chunk in program_io.inputs.chunks(ws) {
        initial_memory_state[index] = jolt_core::zkvm::ram::bytes_to_ram_word(chunk);
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
    let ws = RAM_WORD_SIZE as usize;
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
            .chunks(ws)
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
            .chunks(ws)
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
    for chunk in program_io.inputs.chunks(ws) {
        initial_memory_state[index] = rep3_arith::promote_to_trivial_share(
            party_id,
            F::from_u64(jolt_core::zkvm::ram::bytes_to_ram_word(chunk)),
        );
        index += 1;
    }

    Ok(initial_memory_state)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute `d` eq-weighted address histograms in one pass over the trace.
///
/// For each chunk `i`, computes `H[i][k] = Σ_j weights[j] * [addr_chunk_i(j) == k]`.
///
/// Mirrors vanilla-style single-pass chunk extraction (shift-by-`log_root`).
fn compute_address_chunk_hists<F: JoltField>(
    addresses: &[Option<u64>],
    weights: &[F],
    d: usize,
    chunk_size: usize,
    log_root: usize,
) -> Vec<Vec<F>> {
    use jolt_core::utils::thread::unsafe_allocate_zero_vec;

    debug_assert_eq!(addresses.len(), weights.len());
    let root = DTH_ROOT_OF_K;

    addresses
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, addr_chunk): (usize, &[Option<u64>])| {
            let mut local: Vec<Vec<F>> = (0..d).map(|_| unsafe_allocate_zero_vec(root)).collect();
            let j0 = chunk_index * chunk_size;
            for (off, addr_opt) in addr_chunk.iter().enumerate() {
                let j = j0 + off;
                let w = weights[j];
                if let Some(addr) = addr_opt {
                    let mut x = *addr;
                    for i in (0..d).rev() {
                        let idx = (x % root as u64) as usize;
                        local[i][idx] += w;
                        x >>= log_root;
                    }
                }
            }
            local
        })
        .reduce(
            || (0..d).map(|_| unsafe_allocate_zero_vec(root)).collect(),
            |mut running: Vec<Vec<F>>, new| {
                // Avoid nested rayon in the reduce combiner. The reduction tree already
                // runs in parallel; nesting can oversubscribe or add overhead.
                for (x, y) in running.iter_mut().zip(new) {
                    for (x, y) in x.iter_mut().zip(y) {
                        *x += y;
                    }
                }
                running
            },
        )
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3RamDag;

impl Rep3RamDag {
    /// Create coordinator stage4 instances AND return the init data for workers.
    pub fn stage4_instances_with_init<F, ProofTranscript, PCS>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> (
        Vec<BatchedSumcheckInstance<F, ProofTranscript>>,
        RamStage4Init<F>,
    )
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    {
        use jolt_core::zkvm::ram::{
            booleanity::BooleanitySumcheck as RamBooleanity,
            hamming_weight::HammingWeightSumcheck as RamHammingWeight,
            ra_virtual::RaSumcheck as RamRaSumcheck,
        };

        let ram_K = sm.ram_K;
        let d = compute_d_parameter(ram_K);
        let log_K = ram_K.log_2();

        // === HammingWeight: gamma from transcript, claim from accumulator ===
        let hamming_gamma: F = sm.transcript.challenge_scalar();
        let mut hamming_gamma_powers = vec![F::one(); d];
        for i in 1..d {
            hamming_gamma_powers[i] = hamming_gamma_powers[i - 1] * hamming_gamma;
        }
        let (_, hamming_booleanity_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        );
        let hamming_input_claim = hamming_booleanity_claim * hamming_gamma_powers.iter().sum::<F>();

        let hamming_weight = RamHammingWeight::new_verifier_from_parts(
            hamming_gamma_powers.clone(),
            hamming_input_claim,
        );

        // === Booleanity: r_cycle, r_address, gamma from transcript ===
        let T = sm.trace_length;
        let bool_r_cycle: Vec<F::Challenge> =
            sm.transcript.challenge_vector_optimized::<F>(T.log_2());
        let bool_r_address: Vec<F::Challenge> = sm
            .transcript
            .challenge_vector_optimized::<F>(DTH_ROOT_OF_K.log_2());
        let bool_gamma: F = sm.transcript.challenge_scalar();
        let mut bool_gamma_powers = vec![F::one(); d];
        for i in 1..d {
            bool_gamma_powers[i] = bool_gamma_powers[i - 1] * bool_gamma;
        }

        let booleanity = RamBooleanity::new_verifier_from_parts(
            d,
            T,
            bool_r_cycle.clone(),
            bool_r_address.clone(),
            bool_gamma_powers.clone(),
        );

        // === RaSumcheck: gamma from transcript, r_cycle/r_address from accumulator ===
        let (r_val, ra_claim_val) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );
        let (r_address_val, r_cycle_val) = r_val.split_at_r(log_K);

        let (r_rw, ra_claim_rw) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, r_cycle_rw) = r_rw.split_at_r(log_K);

        let (r_raf, ra_claim_raf) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        let (_, r_cycle_raf) = r_raf.split_at_r(log_K);

        let r_address = if r_address_val.len() % DTH_ROOT_OF_K.log_2() == 0 {
            r_address_val.to_vec()
        } else {
            let pad = DTH_ROOT_OF_K.log_2() - (r_address_val.len() % DTH_ROOT_OF_K.log_2());
            [&vec![F::Challenge::from(0_u128); pad], r_address_val].concat()
        };
        let r_address_chunks: Vec<Vec<F::Challenge>> = r_address
            .chunks(DTH_ROOT_OF_K.log_2())
            .map(|c| c.to_vec())
            .collect();

        let ra_gamma: F = sm.transcript.challenge_scalar();
        let ra_gamma_arr = [F::one(), ra_gamma, ra_gamma.square()];
        let combined_ra_claim = ra_gamma_arr[0] * ra_claim_val
            + ra_gamma_arr[1] * ra_claim_rw
            + ra_gamma_arr[2] * ra_claim_raf;

        let ra_virtual = RamRaSumcheck::new_verifier_from_parts(
            ra_gamma_arr,
            combined_ra_claim,
            d,
            T,
            [
                r_cycle_val.to_vec(),
                r_cycle_rw.to_vec(),
                r_cycle_raf.to_vec(),
            ],
            r_address_chunks.clone(),
        );

        let instances = vec![
            BatchedSumcheckInstance::Public(Box::new(hamming_weight)),
            BatchedSumcheckInstance::Public(Box::new(booleanity)),
            BatchedSumcheckInstance::Public(Box::new(ra_virtual)),
        ];

        let init = RamStage4Init {
            hamming_gamma_powers,
            hamming_input_claim,
            bool_r_cycle,
            bool_r_address,
            bool_gamma_powers,
            ra_gamma: ra_gamma_arr,
            ra_claim: combined_ra_claim,
            ra_r_cycle: [
                r_cycle_val.to_vec(),
                r_cycle_rw.to_vec(),
                r_cycle_raf.to_vec(),
            ],
            ra_r_address_chunks: r_address_chunks,
        };

        (instances, init)
    }
}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>, N>
    SumcheckStagesCoordinator<F, ProofTranscript, PCS, N> for Rep3RamDag
where
    N: mpc_core::protocols::rep3::network::Rep3NetworkCoordinator,
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        let raf_evaluation = Rep3RafEvaluation::new(sm);
        let read_write_checking = Rep3RamReadWriteChecking::new(sm);
        let output_check = Rep3OutputSumcheck::new(sm);

        Ok(vec![
            BatchedSumcheckInstance::Secret(Box::new(raf_evaluation)),
            BatchedSumcheckInstance::Secret(Box::new(read_write_checking)),
            BatchedSumcheckInstance::Secret(Box::new(output_check)),
        ])
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _network: &mut N,
    ) -> Result<Vec<BatchedSumcheckInstance<F, ProofTranscript>>, eyre::Report> {
        use jolt_core::poly::multilinear_polynomial::{
            MultilinearPolynomial, PolynomialEvaluation,
        };
        use jolt_core::utils::math::Math;
        use jolt_core::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheck;

        // Compute init_eval from public initial_ram_state for ValEvaluation input_claim
        let initial_ram_state =
            build_initial_memory_state(&sm.preprocessing.shared.ram, &sm.program_io, sm.ram_K);
        let (opening_point, _) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, _) = opening_point.split_at(sm.ram_K.log_2());
        let val_init_poly: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state);
        let init_eval = val_init_poly.evaluate(&r_address.r);

        let val_eval =
            val_evaluation::Rep3RamValEvaluation::<F>::new::<ProofTranscript, PCS>(sm, init_eval);
        let val_final = Rep3ValFinalSumcheck::new(sm);
        let log_T = sm.trace_length.log_2();
        let hamming_bool = HammingBooleanitySumcheck::<F>::new_verifier_from_parts(log_T);

        // Vanilla ordering: ValEvaluation, ValFinal, HammingBooleanity
        Ok(vec![
            BatchedSumcheckInstance::Secret(Box::new(val_eval)),
            BatchedSumcheckInstance::Secret(Box::new(val_final)),
            BatchedSumcheckInstance::Public(Box::new(hamming_bool)),
        ])
    }
}
