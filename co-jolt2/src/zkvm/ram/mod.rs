use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_common::constants::{RAM_START_ADDRESS, RAM_WORD_SIZE};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::{compute_d_parameter, VirtualPolynomial, DTH_ROOT_OF_K};
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::binary_ring_to_field_many;
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use rayon::prelude::*;

use jolt_core::field::JoltField;
use crate::host::jolt_device::Rep3ProgramIOInput;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::mixed_polynomial::MixedPolynomial;
use crate::utils::types::Rep3Value;
use crate::zkvm::dag::stage::{BatchedSumcheckWorkerInstance, SumcheckStagesWorker};
use crate::zkvm::dag::state_manager::StateManagerWorker;

use self::output_check::{
    Rep3OutputSumcheckWorker, Rep3ValFinalSumcheckWorker,
};
use self::raf_evaluation::Rep3RafEvaluationWorker;
use self::read_write_checking::Rep3RamReadWriteCheckingWorker;

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
// Worker
// ---------------------------------------------------------------------------

pub use co_jolt_coordinator::zkvm::ram::RamStage4Init;

pub struct Rep3RamDagWorker<F: JoltField> {
    /// val_init (SHARED) — initial RAM state as a dense MLE.
    val_init: Rep3DensePolynomial<F>,
    /// val_final (MIXED) — final RAM state as an MLE with public regions kept public.
    /// This avoids materializing a full K-length `Vec<Rep3PrimeFieldShare<F>>`.
    val_final: MixedPolynomial<F>,
    stage2: Option<(F, F, Vec<F::Challenge>)>,
    stage3: Option<(F, F)>,
    stage4: Option<RamStage4Init<F>>,
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
        let ws = RAM_WORD_SIZE as usize;

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

        // Take ownership of the final-memory ring buffer and drop it immediately after
        // ring→field conversion to minimize peak RSS.
        let mut final_memory_ring = std::mem::take(&mut sm.prover_state.final_memory_state.data);
        final_memory_ring.truncate(dram_convert_len);
        let dram_field: Vec<Rep3PrimeFieldShare<F>> =
            binary_ring_to_field_many(&final_memory_ring, io_ctx.main())?;
        drop(final_memory_ring);

        // Build val_final as a mixed polynomial: keep known-public regions public,
        // only storing `Rep3PrimeFieldShare` values where the witness is secret.
        let mut final_memory_mixed: Vec<Rep3Value<F>> = vec![Rep3Value::Public(F::zero()); K];

        // Copy bytecode (PUBLIC)
        let mut index =
            remap_address(ram_preprocessing.min_bytecode_address, memory_layout).unwrap() as usize;
        for word in &ram_preprocessing.bytecode_words {
            final_memory_mixed[index] = Rep3Value::Public(F::from_u64(*word));
            index += 1;
        }

        // Trusted advice (SHARED) — reuse already-converted initial memory shares.
        let trusted_advice_start =
            remap_address(memory_layout.trusted_advice_start, memory_layout).unwrap() as usize;
        let trusted_words_len = advice.trusted_advice.len().div_ceil(ws);
        for i in 0..trusted_words_len {
            final_memory_mixed[trusted_advice_start + i] =
                Rep3Value::Shared(initial_memory_state[trusted_advice_start + i]);
        }

        // Untrusted advice (SHARED) — reuse already-converted initial memory shares.
        let untrusted_advice_start =
            remap_address(memory_layout.untrusted_advice_start, memory_layout).unwrap() as usize;
        let untrusted_words_len = advice.untrusted_advice.len().div_ceil(ws);
        for i in 0..untrusted_words_len {
            final_memory_mixed[untrusted_advice_start + i] =
                Rep3Value::Shared(initial_memory_state[untrusted_advice_start + i]);
        }

        // Copy inputs (PUBLIC)
        index = remap_address(memory_layout.input_start, memory_layout).unwrap() as usize;
        for chunk in sm.program_io.inputs.chunks(ws) {
            final_memory_mixed[index] =
                Rep3Value::Public(F::from_u64(jolt_core::zkvm::ram::bytes_to_ram_word(chunk)));
            index += 1;
        }

        // Overlay DRAM region with SHARED final memory values
        for (i, share) in dram_field.into_iter().enumerate() {
            final_memory_mixed[dram_start_index + i] = Rep3Value::Shared(share);
        }

        // Overlay outputs (PUBLIC) — the verifier knows the expected outputs
        index = remap_address(memory_layout.output_start, memory_layout).unwrap() as usize;
        for chunk in sm.program_io.outputs.chunks(ws) {
            final_memory_mixed[index] =
                Rep3Value::Public(F::from_u64(jolt_core::zkvm::ram::bytes_to_ram_word(chunk)));
            index += 1;
        }

        // Copy panic bit to final state (PUBLIC)
        let panic_index = remap_address(memory_layout.panic, memory_layout).unwrap() as usize;
        final_memory_mixed[panic_index] =
            Rep3Value::Public(F::from_u64(sm.program_io.panic as u64));
        if !sm.program_io.panic {
            let termination_index =
                remap_address(memory_layout.termination, memory_layout).unwrap() as usize;
            final_memory_mixed[termination_index] = Rep3Value::Public(F::one());
        }

        let val_init = Rep3DensePolynomial::new(initial_memory_state);
        let val_final = MixedPolynomial::new(final_memory_mixed, party_id);

        Ok(Self {
            val_init,
            val_final,
            stage2: None,
            stage3: None,
            stage4: None,
        })
    }

    pub fn set_stage2_init(&mut self, gamma: F, input_claim: F, r_address: Vec<F::Challenge>) {
        self.stage2 = Some((gamma, input_claim, r_address));
    }

    pub fn set_stage3_init(&mut self, val_final_input_claim: F, val_eval_input_claim: F) {
        self.stage3 = Some((val_final_input_claim, val_eval_input_claim));
    }

    pub fn set_stage4_init(&mut self, init: RamStage4Init<F>) {
        self.stage4 = Some(init);
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
    SumcheckStagesWorker<F, PCS, N> for Rep3RamDagWorker<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut mpc_core::protocols::rep3::network::IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        let (gamma, input_claim, r_address) = self
            .stage2
            .take()
            .expect("Rep3RamDagWorker stage2 init not set");
        let raf_evaluation = Rep3RafEvaluationWorker::new(sm);
        let read_write_checking =
            Rep3RamReadWriteCheckingWorker::new(self.val_init.coeffs_ref(), sm, gamma, input_claim);
        let output_check = Rep3OutputSumcheckWorker::new(
            self.val_init.clone(),
            self.val_final.clone(),
            r_address,
            sm,
        );

        Ok(vec![
            BatchedSumcheckWorkerInstance::Secret(Box::new(raf_evaluation)),
            BatchedSumcheckWorkerInstance::Secret(Box::new(read_write_checking)),
            BatchedSumcheckWorkerInstance::Secret(Box::new(output_check)),
        ])
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        use jolt_core::poly::opening_proof::SumcheckId;
        use jolt_core::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheck;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        let (val_final_input_claim, val_eval_input_claim) = self
            .stage3
            .take()
            .expect("Rep3RamDagWorker stage3 init not set");

        // ValEvaluation must be constructed BEFORE ValFinal because ValFinal `.take()`s ram_inc.
        let ram_inc = sm.prover_state.cycle_witness.ram_inc_ref().clone();
        let val_eval =
            val_evaluation::Rep3RamValEvaluationWorker::new(sm, val_eval_input_claim, ram_inc);
        let val_final = Rep3ValFinalSumcheckWorker::new(sm, val_final_input_claim);

        let hamming_bool: HammingBooleanitySumcheck<F> = if sm.party_id == PartyID::ID0 {
            let r_cycle = sm
                .accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupOutput,
                    SumcheckId::SpartanOuter,
                )
                .0
                .r;
            let memory_layout = &sm.program_io.memory_layout;
            let ram_addrs: Vec<u64> = sm
                .prover_state
                .cycle_witness
                .meta()
                .iter()
                .map(|m| remap_address(m.ram_addr, memory_layout).unwrap_or(0))
                .collect();
            HammingBooleanitySumcheck::new_prover_from_parts(&ram_addrs, &r_cycle)
        } else {
            let log_T = sm
                .accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupOutput,
                    SumcheckId::SpartanOuter,
                )
                .0
                .r
                .len();
            HammingBooleanitySumcheck::new_verifier_from_parts(log_T)
        };

        // Vanilla ordering: ValEvaluation, ValFinal, HammingBooleanity
        Ok(vec![
            BatchedSumcheckWorkerInstance::Secret(Box::new(val_eval)),
            BatchedSumcheckWorkerInstance::Secret(Box::new(val_final)),
            BatchedSumcheckWorkerInstance::Public(Box::new(hamming_bool)),
        ])
    }

    fn stage4_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut mpc_core::protocols::rep3::network::IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        use jolt_core::poly::dense_mlpoly::DensePolynomial;
        use jolt_core::poly::eq_poly::EqPolynomial;
        use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
        use jolt_core::poly::ra_poly::RaPolynomial;
        use jolt_core::zkvm::ram::{
            booleanity::BooleanitySumcheck as RamBooleanity,
            hamming_weight::HammingWeightSumcheck as RamHammingWeight,
            ra_virtual::RaSumcheck as RamRaSumcheck,
        };
        use std::sync::Arc;

        let init = self
            .stage4
            .take()
            .expect("Rep3RamDagWorker stage4 init not set");

        let ram_K = sm.ram_K;
        let d = compute_d_parameter(ram_K);
        let T = sm.prover_state.cycle_witness.len();

        let instances = if sm.party_id == PartyID::ID0 {
            // ID0 has trace data — create prover instances
            let memory_layout = &sm.program_io.memory_layout;

            // Pre-compute remapped addresses from cycle_witness (public data on ID0)
            let addresses: Vec<Option<u64>> = sm
                .prover_state
                .cycle_witness
                .meta()
                .par_iter()
                .map(|m| remap_address(m.ram_addr, memory_layout))
                .collect();

            // Compute F_arrays (used for HammingWeight)
            let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
            let chunk_size = (T / num_chunks).max(1);
            let dth_root_log = DTH_ROOT_OF_K.log_2();

            // HammingWeight eq_r_cycle from accumulator
            let (r_cycle_point, _) = sm.accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            );
            let eq_r_cycle = EqPolynomial::evals(&r_cycle_point.r);

            let F_arrays = compute_address_chunk_hists::<F>(
                &addresses,
                &eq_r_cycle,
                d,
                chunk_size,
                dth_root_log,
            );

            let hamming_weight = RamHammingWeight::new_prover_from_parts(
                init.hamming_gamma_powers,
                init.hamming_input_claim,
                F_arrays,
            );

            // Booleanity uses its OWN eq_r_cycle (from transcript challenge),
            // which differs from HammingWeight's r_cycle (from accumulator opening).
            let bool_eq_r_cycle = EqPolynomial::evals(&init.bool_r_cycle);
            let G_arrays = compute_address_chunk_hists::<F>(
                &addresses,
                &bool_eq_r_cycle,
                d,
                chunk_size,
                dth_root_log,
            );

            // RaSumcheck: build ra_i_polys and eq_poly from trace
            let eq_tables: Vec<Vec<F>> = init
                .ra_r_address_chunks
                .iter()
                .map(|chunk| EqPolynomial::evals(chunk))
                .collect();

            let eq_polys = [
                &EqPolynomial::<F>::evals(&init.ra_r_cycle[0]).into(),
                &EqPolynomial::<F>::evals(&init.ra_r_cycle[1]).into(),
                &EqPolynomial::<F>::evals(&init.ra_r_cycle[2]).into(),
            ];
            let eq_poly = MultilinearPolynomial::from(
                DensePolynomial::linear_combination(&eq_polys, &init.ra_gamma).Z,
            );

            let ra_i_polys: Vec<RaPolynomial<u8, F>> = (0..d)
                .into_par_iter()
                .zip(eq_tables.into_par_iter())
                .map(|(i, eq_table)| {
                    let ra_i_indices: Vec<Option<u8>> = addresses
                        .par_iter()
                        .map(|addr_opt| {
                            addr_opt.map(|address| {
                                let shift = dth_root_log * (d - 1 - i);
                                let idx = (address >> shift) % DTH_ROOT_OF_K as u64;
                                idx as u8
                            })
                        })
                        .collect();
                    RaPolynomial::new(Arc::new(ra_i_indices), eq_table)
                })
                .collect();

            let ra_virtual = RamRaSumcheck::new_prover_from_parts(
                init.ra_gamma,
                init.ra_claim,
                d,
                T,
                init.ra_r_cycle,
                init.ra_r_address_chunks,
                ra_i_polys,
                eq_poly,
            );

            // Consume addresses last (avoid cloning).
            let booleanity = RamBooleanity::new_prover_from_parts(
                d,
                T,
                ram_K,
                init.bool_r_cycle,
                init.bool_r_address,
                init.bool_gamma_powers,
                G_arrays,
                addresses,
            );

            vec![
                BatchedSumcheckWorkerInstance::Public(Box::new(hamming_weight)),
                BatchedSumcheckWorkerInstance::Public(Box::new(booleanity)),
                BatchedSumcheckWorkerInstance::Public(Box::new(ra_virtual)),
            ]
        } else {
            let hamming_weight = RamHammingWeight::new_verifier_from_parts(
                init.hamming_gamma_powers,
                init.hamming_input_claim,
            );

            let booleanity = RamBooleanity::new_verifier_from_parts(
                d,
                T,
                init.bool_r_cycle,
                init.bool_r_address,
                init.bool_gamma_powers,
            );

            let ra_virtual = RamRaSumcheck::new_verifier_from_parts(
                init.ra_gamma,
                init.ra_claim,
                d,
                T,
                init.ra_r_cycle,
                init.ra_r_address_chunks,
            );

            vec![
                BatchedSumcheckWorkerInstance::Public(Box::new(hamming_weight)),
                BatchedSumcheckWorkerInstance::Public(Box::new(booleanity)),
                BatchedSumcheckWorkerInstance::Public(Box::new(ra_virtual)),
            ]
        };
        Ok(instances)
    }
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
        .map(|(chunk_index, addr_chunk)| {
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
