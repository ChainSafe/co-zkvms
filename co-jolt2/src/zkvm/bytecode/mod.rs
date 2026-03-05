use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::network::Rep3NetworkWorker;
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;
use strum::IntoEnumIterator;

use crate::field::JoltField;
use crate::zkvm::dag::stage::{
    BatchedSumcheckInstance, BatchedSumcheckWorkerInstance, SumcheckStagesWorker,
};
use crate::zkvm::dag::state_manager::{StateManager, StateManagerWorker};

pub mod booleanity;
pub mod hamming_weight;
pub mod read_raf_checking;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3BytecodeDag;

impl Rep3BytecodeDag {
    /// Create coordinator stage4 instances AND return the init data for workers.
    pub fn stage4_instances_with_init<F, ProofTranscript, PCS>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> (
        Vec<BatchedSumcheckInstance<F, ProofTranscript>>,
        BytecodeStage4Init<F>,
    )
    where
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    {
        use jolt_core::zkvm::bytecode::{
            booleanity::BooleanitySumcheck as BytecodeBooleanity,
            hamming_weight::HammingWeightSumcheck as BytecodeHammingWeight,
            read_raf_checking::ReadRafSumcheck as BytecodeReadRaf,
        };

        let K = sm.preprocessing.shared.bytecode.code_size;
        let log_K = K.log_2();
        let d = sm.preprocessing.shared.bytecode.d;
        let log_K_chunk = log_K.div_ceil(d);
        let log_T = sm.trace_length.log_2();

        // ReadRaf draws from transcript first (matching vanilla ordering).
        // The ReadRaf constructor draws gamma, then 3 sets of gamma_powers.
        // For the coordinator we replicate the verifier path.
        let read_raf_gamma: F = sm.transcript.challenge_scalar();
        let read_raf_gamma_sqr = read_raf_gamma.square();
        let read_raf_gamma_cub = read_raf_gamma_sqr * read_raf_gamma;
        let read_raf_gamma_four = read_raf_gamma_sqr.square();

        // Stage1 gamma_powers
        let _gamma_powers_1 = jolt_core::zkvm::bytecode::read_raf_checking::get_gamma_powers::<F>(
            &mut sm.transcript,
            3 + jolt_core::zkvm::instruction::NUM_CIRCUIT_FLAGS,
        );
        // Stage1 rv_claim: needs accumulator openings
        let (_, unexpanded_pc_claim_1) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, imm_claim_1) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (_, rd_claim_1) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rd, SumcheckId::SpartanOuter);
        let mut rv_claim_1 = _gamma_powers_1[0] * unexpanded_pc_claim_1
            + _gamma_powers_1[1] * imm_claim_1
            + _gamma_powers_1[2] * rd_claim_1;
        for (i, flag) in jolt_core::zkvm::instruction::CircuitFlags::iter().enumerate() {
            let (_, flag_claim) = sm.accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(flag),
                SumcheckId::SpartanOuter,
            );
            rv_claim_1 += _gamma_powers_1[3 + i] * flag_claim;
        }

        // Stage2 gamma_powers
        let _gamma_powers_2 = jolt_core::zkvm::bytecode::read_raf_checking::get_gamma_powers::<F>(
            &mut sm.transcript,
            3,
        );
        let (_, rdwa_claim_2) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1ra_claim_2) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2ra_claim_2) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let rv_claim_2 = _gamma_powers_2[0] * rdwa_claim_2
            + _gamma_powers_2[1] * rs1ra_claim_2
            + _gamma_powers_2[2] * rs2ra_claim_2;

        // Stage3 gamma_powers
        use jolt2_common::constants::XLEN;
        use jolt_core::zkvm::lookup_table::LookupTables;
        use strum::EnumCount;
        let _gamma_powers_3 = jolt_core::zkvm::bytecode::read_raf_checking::get_gamma_powers::<F>(
            &mut sm.transcript,
            4 + LookupTables::<XLEN>::COUNT,
        );
        let (_, rd_wa_claim_3) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, unexpanded_pc_claim_3) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, is_noop_claim_3) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(jolt_core::zkvm::instruction::CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
        );
        let (_, raf_flag_claim_3) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );
        let mut rv_claim_3 = _gamma_powers_3[0] * rd_wa_claim_3
            + _gamma_powers_3[1] * unexpanded_pc_claim_3
            + _gamma_powers_3[2] * is_noop_claim_3
            + _gamma_powers_3[3] * raf_flag_claim_3;
        for i in 0..LookupTables::<XLEN>::COUNT {
            let (_, lt_claim) = sm.accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
            );
            rv_claim_3 += _gamma_powers_3[4 + i] * lt_claim;
        }

        let (_, raf_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);

        let rv_claim = rv_claim_1
            + read_raf_gamma * rv_claim_2
            + read_raf_gamma_sqr * rv_claim_3
            + read_raf_gamma_cub * raf_claim
            + read_raf_gamma_four * raf_shift_claim;

        // Val polys must be computed from bytecode (public preprocessing).
        let bytecode = &sm.preprocessing.shared.bytecode.bytecode;
        let val_1 = BytecodeReadRaf::<F>::compute_val_1_from_bytecode(bytecode, &_gamma_powers_1);

        // Val2 needs eq_r_register from the accumulator.
        let r_register_2 = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersReadWriteChecking,
            )
            .0
            .r;
        let eq_r_register_2 = jolt_core::poly::eq_poly::EqPolynomial::<F>::evals(
            &r_register_2[..(jolt2_common::constants::REGISTER_COUNT as usize).log_2()],
        );
        let val_2 = BytecodeReadRaf::<F>::compute_val_2_from_bytecode(
            bytecode,
            &_gamma_powers_2,
            &eq_r_register_2,
        );

        // Val3 needs eq_r_register from a different sumcheck.
        let r_register_3 = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            )
            .0
            .r;
        let eq_r_register_3 = jolt_core::poly::eq_poly::EqPolynomial::<F>::evals(
            &r_register_3[..(jolt2_common::constants::REGISTER_COUNT as usize).log_2()],
        );
        let val_3 = BytecodeReadRaf::<F>::compute_val_3_from_bytecode(
            bytecode,
            &_gamma_powers_3,
            &eq_r_register_3,
        );

        // Compute r_cycles from accumulator (matching vanilla get_r_cycle_verif).
        use jolt2_common::constants::REGISTER_COUNT;
        let r_cycle_1 = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter)
            .0
            .r;
        let r_2 = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::Rs1Ra,
                SumcheckId::RegistersReadWriteChecking,
            )
            .0;
        let (_, r_cycle_2) = r_2.split_at_r((REGISTER_COUNT as usize).log_2());
        let r_3 = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            )
            .0;
        let (_, r_cycle_3) = r_3.split_at_r((REGISTER_COUNT as usize).log_2());
        let r_cycles = [r_cycle_1, r_cycle_2.to_vec(), r_cycle_3.to_vec()];

        let val_polys = [val_1, val_2, val_3];
        let read_raf = BytecodeReadRaf::new_verifier_from_parts(
            read_raf_gamma,
            rv_claim,
            log_K,
            log_T,
            d,
            val_polys.clone(),
        );

        // Booleanity: draws gamma from transcript
        let bool_gamma: F = sm.transcript.challenge_scalar();
        let mut bool_gamma_powers = vec![F::one(); d];
        for i in 1..d {
            bool_gamma_powers[i] = bool_gamma_powers[i - 1] * bool_gamma;
        }
        let bool_r_address: Vec<F::Challenge> =
            sm.transcript.challenge_vector_optimized::<F>(log_K_chunk);

        let booleanity = BytecodeBooleanity::new_verifier_from_parts(
            bool_gamma_powers.clone(),
            bool_r_address.clone(),
            log_T,
            log_K_chunk,
        );

        // HammingWeight: draws gamma from transcript
        let hw_gamma: F = sm.transcript.challenge_scalar();
        let mut hw_gamma_powers = vec![F::one(); d];
        for i in 1..d {
            hw_gamma_powers[i] = hw_gamma_powers[i - 1] * hw_gamma;
        }

        let hamming_weight =
            BytecodeHammingWeight::new_verifier_from_parts(hw_gamma_powers.clone(), log_K_chunk);

        let instances = vec![
            BatchedSumcheckInstance::Public(Box::new(read_raf)),
            BatchedSumcheckInstance::Public(Box::new(booleanity)),
            BatchedSumcheckInstance::Public(Box::new(hamming_weight)),
        ];

        let init = BytecodeStage4Init {
            read_raf_gamma,
            rv_claim,
            val_polys,
            r_cycles,
            bool_gamma_powers,
            bool_r_address,
            hw_gamma_powers,
        };

        (instances, init)
    }
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Init data for Bytecode stage4 instances, broadcast by coordinator.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodeStage4Init<F: JoltField> {
    // ReadRaf
    pub read_raf_gamma: F,
    pub rv_claim: F,
    pub val_polys: [Vec<F>; 3],
    pub r_cycles: [Vec<F::Challenge>; 3],
    // Booleanity
    pub bool_gamma_powers: Vec<F>,
    pub bool_r_address: Vec<F::Challenge>,
    // HammingWeight
    pub hw_gamma_powers: Vec<F>,
}

pub struct Rep3BytecodeDagWorker<F: JoltField> {
    stage4: Option<BytecodeStage4Init<F>>,
}

impl<F: JoltField> Rep3BytecodeDagWorker<F> {
    pub fn new() -> Self {
        Self { stage4: None }
    }

    pub fn set_stage4_init(&mut self, init: BytecodeStage4Init<F>) {
        self.stage4 = Some(init);
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
    SumcheckStagesWorker<F, PCS, N> for Rep3BytecodeDagWorker<F>
{
    fn stage4_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut mpc_core::protocols::rep3::network::IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        use jolt_core::poly::eq_poly::EqPolynomial;
        use jolt_core::zkvm::bytecode::{
            booleanity::BooleanitySumcheck as BytecodeBooleanity,
            hamming_weight::HammingWeightSumcheck as BytecodeHammingWeight,
            read_raf_checking::ReadRafSumcheck as BytecodeReadRaf,
        };

        let init = self
            .stage4
            .take()
            .expect("Rep3BytecodeDagWorker stage4 init not set");

        let d = sm.prover_state.preprocessing.shared.bytecode.d;
        let K = sm.prover_state.preprocessing.shared.bytecode.code_size;
        let log_K = K.log_2();
        let log_K_chunk = log_K.div_ceil(d);
        let T = sm.prover_state.cycle_witness.len();
        let log_T = T.log_2();

        let instances = if sm.party_id == PartyID::ID0 {
            // Use cycle_witness.pc which stores bytecode table indices (from get_pc).
            let meta = sm.prover_state.cycle_witness.meta();
            let (pc_indices, pc): (Vec<u64>, Vec<usize>) = meta
                .par_iter()
                .map(|m| (m.pc_index, m.pc_index as usize))
                .unzip();

            // Compute eq_r_cycle from accumulator r_cycle point
            let r_cycle: Vec<F::Challenge> = sm
                .accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::UnexpandedPC,
                    SumcheckId::SpartanOuter,
                )
                .0
                .r
                .clone();
            let eq_r_cycle: Vec<F> = EqPolynomial::evals(&r_cycle);

            // compute_ra_evals equivalent using pc_indices
            let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
            let chunk_size = (T / num_chunks).max(1);
            let K_chunk = 1 << log_K_chunk;

            // F_polys (3 eq-weighted histograms) for ReadRaf
            let eq_evals = [
                EqPolynomial::evals(&init.r_cycles[0]),
                EqPolynomial::evals(&init.r_cycles[1]),
                EqPolynomial::evals(&init.r_cycles[2]),
            ];

            let (F_1, F_polys) = compute_pc_hists::<F>(
                &pc_indices,
                &eq_r_cycle,
                &eq_evals,
                d,
                log_K_chunk,
                K_chunk,
                K,
                chunk_size,
            );

            let read_raf = BytecodeReadRaf::new_prover_from_parts(
                init.read_raf_gamma,
                init.rv_claim,
                log_K,
                log_T,
                d,
                init.val_polys,
                init.r_cycles,
                pc,
                F_polys,
            );

            let booleanity = BytecodeBooleanity::new_prover_from_pc_indices(
                init.bool_gamma_powers,
                init.bool_r_address,
                log_T,
                log_K_chunk,
                eq_r_cycle,
                F_1.clone(),
                &pc_indices,
            );

            let hamming_weight = BytecodeHammingWeight::new_prover_from_parts(
                init.hw_gamma_powers,
                log_K_chunk,
                F_1,
            );

            vec![
                BatchedSumcheckWorkerInstance::Public(Box::new(read_raf)),
                BatchedSumcheckWorkerInstance::Public(Box::new(booleanity)),
                BatchedSumcheckWorkerInstance::Public(Box::new(hamming_weight)),
            ]
        } else {
            let read_raf = BytecodeReadRaf::new_verifier_from_parts(
                init.read_raf_gamma,
                init.rv_claim,
                log_K,
                log_T,
                d,
                init.val_polys,
            );

            let booleanity = BytecodeBooleanity::new_verifier_from_parts(
                init.bool_gamma_powers,
                init.bool_r_address,
                log_T,
                log_K_chunk,
            );

            let hamming_weight =
                BytecodeHammingWeight::new_verifier_from_parts(init.hw_gamma_powers, log_K_chunk);

            vec![
                BatchedSumcheckWorkerInstance::Public(Box::new(read_raf)),
                BatchedSumcheckWorkerInstance::Public(Box::new(booleanity)),
                BatchedSumcheckWorkerInstance::Public(Box::new(hamming_weight)),
            ]
        };
        Ok(instances)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute bytecode stage4 histograms in one pass over `pc_indices`.
///
/// Returns:
/// - `G`: `d` chunk histograms of `eq_r_cycle[j]` binned by `pc_indices[j]` chunk `i`
/// - `F_polys`: `[r1, r2, r3]` where `rs[pc] = Σ_j eq_evals[s][j] * [pc_indices[j] == pc]`
fn compute_pc_hists<F: JoltField>(
    pc_indices: &[u64],
    eq_r_cycle: &[F],
    eq_evals: &[Vec<F>; 3],
    d: usize,
    log_K_chunk: usize,
    K_chunk: usize,
    K: usize,
    chunk_size: usize,
) -> (Vec<Vec<F>>, [Vec<F>; 3]) {
    use jolt_core::utils::thread::unsafe_allocate_zero_vec;

    debug_assert_eq!(pc_indices.len(), eq_r_cycle.len());
    debug_assert_eq!(eq_evals[0].len(), pc_indices.len());
    debug_assert_eq!(eq_evals[1].len(), pc_indices.len());
    debug_assert_eq!(eq_evals[2].len(), pc_indices.len());

    pc_indices
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, pcs)| {
            let mut local_G: Vec<Vec<F>> =
                (0..d).map(|_| unsafe_allocate_zero_vec(K_chunk)).collect();
            let mut r1: Vec<F> = unsafe_allocate_zero_vec(K);
            let mut r2: Vec<F> = unsafe_allocate_zero_vec(K);
            let mut r3: Vec<F> = unsafe_allocate_zero_vec(K);

            let j0 = chunk_index * chunk_size;
            for (off, &pc_u64) in pcs.iter().enumerate() {
                let j = j0 + off;
                let pc = pc_u64 as usize;

                r1[pc] += eq_evals[0][j];
                r2[pc] += eq_evals[1][j];
                r3[pc] += eq_evals[2][j];

                let mut x = pc;
                let w = eq_r_cycle[j];
                for i in (0..d).rev() {
                    let idx = x % K_chunk;
                    local_G[i][idx] += w;
                    x >>= log_K_chunk;
                }
            }

            (local_G, [r1, r2, r3])
        })
        .reduce(
            || {
                let zeros_G: Vec<Vec<F>> =
                    (0..d).map(|_| unsafe_allocate_zero_vec(K_chunk)).collect();
                let zeros_F = [
                    unsafe_allocate_zero_vec(K),
                    unsafe_allocate_zero_vec(K),
                    unsafe_allocate_zero_vec(K),
                ];
                (zeros_G, zeros_F)
            },
            |(mut running_G, mut running_F), (new_G, new_F)| {
                // NOTE: Avoid nested rayon in the reduce combiner. The outer reduction tree is
                // already parallel; nesting can oversubscribe or add overhead.
                for (x, y) in running_G.iter_mut().zip(new_G) {
                    for (x, y) in x.iter_mut().zip(y) {
                        *x += y;
                    }
                }
                let [nf0, nf1, nf2] = new_F;
                for (x, y) in running_F[0].iter_mut().zip(nf0) {
                    *x += y;
                }
                for (x, y) in running_F[1].iter_mut().zip(nf1) {
                    *x += y;
                }
                for (x, y) in running_F[2].iter_mut().zip(nf2) {
                    *x += y;
                }
                (running_G, running_F)
            },
        )
}
