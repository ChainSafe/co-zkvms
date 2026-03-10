use std::marker::PhantomData;

use strum::{EnumCount, IntoEnumIterator};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{OpeningPoint, SumcheckId};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::{ProofData, ProofKeys, StateManager};
use crate::zkvm::r1cs::inputs::{ALL_R1CS_INPUTS, COMMITTED_R1CS_INPUTS};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use common::constants::XLEN;

// ---------------------------------------------------------------------------
// SpartanDag
// ---------------------------------------------------------------------------

pub struct SpartanDag<F: JoltField> {
    padded_trace_length: usize,
    _marker: PhantomData<F>,
}

impl<F: JoltField> SpartanDag<F> {
    pub fn new<ProofTranscript: Transcript>(padded_trace_length: usize) -> Self {
        Self {
            padded_trace_length,
            _marker: PhantomData,
        }
    }

    pub fn stage1_verify<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        let key = UniformSpartanKey::<F>::new(self.padded_trace_length);
        let num_rounds_x = key.num_rows_bits();

        let tau: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(num_rounds_x);

        // Get stage1 proof
        let proofs = sm.proofs.borrow();
        let proof_data = proofs
            .get(&ProofKeys::Stage1Sumcheck)
            .expect("Stage 1 sumcheck proof not found");
        let proof = match proof_data {
            ProofData::SumcheckProof(p) => p,
            _ => return Err(anyhow::anyhow!("Invalid proof type for stage 1")),
        };

        // Verify the outer sumcheck
        let (final_eval, r) = proof.verify(
            F::zero(), // initial claim is 0
            num_rounds_x,
            3, // degree 3: Az(x)*Bz(x)*Cz(x) with eq folded in
            &mut *sm.transcript.borrow_mut(),
        )?;

        // Reverse r (outer sumcheck binds from top)
        let outer_sumcheck_r: Vec<F::Challenge> = r.into_iter().rev().collect();

        // Compute eq(tau, r)
        let eq_eval = EqPolynomial::<F>::mle(&tau, &outer_sumcheck_r);

        // Get Az/Bz/Cz claims from accumulator
        let accumulator = sm.get_verifier_accumulator();
        let acc = accumulator.borrow();
        let claim_az = acc
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter)
            .1;
        let claim_bz = acc
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter)
            .1;
        let claim_cz = acc
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter)
            .1;
        drop(acc);

        // Verify: final_eval == eq(tau, r) * (Az * Bz - Cz)
        let expected = eq_eval * (claim_az * claim_bz - claim_cz);
        if final_eval != expected {
            return Err(anyhow::anyhow!(
                "Spartan outer sumcheck final eval mismatch"
            ));
        }

        // Append Az/Bz/Cz to transcript
        sm.transcript
            .borrow_mut()
            .append_scalars(&[claim_az, claim_bz, claim_cz]);

        // Store virtual openings with opening points
        let opening_point = OpeningPoint::new(outer_sumcheck_r.clone());
        {
            let mut acc = accumulator.borrow_mut();
            let transcript = &mut *sm.transcript.borrow_mut();

            acc.append_virtual(
                transcript,
                VirtualPolynomial::SpartanAz,
                SumcheckId::SpartanOuter,
                opening_point.clone(),
            );
            acc.append_virtual(
                transcript,
                VirtualPolynomial::SpartanBz,
                SumcheckId::SpartanOuter,
                opening_point.clone(),
            );
            acc.append_virtual(
                transcript,
                VirtualPolynomial::SpartanCz,
                SumcheckId::SpartanOuter,
                opening_point,
            );
        }

        // Compute r_cycle and append committed/virtual openings
        let num_steps_bits = key.num_steps.log_2();
        let (r_cycle, _) = outer_sumcheck_r.split_at(num_steps_bits);

        // Append committed openings (PCS)
        let committed_polys: Vec<CommittedPolynomial> = COMMITTED_R1CS_INPUTS
            .iter()
            .map(|input| CommittedPolynomial::try_from(input).ok().unwrap())
            .collect();
        {
            let mut acc = accumulator.borrow_mut();
            acc.append_dense(
                &mut *sm.transcript.borrow_mut(),
                committed_polys,
                SumcheckId::SpartanOuter,
                r_cycle.to_vec(),
            );
        }

        // Append virtual openings for remaining R1CS inputs
        for input in ALL_R1CS_INPUTS.iter() {
            if COMMITTED_R1CS_INPUTS.contains(input) {
                continue;
            }
            let poly = VirtualPolynomial::try_from(input).ok().unwrap();
            let mut acc = accumulator.borrow_mut();
            acc.append_virtual(
                &mut *sm.transcript.borrow_mut(),
                poly,
                SumcheckId::SpartanOuter,
                OpeningPoint::new(r_cycle.to_vec()),
            );
        }

        drop(proofs);
        Ok(())
    }

    pub fn stage2_verifier_instances<
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::spartan::inner::InnerSumcheck;
        let inner = InnerSumcheck::new_verifier::<ProofTranscript, PCS>(sm);
        vec![Box::new(inner)]
    }

    pub fn stage3_verifier_instances<
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::spartan::pc::PCSumcheck;
        use crate::zkvm::spartan::product::ProductVirtualizationSumcheck;

        let accumulator = sm.get_verifier_accumulator();
        let acc = accumulator.borrow();

        let gamma_pc: F = sm.transcript.borrow_mut().challenge_scalar();
        let (r_cycle_point, next_pc_eval) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let (_, next_unexpanded_pc_eval) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::NextUnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, next_is_noop_eval) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::NextIsNoop,
            SumcheckId::SpartanOuter,
        );
        drop(acc);

        let input_claim_pc =
            next_unexpanded_pc_eval + gamma_pc * next_pc_eval + gamma_pc.square() * next_is_noop_eval;
        let spartan_pc = PCSumcheck::<F>::new_verifier_from_openings(
            input_claim_pc,
            gamma_pc,
            r_cycle_point.r.len(),
        );
        let spartan_product = ProductVirtualizationSumcheck::<F>::new_verifier(sm);

        vec![Box::new(spartan_pc), Box::new(spartan_product)]
    }
}

// ---------------------------------------------------------------------------
// RegistersDag
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct RegistersDag;

impl RegistersDag {
    pub fn stage2_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::registers::read_write_checking::RegistersReadWriteChecking;
        let rwc = RegistersReadWriteChecking::new_verifier::<ProofTranscript, PCS>(sm);
        vec![Box::new(rwc)]
    }

    pub fn stage3_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::registers::val_evaluation::ValEvaluationSumcheck;
        let val_eval = ValEvaluationSumcheck::new_verifier::<ProofTranscript, PCS>(sm);
        vec![Box::new(val_eval)]
    }
}

// ---------------------------------------------------------------------------
// RamDag
// ---------------------------------------------------------------------------

pub struct RamDag {
    initial_ram_state: Vec<u64>,
}

impl RamDag {
    pub fn new_verifier<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let initial_ram_state = crate::zkvm::ram::build_initial_memory_state(
            &sm.preprocessing.shared.ram,
            &sm.program_io,
            sm.ram_K,
        );
        Self { initial_ram_state }
    }

    pub fn stage2_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::ram::output_check::OutputSumcheck;
        use crate::zkvm::ram::raf_evaluation::RafEvaluationSumcheck;
        use crate::zkvm::ram::read_write_checking::RamReadWriteChecking;

        let K = sm.ram_K;
        let accumulator = sm.get_verifier_accumulator();
        let raf_claim = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .1;
        let start_address = sm.preprocessing.shared.ram.min_bytecode_address;
        let ra_claim = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation)
            .1;
        let raf = RafEvaluationSumcheck::new_verifier_from_parts(
            raf_claim,
            K.log_2(),
            sm.program_io.memory_layout.trusted_advice_start,
            ra_claim,
        );
        let rwc = RamReadWriteChecking::new_verifier::<ProofTranscript, PCS>(sm);
        let output = OutputSumcheck::new_verifier::<ProofTranscript, PCS>(sm);

        vec![Box::new(raf), Box::new(rwc), Box::new(output)]
    }

    pub fn stage3_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheck;
        use crate::zkvm::ram::output_check::ValFinalSumcheck;
        use crate::zkvm::ram::val_evaluation::ValEvaluationSumcheck;

        let val_eval = ValEvaluationSumcheck::new_verifier::<ProofTranscript, PCS>(
            &self.initial_ram_state,
            sm,
        );
        let val_final = ValFinalSumcheck::new_verifier::<ProofTranscript, PCS>(
            &self.initial_ram_state,
            sm,
        );
        let log_T = sm.trace_length.log_2();
        let hamming_bool = HammingBooleanitySumcheck::<F>::new_verifier_from_parts(log_T);

        vec![
            Box::new(val_eval),
            Box::new(val_final),
            Box::new(hamming_bool),
        ]
    }

    pub fn stage4_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::ram::booleanity::BooleanitySumcheck;
        use crate::zkvm::ram::hamming_weight::HammingWeightSumcheck;
        use crate::zkvm::ram::ra_virtual::RaSumcheck;
        use crate::zkvm::witness::{compute_d_parameter, DTH_ROOT_OF_K};

        let ram_K = sm.ram_K;
        let d = compute_d_parameter(ram_K);
        let log_K = ram_K.log_2();
        let T = sm.trace_length;

        // HammingWeight
        let hamming_gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut hamming_gamma_powers = vec![F::one(); d];
        for i in 1..d {
            hamming_gamma_powers[i] = hamming_gamma_powers[i - 1] * hamming_gamma;
        }
        let accumulator = sm.get_verifier_accumulator();
        let (_, hamming_booleanity_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            );
        let hamming_input_claim = hamming_booleanity_claim * hamming_gamma_powers.iter().sum::<F>();
        let hamming_weight =
            HammingWeightSumcheck::new_verifier_from_parts(hamming_gamma_powers, hamming_input_claim);

        // Booleanity
        let bool_r_cycle: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(T.log_2());
        let bool_r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(DTH_ROOT_OF_K.log_2());
        let bool_gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut bool_gamma_powers = vec![F::one(); d];
        for i in 1..d {
            bool_gamma_powers[i] = bool_gamma_powers[i - 1] * bool_gamma;
        }
        let booleanity =
            BooleanitySumcheck::new_verifier_from_parts(d, T, bool_r_cycle, bool_r_address, bool_gamma_powers);

        // RaSumcheck
        let acc = accumulator.borrow();
        let (r_val, ra_claim_val) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );
        let (r_address_val, r_cycle_val) = r_val.split_at_r(log_K);
        let (r_rw, ra_claim_rw) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, r_cycle_rw) = r_rw.split_at_r(log_K);
        let (r_raf, ra_claim_raf) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        let (_, r_cycle_raf) = r_raf.split_at_r(log_K);
        drop(acc);

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

        let ra_gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let ra_gamma_arr = [F::one(), ra_gamma, ra_gamma.square()];
        let combined_ra_claim =
            ra_gamma_arr[0] * ra_claim_val + ra_gamma_arr[1] * ra_claim_rw + ra_gamma_arr[2] * ra_claim_raf;

        let ra_virtual = RaSumcheck::new_verifier_from_parts(
            ra_gamma_arr,
            combined_ra_claim,
            d,
            T,
            [
                r_cycle_val.to_vec(),
                r_cycle_rw.to_vec(),
                r_cycle_raf.to_vec(),
            ],
            r_address_chunks,
        );

        vec![
            Box::new(hamming_weight),
            Box::new(booleanity),
            Box::new(ra_virtual),
        ]
    }
}

// ---------------------------------------------------------------------------
// LookupsDag
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct LookupsDag;

impl LookupsDag {
    pub fn stage2_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::instruction_lookups::booleanity::BooleanitySumcheck;
        use crate::zkvm::instruction_lookups::{D, LOG_K_CHUNK};

        let accumulator = sm.get_verifier_accumulator();
        let log_T = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .len();

        // Draw gamma and r_address from transcript (matches coordinator)
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(LOG_K_CHUNK);

        let booleanity =
            BooleanitySumcheck::new_verifier_from_parts(gamma_powers, r_address, log_T);

        vec![Box::new(booleanity)]
    }

    pub fn stage3_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::instruction_lookups::hamming_weight::HammingWeightSumcheck;
        use crate::zkvm::instruction_lookups::read_raf_checking::ReadRafSumcheck;
        use crate::zkvm::instruction_lookups::D;

        let accumulator = sm.get_verifier_accumulator();
        let acc = accumulator.borrow();

        let (_, rv_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_operand_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let (_, right_operand_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let log_T = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .len();
        drop(acc);

        let read_raf = ReadRafSumcheck::new_verifier(
            &mut *sm.transcript.borrow_mut(),
            rv_claim,
            left_operand_claim,
            right_operand_claim,
            log_T,
        );

        // HammingWeight: draw gamma from transcript
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let hamming_weight = HammingWeightSumcheck::new_verifier_from_parts(gamma_powers);

        vec![Box::new(read_raf), Box::new(hamming_weight)]
    }

    pub fn stage4_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::instruction_lookups::ra_virtual::InstructionRaSumcheck;
        use crate::zkvm::instruction_lookups::{D, LOG_K_CHUNK};

        let accumulator = sm.get_verifier_accumulator();
        let acc = accumulator.borrow();

        let (ra_point, ra_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        let (r_address, r_cycle) = ra_point.r.split_at(D * LOG_K_CHUNK);
        let r_address_chunks: Vec<Vec<F::Challenge>> =
            r_address.chunks(LOG_K_CHUNK).map(|c| c.to_vec()).collect();
        drop(acc);

        let ra = InstructionRaSumcheck::new(ra_claim, r_cycle.to_vec(), r_address_chunks);
        vec![Box::new(ra)]
    }
}

// ---------------------------------------------------------------------------
// BytecodeDag
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct BytecodeDag;

impl BytecodeDag {
    pub fn stage4_verifier_instances<
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        &mut self,
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F, ProofTranscript>>> {
        use crate::zkvm::bytecode::booleanity::BooleanitySumcheck as BytecodeBooleanity;
        use crate::zkvm::bytecode::hamming_weight::HammingWeightSumcheck as BytecodeHammingWeight;
        use crate::zkvm::bytecode::read_raf_checking::ReadRafSumcheck as BytecodeReadRaf;
        use crate::zkvm::instruction::CircuitFlags;
        use crate::zkvm::lookup_table::LookupTables;
        use strum::EnumCount;

        let K = sm.preprocessing.shared.bytecode.code_size;
        let log_K = K.log_2();
        let d = sm.preprocessing.shared.bytecode.d;
        let log_K_chunk = log_K.div_ceil(d);
        let log_T = sm.trace_length.log_2();

        let accumulator = sm.get_verifier_accumulator();

        // ReadRaf: draw gamma from transcript
        let read_raf_gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let read_raf_gamma_sqr = read_raf_gamma.square();
        let read_raf_gamma_cub = read_raf_gamma_sqr * read_raf_gamma;
        let read_raf_gamma_four = read_raf_gamma_sqr.square();

        // Stage1 gamma_powers + rv_claim
        let gamma_powers_1 = crate::zkvm::bytecode::read_raf_checking::get_gamma_powers::<F>(
            &mut *sm.transcript.borrow_mut(),
            3 + crate::zkvm::instruction::NUM_CIRCUIT_FLAGS,
        );
        let acc = accumulator.borrow();
        let (_, unexpanded_pc_claim_1) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
        );
        let (_, imm_claim_1) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (_, rd_claim_1) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::Rd, SumcheckId::SpartanOuter);
        let mut rv_claim_1 = gamma_powers_1[0] * unexpanded_pc_claim_1
            + gamma_powers_1[1] * imm_claim_1
            + gamma_powers_1[2] * rd_claim_1;
        for (i, flag) in CircuitFlags::iter().enumerate() {
            let (_, flag_claim) = acc.get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(flag),
                SumcheckId::SpartanOuter,
            );
            rv_claim_1 += gamma_powers_1[3 + i] * flag_claim;
        }
        drop(acc);

        // Stage2 gamma_powers
        let gamma_powers_2 = crate::zkvm::bytecode::read_raf_checking::get_gamma_powers::<F>(
            &mut *sm.transcript.borrow_mut(),
            3,
        );
        let acc = accumulator.borrow();
        let (_, rdwa_claim_2) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1ra_claim_2) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2ra_claim_2) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let rv_claim_2 = gamma_powers_2[0] * rdwa_claim_2
            + gamma_powers_2[1] * rs1ra_claim_2
            + gamma_powers_2[2] * rs2ra_claim_2;

        // Stage3 gamma_powers
        drop(acc);
        let gamma_powers_3 = crate::zkvm::bytecode::read_raf_checking::get_gamma_powers::<F>(
            &mut *sm.transcript.borrow_mut(),
            4 + LookupTables::<XLEN>::COUNT,
        );
        let acc = accumulator.borrow();
        let (_, rd_wa_claim_3) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, unexpanded_pc_claim_3) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, is_noop_claim_3) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
        );
        let (_, raf_flag_claim_3) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
        );
        let mut rv_claim_3 = gamma_powers_3[0] * rd_wa_claim_3
            + gamma_powers_3[1] * unexpanded_pc_claim_3
            + gamma_powers_3[2] * is_noop_claim_3
            + gamma_powers_3[3] * raf_flag_claim_3;
        for i in 0..LookupTables::<XLEN>::COUNT {
            let (_, lt_claim) = acc.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
            );
            rv_claim_3 += gamma_powers_3[4 + i] * lt_claim;
        }

        let (_, raf_claim) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        drop(acc);

        let rv_claim = rv_claim_1
            + read_raf_gamma * rv_claim_2
            + read_raf_gamma_sqr * rv_claim_3
            + read_raf_gamma_cub * raf_claim
            + read_raf_gamma_four * raf_shift_claim;

        // Compute val polynomials from bytecode preprocessing
        let bytecode = &sm.preprocessing.shared.bytecode.bytecode;
        let val_1 = BytecodeReadRaf::<F>::compute_val_1_from_bytecode(bytecode, &gamma_powers_1);

        // Val2 needs eq_r_register
        let acc = accumulator.borrow();
        let r_register_2 = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersReadWriteChecking,
            )
            .0
            .r;
        drop(acc);
        let eq_r_register_2 = EqPolynomial::<F>::evals(
            &r_register_2[..(common::constants::REGISTER_COUNT as usize).log_2()],
        );
        let val_2 =
            BytecodeReadRaf::<F>::compute_val_2_from_bytecode(bytecode, &gamma_powers_2, &eq_r_register_2);

        // Val3 needs eq_r_register from val evaluation
        let acc = accumulator.borrow();
        let r_register_3 = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            )
            .0
            .r;
        drop(acc);
        let eq_r_register_3 = EqPolynomial::<F>::evals(
            &r_register_3[..(common::constants::REGISTER_COUNT as usize).log_2()],
        );
        let val_3 =
            BytecodeReadRaf::<F>::compute_val_3_from_bytecode(bytecode, &gamma_powers_3, &eq_r_register_3);

        // r_cycles from accumulator
        let acc = accumulator.borrow();
        let r_cycle_1 = acc
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter)
            .0
            .r;
        let r_2 = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::Rs1Ra,
                SumcheckId::RegistersReadWriteChecking,
            )
            .0;
        let (_, r_cycle_2) = r_2.split_at_r((common::constants::REGISTER_COUNT as usize).log_2());
        let r_3 = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RdWa,
                SumcheckId::RegistersValEvaluation,
            )
            .0;
        let (_, r_cycle_3) = r_3.split_at_r((common::constants::REGISTER_COUNT as usize).log_2());
        drop(acc);

        let val_polys = [val_1, val_2, val_3];
        let read_raf = BytecodeReadRaf::new_verifier_from_parts(
            read_raf_gamma,
            rv_claim,
            log_K,
            log_T,
            d,
            val_polys,
        );

        // Booleanity
        let bool_gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut bool_gamma_powers = vec![F::one(); d];
        for i in 1..d {
            bool_gamma_powers[i] = bool_gamma_powers[i - 1] * bool_gamma;
        }
        let bool_r_address: Vec<F::Challenge> = sm
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(log_K_chunk);
        let booleanity = BytecodeBooleanity::new_verifier_from_parts(
            bool_gamma_powers,
            bool_r_address,
            log_T,
            log_K_chunk,
        );

        // HammingWeight
        let hw_gamma: F = sm.transcript.borrow_mut().challenge_scalar();
        let mut hw_gamma_powers = vec![F::one(); d];
        for i in 1..d {
            hw_gamma_powers[i] = hw_gamma_powers[i - 1] * hw_gamma;
        }
        let hamming_weight =
            BytecodeHammingWeight::new_verifier_from_parts(hw_gamma_powers, log_K_chunk);

        vec![
            Box::new(read_raf),
            Box::new(booleanity),
            Box::new(hamming_weight),
        ]
    }
}
