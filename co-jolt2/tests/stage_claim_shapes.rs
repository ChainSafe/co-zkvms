use ark_bn254::Fr;
use ark_ff::One;
use jolt2_common::constants::RAM_START_ADDRESS;
use jolt2_common::jolt_device::MemoryConfig;
use jolt_core::poly::commitment::mock::MockCommitScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::program_io_polynomial::ProgramIOPolynomial;
use jolt_core::poly::range_mask_polynomial::RangeMaskPolynomial;
use jolt_core::transcripts::KeccakTranscript;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::{remap_address, RAMPreprocessing};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use jolt_core::zkvm::{
    JoltProverPreprocessing, JoltSharedPreprocessing, JoltVerifierPreprocessing,
};
use tracer::instruction::add::ADD;
use tracer::instruction::Instruction;
use tracer::JoltDevice;

use co_jolt2::subprotocols::sumcheck::Rep3SumcheckInstance;
use co_jolt2::zkvm::dag::state_manager::StateManager;
use co_jolt2::zkvm::ram::output_check::{Rep3OutputSumcheck, Rep3ValFinalSumcheck};
use co_jolt2::zkvm::ram::raf_evaluation::Rep3RafEvaluation;
use co_jolt2::zkvm::ram::read_write_checking::Rep3RamReadWriteChecking;
use co_jolt2::zkvm::registers::read_write_checking::Rep3RegistersReadWriteChecking;
use co_jolt2::zkvm::registers::val_evaluation::Rep3ValEvaluation;

type F = Fr;
type PCS = MockCommitScheme<F>;
type ProofTranscript = KeccakTranscript;

fn make_state_manager(
    trace_length: usize,
    ram_k: usize,
) -> StateManager<'static, F, ProofTranscript, PCS> {
    let mut mem_cfg = MemoryConfig::default();
    mem_cfg.program_size = Some(0);
    let program_io = JoltDevice::new(&mem_cfg);

    let mut dummy_add = ADD::default();
    dummy_add.address = RAM_START_ADDRESS;
    let bytecode = vec![Instruction::ADD(dummy_add)];

    let shared = JoltSharedPreprocessing {
        memory_layout: program_io.memory_layout.clone(),
        bytecode: BytecodePreprocessing::preprocess(bytecode),
        ram: RAMPreprocessing::preprocess(vec![(RAM_START_ADDRESS, 0u8)]),
    };
    let prover_preprocessing = JoltProverPreprocessing::<F, PCS> {
        generators: (),
        shared,
    };

    // Leak preprocessing to satisfy the coordinator's `'a` lifetime without carrying
    // the whole struct graph through test return types.
    let verifier_preprocessing: &'static JoltVerifierPreprocessing<F, PCS> = Box::leak(Box::new(
        JoltVerifierPreprocessing::from(&prover_preprocessing),
    ));

    let mut sm = StateManager::<F, ProofTranscript, PCS>::new(
        verifier_preprocessing,
        program_io,
        ram_k,
        0, // twist_sumcheck_switch_index
    );
    sm.trace_length = trace_length;
    sm.fiat_shamir_preamble(trace_length);
    sm
}

fn seed_spartan_outer_openings(sm: &mut StateManager<'_, F, ProofTranscript, PCS>, log_t: usize) {
    let r_cycle = sm.transcript.challenge_vector_optimized::<F>(log_t);
    let r_cycle_point = OpeningPoint::<BIG_ENDIAN, F>::new(r_cycle);

    // Registers
    sm.accumulator.append_virtual(
        &mut sm.transcript,
        VirtualPolynomial::Rs1Value,
        SumcheckId::SpartanOuter,
        r_cycle_point.clone(),
        F::from(11u64),
    );
    sm.accumulator.append_virtual(
        &mut sm.transcript,
        VirtualPolynomial::Rs2Value,
        SumcheckId::SpartanOuter,
        r_cycle_point.clone(),
        F::from(22u64),
    );
    sm.accumulator.append_virtual(
        &mut sm.transcript,
        VirtualPolynomial::RdWriteValue,
        SumcheckId::SpartanOuter,
        r_cycle_point.clone(),
        F::from(33u64),
    );

    // RAM
    sm.accumulator.append_virtual(
        &mut sm.transcript,
        VirtualPolynomial::RamReadValue,
        SumcheckId::SpartanOuter,
        r_cycle_point.clone(),
        F::from(44u64),
    );
    sm.accumulator.append_virtual(
        &mut sm.transcript,
        VirtualPolynomial::RamWriteValue,
        SumcheckId::SpartanOuter,
        r_cycle_point.clone(),
        F::from(55u64),
    );
    sm.accumulator.append_virtual(
        &mut sm.transcript,
        VirtualPolynomial::RamAddress,
        SumcheckId::SpartanOuter,
        r_cycle_point,
        F::from(66u64),
    );
}

#[test]
fn stage_cache_openings_shapes_do_not_panic() {
    let trace_length = 4;
    let log_t = trace_length.log_2();
    let ram_k = 8192;

    let mut sm = make_state_manager(trace_length, ram_k);
    seed_spartan_outer_openings(&mut sm, log_t);

    // --- Stage 2: Registers R/W checking ---
    let registers_rwc = Rep3RegistersReadWriteChecking::<F>::new(&mut sm);
    let reg_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
        sm.transcript.challenge_vector_optimized::<F>(5 + log_t),
    );
    registers_rwc.cache_openings(
        &mut sm.accumulator,
        &mut sm.transcript,
        reg_opening_point,
        vec![F::one(); 5],
    );
    let _ = sm.accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RegistersVal,
        SumcheckId::RegistersReadWriteChecking,
    );
    let _ = sm.accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersReadWriteChecking,
    );

    // --- Stage 2: RAM RAF evaluation ---
    let ram_raf = Rep3RafEvaluation::<F>::new(&mut sm);
    let raf_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
        sm.transcript.challenge_vector_optimized::<F>(ram_k.log_2()),
    );
    ram_raf.cache_openings(
        &mut sm.accumulator,
        &mut sm.transcript,
        raf_opening_point,
        vec![F::one(); 1],
    );
    let _ = sm
        .accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);

    // --- Stage 2: RAM R/W checking ---
    let ram_rwc = Rep3RamReadWriteChecking::<F>::new(&mut sm);
    let ram_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
        sm.transcript
            .challenge_vector_optimized::<F>(ram_k.log_2() + log_t),
    );
    ram_rwc.cache_openings(
        &mut sm.accumulator,
        &mut sm.transcript,
        ram_opening_point,
        vec![F::one(); 3],
    );
    let _ = sm.accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RamVal,
        SumcheckId::RamReadWriteChecking,
    );
    let _ = sm.accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RamInc,
        SumcheckId::RamReadWriteChecking,
    );

    // --- Stage 2: RAM Output check ---
    let ram_output = Rep3OutputSumcheck::<F>::new(&mut sm);
    let output_opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
        sm.transcript.challenge_vector_optimized::<F>(ram_k.log_2()),
    );
    ram_output.cache_openings(
        &mut sm.accumulator,
        &mut sm.transcript,
        output_opening_point,
        vec![F::one(); 1],
    );
    let _ = sm
        .accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::RamValFinal, SumcheckId::RamOutputCheck);
    let _ = sm
        .accumulator
        .get_virtual_polynomial_opening(VirtualPolynomial::RamValInit, SumcheckId::RamOutputCheck);

    // --- Stage 3: Registers val evaluation ---
    let registers_val = Rep3ValEvaluation::<F>::new(&mut sm);
    let r_cycle_prime =
        OpeningPoint::<BIG_ENDIAN, F>::new(sm.transcript.challenge_vector_optimized::<F>(log_t));
    registers_val.cache_openings(
        &mut sm.accumulator,
        &mut sm.transcript,
        r_cycle_prime,
        vec![F::one(); 2],
    );
    let _ = sm.accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RdWa,
        SumcheckId::RegistersValEvaluation,
    );
    let _ = sm.accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersValEvaluation,
    );

    // --- Stage 3: RAM val_final evaluation ---
    let ram_val_final = Rep3ValFinalSumcheck::<F>::new(&mut sm);
    let r_cycle_prime =
        OpeningPoint::<BIG_ENDIAN, F>::new(sm.transcript.challenge_vector_optimized::<F>(log_t));
    ram_val_final.cache_openings(
        &mut sm.accumulator,
        &mut sm.transcript,
        r_cycle_prime,
        vec![F::one(); 2],
    );
    let _ = sm.accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::RamRa,
        SumcheckId::RamValFinalEvaluation,
    );
    let _ = sm.accumulator.get_committed_polynomial_opening(
        CommittedPolynomial::RamInc,
        SumcheckId::RamValFinalEvaluation,
    );
}

#[test]
fn output_sumcheck_expected_output_claim_matches_formula() {
    let trace_length = 4;
    let log_t = trace_length.log_2();
    let ram_k = 8192;

    let mut sm = make_state_manager(trace_length, ram_k);
    seed_spartan_outer_openings(&mut sm, log_t);

    let output = Rep3OutputSumcheck::<F>::new(&mut sm);
    let r = sm.transcript.challenge_vector_optimized::<F>(ram_k.log_2());
    let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(r);
    let val_final_claim = F::from(777u64);

    // Seed the needed RamValFinal opening and also populate RamValInit via `cache_openings`.
    output.cache_openings(
        &mut sm.accumulator,
        &mut sm.transcript,
        opening_point.clone(),
        vec![val_final_claim],
    );

    let got =
        <Rep3OutputSumcheck<F> as Rep3SumcheckInstance<F, ProofTranscript>>::expected_output_claim(
            &output,
            &sm.accumulator,
            &opening_point.r,
        );

    let r_address_prime = &opening_point.r[..output.r_address().len()];
    let eq_eval: F = EqPolynomial::<F>::mle(output.r_address(), r_address_prime);

    let io_mask = RangeMaskPolynomial::<F>::new(
        remap_address(
            sm.program_io.memory_layout.input_start,
            &sm.program_io.memory_layout,
        )
        .unwrap() as u128,
        remap_address(RAM_START_ADDRESS, &sm.program_io.memory_layout).unwrap() as u128,
    );
    let val_io = ProgramIOPolynomial::new(&sm.program_io);
    let io_mask_eval = io_mask.evaluate_mle(r_address_prime);
    let val_io_eval: F = val_io.evaluate(r_address_prime);

    let expected = eq_eval * io_mask_eval * (val_final_claim - val_io_eval);
    assert_eq!(got, expected);
}
