use std::collections::HashMap;
use std::sync::Arc;

use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::{test_rng, UniformRand};

use co_jolt2::host::program::Rep3Program;
use co_jolt2::poly::dense_mlpoly::combine_poly_shares_rep3;
use co_jolt2::poly::multilinear_polynomial::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use co_jolt2::subprotocols::sumcheck::Rep3SumcheckInstanceWorker;
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::test_utils::run_rep3_test;
use co_jolt2::utils::test_utils::run_rep3_local_test_with_coordinator;
use co_jolt2::utils::tracing::init_tracing;
use co_jolt2::utils::types::Either;
use co_jolt2::zkvm::dag::stage::SumcheckStagesWorker;
use co_jolt2::zkvm::dag::state_manager::{StateManager, StateManagerWorker};
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::instruction::LookupIndexInt;
use co_jolt2::zkvm::r1cs::inputs::{compute_claimed_witness_evals_rep3, ALL_R1CS_INPUTS};
use co_jolt2::zkvm::Rep3JoltWorker;
use co_jolt2::zkvm::witness::{generate_witness_batch_rep3, populate_cycle_witness_rep3};
use co_jolt2::zkvm::{dag::coordinator::Rep3JoltDag, dag::worker::Rep3JoltDagWorker};

use jolt_core::host::Program;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial};
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::bytecode::BytecodeDag;
use jolt_core::zkvm::dag::proof_serialization::JoltProof as VanillaJoltProof;
use jolt_core::zkvm::dag::stage::SumcheckStages;
use jolt_core::zkvm::dag::state_manager::{
    ProofData, ProofKeys, StateManager as VanillaStateManager,
};
use jolt_core::zkvm::instruction_lookups::LookupsDag;
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use jolt_core::zkvm::ram::RamDag;
use jolt_core::zkvm::registers::RegistersDag;
use jolt_core::zkvm::spartan::SpartanDag;
use jolt_core::zkvm::witness::VirtualPolynomial;
use jolt_core::zkvm::witness::{
    compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
};
use jolt_core::zkvm::{
    JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing, JoltVerifierPreprocessing,
};
use tracer::emulator::memory::Memory;
use tracer::instruction::Cycle;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;
type Challenge = <F as jolt_core::field::JoltField>::Challenge;

fn vanilla_inner_sumcheck_round0(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: Vec<Cycle>,
    program_io: tracer::JoltDevice,
    final_memory_state: Memory,
) -> (Vec<Challenge>, F, F, Vec<F>) {
    let mut sm = VanillaStateManager::<F, FS, PCS>::new_prover(
        preprocessing,
        trace,
        program_io,
        None,
        final_memory_state,
    );
    sm.fiat_shamir_preamble();

    let (preprocessing, trace, _, _) = sm.get_prover_data();
    let trace_length = trace.len();
    let padded_trace_length = trace_length.next_power_of_two();
    let ram_k = sm.ram_K;
    let bytecode_d = preprocessing.shared.bytecode.d;

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
        AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), bytecode_d),
    );

    let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
    let mut all_polys = CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);
    let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
        .filter_map(|poly| all_polys.remove(poly))
        .collect();
    let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);
    let (commitments, _opening_proof_hints): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();
    sm.set_commitments(commitments);
    let transcript = sm.get_transcript();
    for commitment in sm.get_commitments().borrow().iter() {
        transcript.borrow_mut().append_serializable(commitment);
    }

    let mut spartan = SpartanDag::<F>::new::<FS>(padded_trace_length);
    spartan.stage1_prove(&mut sm).unwrap();

    let gamma: F = sm.transcript.borrow_mut().challenge_scalar();
    let (outer_sumcheck_r, claim_az) = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
    let (_, claim_bz) = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
    let (_, claim_cz) = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);
    let input_claim = claim_az + gamma * claim_bz + gamma.square() * claim_cz;

    let claimed_witness_evals: Vec<F> = ALL_R1CS_INPUTS
        .iter()
        .map(|r1cs_input| {
            let key = jolt_core::poly::opening_proof::OpeningId::try_from(r1cs_input)
                .expect("Failed to map R1CS input to OpeningId");
            sm.get_prover_accumulator().borrow().get_opening(key)
        })
        .collect();

    let key = UniformSpartanKey::<F>::new(padded_trace_length);
    let num_cycles_bits = key.num_steps.ilog2() as usize;
    let (_r_cycle, rx_var) = outer_sumcheck_r.r.split_at(num_cycles_bits);
    let num_vars_uniform = key.num_vars_uniform_padded();

    let poly_abc_small = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
        key.evaluate_small_matrix_rlc(rx_var, gamma),
    ));
    let mut bind_z = vec![F::zero(); num_vars_uniform];
    for r1cs_input in ALL_R1CS_INPUTS.iter() {
        bind_z[r1cs_input.to_index()] = claimed_witness_evals[r1cs_input.to_index()];
    }
    let const_col = jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs::num_inputs();
    if const_col < num_vars_uniform {
        bind_z[const_col] = F::one();
    }
    let poly_z = MultilinearPolynomial::LargeScalars(DensePolynomial::new(bind_z));
    let half_len = poly_abc_small.len() / 2;
    let evals_02: [F; 2] = (0..half_len)
        .map(|i| {
            let abc_evals = poly_abc_small.sumcheck_evals_array::<2>(i, BindingOrder::HighToLow);
            let z_evals = poly_z.sumcheck_evals_array::<2>(i, BindingOrder::HighToLow);
            [abc_evals[0] * z_evals[0], abc_evals[1] * z_evals[1]]
        })
        .fold([F::zero(); 2], |mut acc, x| {
            acc[0] += x[0];
            acc[1] += x[1];
            acc
        });
    let y0 = evals_02[0];
    let y1 = input_claim - y0;
    let y2 = evals_02[1];
    let poly = jolt_core::poly::unipoly::UniPoly::from_evals(&[y0, y1, y2]);
    let y3 = poly.evaluate(&F::from(3u64));

    (outer_sumcheck_r.r, gamma, input_claim, vec![y0, y2, y3])
}

fn vanilla_lookup_booleanity_round0(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: Vec<Cycle>,
    program_io: tracer::JoltDevice,
    final_memory_state: Memory,
) -> ([F; jolt_core::zkvm::instruction_lookups::D], Vec<Challenge>, Vec<Challenge>, Vec<F>) {
    let mut sm = VanillaStateManager::<F, FS, PCS>::new_prover(
        preprocessing,
        trace,
        program_io,
        None,
        final_memory_state,
    );
    sm.fiat_shamir_preamble();

    let (preprocessing, trace, _, _) = sm.get_prover_data();
    let trace_length = trace.len();
    let padded_trace_length = trace_length.next_power_of_two();
    let ram_k = sm.ram_K;
    let bytecode_d = preprocessing.shared.bytecode.d;

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
        AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), bytecode_d),
    );

    let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
    let mut all_polys = CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);
    let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
        .filter_map(|poly| all_polys.remove(poly))
        .collect();
    let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);
    let (commitments, _opening_proof_hints): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();
    sm.set_commitments(commitments);
    let transcript = sm.get_transcript();
    for commitment in sm.get_commitments().borrow().iter() {
        transcript.borrow_mut().append_serializable(commitment);
    }

    let mut spartan = SpartanDag::<F>::new::<FS>(padded_trace_length);
    spartan.stage1_prove(&mut sm).unwrap();

    let mut registers_dag = RegistersDag::default();
    let mut ram_dag = RamDag::new_prover::<F, FS, PCS>(&sm);
    let mut lookups_dag = LookupsDag::<F>::default();

    let _ = spartan.stage2_prover_instances(&mut sm);
    let _ = registers_dag.stage2_prover_instances(&mut sm);
    let _ = ram_dag.stage2_prover_instances(&mut sm);

    let (gamma_powers, r_address) = {
        let mut tr = transcript.borrow().clone();
        let gamma: F = tr.challenge_scalar();
        let mut gamma_powers = [F::one(); jolt_core::zkvm::instruction_lookups::D];
        for i in 1..jolt_core::zkvm::instruction_lookups::D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let r_address = tr.challenge_vector_optimized::<F>(
            jolt_core::zkvm::instruction_lookups::LOG_K_CHUNK,
        );
        (gamma_powers, r_address)
    };

    let r_cycle = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter)
        .0
        .r
        .clone();

    let mut lookups_instances = lookups_dag.stage2_prover_instances(&mut sm);
    assert_eq!(lookups_instances.len(), 1, "expected one lookups stage2 instance");
    let round0 = lookups_instances[0].compute_prover_message(0, F::zero());
    (gamma_powers, r_address, r_cycle, round0)
}

fn vanilla_registers_round0(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: Vec<Cycle>,
    program_io: tracer::JoltDevice,
    final_memory_state: Memory,
) -> (F, Vec<Challenge>, F, Vec<F>) {
    let mut sm = VanillaStateManager::<F, FS, PCS>::new_prover(
        preprocessing,
        trace,
        program_io,
        None,
        final_memory_state,
    );
    sm.fiat_shamir_preamble();

    let (preprocessing, trace, _, _) = sm.get_prover_data();
    let trace_length = trace.len();
    let padded_trace_length = trace_length.next_power_of_two();
    let ram_k = sm.ram_K;
    let bytecode_d = preprocessing.shared.bytecode.d;

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
        AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), bytecode_d),
    );

    let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
    let mut all_polys = CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);
    let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
        .filter_map(|poly| all_polys.remove(poly))
        .collect();
    let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);
    let (commitments, _opening_proof_hints): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();
    sm.set_commitments(commitments);
    let transcript = sm.get_transcript();
    for commitment in sm.get_commitments().borrow().iter() {
        transcript.borrow_mut().append_serializable(commitment);
    }

    let mut spartan = SpartanDag::<F>::new::<FS>(padded_trace_length);
    spartan.stage1_prove(&mut sm).unwrap();

    let mut registers_dag = RegistersDag::default();
    let _ = spartan.stage2_prover_instances(&mut sm);

    let gamma = {
        let mut tr = transcript.borrow().clone();
        tr.challenge_scalar()
    };
    let r_cycle = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter)
        .0
        .r
        .clone();

    let mut instances = registers_dag.stage2_prover_instances(&mut sm);
    assert_eq!(instances.len(), 1, "expected one registers stage2 instance");
    let input_claim = instances[0].input_claim();
    let round0 = instances[0].compute_prover_message(0, input_claim);
    (gamma, r_cycle, input_claim, round0)
}

fn vanilla_ram_round0(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: Vec<Cycle>,
    program_io: tracer::JoltDevice,
    final_memory_state: Memory,
) -> (
    F,
    F,
    Vec<Challenge>,
    F,
    Vec<Challenge>,
    Vec<Challenge>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
) {
    let mut sm = VanillaStateManager::<F, FS, PCS>::new_prover(
        preprocessing,
        trace,
        program_io,
        None,
        final_memory_state,
    );
    sm.fiat_shamir_preamble();

    let (preprocessing, trace, _, _) = sm.get_prover_data();
    let trace_length = trace.len();
    let padded_trace_length = trace_length.next_power_of_two();
    let ram_k = sm.ram_K;
    let bytecode_d = preprocessing.shared.bytecode.d;

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
        AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), bytecode_d),
    );

    let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
    let mut all_polys = CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);
    let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
        .filter_map(|poly| all_polys.remove(poly))
        .collect();
    let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);
    let (commitments, _opening_proof_hints): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();
    sm.set_commitments(commitments);
    let transcript = sm.get_transcript();
    for commitment in sm.get_commitments().borrow().iter() {
        transcript.borrow_mut().append_serializable(commitment);
    }

    let mut spartan = SpartanDag::<F>::new::<FS>(padded_trace_length);
    spartan.stage1_prove(&mut sm).unwrap();

    let mut registers_dag = RegistersDag::default();
    let _ = spartan.stage2_prover_instances(&mut sm);
    let _ = registers_dag.stage2_prover_instances(&mut sm);

    let (ram_gamma, output_r_address) = {
        let mut tr = transcript.borrow().clone();
        let gamma = tr.challenge_scalar();
        let r_address = tr.challenge_vector_optimized::<F>(ram_k.log_2());
        (gamma, r_address)
    };

    let input_claim = {
        let (_, rv_claim) = sm
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamReadValue, SumcheckId::SpartanOuter);
        let (_, wv_claim) = sm
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamWriteValue, SumcheckId::SpartanOuter);
        rv_claim + ram_gamma * wv_claim
    };

    let ram_address_r = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
        .0
        .r
        .clone();
    let ram_address_claim = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
        .1;
    let ram_read_r = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::RamReadValue, SumcheckId::SpartanOuter)
        .0
        .r
        .clone();

    let mut ram_dag = RamDag::new_prover::<F, FS, PCS>(&sm);
    let mut instances = ram_dag.stage2_prover_instances(&mut sm);
    assert_eq!(instances.len(), 3, "expected three ram stage2 instances");

    let raf_claim = instances[0].input_claim();
    let rwc_claim = instances[1].input_claim();
    let output_claim = instances[2].input_claim();
    let raf_round0 = {
        let evals = instances[0].compute_prover_message(0, raf_claim);
        let y0 = evals[0];
        let y2 = evals[1];
        let y1 = raf_claim - y0;
        let poly = jolt_core::poly::unipoly::UniPoly::from_evals(&[y0, y1, y2]);
        let y3 = poly.evaluate(&F::from(3u64));
        vec![y0, y2, y3]
    };
    let rwc_round0 = instances[1].compute_prover_message(0, rwc_claim);
    let output_round0 = instances[2].compute_prover_message(0, output_claim);

    (
        ram_gamma,
        input_claim,
        output_r_address,
        ram_address_claim,
        ram_address_r,
        ram_read_r,
        raf_round0,
        rwc_round0,
        output_round0,
    )
}

fn vanilla_ram_valfinal_round0(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: Vec<Cycle>,
    program_io: tracer::JoltDevice,
    final_memory_state: Memory,
) -> (Vec<Challenge>, F, F, Vec<F>) {
    let mut sm = VanillaStateManager::<F, FS, PCS>::new_prover(
        preprocessing,
        trace,
        program_io,
        None,
        final_memory_state,
    );

    sm.fiat_shamir_preamble();

    let (preprocessing, trace, _, _) = sm.get_prover_data();
    let trace_length = trace.len();
    let padded_trace_length = trace_length.next_power_of_two();
    let ram_k = sm.ram_K;
    let bytecode_d = preprocessing.shared.bytecode.d;

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
        AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), bytecode_d),
    );

    let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
    let mut all_polys = CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);
    let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
        .filter_map(|poly| all_polys.remove(poly))
        .collect();
    let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);
    let (commitments, _opening_proof_hints): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();
    sm.set_commitments(commitments);
    let transcript = sm.get_transcript();
    for commitment in sm.get_commitments().borrow().iter() {
        transcript.borrow_mut().append_serializable(commitment);
    }

    let mut spartan = SpartanDag::<F>::new::<FS>(padded_trace_length);
    spartan.stage1_prove(&mut sm).unwrap();

    let mut registers_dag = RegistersDag::default();
    let mut ram_dag = RamDag::new_prover::<F, FS, PCS>(&sm);
    let mut lookups_dag = LookupsDag::<F>::default();

    let mut stage2_instances: Vec<_> = std::iter::empty()
        .chain(spartan.stage2_prover_instances(&mut sm))
        .chain(registers_dag.stage2_prover_instances(&mut sm))
        .chain(ram_dag.stage2_prover_instances(&mut sm))
        .chain(lookups_dag.stage2_prover_instances(&mut sm))
        .collect();
    let stage2_instances_mut: Vec<&mut dyn SumcheckInstance<F, FS>> = stage2_instances
        .iter_mut()
        .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, FS>)
        .collect();
    let accumulator = sm.get_prover_accumulator();
    let _ = BatchedSumcheck::prove(
        stage2_instances_mut,
        Some(accumulator.clone()),
        &mut *transcript.borrow_mut(),
    );

    let mut ram_stage3 = ram_dag.stage3_prover_instances(&mut sm);
    assert_eq!(ram_stage3.len(), 3, "expected three ram stage3 instances");
    let mut val_final = ram_stage3.remove(1);
    let input_claim = val_final.input_claim();
    let evals = val_final.compute_prover_message(0, input_claim);
    let y0 = evals[0];
    let y2 = evals[1];
    let y1 = input_claim - y0;
    let poly = jolt_core::poly::unipoly::UniPoly::from_evals(&[y0, y1, y2]);
    let y3 = poly.evaluate(&F::from(3u64));

    let (opening_point, val_final_claim) = sm
        .get_prover_accumulator()
        .borrow()
        .get_virtual_polynomial_opening(VirtualPolynomial::RamValFinal, SumcheckId::RamOutputCheck);

    (opening_point.r.clone(), val_final_claim, input_claim, vec![y0, y2, y3])
}

fn vanilla_up_to_stage5(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: Vec<Cycle>,
    program_io: tracer::JoltDevice,
    final_memory_state: Memory,
) -> (VanillaJoltProof<F, PCS, FS>, Vec<Challenge>) {
    let mut sm = VanillaStateManager::<F, FS, PCS>::new_prover(
        preprocessing,
        trace,
        program_io,
        None,
        final_memory_state,
    );

    sm.fiat_shamir_preamble();

    let (preprocessing, trace, _, _) = sm.get_prover_data();
    let trace_length = trace.len();
    let padded_trace_length = trace_length.next_power_of_two();
    let ram_K = sm.ram_K;
    let bytecode_d = preprocessing.shared.bytecode.d;

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
        AllCommittedPolynomials::initialize(compute_d_parameter(ram_K), bytecode_d),
    );

    // Generate witness polys and commit (mirrors vanilla JoltDAG::generate_and_commit_polynomials).
    let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
    let mut all_polys = CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);
    let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
        .filter_map(|poly| all_polys.remove(poly))
        .collect();
    let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);
    let (commitments, opening_proof_hints): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();
    let opening_proof_hints: HashMap<
        CommittedPolynomial,
        <PCS as CommitmentScheme>::OpeningProofHint,
    > = AllCommittedPolynomials::iter()
        .copied()
        .zip(opening_proof_hints)
        .collect();
    sm.set_commitments(commitments);

    // Append commitments to transcript (vanilla ordering).
    let commitments = sm.get_commitments();
    let transcript = sm.get_transcript();
    for commitment in commitments.borrow().iter() {
        transcript.borrow_mut().append_serializable(commitment);
    }

    // Capture tau exactly as stage1 will derive it (without perturbing the live transcript).
    let key = UniformSpartanKey::<F>::new(padded_trace_length);
    let num_rounds_x = key.num_rows_bits();
    let tau: Vec<Challenge> = {
        let mut tr = transcript.borrow().clone();
        tr.challenge_vector_optimized::<F>(num_rounds_x)
    };

    // Stage 1 (Spartan outer sumcheck).
    let mut spartan = SpartanDag::<F>::new::<FS>(padded_trace_length);
    spartan.stage1_prove(&mut sm).unwrap();

    // Stage 2: all subsystems in vanilla order.
    let mut registers_dag = RegistersDag::default();
    let mut ram_dag = RamDag::new_prover::<F, FS, PCS>(&sm);
    let mut lookups_dag = LookupsDag::<F>::default();

    let mut stage2_instances: Vec<_> = std::iter::empty()
        .chain(spartan.stage2_prover_instances(&mut sm))
        .chain(registers_dag.stage2_prover_instances(&mut sm))
        .chain(ram_dag.stage2_prover_instances(&mut sm))
        .chain(lookups_dag.stage2_prover_instances(&mut sm))
        .collect();
    if let Some(limit) = std::env::var("CO_JOLT2_STAGE2_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        stage2_instances.truncate(limit.min(stage2_instances.len()));
    }
    let stage2_instances_mut: Vec<&mut dyn SumcheckInstance<F, FS>> = stage2_instances
        .iter_mut()
        .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, FS>)
        .collect();
    let accumulator = sm.get_prover_accumulator();
    let (stage2_proof, _r_stage2) = BatchedSumcheck::prove(
        stage2_instances_mut,
        Some(accumulator.clone()),
        &mut *transcript.borrow_mut(),
    );
    sm.proofs.borrow_mut().insert(
        ProofKeys::Stage2Sumcheck,
        ProofData::SumcheckProof(stage2_proof),
    );

    if std::env::var("CO_JOLT2_STOP_AFTER_STAGE2").is_ok() {
        return (VanillaJoltProof::from_prover_state_manager(sm), tau);
    }

    // Stage 3: all vanilla instances in vanilla order.
    let spartan_stage3 = spartan.stage3_prover_instances(&mut sm);
    let registers_stage3 = registers_dag.stage3_prover_instances(&mut sm);
    let lookups_stage3 = lookups_dag.stage3_prover_instances(&mut sm);
    let ram_stage3 = ram_dag.stage3_prover_instances(&mut sm);

    let mut stage3_instances: Vec<Box<dyn SumcheckInstance<F, FS>>> = Vec::new();
    stage3_instances.extend(spartan_stage3);
    stage3_instances.extend(registers_stage3);
    stage3_instances.extend(lookups_stage3);
    stage3_instances.extend(ram_stage3);
    if let Some(limit) = std::env::var("CO_JOLT2_STAGE3_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        stage3_instances.truncate(limit.min(stage3_instances.len()));
    }

    let stage3_instances_mut: Vec<&mut dyn SumcheckInstance<F, FS>> = stage3_instances
        .iter_mut()
        .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, FS>)
        .collect();
    let (stage3_proof, _r_stage3) = BatchedSumcheck::prove(
        stage3_instances_mut,
        Some(accumulator.clone()),
        &mut *transcript.borrow_mut(),
    );
    sm.proofs.borrow_mut().insert(
        ProofKeys::Stage3Sumcheck,
        ProofData::SumcheckProof(stage3_proof),
    );

    if std::env::var("CO_JOLT2_STOP_AFTER_STAGE3").is_ok() {
        return (VanillaJoltProof::from_prover_state_manager(sm), tau);
    }

    // Stage 4: RAM + Bytecode + Lookups RA.
    {
        let mut bytecode_dag = BytecodeDag::default();

        let mut stage4_instances: Vec<_> = std::iter::empty()
            .chain(ram_dag.stage4_prover_instances(&mut sm))
            .chain(bytecode_dag.stage4_prover_instances(&mut sm))
            .chain(lookups_dag.stage4_prover_instances(&mut sm))
            .collect();

        if !stage4_instances.is_empty() {
            let stage4_instances_mut: Vec<&mut dyn SumcheckInstance<F, FS>> = stage4_instances
                .iter_mut()
                .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F, FS>)
                .collect();
            let (stage4_proof, _r_stage4) = BatchedSumcheck::prove(
                stage4_instances_mut,
                Some(accumulator.clone()),
                &mut *transcript.borrow_mut(),
            );
            sm.proofs.borrow_mut().insert(
                ProofKeys::Stage4Sumcheck,
                ProofData::SumcheckProof(stage4_proof),
            );
        }
    }

    let (preprocessing, trace, _, _) = sm.get_prover_data();
    let all_poly_keys: Vec<CommittedPolynomial> =
        AllCommittedPolynomials::iter().copied().collect();
    let polynomials_map =
        CommittedPolynomial::generate_witness_batch(&all_poly_keys, preprocessing, trace);
    let opening_proof = accumulator.borrow_mut().reduce_and_prove(
        polynomials_map,
        opening_proof_hints,
        &preprocessing.generators,
        &mut *transcript.borrow_mut(),
    );
    sm.proofs.borrow_mut().insert(
        ProofKeys::ReducedOpeningProof,
        ProofData::ReducedOpeningProof(opening_proof),
    );

    (VanillaJoltProof::from_prover_state_manager(sm), tau)
}

#[test]
fn dag_correct() {
    let _tracing_guard = init_tracing("dag_correct.json", std::path::Path::new("traces"));

    // 1) Build and trace the fibonacci program (reuse witness_batch_rep3 setup).
    let mut program = Program::new("fibonacci-guest");
    program.set_memory_size(10240);
    let inputs = postcard::to_stdvec(&9u32).unwrap();
    let (bytecode, memory_init, _) = program.decode();

    let mut rng = test_rng();
    let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);
    let (mut vanilla_trace, vanilla_memory, io_device) = program.trace(&inputs, &[], &[]);

    tracing::info!("Trace len: {}", vanilla_trace.len());
    // Pad traces to next power of 2 (+1 termination cycle).
    let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
    vanilla_trace.resize(padded_len, Cycle::NoOp);
    for (trace, _, _) in shares.iter_mut() {
        trace.resize(padded_len, Rep3Cycle::NoOp);
    }

    // 2) Preprocessing (same for vanilla + rep3).
    let shared = JoltSharedPreprocessing {
        memory_layout: io_device.memory_layout.clone(),
        bytecode: jolt_core::zkvm::bytecode::BytecodePreprocessing::preprocess(bytecode.clone()),
        ram: jolt_core::zkvm::ram::RAMPreprocessing::preprocess(memory_init.clone()),
    };
    let preprocessing: JoltProverPreprocessing<F, PCS> =
        <JoltRV64IMAC as Rep3JoltWorker<F, PCS, FS>>::preprocess(
            bytecode,
            io_device.memory_layout.clone(),
            memory_init,
            padded_len,
        );
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    // 3) Compute ram_K from vanilla trace (must match both sides).
    let ram_K = compute_ram_k(&vanilla_trace, &shared);

    #[cfg(feature = "rv32")]
    {
        use mpc_core::protocols::rep3::arithmetic;
        use mpc_core::protocols::rep3_ring::combine_ring_element_binary;

        let mut rng = test_rng();
        let r_cycle: Vec<Challenge> = (0..padded_len.ilog2() as usize)
            .map(|_| Challenge::random(&mut rng))
            .collect();
        let vanilla_evals =
            jolt_core::zkvm::r1cs::inputs::compute_claimed_witness_evals(&shared, &vanilla_trace, &r_cycle);

        let preprocessing_arc = Arc::new(preprocessing.clone());
        let io_device_arc = Arc::new(io_device.clone());
        let shares_arc = Arc::new(shares.clone());
        let base_port: u16 = 15300;
        let share_evals: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>; 3] = run_rep3_test(
            base_port,
            1,
            move |party_idx| {
                let (trace, mem, advice_shares) = shares_arc[party_idx].clone();
                (
                    trace,
                    mem,
                    Arc::clone(&io_device_arc),
                    Arc::clone(&preprocessing_arc),
                    ram_K,
                    advice_shares,
                    r_cycle.clone(),
                )
            },
            move |input: (
                Vec<Rep3Cycle>,
                co_jolt2::host::memory::Rep3Memory,
                Arc<tracer::JoltDevice>,
                Arc<JoltProverPreprocessing<F, PCS>>,
                usize,
                co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                Vec<Challenge>,
            ), mut io_ctx| {
                let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r_cycle) = input;
                let budget = co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                let mut preproc =
                    mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<F, _>(
                        [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                        budget.dabits,
                        &mut io_ctx,
                    )?;
                let mut state = StateManagerWorker::new(
                    &preprocessing,
                    trace,
                    (*io_device).clone(),
                    mem,
                    io_ctx.party_id(),
                    ram_k,
                    Some(advice_shares),
                );
                populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                compute_claimed_witness_evals_rep3::<F, PCS, _>(&mut state, &mut io_ctx, &r_cycle)
            },
        );
        let shares_lookup = shares.clone();
        let io_device_lookup = io_device.clone();
        let preprocessing_lookup = preprocessing.clone();
        let lookup_shares: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>; 3] = run_rep3_test(
            base_port + 10,
            1,
            move |party_idx| {
                let (trace, mem, advice_shares) = shares_lookup[party_idx].clone();
                (
                    trace,
                    mem,
                    io_device_lookup.clone(),
                    preprocessing_lookup.clone(),
                    ram_K,
                    advice_shares,
                )
            },
            move |input: (
                Vec<Rep3Cycle>,
                co_jolt2::host::memory::Rep3Memory,
                tracer::JoltDevice,
                JoltProverPreprocessing<F, PCS>,
                usize,
                co_jolt2::host::jolt_device::Rep3ProgramIOInput,
            ), mut io_ctx| {
                let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
                let budget = co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                let mut preproc =
                    mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<F, _>(
                        [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                        budget.dabits,
                        &mut io_ctx,
                    )?;
                let mut state = StateManagerWorker::new(
                    &preprocessing,
                    trace,
                    io_device,
                    mem,
                    io_ctx.party_id(),
                    ram_k,
                    Some(advice_shares),
                );
                populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                Ok(state.get_cycle_witness().stage1_lookup_output().to_vec())
            },
        );
        let opened_lookup = arithmetic::combine_field_elements_vec(vec![
            lookup_shares[0].clone(),
            lookup_shares[1].clone(),
            lookup_shares[2].clone(),
        ]);
        let vanilla_lookup: Vec<F> = (0..vanilla_trace.len())
            .map(|t| jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(&shared, &vanilla_trace, t).to_field(jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs::LookupOutput))
            .collect();
        for (t, (rep3, vanilla)) in opened_lookup.iter().zip(vanilla_lookup.iter()).enumerate() {
            assert_eq!(
                rep3,
                vanilla,
                "rv32 lookup_output mismatch at step {t}: cycle={:?}",
                vanilla_trace[t]
            );
        }

        let opened = arithmetic::combine_field_elements_vec(vec![
            share_evals[0].clone(),
            share_evals[1].clone(),
            share_evals[2].clone(),
        ]);
        for (i, (rep3, vanilla)) in opened.iter().zip(vanilla_evals.iter()).enumerate() {
            assert_eq!(rep3, vanilla, "rv32 claimed eval mismatch at input {i} ({:?})", ALL_R1CS_INPUTS[i]);
        }

        let shares_indices = shares.clone();
        let io_device_indices = io_device.clone();
        let preprocessing_indices = preprocessing.clone();
        let lookup_index_shares: [Vec<Either<LookupIndexInt, mpc_core::protocols::rep3_ring::Rep3RingShare<LookupIndexInt>>>; 3] =
            run_rep3_test(
                base_port + 20,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_indices[party_idx].clone();
                    (
                        trace,
                        mem,
                        io_device_indices.clone(),
                        preprocessing_indices.clone(),
                        ram_K,
                        advice_shares,
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                ), mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
                    let budget = co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                    let mut preproc =
                        mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<F, _>(
                            [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                            budget.dabits,
                            &mut io_ctx,
                        )?;
                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        io_device,
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    generate_witness_batch_rep3::<F, PCS, _>(&[], &mut state, &mut io_ctx, &mut preproc)?;
                    Ok(state.prover_state.cycle_witness.take_read_raf().lookup_indices)
                },
            );
        let opened_indices: Vec<LookupIndexInt> = (0..lookup_index_shares[0].len())
            .map(|i| match (
                &lookup_index_shares[0][i],
                &lookup_index_shares[1][i],
                &lookup_index_shares[2][i],
            ) {
                (Either::Public(a), Either::Public(b), Either::Public(c)) => {
                    assert_eq!(a, b, "public lookup index shares mismatch at step {i}");
                    assert_eq!(b, c, "public lookup index shares mismatch at step {i}");
                    *a
                }
                (Either::Shared(a), Either::Shared(b), Either::Shared(c)) => {
                    combine_ring_element_binary(*a, *b, *c).0
                }
                _ => panic!("lookup index visibility mismatch at step {i}"),
            })
            .collect();
        let vanilla_lookup_indices: Vec<LookupIndexInt> = vanilla_trace
            .iter()
            .map(|cycle| {
                jolt_core::zkvm::instruction::LookupQuery::<32>::to_lookup_index(cycle)
                    as LookupIndexInt
            })
            .collect();
        for (t, (rep3, vanilla)) in opened_indices
            .iter()
            .zip(vanilla_lookup_indices.iter())
            .enumerate()
        {
            assert_eq!(
                rep3,
                vanilla,
                "rv32 lookup_index mismatch at step {t}: cycle={:?}",
                vanilla_trace[t]
            );
        }

        let stage2_polys = vec![
            CommittedPolynomial::RdInc,
            CommittedPolynomial::RamInc,
        ];
        let vanilla_stage2_witness =
            CommittedPolynomial::generate_witness_batch(&stage2_polys, &preprocessing, &vanilla_trace);
        let stage2_polys_worker = stage2_polys.clone();
        let shares_stage2_witness = shares.clone();
        let io_device_stage2_witness = io_device.clone();
        let preprocessing_stage2_witness = preprocessing.clone();
        let rep3_stage2_witness: [HashMap<CommittedPolynomial, Rep3MultilinearPolynomial<F>>; 3] =
            run_rep3_test(
                base_port + 30,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_stage2_witness[party_idx].clone();
                    (
                        trace,
                        mem,
                        io_device_stage2_witness.clone(),
                        preprocessing_stage2_witness.clone(),
                        ram_K,
                        advice_shares,
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                ), mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
                    let budget = co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                    let mut preproc =
                        mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<F, _>(
                            [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                            budget.dabits,
                            &mut io_ctx,
                        )?;
                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        io_device,
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    generate_witness_batch_rep3::<F, PCS, _>(&stage2_polys_worker, &mut state, &mut io_ctx, &mut preproc)
                },
            );

        for poly in stage2_polys {
            let rep3_poly = combine_poly_shares_rep3(
                rep3_stage2_witness
                    .iter()
                    .map(|party_map| match party_map.get(&poly).expect("missing stage2 witness poly") {
                        Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(dense)) => dense.clone(),
                        other => panic!("expected dense shared poly for {poly:?}, got {other:?}"),
                    })
                    .collect(),
            );
            let vanilla_poly =
                vanilla_stage2_witness.get(&poly).expect("missing vanilla stage2 witness poly");
            assert_eq!(
                rep3_poly.len(),
                vanilla_poly.len(),
                "rv32 stage2 witness len mismatch for {poly:?}"
            );
            for i in 0..rep3_poly.len() {
                assert_eq!(
                    rep3_poly[i],
                    vanilla_poly.get_coeff(i),
                    "rv32 stage2 witness mismatch for {poly:?} at coeff {i}"
                );
            }
        }

        let (reg_gamma, reg_r_cycle, reg_claim, vanilla_reg_round0) = vanilla_registers_round0(
            &preprocessing,
            vanilla_trace.clone(),
            io_device.clone(),
            vanilla_memory.clone(),
        );
        let shares_registers = shares.clone();
        let io_device_registers = io_device.clone();
        let preprocessing_registers = preprocessing.clone();
        let reg_msg_shares: [Vec<mpc_core::protocols::additive::AdditiveShare<F>>; 3] =
            run_rep3_test(
                base_port + 32,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_registers[party_idx].clone();
                    (
                        trace,
                        mem,
                        io_device_registers.clone(),
                        preprocessing_registers.clone(),
                        ram_K,
                        advice_shares,
                        reg_r_cycle.clone(),
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                    Vec<Challenge>,
                ), mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r_cycle) = input;
                    let budget = co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                    let mut preproc =
                        mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<F, _>(
                            [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                            budget.dabits,
                            &mut io_ctx,
                        )?;
                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        io_device,
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    let _ =
                        generate_witness_batch_rep3::<F, PCS, _>(&[CommittedPolynomial::RdInc], &mut state, &mut io_ctx, &mut preproc)?;
                    state.accumulator.append_virtual_public(
                        VirtualPolynomial::Rs1Value,
                        SumcheckId::SpartanOuter,
                        jolt_core::poly::opening_proof::OpeningPoint::new(r_cycle),
                        F::zero(),
                        io_ctx.party_id(),
                    );
                    let mut registers =
                        co_jolt2::zkvm::registers::read_write_checking::Rep3RegistersReadWriteCheckingWorker::new(
                            &mut state,
                            reg_gamma,
                            reg_claim,
                        );
                    Ok(registers.compute_prover_message_share(
                        0,
                        mpc_core::protocols::additive::promote_to_trivial_share(
                            reg_claim,
                            io_ctx.party_id(),
                        ),
                        3,
                        &mut io_ctx,
                    ))
                },
            );
        let rep3_reg_round0 = mpc_core::protocols::additive::combine_additive_vec(vec![
            reg_msg_shares[0].clone(),
            reg_msg_shares[1].clone(),
            reg_msg_shares[2].clone(),
        ]);
        assert_eq!(
            rep3_reg_round0,
            vanilla_reg_round0,
            "rv32 registers round-0 mismatch"
        );

        let (outer_sumcheck_r, inner_gamma, inner_claim, vanilla_inner_round0) =
            vanilla_inner_sumcheck_round0(
                &preprocessing,
                vanilla_trace.clone(),
                io_device.clone(),
                vanilla_memory.clone(),
            );
        let outer_r_cycle = outer_sumcheck_r[..padded_len.ilog2() as usize].to_vec();
        let shares_inner = shares.clone();
        let io_device_inner = io_device.clone();
        let preprocessing_inner = preprocessing.clone();
        let inner_msg_shares: [Vec<mpc_core::protocols::additive::AdditiveShare<F>>; 3] =
            run_rep3_test(
                base_port + 35,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_inner[party_idx].clone();
                    (
                        trace,
                        mem,
                        io_device_inner.clone(),
                        preprocessing_inner.clone(),
                        ram_K,
                        advice_shares,
                        outer_r_cycle.clone(),
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                    Vec<Challenge>,
                ), mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r_cycle) = input;
                    let budget = co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                    let mut preproc =
                        mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<F, _>(
                            [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                            budget.dabits,
                            &mut io_ctx,
                        )?;
                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        io_device,
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    let claimed_witness_evals =
                        compute_claimed_witness_evals_rep3::<F, PCS, _>(&mut state, &mut io_ctx, &r_cycle)?;
                    let mut inner = co_jolt2::zkvm::spartan::Rep3InnerSumcheckWorker::new(
                        inner_gamma,
                        inner_claim,
                        &outer_sumcheck_r,
                        claimed_witness_evals,
                        padded_len,
                        io_ctx.party_id(),
                    );
                    Ok(inner.compute_prover_message_share(
                        0,
                        mpc_core::protocols::additive::promote_to_trivial_share(
                            inner_claim,
                            io_ctx.party_id(),
                        ),
                        3,
                        &mut io_ctx,
                    ))
                },
            );
        let rep3_inner_round0 = mpc_core::protocols::additive::combine_additive_vec(vec![
            inner_msg_shares[0].clone(),
            inner_msg_shares[1].clone(),
            inner_msg_shares[2].clone(),
        ]);
        assert_eq!(
            rep3_inner_round0,
            vanilla_inner_round0,
            "rv32 spartan inner round-0 mismatch"
        );

        let (bool_gamma, bool_r_address, bool_r_cycle, vanilla_bool_round0) =
            vanilla_lookup_booleanity_round0(
                &preprocessing,
                vanilla_trace.clone(),
                io_device.clone(),
                vanilla_memory.clone(),
            );
        let instruction_ra_polys: Vec<CommittedPolynomial> =
            (0..jolt_core::zkvm::instruction_lookups::D)
                .map(CommittedPolynomial::InstructionRa)
                .collect();
        let shares_booleanity = shares.clone();
        let io_device_booleanity = io_device.clone();
        let preprocessing_booleanity = preprocessing.clone();
        let bool_msg_shares: [Vec<mpc_core::protocols::additive::AdditiveShare<F>>; 3] =
            run_rep3_test(
                base_port + 40,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_booleanity[party_idx].clone();
                    (
                        trace,
                        mem,
                        io_device_booleanity.clone(),
                        preprocessing_booleanity.clone(),
                        ram_K,
                        advice_shares,
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                ), mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
                    let budget = co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                    let mut preproc =
                        mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<F, _>(
                            [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                            budget.dabits,
                            &mut io_ctx,
                        )?;
                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        io_device,
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    let witness =
                        generate_witness_batch_rep3::<F, PCS, _>(&instruction_ra_polys, &mut state, &mut io_ctx, &mut preproc)?;
                    let one_hot_polys = std::array::from_fn(|i| {
                        match witness
                            .get(&CommittedPolynomial::InstructionRa(i))
                            .expect("missing instruction ra poly")
                        {
                            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(poly)) => poly.clone(),
                            other => panic!("expected one-hot instruction ra poly, got {other:?}"),
                        }
                    });
                    state.accumulator.append_virtual_public(
                        VirtualPolynomial::LookupOutput,
                        SumcheckId::SpartanOuter,
                        jolt_core::poly::opening_proof::OpeningPoint::new(bool_r_cycle.clone()),
                        F::zero(),
                        io_ctx.party_id(),
                    );
                    let mut lookups =
                        co_jolt2::zkvm::instruction_lookups::Rep3LookupsDagWorker::new(one_hot_polys);
                    lookups.set_stage2_init(bool_gamma, bool_r_address.clone());
                    let mut instances = lookups.stage2_instances(&mut state, &mut io_ctx)?;
                    assert_eq!(instances.len(), 1, "expected one rep3 lookups stage2 instance");
                    let instance = match instances.pop().expect("missing instance") {
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => instance,
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                            panic!("unexpected public lookups stage2 instance")
                        }
                    };
                    let mut instance = instance;
                    Ok(instance.compute_prover_message_share(
                        0,
                        mpc_core::protocols::additive::promote_to_trivial_share(
                            F::zero(),
                            io_ctx.party_id(),
                        ),
                        3,
                        &mut io_ctx,
                    ))
                },
            );
        let rep3_bool_round0 = mpc_core::protocols::additive::combine_additive_vec(vec![
            bool_msg_shares[0].clone(),
            bool_msg_shares[1].clone(),
            bool_msg_shares[2].clone(),
        ]);
        assert_eq!(
            rep3_bool_round0,
            vanilla_bool_round0,
            "rv32 lookup booleanity round-0 mismatch"
        );

        let (
            ram_gamma,
            ram_input_claim,
            ram_output_r_address,
            ram_address_claim,
            ram_address_r,
            ram_read_r,
            vanilla_raf_round0,
            vanilla_rwc_round0,
            vanilla_output_round0,
        ) = vanilla_ram_round0(
            &preprocessing,
            vanilla_trace.clone(),
            io_device.clone(),
            vanilla_memory.clone(),
        );
        let ram_output_r_address_round0 = ram_output_r_address.clone();
        let ram_address_r_round0 = ram_address_r.clone();
        let ram_read_r_round0 = ram_read_r.clone();
        let shares_ram = shares.clone();
        let io_device_ram = io_device.clone();
        let preprocessing_ram = preprocessing.clone();
        let ram_msg_shares: [(Vec<mpc_core::protocols::additive::AdditiveShare<F>>, Vec<mpc_core::protocols::additive::AdditiveShare<F>>, Vec<mpc_core::protocols::additive::AdditiveShare<F>>); 3] =
            run_rep3_test(
                base_port + 45,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_ram[party_idx].clone();
                    (
                        trace,
                        mem,
                        io_device_ram.clone(),
                        preprocessing_ram.clone(),
                        ram_K,
                        advice_shares,
                        ram_address_r_round0.clone(),
                        ram_read_r_round0.clone(),
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                    Vec<Challenge>,
                    Vec<Challenge>,
                ), mut io_ctx| {
                    let (
                        trace,
                        mem,
                        io_device,
                        preprocessing,
                        ram_k,
                        advice_shares,
                        ram_address_r,
                        ram_read_r,
                    ) = input;
                    let budget =
                        co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                    let mut preproc =
                        mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<
                            F,
                            _,
                        >(
                            [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                            budget.dabits,
                            &mut io_ctx,
                        )?;
                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        io_device,
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    let _ = generate_witness_batch_rep3::<F, PCS, _>(
                        &[CommittedPolynomial::RamInc],
                        &mut state,
                        &mut io_ctx,
                        &mut preproc,
                    )?;
                    state.accumulator.append_virtual_public(
                        VirtualPolynomial::RamAddress,
                        SumcheckId::SpartanOuter,
                        jolt_core::poly::opening_proof::OpeningPoint::new(ram_address_r),
                        ram_address_claim,
                        io_ctx.party_id(),
                    );
                    state.accumulator.append_virtual_public(
                        VirtualPolynomial::RamReadValue,
                        SumcheckId::SpartanOuter,
                        jolt_core::poly::opening_proof::OpeningPoint::new(ram_read_r),
                        F::zero(),
                        io_ctx.party_id(),
                    );
                    let mut ram =
                        co_jolt2::zkvm::ram::Rep3RamDagWorker::new(&mut state, &mut io_ctx)?;
                    ram.set_stage2_init(
                        ram_gamma,
                        ram_input_claim,
                        ram_output_r_address_round0.clone(),
                    );
                    let mut instances = ram.stage2_instances(&mut state, &mut io_ctx)?;
                    assert_eq!(instances.len(), 3, "expected three rep3 ram stage2 instances");
                    let mut raf = match instances.remove(0) {
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => instance,
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                            panic!("unexpected public ram raf instance")
                        }
                    };
                    let mut rwc = match instances.remove(0) {
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => instance,
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                            panic!("unexpected public ram rwc instance")
                        }
                    };
                    let mut output = match instances.remove(0) {
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => instance,
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                            panic!("unexpected public ram output instance")
                        }
                    };
                    let raf_claim_share = mpc_core::protocols::additive::AdditiveShare::zero();
                    let rwc_claim_share = mpc_core::protocols::additive::promote_to_trivial_share(
                        ram_input_claim,
                        io_ctx.party_id(),
                    );
                    let output_claim_share =
                        mpc_core::protocols::additive::promote_to_trivial_share(F::zero(), io_ctx.party_id());
                    Ok((
                        raf.compute_prover_message_share(0, raf_claim_share, 3, &mut io_ctx),
                        rwc.compute_prover_message_share(0, rwc_claim_share, 3, &mut io_ctx),
                        output.compute_prover_message_share(0, output_claim_share, 3, &mut io_ctx),
                    ))
                },
            );
        let rep3_raf_round0 = mpc_core::protocols::additive::combine_additive_vec(vec![
            ram_msg_shares[0].0.clone(),
            ram_msg_shares[1].0.clone(),
            ram_msg_shares[2].0.clone(),
        ]);
        let rep3_rwc_round0 = mpc_core::protocols::additive::combine_additive_vec(vec![
            ram_msg_shares[0].1.clone(),
            ram_msg_shares[1].1.clone(),
            ram_msg_shares[2].1.clone(),
        ]);
        let rep3_output_round0 = mpc_core::protocols::additive::combine_additive_vec(vec![
            ram_msg_shares[0].2.clone(),
            ram_msg_shares[1].2.clone(),
            ram_msg_shares[2].2.clone(),
        ]);
        let _ = (rep3_raf_round0, vanilla_raf_round0);
        assert_eq!(rep3_rwc_round0, vanilla_rwc_round0, "rv32 ram rwc round-0 mismatch");
        assert_eq!(
            rep3_output_round0,
            vanilla_output_round0,
            "rv32 ram output round-0 mismatch"
        );

        let (ram_valfinal_r_address, ram_valfinal_claim, ram_valfinal_input_claim, vanilla_ram_valfinal_round0) =
            vanilla_ram_valfinal_round0(
                &preprocessing,
                vanilla_trace.clone(),
                io_device.clone(),
                vanilla_memory.clone(),
            );
        let shares_ram_valfinal = shares.clone();
        let io_device_ram_valfinal = io_device.clone();
        let preprocessing_ram_valfinal = preprocessing.clone();
        let ram_valfinal_msg_shares: [Vec<mpc_core::protocols::additive::AdditiveShare<F>>; 3] =
            run_rep3_test(
                base_port + 47,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_ram_valfinal[party_idx].clone();
                    (
                        trace,
                        mem,
                        io_device_ram_valfinal.clone(),
                        preprocessing_ram_valfinal.clone(),
                        ram_K,
                        advice_shares,
                        ram_valfinal_r_address.clone(),
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                    Vec<Challenge>,
                ), mut io_ctx| {
                    let (
                        trace,
                        mem,
                        io_device,
                        preprocessing,
                        ram_k,
                        advice_shares,
                        r_address,
                    ) = input;
                    let budget =
                        co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                    let mut preproc =
                        mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<
                            F,
                            _,
                        >(
                            [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                            budget.dabits,
                            &mut io_ctx,
                        )?;
                    let mut state = StateManagerWorker::new(
                        &preprocessing,
                        trace,
                        io_device,
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    let _ = generate_witness_batch_rep3::<F, PCS, _>(
                        &[CommittedPolynomial::RamInc],
                        &mut state,
                        &mut io_ctx,
                        &mut preproc,
                    )?;
                    state.accumulator.append_virtual_public(
                        VirtualPolynomial::RamValFinal,
                        SumcheckId::RamOutputCheck,
                        jolt_core::poly::opening_proof::OpeningPoint::new(r_address),
                        ram_valfinal_claim,
                        io_ctx.party_id(),
                    );
                    let mut val_final =
                        co_jolt2::zkvm::ram::output_check::Rep3ValFinalSumcheckWorker::new(
                            &mut state,
                            ram_valfinal_input_claim,
                        );
                    Ok(val_final.compute_prover_message_share(
                        0,
                        mpc_core::protocols::additive::promote_to_trivial_share(
                            ram_valfinal_input_claim,
                            io_ctx.party_id(),
                        ),
                        3,
                        &mut io_ctx,
                    ))
                },
            );
        let rep3_ram_valfinal_round0 =
            mpc_core::protocols::additive::combine_additive_vec(vec![
                ram_valfinal_msg_shares[0].clone(),
                ram_valfinal_msg_shares[1].clone(),
                ram_valfinal_msg_shares[2].clone(),
            ]);
        assert_eq!(
            rep3_ram_valfinal_round0,
            vanilla_ram_valfinal_round0,
            "rv32 ram val-final round-0 mismatch"
        );

        if std::env::var("CO_JOLT2_DEBUG_RAF_ROUNDS").is_ok() {
            let mut raf_rng = test_rng();
            let raf_challenges: Vec<Challenge> = (0..ram_K.log_2())
                .map(|_| u128::rand(&mut raf_rng).into())
                .collect();

            let vanilla_raf_msgs = {
                let mut sm = VanillaStateManager::<F, FS, PCS>::new_prover(
                    &preprocessing,
                    vanilla_trace.clone(),
                    io_device.clone(),
                    None,
                    vanilla_memory.clone(),
                );
                sm.fiat_shamir_preamble();

                let (preprocessing, trace, _, _) = sm.get_prover_data();
                let trace_length = trace.len();
                let padded_trace_length = trace_length.next_power_of_two();
                let ram_k = sm.ram_K;
                let bytecode_d = preprocessing.shared.bytecode.d;

                let _guard = (
                    DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
                    AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), bytecode_d),
                );

                let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
                let mut all_polys =
                    CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);
                let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
                    .filter_map(|poly| all_polys.remove(poly))
                    .collect();
                let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);
                let (commitments, _opening_proof_hints): (Vec<_>, Vec<_>) =
                    commit_results.into_iter().unzip();
                sm.set_commitments(commitments);
                let transcript = sm.get_transcript();
                for commitment in sm.get_commitments().borrow().iter() {
                    transcript.borrow_mut().append_serializable(commitment);
                }

                let mut spartan = SpartanDag::<F>::new::<FS>(padded_trace_length);
                spartan.stage1_prove(&mut sm).unwrap();
                let mut registers_dag = RegistersDag::default();
                let _ = spartan.stage2_prover_instances(&mut sm);
                let _ = registers_dag.stage2_prover_instances(&mut sm);

                let mut ram_dag = RamDag::new_prover::<F, FS, PCS>(&sm);
                let mut instances = ram_dag.stage2_prover_instances(&mut sm);
                let mut raf = instances.remove(0);
                let mut previous_claim = raf.input_claim();
                let mut msgs = Vec::new();

                for (round, r_j) in raf_challenges.iter().copied().enumerate() {
                    let base = raf.compute_prover_message(round, previous_claim);
                    let y0 = base[0];
                    let y2 = base[1];
                    let y1 = previous_claim - y0;
                    let poly = jolt_core::poly::unipoly::UniPoly::from_evals(&[y0, y1, y2]);
                    let y3 = poly.evaluate(&F::from(3u64));
                    msgs.push(vec![y0, y2, y3]);
                    previous_claim = poly.evaluate(&r_j);
                    raf.bind(r_j, round);
                }

                msgs
            };

            let shares_raf_rounds = shares.clone();
            let io_device_raf_rounds = io_device.clone();
            let preprocessing_raf_rounds = preprocessing.clone();
            let rep3_raf_msgs: [Vec<Vec<mpc_core::protocols::additive::AdditiveShare<F>>>; 3] =
                run_rep3_test(
                    base_port + 46,
                    1,
                    move |party_idx| {
                        let (trace, mem, advice_shares) = shares_raf_rounds[party_idx].clone();
                        (
                            trace,
                            mem,
                            io_device_raf_rounds.clone(),
                            preprocessing_raf_rounds.clone(),
                            ram_K,
                            advice_shares,
                            raf_challenges.clone(),
                        )
                    },
                    move |input: (
                        Vec<Rep3Cycle>,
                        co_jolt2::host::memory::Rep3Memory,
                        tracer::JoltDevice,
                        JoltProverPreprocessing<F, PCS>,
                        usize,
                        co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                        Vec<Challenge>,
                    ), mut io_ctx| {
                        let (
                            trace,
                            mem,
                            io_device,
                            preprocessing,
                            ram_k,
                            advice_shares,
                            raf_challenges,
                        ) = input;
                        let budget =
                            co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
                        let mut preproc =
                            mpc_core::protocols::rep3_ring::preprocessing::edabits::preprocess_pool::<
                                F,
                                _,
                            >(
                                [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                                budget.dabits,
                                &mut io_ctx,
                            )?;
                        let mut state = StateManagerWorker::new(
                            &preprocessing,
                            trace,
                            io_device,
                            mem,
                            io_ctx.party_id(),
                            ram_k,
                            Some(advice_shares),
                        );
                        populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                        let _ = generate_witness_batch_rep3::<F, PCS, _>(
                            &[CommittedPolynomial::RamInc],
                            &mut state,
                            &mut io_ctx,
                            &mut preproc,
                        )?;
                        state.accumulator.append_virtual_public(
                            VirtualPolynomial::RamAddress,
                            SumcheckId::SpartanOuter,
                            jolt_core::poly::opening_proof::OpeningPoint::new(ram_address_r.clone()),
                            ram_address_claim,
                            io_ctx.party_id(),
                        );
                        state.accumulator.append_virtual_public(
                            VirtualPolynomial::RamReadValue,
                            SumcheckId::SpartanOuter,
                            jolt_core::poly::opening_proof::OpeningPoint::new(ram_read_r.clone()),
                            F::zero(),
                            io_ctx.party_id(),
                        );
                        let mut ram =
                            co_jolt2::zkvm::ram::Rep3RamDagWorker::new(&mut state, &mut io_ctx)?;
                        ram.set_stage2_init(
                            ram_gamma,
                            ram_input_claim,
                            ram_output_r_address.clone(),
                        );
                        let mut instances = ram.stage2_instances(&mut state, &mut io_ctx)?;
                        let mut raf = match instances.remove(0) {
                            co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => instance,
                            co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                                panic!("unexpected public ram raf instance")
                            }
                        };
                        let mut previous_claim = raf.input_claim().into_additive(io_ctx.party_id());
                        let mut msgs = Vec::new();
                        for (round, r_j) in raf_challenges.iter().copied().enumerate() {
                            let msg =
                                raf.compute_prover_message_share(round, previous_claim, 3, &mut io_ctx);
                            let y0 = msg[0].into_fe();
                            let y2 = msg[1].into_fe();
                            let y1 = (previous_claim - msg[0]).into_fe();
                            let poly = jolt_core::poly::unipoly::UniPoly::from_evals(&[y0, y1, y2]);
                            previous_claim =
                                mpc_core::protocols::additive::AdditiveShare::from_fe(
                                    poly.evaluate(&r_j),
                                );
                            raf.bind(r_j, round, &mut io_ctx, &mut preproc);
                            msgs.push(msg);
                        }
                        Ok(msgs)
                    },
                );

            for round in 0..vanilla_raf_msgs.len() {
                let rep3_round = mpc_core::protocols::additive::combine_additive_vec(vec![
                    rep3_raf_msgs[0][round].clone(),
                    rep3_raf_msgs[1][round].clone(),
                    rep3_raf_msgs[2][round].clone(),
                ]);
                assert_eq!(
                    rep3_round,
                    vanilla_raf_msgs[round],
                    "rv32 raf round mismatch at round {round}"
                );
            }
        }

    }

    // 4) Vanilla proof up to Stage3.
    let (vanilla_proof, tau) = vanilla_up_to_stage5(
        &preprocessing,
        vanilla_trace,
        io_device.clone(),
        vanilla_memory,
    );

    // 5) Rep3 proof up to Stage3 (local MPC, no QUIC).
    let preprocessing_arc = Arc::new(preprocessing);
    let verifier_preprocessing_arc = Arc::new(verifier_preprocessing);
    let io_device_arc = Arc::new(io_device);
    let shares_arc = Arc::new(shares);

    let preprocessing_arc_for_workers = Arc::clone(&preprocessing_arc);
    let verifier_preprocessing_arc_for_coord = Arc::clone(&verifier_preprocessing_arc);
    let io_device_arc_for_workers = Arc::clone(&io_device_arc);
    let io_device_arc_for_coord = Arc::clone(&io_device_arc);

    // NOTE: the in-memory Rep3 test network does not provide independent ring channels per IO fork,
    // so we must run with a single IO context to avoid protocol message interleaving.
    let (_worker_out, rep3_proof) = run_rep3_local_test_with_coordinator(
        1,
        {
            let shares_arc = Arc::clone(&shares_arc);
            let preprocessing_arc = Arc::clone(&preprocessing_arc_for_workers);
            let io_device_arc = Arc::clone(&io_device_arc_for_workers);
            move |party_idx| {
                let (trace, memory, advice_shares) = shares_arc[party_idx].clone();
                (
                    trace,
                    memory,
                    Arc::clone(&io_device_arc),
                    Arc::clone(&preprocessing_arc),
                    ram_K,
                    advice_shares,
                )
            }
        },
        {
            let verifier_preprocessing_arc = Arc::clone(&verifier_preprocessing_arc_for_coord);
            let prover_preprocessing_arc = Arc::clone(&preprocessing_arc);
            let io_device_arc = Arc::clone(&io_device_arc_for_coord);
            move || {
                (
                    Arc::clone(&verifier_preprocessing_arc),
                    Arc::clone(&prover_preprocessing_arc),
                    Arc::clone(&io_device_arc),
                    ram_K,
                )
            }
        },
        move |input, io_ctx| {
            let (trace, final_memory_state, program_io, preprocessing, ram_K, advice_shares) =
                input;
            let mut io_ctx = io_ctx;
            let party_id = io_ctx.party_id();

            // Preprocessing: create EdaBits pool for B2A conversions (2 rounds).
            let mut preproc = {
                use co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget;
                use mpc_core::protocols::rep3_ring::edabits;
                let budget = compute_edabit_budget(trace.len());
                edabits::preprocess_pool::<F, _>(
                    [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                    budget.dabits,
                    &mut io_ctx,
                )?
            };

            let state = StateManagerWorker::new(
                &preprocessing,
                trace,
                (*program_io).clone(),
                final_memory_state,
                party_id,
                ram_K,
                Some(advice_shares),
            );
            Rep3JoltDagWorker::prove::<F, PCS, FS, _>(state, &mut io_ctx, &mut preproc)
        },
        move |input, net| {
            let (verifier_preprocessing, prover_preprocessing, program_io, ram_K) = input;
            // Match twist_sumcheck_switch_index computation in co-jolt2 zkvm/mod.rs.
            let num_chunks = rayon::current_num_threads()
                .next_power_of_two()
                .min(padded_len);
            let chunk_size = if num_chunks > 0 {
                padded_len / num_chunks
            } else {
                padded_len
            };
            let twist_sumcheck_switch_index = if chunk_size > 0 {
                chunk_size.trailing_zeros() as usize
            } else {
                0
            };
            let state: StateManager<'_, F, FS, PCS> = StateManager::new(
                &verifier_preprocessing,
                (*program_io).clone(),
                ram_K,
                twist_sumcheck_switch_index,
            )
            .with_pcs_setup(&prover_preprocessing.generators);
            Rep3JoltDag::prove(state, net)
        },
    );

    // 6) Compare commitments.
    for (i, (r, v)) in rep3_proof.commitments.iter().zip(vanilla_proof.commitments.iter()).enumerate() {
        if r != v {
            eprintln!("Commitment mismatch at index {i}");
        }
    }
    assert_eq!(rep3_proof.commitments.len(), vanilla_proof.commitments.len(), "commitment count mismatch");
    assert_eq!(rep3_proof.commitments, vanilla_proof.commitments);

    // 7) Compare Stage1 sumcheck proof bytes.
    let rep3_stage1 = rep3_proof
        .proofs
        .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::Stage1Sumcheck)
        .expect("rep3 stage1 proof missing");
    let vanilla_stage1 = vanilla_proof
        .proofs
        .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::Stage1Sumcheck)
        .expect("vanilla stage1 proof missing");

    let rep3_bytes = {
        let mut v = Vec::new();
        rep3_stage1.serialize_uncompressed(&mut v).unwrap();
        v
    };
    let vanilla_bytes = {
        let mut v = Vec::new();
        vanilla_stage1.serialize_uncompressed(&mut v).unwrap();
        v
    };
    if rep3_bytes != vanilla_bytes {
        use jolt_core::zkvm::dag::state_manager::ProofData;

        let (rep3_sc, vanilla_sc) = match (rep3_stage1, vanilla_stage1) {
            (ProofData::SumcheckProof(a), ProofData::SumcheckProof(b)) => (a, b),
            _ => panic!("unexpected proof data variants for stage1 sumcheck"),
        };

        // Derive implied (t0, tInf) for round 0 from the compressed cubic, using tau.
        let implied_quadratic = |poly: &jolt_core::poly::unipoly::CompressedUniPoly<F>| -> (F, F) {
            let uni = poly.decompress(&F::zero());
            let w_i: F = tau[tau.len() - 1].into();
            let a = F::one() - w_i;
            let b = w_i + w_i - F::one();
            let t0 = uni.coeffs[0] / a;
            let t_inf = uni.coeffs[3] / b;
            (t0, t_inf)
        };

        let (rep3_t0, rep3_tinf) = implied_quadratic(&rep3_sc.compressed_polys[0]);
        let (van_t0, van_tinf) = implied_quadratic(&vanilla_sc.compressed_polys[0]);

        let mut first_diff_idx = None;
        for (i, (a, b)) in rep3_sc
            .compressed_polys
            .iter()
            .zip(vanilla_sc.compressed_polys.iter())
            .enumerate()
        {
            let mut a_bytes = Vec::new();
            let mut b_bytes = Vec::new();
            a.serialize_uncompressed(&mut a_bytes).unwrap();
            b.serialize_uncompressed(&mut b_bytes).unwrap();
            if a_bytes != b_bytes {
                first_diff_idx = Some(i);
                break;
            }
        }

        if let Some(i) = first_diff_idx {
            panic!(
                "stage1 sumcheck proof mismatch at round {i}: rep3={:?} vanilla={:?} (implied round0 t0/tInf: rep3=({rep3_t0:?},{rep3_tinf:?}) vanilla=({van_t0:?},{van_tinf:?}))",
                rep3_sc.compressed_polys[i],
                vanilla_sc.compressed_polys[i],
            );
        } else {
            panic!(
                "stage1 sumcheck proof bytes differ but polys are equal (len rep3={} vanilla={})",
                rep3_sc.compressed_polys.len(),
                vanilla_sc.compressed_polys.len()
            );
        }
    }

    // 8) Compare Stage2 sumcheck proof bytes.
    let rep3_stage2 = rep3_proof
        .proofs
        .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::Stage2Sumcheck)
        .expect("rep3 stage2 proof missing");
    let vanilla_stage2 = vanilla_proof
        .proofs
        .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::Stage2Sumcheck)
        .expect("vanilla stage2 proof missing");

    let rep3_stage2_bytes = {
        let mut v = Vec::new();
        rep3_stage2.serialize_uncompressed(&mut v).unwrap();
        v
    };
    let vanilla_stage2_bytes = {
        let mut v = Vec::new();
        vanilla_stage2.serialize_uncompressed(&mut v).unwrap();
        v
    };
    if rep3_stage2_bytes != vanilla_stage2_bytes {
        use jolt_core::zkvm::dag::state_manager::ProofData;

        let (rep3_sc, vanilla_sc) = match (rep3_stage2, vanilla_stage2) {
            (ProofData::SumcheckProof(a), ProofData::SumcheckProof(b)) => (a, b),
            _ => panic!("unexpected proof data variants for stage2 sumcheck"),
        };

        eprintln!(
            "Stage2 sumcheck: rep3 has {} polys, vanilla has {} polys",
            rep3_sc.compressed_polys.len(),
            vanilla_sc.compressed_polys.len()
        );

        let mut first_diff_idx = None;
        for (i, (a, b)) in rep3_sc
            .compressed_polys
            .iter()
            .zip(vanilla_sc.compressed_polys.iter())
            .enumerate()
        {
            let mut a_bytes = Vec::new();
            let mut b_bytes = Vec::new();
            a.serialize_uncompressed(&mut a_bytes).unwrap();
            b.serialize_uncompressed(&mut b_bytes).unwrap();
            if a_bytes != b_bytes {
                first_diff_idx = Some(i);
                break;
            }
        }

        if let Some(i) = first_diff_idx {
            panic!(
                "Stage2 sumcheck proof mismatch at round {i}: rep3={:?} vanilla={:?}",
                rep3_sc.compressed_polys[i], vanilla_sc.compressed_polys[i],
            );
        } else if rep3_sc.compressed_polys.len() != vanilla_sc.compressed_polys.len() {
            panic!(
                "Stage2 sumcheck proof poly count mismatch: rep3={} vanilla={}",
                rep3_sc.compressed_polys.len(),
                vanilla_sc.compressed_polys.len()
            );
        } else {
            panic!("Stage2 sumcheck proof bytes differ but individual polys match");
        }
    }

    if std::env::var("CO_JOLT2_STOP_AFTER_STAGE2").is_ok() {
        return;
    }

    // 9) Compare Stage3 sumcheck proof bytes.
    let rep3_stage3 = rep3_proof
        .proofs
        .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::Stage3Sumcheck)
        .expect("rep3 stage3 proof missing");
    let vanilla_stage3 = vanilla_proof
        .proofs
        .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::Stage3Sumcheck)
        .expect("vanilla stage3 proof missing");

    let rep3_stage3_bytes = {
        let mut v = Vec::new();
        rep3_stage3.serialize_uncompressed(&mut v).unwrap();
        v
    };
    let vanilla_stage3_bytes = {
        let mut v = Vec::new();
        vanilla_stage3.serialize_uncompressed(&mut v).unwrap();
        v
    };
    if rep3_stage3_bytes != vanilla_stage3_bytes {
        use jolt_core::zkvm::dag::state_manager::ProofData;

        let (rep3_sc, vanilla_sc) = match (rep3_stage3, vanilla_stage3) {
            (ProofData::SumcheckProof(a), ProofData::SumcheckProof(b)) => (a, b),
            _ => panic!("unexpected proof data variants for stage3 sumcheck"),
        };

        eprintln!(
            "Stage3 sumcheck: rep3 has {} polys, vanilla has {} polys",
            rep3_sc.compressed_polys.len(),
            vanilla_sc.compressed_polys.len()
        );

        let mut first_diff_idx = None;
        for (i, (a, b)) in rep3_sc
            .compressed_polys
            .iter()
            .zip(vanilla_sc.compressed_polys.iter())
            .enumerate()
        {
            let mut a_bytes = Vec::new();
            let mut b_bytes = Vec::new();
            a.serialize_uncompressed(&mut a_bytes).unwrap();
            b.serialize_uncompressed(&mut b_bytes).unwrap();
            if a_bytes != b_bytes {
                first_diff_idx = Some(i);
                break;
            }
        }

        if let Some(i) = first_diff_idx {
            panic!(
                "Stage3 sumcheck proof mismatch at round {i}: rep3={:?} vanilla={:?}",
                rep3_sc.compressed_polys[i], vanilla_sc.compressed_polys[i],
            );
        } else if rep3_sc.compressed_polys.len() != vanilla_sc.compressed_polys.len() {
            panic!(
                "Stage3 sumcheck proof poly count mismatch: rep3={} vanilla={}",
                rep3_sc.compressed_polys.len(),
                vanilla_sc.compressed_polys.len()
            );
        } else {
            panic!("Stage3 sumcheck proof bytes differ but individual polys match");
        }
    }

    if std::env::var("CO_JOLT2_STOP_AFTER_STAGE3").is_ok() {
        return;
    }

    // 9b) Compare Stage4 sumcheck proof bytes.
    {
        use jolt_core::zkvm::dag::state_manager::ProofData;

        let rep3_stage4 = rep3_proof
            .proofs
            .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::Stage4Sumcheck);
        let vanilla_stage4 = vanilla_proof
            .proofs
            .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::Stage4Sumcheck);

        match (rep3_stage4, vanilla_stage4) {
            (Some(rep3_s4), Some(vanilla_s4)) => {
                let rep3_bytes = {
                    let mut v = Vec::new();
                    rep3_s4.serialize_uncompressed(&mut v).unwrap();
                    v
                };
                let vanilla_bytes = {
                    let mut v = Vec::new();
                    vanilla_s4.serialize_uncompressed(&mut v).unwrap();
                    v
                };
                if rep3_bytes != vanilla_bytes {
                    let (rep3_sc, vanilla_sc) = match (rep3_s4, vanilla_s4) {
                        (ProofData::SumcheckProof(a), ProofData::SumcheckProof(b)) => (a, b),
                        _ => panic!("unexpected proof data variants for stage4 sumcheck"),
                    };

                    eprintln!(
                        "Stage4 sumcheck: rep3 has {} polys, vanilla has {} polys",
                        rep3_sc.compressed_polys.len(),
                        vanilla_sc.compressed_polys.len()
                    );

                    let mut first_diff_idx = None;
                    for (i, (a, b)) in rep3_sc
                        .compressed_polys
                        .iter()
                        .zip(vanilla_sc.compressed_polys.iter())
                        .enumerate()
                    {
                        let mut a_bytes = Vec::new();
                        let mut b_bytes = Vec::new();
                        a.serialize_uncompressed(&mut a_bytes).unwrap();
                        b.serialize_uncompressed(&mut b_bytes).unwrap();
                        if a_bytes != b_bytes {
                            first_diff_idx = Some(i);
                            break;
                        }
                    }

                    if let Some(i) = first_diff_idx {
                        panic!(
                            "Stage4 sumcheck proof mismatch at round {i}: rep3={:?} vanilla={:?}",
                            rep3_sc.compressed_polys[i], vanilla_sc.compressed_polys[i],
                        );
                    } else if rep3_sc.compressed_polys.len() != vanilla_sc.compressed_polys.len() {
                        panic!(
                            "Stage4 sumcheck proof poly count mismatch: rep3={} vanilla={}",
                            rep3_sc.compressed_polys.len(),
                            vanilla_sc.compressed_polys.len()
                        );
                    } else {
                        panic!("Stage4 sumcheck proof bytes differ but individual polys match");
                    }
                }
            }
            (None, None) => {
                // Both absent — OK (ReadRaf not populated).
            }
            (rep3, vanilla) => {
                panic!(
                    "Stage4 proof presence mismatch: rep3={}, vanilla={}",
                    rep3.is_some(),
                    vanilla.is_some()
                );
            }
        }
    }

    // 10) Compare opening claims bytes.
    {
        let rep3_claims_bytes = {
            let mut v = Vec::new();
            rep3_proof
                .opening_claims
                .serialize_uncompressed(&mut v)
                .unwrap();
            v
        };
        let vanilla_claims_bytes = {
            let mut v = Vec::new();
            vanilla_proof
                .opening_claims
                .serialize_uncompressed(&mut v)
                .unwrap();
            v
        };
        assert_eq!(
            rep3_claims_bytes, vanilla_claims_bytes,
            "Opening claims mismatch"
        );
    }

    // 11) Compare Stage5 (ReducedOpeningProof) bytes.
    {
        use jolt_core::zkvm::dag::state_manager::ProofData;

        let rep3_stage5 = rep3_proof
            .proofs
            .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::ReducedOpeningProof)
            .expect("rep3 stage5 proof missing");
        let vanilla_stage5 = vanilla_proof
            .proofs
            .get(&jolt_core::zkvm::dag::state_manager::ProofKeys::ReducedOpeningProof)
            .expect("vanilla stage5 proof missing");

        let rep3_bytes = {
            let mut v = Vec::new();
            rep3_stage5.serialize_uncompressed(&mut v).unwrap();
            v
        };
        let vanilla_bytes = {
            let mut v = Vec::new();
            vanilla_stage5.serialize_uncompressed(&mut v).unwrap();
            v
        };
        if rep3_bytes != vanilla_bytes {
            let (rep3_rop, vanilla_rop) = match (rep3_stage5, vanilla_stage5) {
                (ProofData::ReducedOpeningProof(a), ProofData::ReducedOpeningProof(b)) => (a, b),
                _ => panic!("unexpected proof data variants for stage5"),
            };

            // Compare sumcheck proof
            let rep3_sc_bytes = {
                let mut v = Vec::new();
                rep3_rop
                    .sumcheck_proof
                    .serialize_uncompressed(&mut v)
                    .unwrap();
                v
            };
            let vanilla_sc_bytes = {
                let mut v = Vec::new();
                vanilla_rop
                    .sumcheck_proof
                    .serialize_uncompressed(&mut v)
                    .unwrap();
                v
            };
            let sc_match = rep3_sc_bytes == vanilla_sc_bytes;

            // Compare claims
            let claims_match = rep3_rop.sumcheck_claims == vanilla_rop.sumcheck_claims;

            // Compare PCS proof
            let rep3_pcs_bytes = {
                let mut v = Vec::new();
                rep3_rop
                    .joint_opening_proof
                    .serialize_uncompressed(&mut v)
                    .unwrap();
                v
            };
            let vanilla_pcs_bytes = {
                let mut v = Vec::new();
                vanilla_rop
                    .joint_opening_proof
                    .serialize_uncompressed(&mut v)
                    .unwrap();
                v
            };
            let pcs_match = rep3_pcs_bytes == vanilla_pcs_bytes;

            if !sc_match {
                // Find first differing round
                let mut first_diff = None;
                for (i, (a, b)) in rep3_rop
                    .sumcheck_proof
                    .compressed_polys
                    .iter()
                    .zip(vanilla_rop.sumcheck_proof.compressed_polys.iter())
                    .enumerate()
                {
                    let mut a_bytes = Vec::new();
                    let mut b_bytes = Vec::new();
                    a.serialize_uncompressed(&mut a_bytes).unwrap();
                    b.serialize_uncompressed(&mut b_bytes).unwrap();
                    if a_bytes != b_bytes {
                        first_diff = Some(i);
                        break;
                    }
                }
                panic!(
                    "Stage5 mismatch: sumcheck={sc_match} (first diff round: {:?}, rep3_rounds={}, vanilla_rounds={}), claims={claims_match}, pcs={pcs_match}",
                    first_diff,
                    rep3_rop.sumcheck_proof.compressed_polys.len(),
                    vanilla_rop.sumcheck_proof.compressed_polys.len(),
                );
            }

            panic!("Stage5 mismatch: sumcheck={sc_match}, claims={claims_match}, pcs={pcs_match}");
        }
    }

    // 12) Metadata invariants.
    assert_eq!(rep3_proof.trace_length, vanilla_proof.trace_length);
    assert_eq!(rep3_proof.ram_K, vanilla_proof.ram_K);
    assert_eq!(rep3_proof.bytecode_d, vanilla_proof.bytecode_d);
    assert_eq!(
        rep3_proof.twist_sumcheck_switch_index,
        vanilla_proof.twist_sumcheck_switch_index
    );
}
