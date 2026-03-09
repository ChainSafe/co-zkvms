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
use co_jolt2::utils::test_utils::run_rep3_local_test_with_coordinator;
use co_jolt2::utils::test_utils::run_rep3_test;
use co_jolt2::utils::tracing::init_tracing;
use co_jolt2::utils::types::Either;
use co_jolt2::zkvm::dag::stage::SumcheckStagesWorker;
use co_jolt2::zkvm::dag::state_manager::{StateManager, StateManagerWorker};
use co_jolt2::zkvm::instruction::LookupIndexInt;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::r1cs::inputs::{compute_claimed_witness_evals_rep3, ALL_R1CS_INPUTS};
use co_jolt2::zkvm::witness::{generate_witness_batch_rep3, populate_cycle_witness_rep3};
use co_jolt2::zkvm::Rep3JoltWorker;
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
use jolt_core::zkvm::r1cs::constraints::UNIFORM_R1CS;
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use jolt_core::zkvm::ram::RamDag;
use jolt_core::zkvm::registers::RegistersDag;
use jolt_core::zkvm::spartan::SpartanDag;
use jolt_core::zkvm::witness::VirtualPolynomial;
use jolt_core::zkvm::witness::{
    compute_d_parameter, AllCommittedPolynomials, CommittedPolynomial, DTH_ROOT_OF_K,
};
use jolt_core::zkvm::{
    Jolt, JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing, JoltVerifierPreprocessing,
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
) -> (
    [F; jolt_core::zkvm::instruction_lookups::D],
    Vec<Challenge>,
    Vec<Challenge>,
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
        let r_address =
            tr.challenge_vector_optimized::<F>(jolt_core::zkvm::instruction_lookups::LOG_K_CHUNK);
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
    assert_eq!(
        lookups_instances.len(),
        1,
        "expected one lookups stage2 instance"
    );
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
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamReadValue,
                SumcheckId::SpartanOuter,
            );
        let (_, wv_claim) = sm
            .get_prover_accumulator()
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamWriteValue,
                SumcheckId::SpartanOuter,
            );
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

fn vanilla_ram_output_rounds(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: Vec<Cycle>,
    program_io: tracer::JoltDevice,
    final_memory_state: Memory,
    challenges: &[Challenge],
) -> Vec<Vec<F>> {
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

    let mut ram_dag = RamDag::new_prover::<F, FS, PCS>(&sm);
    let mut instances = ram_dag.stage2_prover_instances(&mut sm);
    let mut output = instances.remove(2);

    let mut rounds = Vec::with_capacity(challenges.len());
    for (round, challenge) in challenges.iter().copied().enumerate() {
        rounds.push(output.compute_prover_message(round, F::zero()));
        output.bind(challenge, round);
    }

    rounds
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

    (
        opening_point.r.clone(),
        val_final_claim,
        input_claim,
        vec![y0, y2, y3],
    )
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

    // 1) Build and trace the guest program.
    let guest = std::env::var("CO_JOLT2_GUEST").unwrap_or_else(|_| "fibonacci-guest".to_string());
    let mut program = Program::new(&guest);
    let inputs = if guest == "sha2-chain-guest" {
        program.set_stack_size(65536);
        program.set_memory_size(10240);
        let mut inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
        inputs.append(&mut postcard::to_stdvec(&1u32).unwrap());
        inputs
    } else {
        program.set_memory_size(10240);
        postcard::to_stdvec(&9u32).unwrap()
    };
    let (bytecode, memory_init, _) = program.decode();

    let mut rng = test_rng();
    let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);
    let (mut vanilla_trace, vanilla_memory, mut io_device) = program.trace(&inputs, &[], &[]);

    // Truncate trailing zeros on device outputs, matching what Jolt::prove does.
    io_device.outputs.truncate(
        io_device
            .outputs
            .iter()
            .rposition(|&b| b != 0)
            .map_or(0, |pos| pos + 1),
    );

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

    #[cfg(not(feature = "rv64"))]
    {
        let mut failures = Vec::new();
        for (step_idx, cycle) in vanilla_trace.iter().enumerate() {
            if matches!(cycle, Cycle::NoOp) {
                continue;
            }
            let row = jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                &shared,
                &vanilla_trace,
                step_idx,
            );
            for constraint in UNIFORM_R1CS.iter() {
                let a = constraint.cons.a.evaluate_row_with::<F>(&row);
                let b = constraint.cons.b.evaluate_row_with::<F>(&row);
                let c = constraint.cons.c.evaluate_row_with::<F>(&row);
                let residual = a * b - c;
                if !residual.is_zero() {
                    failures.push((
                        step_idx,
                        constraint.name,
                        a,
                        b,
                        c,
                        residual,
                        row.unexpanded_pc,
                        row.next_unexpanded_pc,
                        row.imm.to_i128(),
                        row.lookup_output,
                        row.should_branch,
                    ));
                    if failures.len() >= 16 {
                        break;
                    }
                }
            }
            if failures.len() >= 16 {
                break;
            }
        }
        if !failures.is_empty() {
            for (
                step_idx,
                name,
                a,
                b,
                c,
                residual,
                unexpanded_pc,
                next_unexpanded_pc,
                imm,
                lookup_output,
                should_branch,
            ) in failures
            {
                eprintln!(
                    "rv32 r1cs failure: step={} constraint={:?} a={:?} b={:?} c={:?} residual={:?} pc={} next_pc={} imm={} lookup_output={} should_branch={}",
                    step_idx, name, a, b, c, residual, unexpanded_pc, next_unexpanded_pc, imm, lookup_output, should_branch
                );
            }
            panic!("rv32 r1cs constraints are unsatisfied");
        }
    }

    // 3) Compute ram_K from vanilla trace (must match both sides).
    let ram_K = compute_ram_k(&vanilla_trace, &shared);

    #[cfg(feature = "rv64")]
    if std::env::var("CO_JOLT2_COMPARE_STAGE1_EVALS").is_ok() {
        use jolt_core::field::JoltField as _;
        use jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs;
        use jolt_core::zkvm::instruction::CircuitFlags;
        use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
        use mpc_core::protocols::rep3::arithmetic;

        let mut rng = test_rng();
        let r_cycle: Vec<Challenge> = (0..padded_len.ilog2() as usize)
            .map(|_| Challenge::random(&mut rng))
            .collect();
        let vanilla_evals = jolt_core::zkvm::r1cs::inputs::compute_claimed_witness_evals(
            &shared,
            &vanilla_trace,
            &r_cycle,
        );

        let preprocessing_arc = Arc::new(preprocessing.clone());
        let io_device_arc = Arc::new(io_device.clone());
        let shares_arc = Arc::new(shares.clone());
        let base_port: u16 = 15300;
        let share_evals: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>; 3] =
            run_rep3_test(
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r_cycle) =
                        input;
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
                        (*io_device).clone(),
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    compute_claimed_witness_evals_rep3::<F, PCS, _>(
                        &mut state,
                        &mut io_ctx,
                        &r_cycle,
                    )
                },
            );
        let shares_lookup = shares.clone();
        let io_device_lookup = io_device.clone();
        let preprocessing_lookup = preprocessing.clone();
        let lookup_shares: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>; 3] =
            run_rep3_test(
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
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
                    Ok(state.get_cycle_witness().stage1_lookup_output().to_vec())
                },
            );
        let opened_lookup = arithmetic::combine_field_elements_vec(vec![
            lookup_shares[0].clone(),
            lookup_shares[1].clone(),
            lookup_shares[2].clone(),
        ]);
        let vanilla_lookup: Vec<F> = (0..vanilla_trace.len())
            .map(|t| {
                jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                    &shared,
                    &vanilla_trace,
                    t,
                )
                .to_field(jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs::LookupOutput)
            })
            .collect();
        for (t, (rep3, vanilla)) in opened_lookup.iter().zip(vanilla_lookup.iter()).enumerate() {
            assert_eq!(
                rep3, vanilla,
                "lookup_output mismatch at step {t}: cycle={:?}",
                vanilla_trace[t]
            );
        }
        let opened = arithmetic::combine_field_elements_vec(vec![
            share_evals[0].clone(),
            share_evals[1].clone(),
            share_evals[2].clone(),
        ]);
        for (i, (rep3, vanilla)) in opened.iter().zip(vanilla_evals.iter()).enumerate() {
            assert_eq!(
                rep3, vanilla,
                "claimed eval mismatch at input {i} ({:?})",
                ALL_R1CS_INPUTS[i]
            );
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_ROWS").is_ok() {
            let shares_rows = shares.clone();
            let io_device_rows = io_device.clone();
            let preprocessing_rows = preprocessing.clone();
            let row_shares: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>; 3] =
                run_rep3_test(
                    base_port + 20,
                    1,
                    move |party_idx| {
                        let (trace, mem, advice_shares) = shares_rows[party_idx].clone();
                        (
                            trace,
                            mem,
                            io_device_rows.clone(),
                            preprocessing_rows.clone(),
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
                    ),
                          mut io_ctx| {
                        let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
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

                        let party_id = io_ctx.party_id();
                        let cycle_witness = state.get_cycle_witness();
                        let num_steps = cycle_witness.len();
                        let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                        let mask_left_rs1 =
                            1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                        let mask_right_rs2 =
                            1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                        let mask_both_shared = mask_left_rs1 | mask_right_rs2;

                        let mut shared_mul_rows = Vec::new();
                        let mut mul_map = vec![u32::MAX; num_steps];
                        for (t, &fb) in flags_bits.iter().enumerate() {
                            if (fb & mask_both_shared) == mask_both_shared {
                                mul_map[t] = shared_mul_rows.len() as u32;
                                shared_mul_rows.push(t);
                            }
                        }

                        let mul_products = if shared_mul_rows.is_empty() {
                            vec![]
                        } else {
                            let lhs: Vec<_> = shared_mul_rows
                                .iter()
                                .map(|&t| cycle_witness.row_stage1(t).rs1_value())
                                .collect();
                            let rhs: Vec<_> = shared_mul_rows
                                .iter()
                                .map(|&t| cycle_witness.row_stage1(t).rs2_value())
                                .collect();
                            rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
                        };

                        let mut out = Vec::with_capacity(num_steps * ALL_R1CS_INPUTS.len());
                        for t in 0..num_steps {
                            let row = cycle_witness.row_stage1(t);
                            let fb = flags_bits[t];
                            let left_shared = (fb & mask_left_rs1) != 0;
                            let right_shared = (fb & mask_right_rs2) != 0;
                            let product = if mul_map[t] != u32::MAX {
                                mul_products[mul_map[t] as usize]
                            } else {
                                match (left_shared, right_shared) {
                                    (true, false) => rep3_arithmetic::mul_public(
                                        row.rs1_value(),
                                        row.to_right_public_input(),
                                    ),
                                    (false, true) => rep3_arithmetic::mul_public(
                                        row.rs2_value(),
                                        row.to_left_public_input(),
                                    ),
                                    (false, false) => rep3_arithmetic::promote_to_trivial_share(
                                        party_id,
                                        row.to_left_public_input() * row.to_right_public_input(),
                                    ),
                                    (true, true) => unreachable!(),
                                }
                            };
                            let (left_input, right_input) = row.to_instruction_inputs(party_id);
                            let (left_lookup, right_lookup) =
                                row.to_lookup_operands(party_id, product);
                            let lookup_output = row.to_lookup_output();
                            let should_branch = if row.flag(CircuitFlags::Branch) {
                                lookup_output
                            } else {
                                mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share()
                            };
                            for input in ALL_R1CS_INPUTS.iter() {
                                let share = match input {
                                    JoltR1CSInputs::LeftInstructionInput => left_input,
                                    JoltR1CSInputs::RightInstructionInput => right_input,
                                    JoltR1CSInputs::Product => product,
                                    JoltR1CSInputs::LeftLookupOperand => left_lookup,
                                    JoltR1CSInputs::RightLookupOperand => right_lookup,
                                    JoltR1CSInputs::LookupOutput => lookup_output,
                                    JoltR1CSInputs::Rs1Value => row.rs1_value(),
                                    JoltR1CSInputs::Rs2Value => row.rs2_value(),
                                    JoltR1CSInputs::RdWriteValue => row.rd_write_value(),
                                    JoltR1CSInputs::RamReadValue => row.ram_read_value(),
                                    JoltR1CSInputs::RamWriteValue => row.ram_write_value(),
                                    JoltR1CSInputs::ShouldBranch => should_branch,
                                    JoltR1CSInputs::WriteLookupOutputToRD => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_u64(
                                                if row.flag(CircuitFlags::WriteLookupOutputToRD) {
                                                    row.rd_addr() as u64
                                                } else {
                                                    0
                                                },
                                            ),
                                        )
                                    }
                                    JoltR1CSInputs::WritePCtoRD => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_u64(
                                                if row.flag(CircuitFlags::Jump) {
                                                    row.rd_addr() as u64
                                                } else {
                                                    0
                                                },
                                            ),
                                        )
                                    }
                                    JoltR1CSInputs::PC => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_u64(row.pc_index()),
                                        )
                                    }
                                    JoltR1CSInputs::NextPC => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_u64(row.next_pc_index()),
                                        )
                                    }
                                    JoltR1CSInputs::UnexpandedPC => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_u64(row.unexpanded_pc()),
                                        )
                                    }
                                    JoltR1CSInputs::NextUnexpandedPC => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_u64(row.next_unexpanded_pc()),
                                        )
                                    }
                                    JoltR1CSInputs::Imm => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_i128(row.imm()),
                                        )
                                    }
                                    JoltR1CSInputs::Rd => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_u64(row.rd_addr() as u64),
                                        )
                                    }
                                    JoltR1CSInputs::RamAddress => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_u64(row.ram_addr()),
                                        )
                                    }
                                    JoltR1CSInputs::NextIsNoop => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_bool(row.next_is_noop()),
                                        )
                                    }
                                    JoltR1CSInputs::ShouldJump => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_bool(row.should_jump()),
                                        )
                                    }
                                    JoltR1CSInputs::OpFlags(flag) => {
                                        mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            F::from_bool(row.flag(*flag)),
                                        )
                                    }
                                };
                                out.push(share);
                            }
                        }
                        Ok(out)
                    },
                );
            let opened_rows = arithmetic::combine_field_elements_vec(vec![
                row_shares[0].clone(),
                row_shares[1].clone(),
                row_shares[2].clone(),
            ]);
            for t in 0..vanilla_trace.len() {
                let vanilla_row = jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                    &shared,
                    &vanilla_trace,
                    t,
                );
                for (field_idx, input) in ALL_R1CS_INPUTS.iter().enumerate() {
                    let rep3 = opened_rows[t * ALL_R1CS_INPUTS.len() + field_idx];
                    let vanilla = vanilla_row.to_field(*input);
                    assert_eq!(
                        rep3, vanilla,
                        "stage1 row mismatch at step {t} input {:?}: cycle={:?}",
                        input, vanilla_trace[t]
                    );
                }
            }
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_ROUND0_B").is_ok() {
            use jolt_core::zkvm::r1cs::constraints::UNIFORM_R1CS;

            let r0 = Challenge::random(&mut rng);
            let shares_round0 = shares.clone();
            let io_device_round0 = io_device.clone();
            let preprocessing_round0 = preprocessing.clone();
            let round0_shares: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>; 3] =
                run_rep3_test(
                    base_port + 30,
                    1,
                    move |party_idx| {
                        let (trace, mem, advice_shares) = shares_round0[party_idx].clone();
                        (
                            trace,
                            mem,
                            io_device_round0.clone(),
                            preprocessing_round0.clone(),
                            ram_K,
                            advice_shares,
                            r0,
                        )
                    },
                    move |input: (
                        Vec<Rep3Cycle>,
                        co_jolt2::host::memory::Rep3Memory,
                        tracer::JoltDevice,
                        JoltProverPreprocessing<F, PCS>,
                        usize,
                        co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                        Challenge,
                    ),
                          mut io_ctx| {
                        let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r0) =
                            input;
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

                        let party_id = io_ctx.party_id();
                        let cycle_witness = state.get_cycle_witness();
                        let num_steps = cycle_witness.len();
                        let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                        let mask_left_rs1 =
                            1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                        let mask_right_rs2 =
                            1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                        let mask_both_shared = mask_left_rs1 | mask_right_rs2;

                        let mut shared_mul_rows = Vec::new();
                        let mut mul_map = vec![u32::MAX; num_steps];
                        for (t, &fb) in flags_bits.iter().enumerate() {
                            if (fb & mask_both_shared) == mask_both_shared {
                                mul_map[t] = shared_mul_rows.len() as u32;
                                shared_mul_rows.push(t);
                            }
                        }

                        let mul_products = if shared_mul_rows.is_empty() {
                            vec![]
                        } else {
                            let lhs: Vec<_> = shared_mul_rows
                                .iter()
                                .map(|&t| cycle_witness.row_stage1(t).rs1_value())
                                .collect();
                            let rhs: Vec<_> = shared_mul_rows
                                .iter()
                                .map(|&t| cycle_witness.row_stage1(t).rs2_value())
                                .collect();
                            rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
                        };

                        let input_share = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                           product: mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>,
                                           input: JoltR1CSInputs| {
                            match input {
                                JoltR1CSInputs::LeftInstructionInput => row.to_instruction_inputs(party_id).0,
                                JoltR1CSInputs::RightInstructionInput => row.to_instruction_inputs(party_id).1,
                                JoltR1CSInputs::Product => product,
                                JoltR1CSInputs::LeftLookupOperand => row.to_lookup_operands(party_id, product).0,
                                JoltR1CSInputs::RightLookupOperand => row.to_lookup_operands(party_id, product).1,
                                JoltR1CSInputs::LookupOutput => row.to_lookup_output(),
                                JoltR1CSInputs::Rs1Value => row.rs1_value(),
                                JoltR1CSInputs::Rs2Value => row.rs2_value(),
                                JoltR1CSInputs::RdWriteValue => row.rd_write_value(),
                                JoltR1CSInputs::RamReadValue => row.ram_read_value(),
                                JoltR1CSInputs::RamWriteValue => row.ram_write_value(),
                                JoltR1CSInputs::ShouldBranch => {
                                    if row.flag(CircuitFlags::Branch) {
                                        row.to_lookup_output()
                                    } else {
                                        mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share()
                                    }
                                }
                                JoltR1CSInputs::WriteLookupOutputToRD => rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(if row.flag(CircuitFlags::WriteLookupOutputToRD) { row.rd_addr() as u64 } else { 0 }),
                                ),
                                JoltR1CSInputs::WritePCtoRD => rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(if row.flag(CircuitFlags::Jump) { row.rd_addr() as u64 } else { 0 }),
                                ),
                                JoltR1CSInputs::PC => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.pc_index())),
                                JoltR1CSInputs::NextPC => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.next_pc_index())),
                                JoltR1CSInputs::UnexpandedPC => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.unexpanded_pc())),
                                JoltR1CSInputs::NextUnexpandedPC => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.next_unexpanded_pc())),
                                JoltR1CSInputs::Imm => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_i128(row.imm())),
                                JoltR1CSInputs::Rd => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.rd_addr() as u64)),
                                JoltR1CSInputs::RamAddress => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.ram_addr())),
                                JoltR1CSInputs::NextIsNoop => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_bool(row.next_is_noop())),
                                JoltR1CSInputs::ShouldJump => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_bool(row.should_jump())),
                                JoltR1CSInputs::OpFlags(flag) => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_bool(row.flag(flag))),
                            }
                        };

                        let num_pairs = UNIFORM_R1CS.len().next_power_of_two() / 2;
                        let r0_f: F = r0.into();
                        let mut out = Vec::with_capacity(num_steps * num_pairs);
                        for t in 0..num_steps {
                            let row = cycle_witness.row_stage1(t);
                            let fb = flags_bits[t];
                            let left_shared = (fb & mask_left_rs1) != 0;
                            let right_shared = (fb & mask_right_rs2) != 0;
                            let product = if mul_map[t] != u32::MAX {
                                mul_products[mul_map[t] as usize]
                            } else {
                                match (left_shared, right_shared) {
                                    (true, false) => rep3_arithmetic::mul_public(
                                        row.rs1_value(),
                                        row.to_right_public_input(),
                                    ),
                                    (false, true) => rep3_arithmetic::mul_public(
                                        row.rs2_value(),
                                        row.to_left_public_input(),
                                    ),
                                    (false, false) => rep3_arithmetic::promote_to_trivial_share(
                                        party_id,
                                        row.to_left_public_input() * row.to_right_public_input(),
                                    ),
                                    (true, true) => unreachable!(),
                                }
                            };
                            for pair in 0..num_pairs {
                                let eval_b = |constraint_idx: usize| {
                                    if constraint_idx >= UNIFORM_R1CS.len() {
                                        return mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share();
                                    }
                                    let mut acc =
                                        mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share();
                                    UNIFORM_R1CS[constraint_idx].cons.b.for_each_term(|input_index, coeff| {
                                        let scalar = F::from_i128(coeff.to_i128());
                                        acc += rep3_arithmetic::mul_public(
                                            input_share(row, product, JoltR1CSInputs::from_index(input_index)),
                                            scalar,
                                        );
                                    });
                                    if let Some(c) = UNIFORM_R1CS[constraint_idx].cons.b.const_term() {
                                        acc = rep3_arithmetic::add_public(acc, F::from_i128(c.to_i128()), party_id);
                                    }
                                    acc
                                };
                                let c0 = pair * 2;
                                let c1 = c0 + 1;
                                let b0 = eval_b(c0);
                                let b1 = eval_b(c1);
                                out.push(b0 + rep3_arithmetic::mul_public(b1 - b0, r0_f));
                            }
                        }
                        Ok(out)
                    },
                );

            let opened_round0 = arithmetic::combine_field_elements_vec(vec![
                round0_shares[0].clone(),
                round0_shares[1].clone(),
                round0_shares[2].clone(),
            ]);
            let num_pairs = UNIFORM_R1CS.len().next_power_of_two() / 2;
            let r0_f: F = r0.into();
            for t in 0..vanilla_trace.len() {
                let row_inputs = jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                    &shared,
                    &vanilla_trace,
                    t,
                );
                for pair in 0..num_pairs {
                    let c0 = pair * 2;
                    let c1 = c0 + 1;
                    let b0 = if c0 < UNIFORM_R1CS.len() {
                        UNIFORM_R1CS[c0].cons.b.evaluate_row_with::<F>(&row_inputs)
                    } else {
                        F::zero()
                    };
                    let b1 = if c1 < UNIFORM_R1CS.len() {
                        UNIFORM_R1CS[c1].cons.b.evaluate_row_with::<F>(&row_inputs)
                    } else {
                        F::zero()
                    };
                    let vanilla_bound = b0 + (b1 - b0) * r0_f;
                    let rep3_bound = opened_round0[t * num_pairs + pair];
                    assert_eq!(
                        rep3_bound, vanilla_bound,
                        "round0 B mismatch at step {t}, pair {pair}, constraints ({:?}, {:?})",
                        UNIFORM_R1CS.get(c0).map(|c| c.name),
                        UNIFORM_R1CS.get(c1).map(|c| c.name),
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "rv64"))]
    {
        use mpc_core::protocols::rep3::arithmetic;
        use mpc_core::protocols::rep3_ring::combine_ring_element_binary;

        let mut rng = test_rng();
        let r_cycle: Vec<Challenge> = (0..padded_len.ilog2() as usize)
            .map(|_| Challenge::random(&mut rng))
            .collect();
        let vanilla_evals = jolt_core::zkvm::r1cs::inputs::compute_claimed_witness_evals(
            &shared,
            &vanilla_trace,
            &r_cycle,
        );

        let preprocessing_arc = Arc::new(preprocessing.clone());
        let io_device_arc = Arc::new(io_device.clone());
        let shares_arc = Arc::new(shares.clone());
        let base_port: u16 = 15300;
        let share_evals: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>; 3] =
            run_rep3_test(
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r_cycle) =
                        input;
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
                        (*io_device).clone(),
                        mem,
                        io_ctx.party_id(),
                        ram_k,
                        Some(advice_shares),
                    );
                    populate_cycle_witness_rep3(&mut state, &mut io_ctx, &mut preproc)?;
                    compute_claimed_witness_evals_rep3::<F, PCS, _>(
                        &mut state,
                        &mut io_ctx,
                        &r_cycle,
                    )
                },
            );
        let shares_lookup = shares.clone();
        let io_device_lookup = io_device.clone();
        let preprocessing_lookup = preprocessing.clone();
        let lookup_shares: [Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>>; 3] =
            run_rep3_test(
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
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
                    Ok(state.get_cycle_witness().stage1_lookup_output().to_vec())
                },
            );
        let opened_lookup = arithmetic::combine_field_elements_vec(vec![
            lookup_shares[0].clone(),
            lookup_shares[1].clone(),
            lookup_shares[2].clone(),
        ]);
        let vanilla_lookup: Vec<F> = (0..vanilla_trace.len())
            .map(|t| {
                jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                    &shared,
                    &vanilla_trace,
                    t,
                )
                .to_field(jolt_core::zkvm::r1cs::inputs::JoltR1CSInputs::LookupOutput)
            })
            .collect();
        for (t, (rep3, vanilla)) in opened_lookup.iter().zip(vanilla_lookup.iter()).enumerate() {
            assert_eq!(
                rep3, vanilla,
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
            assert_eq!(
                rep3, vanilla,
                "rv32 claimed eval mismatch at input {i} ({:?})",
                ALL_R1CS_INPUTS[i]
            );
        }

        let shares_indices = shares.clone();
        let io_device_indices = io_device.clone();
        let preprocessing_indices = preprocessing.clone();
        let lookup_index_shares: [Vec<
            Either<LookupIndexInt, mpc_core::protocols::rep3_ring::Rep3RingShare<LookupIndexInt>>,
        >; 3] = run_rep3_test(
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
            ),
                  mut io_ctx| {
                let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
                let budget =
                    co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
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
                generate_witness_batch_rep3::<F, PCS, _>(
                    &[],
                    &mut state,
                    &mut io_ctx,
                    &mut preproc,
                )?;
                Ok(state
                    .prover_state
                    .cycle_witness
                    .take_read_raf()
                    .lookup_indices)
            },
        );
        let opened_indices: Vec<LookupIndexInt> = (0..lookup_index_shares[0].len())
            .map(|i| {
                match (
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
                }
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
                rep3, vanilla,
                "rv32 lookup_index mismatch at step {t}: cycle={:?}",
                vanilla_trace[t]
            );
        }

        let stage2_polys = vec![CommittedPolynomial::RdInc, CommittedPolynomial::RamInc];
        let vanilla_stage2_witness = CommittedPolynomial::generate_witness_batch(
            &stage2_polys,
            &preprocessing,
            &vanilla_trace,
        );
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
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
                    generate_witness_batch_rep3::<F, PCS, _>(
                        &stage2_polys_worker,
                        &mut state,
                        &mut io_ctx,
                        &mut preproc,
                    )
                },
            );

        for poly in stage2_polys {
            let rep3_poly = combine_poly_shares_rep3(
                rep3_stage2_witness
                    .iter()
                    .map(|party_map| {
                        match party_map.get(&poly).expect("missing stage2 witness poly") {
                            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(dense)) => {
                                dense.clone()
                            }
                            other => {
                                panic!("expected dense shared poly for {poly:?}, got {other:?}")
                            }
                        }
                    })
                    .collect(),
            );
            let vanilla_poly = vanilla_stage2_witness
                .get(&poly)
                .expect("missing vanilla stage2 witness poly");
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r_cycle) =
                        input;
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
                        &[CommittedPolynomial::RdInc],
                        &mut state,
                        &mut io_ctx,
                        &mut preproc,
                    )?;
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
            rep3_reg_round0, vanilla_reg_round0,
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r_cycle) =
                        input;
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
                    let claimed_witness_evals = compute_claimed_witness_evals_rep3::<F, PCS, _>(
                        &mut state,
                        &mut io_ctx,
                        &r_cycle,
                    )?;
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
            rep3_inner_round0, vanilla_inner_round0,
            "rv32 spartan inner round-0 mismatch"
        );

        let (bool_gamma, bool_r_address, bool_r_cycle, vanilla_bool_round0) =
            vanilla_lookup_booleanity_round0(
                &preprocessing,
                vanilla_trace.clone(),
                io_device.clone(),
                vanilla_memory.clone(),
            );
        let instruction_ra_polys: Vec<CommittedPolynomial> = (0
            ..jolt_core::zkvm::instruction_lookups::D)
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
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
                    let witness = generate_witness_batch_rep3::<F, PCS, _>(
                        &instruction_ra_polys,
                        &mut state,
                        &mut io_ctx,
                        &mut preproc,
                    )?;
                    let one_hot_polys = std::array::from_fn(|i| {
                        match witness
                            .get(&CommittedPolynomial::InstructionRa(i))
                            .expect("missing instruction ra poly")
                        {
                            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(poly)) => {
                                poly.clone()
                            }
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
                        co_jolt2::zkvm::instruction_lookups::Rep3LookupsDagWorker::new(
                            one_hot_polys,
                        );
                    lookups.set_stage2_init(bool_gamma, bool_r_address.clone());
                    let mut instances = lookups.stage2_instances(&mut state, &mut io_ctx)?;
                    assert_eq!(
                        instances.len(),
                        1,
                        "expected one rep3 lookups stage2 instance"
                    );
                    let instance = match instances.pop().expect("missing instance") {
                        co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(
                            instance,
                        ) => instance,
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
            rep3_bool_round0, vanilla_bool_round0,
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
        let ram_output_r_address_for_round0 = ram_output_r_address_round0.clone();
        let ram_address_r_for_round0 = ram_address_r_round0.clone();
        let ram_read_r_for_round0 = ram_read_r_round0.clone();
        let shares_ram = shares.clone();
        let io_device_ram = io_device.clone();
        let preprocessing_ram = preprocessing.clone();
        let ram_msg_shares: [(
            Vec<mpc_core::protocols::additive::AdditiveShare<F>>,
            Vec<mpc_core::protocols::additive::AdditiveShare<F>>,
            Vec<mpc_core::protocols::additive::AdditiveShare<F>>,
        ); 3] = run_rep3_test(
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
                    ram_address_r_for_round0.clone(),
                    ram_read_r_for_round0.clone(),
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
            ),
                  mut io_ctx| {
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
                let mut ram = co_jolt2::zkvm::ram::Rep3RamDagWorker::new(&mut state, &mut io_ctx)?;
                ram.set_stage2_init(
                    ram_gamma,
                    ram_input_claim,
                    ram_output_r_address_for_round0.clone(),
                );
                let mut instances = ram.stage2_instances(&mut state, &mut io_ctx)?;
                assert_eq!(
                    instances.len(),
                    3,
                    "expected three rep3 ram stage2 instances"
                );
                let mut raf = match instances.remove(0) {
                    co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => {
                        instance
                    }
                    co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                        panic!("unexpected public ram raf instance")
                    }
                };
                let mut rwc = match instances.remove(0) {
                    co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => {
                        instance
                    }
                    co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                        panic!("unexpected public ram rwc instance")
                    }
                };
                let mut output = match instances.remove(0) {
                    co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => {
                        instance
                    }
                    co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                        panic!("unexpected public ram output instance")
                    }
                };
                let raf_claim_share = mpc_core::protocols::additive::AdditiveShare::zero();
                let rwc_claim_share = mpc_core::protocols::additive::promote_to_trivial_share(
                    ram_input_claim,
                    io_ctx.party_id(),
                );
                let output_claim_share = mpc_core::protocols::additive::promote_to_trivial_share(
                    F::zero(),
                    io_ctx.party_id(),
                );
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
        assert_eq!(
            rep3_rwc_round0, vanilla_rwc_round0,
            "rv32 ram rwc round-0 mismatch"
        );
        assert_eq!(
            rep3_output_round0, vanilla_output_round0,
            "rv32 ram output round-0 mismatch"
        );

        let vanilla_output_rounds = vanilla_ram_output_rounds(
            &preprocessing,
            vanilla_trace.clone(),
            io_device.clone(),
            vanilla_memory.clone(),
            &ram_output_r_address_round0,
        );
        let shares_ram_output = shares.clone();
        let io_device_ram_output = io_device.clone();
        let preprocessing_ram_output = preprocessing.clone();
        let rep3_output_round_shares: [Vec<Vec<mpc_core::protocols::additive::AdditiveShare<F>>>;
            3] = run_rep3_test(
            base_port + 46,
            1,
            move |party_idx| {
                let (trace, mem, advice_shares) = shares_ram_output[party_idx].clone();
                (
                    trace,
                    mem,
                    io_device_ram_output.clone(),
                    preprocessing_ram_output.clone(),
                    ram_K,
                    advice_shares,
                    ram_output_r_address_round0.clone(),
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
            ),
                  mut io_ctx| {
                let (trace, mem, io_device, preprocessing, ram_k, advice_shares, challenges) =
                    input;
                let budget =
                    co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget(trace.len());
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
                let _ = generate_witness_batch_rep3::<F, PCS, _>(
                    &[CommittedPolynomial::RamInc],
                    &mut state,
                    &mut io_ctx,
                    &mut preproc,
                )?;
                state.accumulator.append_virtual_public(
                    VirtualPolynomial::RamAddress,
                    SumcheckId::SpartanOuter,
                    jolt_core::poly::opening_proof::OpeningPoint::new(ram_address_r_round0.clone()),
                    ram_address_claim,
                    io_ctx.party_id(),
                );
                state.accumulator.append_virtual_public(
                    VirtualPolynomial::RamReadValue,
                    SumcheckId::SpartanOuter,
                    jolt_core::poly::opening_proof::OpeningPoint::new(ram_read_r_round0.clone()),
                    F::zero(),
                    io_ctx.party_id(),
                );
                let mut ram = co_jolt2::zkvm::ram::Rep3RamDagWorker::new(&mut state, &mut io_ctx)?;
                ram.set_stage2_init(ram_gamma, ram_input_claim, challenges.clone());
                let mut instances = ram.stage2_instances(&mut state, &mut io_ctx)?;
                let mut output = match instances.remove(2) {
                    co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(instance) => {
                        instance
                    }
                    co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(_) => {
                        panic!("unexpected public ram output instance")
                    }
                };
                let zero_claim = mpc_core::protocols::additive::promote_to_trivial_share(
                    F::zero(),
                    io_ctx.party_id(),
                );
                let mut rounds = Vec::with_capacity(challenges.len());
                for (round, challenge) in challenges.into_iter().enumerate() {
                    rounds.push(output.compute_prover_message_share(
                        round,
                        zero_claim,
                        3,
                        &mut io_ctx,
                    ));
                    output.bind(challenge, round, &mut io_ctx, &mut preproc);
                }
                Ok(rounds)
            },
        );
        let rep3_output_rounds: Vec<Vec<F>> = (0..vanilla_output_rounds.len())
            .map(|round| {
                mpc_core::protocols::additive::combine_additive_vec(vec![
                    rep3_output_round_shares[0][round].clone(),
                    rep3_output_round_shares[1][round].clone(),
                    rep3_output_round_shares[2][round].clone(),
                ])
            })
            .collect();
        assert_eq!(
            rep3_output_rounds, vanilla_output_rounds,
            "rv32 ram output full-round mismatch"
        );

        let (
            ram_valfinal_r_address,
            ram_valfinal_claim,
            ram_valfinal_input_claim,
            vanilla_ram_valfinal_round0,
        ) = vanilla_ram_valfinal_round0(
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, r_address) =
                        input;
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
        let rep3_ram_valfinal_round0 = mpc_core::protocols::additive::combine_additive_vec(vec![
            ram_valfinal_msg_shares[0].clone(),
            ram_valfinal_msg_shares[1].clone(),
            ram_valfinal_msg_shares[2].clone(),
        ]);
        assert_eq!(
            rep3_ram_valfinal_round0, vanilla_ram_valfinal_round0,
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
                    ),
                          mut io_ctx| {
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
                            jolt_core::poly::opening_proof::OpeningPoint::new(
                                ram_address_r.clone(),
                            ),
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
                            co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Secret(
                                instance,
                            ) => instance,
                            co_jolt2::zkvm::dag::stage::BatchedSumcheckWorkerInstance::Public(
                                _,
                            ) => {
                                panic!("unexpected public ram raf instance")
                            }
                        };
                        let mut previous_claim = raf.input_claim().into_additive(io_ctx.party_id());
                        let mut msgs = Vec::new();
                        for (round, r_j) in raf_challenges.iter().copied().enumerate() {
                            let msg = raf.compute_prover_message_share(
                                round,
                                previous_claim,
                                3,
                                &mut io_ctx,
                            );
                            let y0 = msg[0].into_fe();
                            let y2 = msg[1].into_fe();
                            let y1 = (previous_claim - msg[0]).into_fe();
                            let poly = jolt_core::poly::unipoly::UniPoly::from_evals(&[y0, y1, y2]);
                            previous_claim = mpc_core::protocols::additive::AdditiveShare::from_fe(
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
                    rep3_round, vanilla_raf_msgs[round],
                    "rv32 raf round mismatch at round {round}"
                );
            }
        }
    }

    // 4) Vanilla proof up to Stage3.
    let vanilla_trace_for_direct = if std::env::var("CO_JOLT2_COMPARE_STAGE1_DIRECT_CLAIMS").is_ok()
    {
        Some(vanilla_trace.clone())
    } else {
        None
    };
    let (vanilla_proof, tau) = vanilla_up_to_stage5(
        &preprocessing,
        vanilla_trace,
        io_device.clone(),
        vanilla_memory,
    );

    // 5) Rep3 proof up to Stage3 (local MPC, no QUIC).
    let preprocessing_arc = Arc::new(preprocessing);
    let verifier_preprocessing_arc = Arc::new(verifier_preprocessing.clone());
    let io_device_arc = Arc::new(io_device.clone());
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
    for (i, (r, v)) in rep3_proof
        .commitments
        .iter()
        .zip(vanilla_proof.commitments.iter())
        .enumerate()
    {
        if r != v {
            eprintln!("Commitment mismatch at index {i}");
        }
    }
    assert_eq!(
        rep3_proof.commitments.len(),
        vanilla_proof.commitments.len(),
        "commitment count mismatch"
    );
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
    if std::env::var("CO_JOLT2_COMPARE_STAGE1_DIRECT_CLAIMS").is_ok() {
        use jolt_core::poly::eq_poly::EqPolynomial;
        use jolt_core::poly::opening_proof::{OpeningId, SumcheckId};
        use jolt_core::zkvm::r1cs::constraints::UNIFORM_R1CS;
        use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
        use jolt_core::zkvm::witness::VirtualPolynomial;

        let az_id = OpeningId::Virtual(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let bz_id = OpeningId::Virtual(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let cz_id = OpeningId::Virtual(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);

        let (outer_point, rep3_claim_az) = rep3_proof.opening_claims.0.get(&az_id).unwrap();
        let (_, rep3_claim_bz) = rep3_proof.opening_claims.0.get(&bz_id).unwrap();
        let (_, rep3_claim_cz) = rep3_proof.opening_claims.0.get(&cz_id).unwrap();

        let key = UniformSpartanKey::<F>::new(padded_len);
        let num_cycles_bits = key.num_steps.ilog2() as usize;
        let (r_cycle, r_constr) = outer_point.r.split_at(num_cycles_bits);
        let eq_cycle = EqPolynomial::<F>::evals(r_cycle);
        let eq_constr = EqPolynomial::<F>::evals(r_constr);

        let mut direct_az = F::zero();
        let mut direct_bz = F::zero();
        let mut direct_cz = F::zero();
        let vanilla_trace = vanilla_trace_for_direct
            .as_ref()
            .expect("vanilla trace clone missing for direct stage1 claim check");
        for (t, eq_t) in eq_cycle.iter().enumerate().take(vanilla_trace.len()) {
            let row_inputs = jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                &shared,
                &vanilla_trace,
                t,
            );
            for (row_idx, named) in UNIFORM_R1CS.iter().enumerate() {
                let w = *eq_t * eq_constr[row_idx];
                direct_az += w * named.cons.a.evaluate_row_with::<F>(&row_inputs);
                direct_bz += w * named.cons.b.evaluate_row_with::<F>(&row_inputs);
                direct_cz += w * named.cons.c.evaluate_row_with::<F>(&row_inputs);
            }
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_FINAL_B_PER_CONSTRAINT").is_ok() {
            use co_jolt2::zkvm::r1cs::inputs::JoltR1CSInputs;
            use jolt_core::field::JoltField;
            use jolt_core::zkvm::instruction::CircuitFlags;
            use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;

            let shares_stage1_final = Arc::clone(&shares_arc);
            let io_device_stage1_final = Arc::clone(&io_device_arc);
            let preprocessing_stage1_final = Arc::clone(&preprocessing_arc);
            let eq_cycle_debug = eq_cycle.clone();
            let eq_constr_debug = eq_constr.clone();

            let per_constraint_shares: [Vec<mpc_core::protocols::additive::AdditiveShare<F>>; 3] =
                run_rep3_test(
                    15331,
                    1,
                    move |party_idx| {
                        let (trace, mem, advice_shares) = shares_stage1_final[party_idx].clone();
                        (
                            trace,
                            mem,
                            (*io_device_stage1_final).clone(),
                            (*preprocessing_stage1_final).clone(),
                            ram_K,
                            advice_shares,
                            eq_cycle_debug.clone(),
                            eq_constr_debug.clone(),
                        )
                    },
                    move |input: (
                        Vec<Rep3Cycle>,
                        co_jolt2::host::memory::Rep3Memory,
                        tracer::JoltDevice,
                        JoltProverPreprocessing<F, PCS>,
                        usize,
                        co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                        Vec<F>,
                        Vec<F>,
                    ),
                          mut io_ctx| {
                        let (
                            trace,
                            mem,
                            io_device,
                            preprocessing,
                            ram_k,
                            advice_shares,
                            eq_cycle,
                            eq_constr,
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

                        let party_id = io_ctx.party_id();
                        let cycle_witness = state.get_cycle_witness();
                        let num_steps = cycle_witness.len();
                        let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                        let mask_left_rs1 =
                            1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                        let mask_right_rs2 =
                            1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                        let mask_both_shared = mask_left_rs1 | mask_right_rs2;

                        let mut shared_mul_rows = Vec::new();
                        let mut mul_map = vec![u32::MAX; num_steps];
                        for (t, &fb) in flags_bits.iter().enumerate() {
                            if (fb & mask_both_shared) == mask_both_shared {
                                mul_map[t] = shared_mul_rows.len() as u32;
                                shared_mul_rows.push(t);
                            }
                        }

                        let mul_products = if shared_mul_rows.is_empty() {
                            vec![]
                        } else {
                            let lhs: Vec<_> = shared_mul_rows
                                .iter()
                                .map(|&t| cycle_witness.row_stage1(t).rs1_value())
                                .collect();
                            let rhs: Vec<_> = shared_mul_rows
                                .iter()
                                .map(|&t| cycle_witness.row_stage1(t).rs2_value())
                                .collect();
                            rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
                        };

                        let input_share = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                           product: mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>,
                                           input: JoltR1CSInputs| {
                            match input {
                                JoltR1CSInputs::LeftInstructionInput => row.to_instruction_inputs(party_id).0,
                                JoltR1CSInputs::RightInstructionInput => row.to_instruction_inputs(party_id).1,
                                JoltR1CSInputs::Product => product,
                                JoltR1CSInputs::LeftLookupOperand => row.to_lookup_operands(party_id, product).0,
                                JoltR1CSInputs::RightLookupOperand => row.to_lookup_operands(party_id, product).1,
                                JoltR1CSInputs::LookupOutput => row.to_lookup_output(),
                                JoltR1CSInputs::Rs1Value => row.rs1_value(),
                                JoltR1CSInputs::Rs2Value => row.rs2_value(),
                                JoltR1CSInputs::RdWriteValue => row.rd_write_value(),
                                JoltR1CSInputs::RamReadValue => row.ram_read_value(),
                                JoltR1CSInputs::RamWriteValue => row.ram_write_value(),
                                JoltR1CSInputs::ShouldBranch => {
                                    if row.flag(CircuitFlags::Branch) {
                                        row.to_lookup_output()
                                    } else {
                                        mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share()
                                    }
                                }
                                JoltR1CSInputs::WriteLookupOutputToRD => rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(if row.flag(CircuitFlags::WriteLookupOutputToRD) { row.rd_addr() as u64 } else { 0 }),
                                ),
                                JoltR1CSInputs::WritePCtoRD => rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(if row.flag(CircuitFlags::Jump) { row.rd_addr() as u64 } else { 0 }),
                                ),
                                JoltR1CSInputs::PC => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.pc_index())),
                                JoltR1CSInputs::NextPC => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.next_pc_index())),
                                JoltR1CSInputs::UnexpandedPC => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.unexpanded_pc())),
                                JoltR1CSInputs::NextUnexpandedPC => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.next_unexpanded_pc())),
                                JoltR1CSInputs::Imm => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_i128(row.imm())),
                                JoltR1CSInputs::Rd => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.rd_addr() as u64)),
                                JoltR1CSInputs::RamAddress => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_u64(row.ram_addr())),
                                JoltR1CSInputs::NextIsNoop => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_bool(row.next_is_noop())),
                                JoltR1CSInputs::ShouldJump => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_bool(row.should_jump())),
                                JoltR1CSInputs::OpFlags(flag) => rep3_arithmetic::promote_to_trivial_share(party_id, F::from_bool(row.flag(flag))),
                            }
                        };

                        let mut out = vec![mpc_core::protocols::additive::AdditiveShare::zero(); UNIFORM_R1CS.len()];
                        for t in 0..num_steps {
                            let row = cycle_witness.row_stage1(t);
                            let fb = flags_bits[t];
                            let left_shared = (fb & mask_left_rs1) != 0;
                            let right_shared = (fb & mask_right_rs2) != 0;
                            let product = if mul_map[t] != u32::MAX {
                                mul_products[mul_map[t] as usize]
                            } else {
                                match (left_shared, right_shared) {
                                    (true, false) => rep3_arithmetic::mul_public(
                                        row.rs1_value(),
                                        row.to_right_public_input(),
                                    ),
                                    (false, true) => rep3_arithmetic::mul_public(
                                        row.rs2_value(),
                                        row.to_left_public_input(),
                                    ),
                                    (false, false) => rep3_arithmetic::promote_to_trivial_share(
                                        party_id,
                                        row.to_left_public_input() * row.to_right_public_input(),
                                    ),
                                    (true, true) => unreachable!(),
                                }
                            };
                            for (constraint_idx, named) in UNIFORM_R1CS.iter().enumerate() {
                                let mut acc =
                                    mpc_core::protocols::rep3::Rep3PrimeFieldShare::zero_share();
                                named.cons.b.for_each_term(|input_index, coeff| {
                                    let scalar = F::from_i128(coeff.to_i128());
                                    acc += rep3_arithmetic::mul_public(
                                        input_share(
                                            row,
                                            product,
                                            JoltR1CSInputs::from_index(input_index),
                                        ),
                                        scalar,
                                    );
                                });
                                if let Some(c) = named.cons.b.const_term() {
                                    acc = rep3_arithmetic::add_public(
                                        acc,
                                        F::from_i128(c.to_i128()),
                                        party_id,
                                    );
                                }
                                out[constraint_idx] +=
                                    rep3_arithmetic::mul_public(
                                        acc,
                                        eq_cycle[t] * eq_constr[constraint_idx],
                                    )
                                    .into_additive();
                            }
                        }
                        Ok(out)
                    },
                );

            let rep3_per_constraint =
                mpc_core::protocols::additive::combine_additive_vec(vec![
                    per_constraint_shares[0].clone(),
                    per_constraint_shares[1].clone(),
                    per_constraint_shares[2].clone(),
                ]);
            let direct_per_constraint: Vec<F> = (0..UNIFORM_R1CS.len())
                .map(|constraint_idx| {
                    let mut acc = F::zero();
                    for (t, eq_t) in eq_cycle.iter().enumerate().take(vanilla_trace.len()) {
                        let row_inputs =
                            jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                                &shared,
                                &vanilla_trace,
                                t,
                            );
                        acc += *eq_t
                            * eq_constr[constraint_idx]
                            * UNIFORM_R1CS[constraint_idx]
                                .cons
                                .b
                                .evaluate_row_with::<F>(&row_inputs);
                    }
                    acc
                })
                .collect();

            let rep3_sum: F = rep3_per_constraint.iter().copied().sum();
            assert_eq!(rep3_sum, *rep3_claim_bz, "rep3 per-constraint B sum mismatch");

            let diffs: Vec<_> = direct_per_constraint
                .iter()
                .zip(rep3_per_constraint.iter())
                .enumerate()
                .filter_map(|(idx, (a, b))| if a != b { Some(idx) } else { None })
                .collect();
            if let Some(&first_diff) = diffs.first() {
                panic!(
                    "stage1 B per-constraint mismatch at constraint {} ({:?}): direct={:?} rep3={:?}; all_diffs={:?}",
                    first_diff,
                    UNIFORM_R1CS[first_diff].name,
                    direct_per_constraint[first_diff],
                    rep3_per_constraint[first_diff],
                    diffs
                );
            }
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_LOOKUP_PATHS").is_ok() {
            use co_jolt2::zkvm::r1cs::inputs::{JoltR1CSInputs, Rep3R1CSCycleInputs};
            use jolt_core::field::JoltField;
            use jolt_core::zkvm::instruction::CircuitFlags;
            use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
            use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

            let target_constraints = [14usize, 15, 17, 21, 22];
            let shares_stage1_paths = Arc::clone(&shares_arc);
            let io_device_stage1_paths = Arc::clone(&io_device_arc);
            let preprocessing_stage1_paths = Arc::clone(&preprocessing_arc);

            let path_shares: [Vec<Rep3PrimeFieldShare<F>>; 3] = run_rep3_test(
                15332,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_stage1_paths[party_idx].clone();
                    (
                        trace,
                        mem,
                        (*io_device_stage1_paths).clone(),
                        (*preprocessing_stage1_paths).clone(),
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
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

                    let party_id = io_ctx.party_id();
                    let cycle_witness = state.get_cycle_witness();
                    let num_steps = cycle_witness.len();
                    let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                    let mask_left_rs1 =
                        1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                    let mask_right_rs2 =
                        1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                    let mask_both_shared = mask_left_rs1 | mask_right_rs2;

                    let mut shared_mul_rows = Vec::new();
                    let mut mul_map = vec![u32::MAX; num_steps];
                    for (t, &fb) in flags_bits.iter().enumerate() {
                        if (fb & mask_both_shared) == mask_both_shared {
                            mul_map[t] = shared_mul_rows.len() as u32;
                            shared_mul_rows.push(t);
                        }
                    }

                    let mul_products = if shared_mul_rows.is_empty() {
                        vec![]
                    } else {
                        let lhs: Vec<_> = shared_mul_rows
                            .iter()
                            .map(|&t| cycle_witness.row_stage1(t).rs1_value())
                            .collect();
                        let rhs: Vec<_> = shared_mul_rows
                            .iter()
                            .map(|&t| cycle_witness.row_stage1(t).rs2_value())
                            .collect();
                        rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
                    };

                    let input_share = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                       product: Rep3PrimeFieldShare<F>,
                                       input: JoltR1CSInputs| {
                        match input {
                            JoltR1CSInputs::LeftInstructionInput => {
                                row.to_instruction_inputs(party_id).0
                            }
                            JoltR1CSInputs::RightInstructionInput => {
                                row.to_instruction_inputs(party_id).1
                            }
                            JoltR1CSInputs::Product => product,
                            JoltR1CSInputs::LeftLookupOperand => {
                                row.to_lookup_operands(party_id, product).0
                            }
                            JoltR1CSInputs::RightLookupOperand => {
                                row.to_lookup_operands(party_id, product).1
                            }
                            JoltR1CSInputs::LookupOutput => row.to_lookup_output(),
                            JoltR1CSInputs::Rs1Value => row.rs1_value(),
                            JoltR1CSInputs::Rs2Value => row.rs2_value(),
                            JoltR1CSInputs::RdWriteValue => row.rd_write_value(),
                            JoltR1CSInputs::RamReadValue => row.ram_read_value(),
                            JoltR1CSInputs::RamWriteValue => row.ram_write_value(),
                            JoltR1CSInputs::ShouldBranch => {
                                if row.flag(CircuitFlags::Branch) {
                                    row.to_lookup_output()
                                } else {
                                    Rep3PrimeFieldShare::zero_share()
                                }
                            }
                            JoltR1CSInputs::WriteLookupOutputToRD => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(
                                        if row.flag(CircuitFlags::WriteLookupOutputToRD) {
                                            row.rd_addr() as u64
                                        } else {
                                            0
                                        },
                                    ),
                                )
                            }
                            JoltR1CSInputs::WritePCtoRD => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(if row.flag(CircuitFlags::Jump) {
                                    row.rd_addr() as u64
                                } else {
                                    0
                                }),
                            ),
                            JoltR1CSInputs::PC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.pc_index()),
                            ),
                            JoltR1CSInputs::NextPC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.next_pc_index()),
                            ),
                            JoltR1CSInputs::UnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.unexpanded_pc()),
                                )
                            }
                            JoltR1CSInputs::NextUnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.next_unexpanded_pc()),
                                )
                            }
                            JoltR1CSInputs::Imm => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_i128(row.imm()),
                            ),
                            JoltR1CSInputs::Rd => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.rd_addr() as u64),
                            ),
                            JoltR1CSInputs::RamAddress => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.ram_addr()),
                                )
                            }
                            JoltR1CSInputs::NextIsNoop => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.next_is_noop()),
                                )
                            }
                            JoltR1CSInputs::ShouldJump => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.should_jump()),
                                )
                            }
                            JoltR1CSInputs::OpFlags(flag) => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.flag(flag)),
                                )
                            }
                        }
                    };

                    let input_cycle = |inputs: &Rep3R1CSCycleInputs<F>,
                                       input: JoltR1CSInputs|
                     -> Rep3PrimeFieldShare<F> {
                        match input {
                            JoltR1CSInputs::LeftInstructionInput => inputs.left_input,
                            JoltR1CSInputs::RightInstructionInput => inputs.right_input,
                            JoltR1CSInputs::Product => inputs.product,
                            JoltR1CSInputs::LeftLookupOperand => inputs.left_lookup,
                            JoltR1CSInputs::RightLookupOperand => inputs.right_lookup,
                            JoltR1CSInputs::LookupOutput => inputs.lookup_output,
                            JoltR1CSInputs::Rs1Value => inputs.rs1_read_value,
                            JoltR1CSInputs::Rs2Value => inputs.rs2_read_value,
                            JoltR1CSInputs::RdWriteValue => inputs.rd_write_value,
                            JoltR1CSInputs::RamReadValue => inputs.ram_read_value,
                            JoltR1CSInputs::RamWriteValue => inputs.ram_write_value,
                            JoltR1CSInputs::ShouldBranch => inputs.should_branch,
                            JoltR1CSInputs::WriteLookupOutputToRD => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(inputs.write_lookup_output_to_rd_addr as u64),
                                )
                            }
                            JoltR1CSInputs::WritePCtoRD => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(inputs.write_pc_to_rd_addr as u64),
                            ),
                            JoltR1CSInputs::PC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(inputs.pc),
                            ),
                            JoltR1CSInputs::NextPC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(inputs.next_pc),
                            ),
                            JoltR1CSInputs::UnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(inputs.unexpanded_pc),
                                )
                            }
                            JoltR1CSInputs::NextUnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(inputs.next_unexpanded_pc),
                                )
                            }
                            JoltR1CSInputs::Imm => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_i128(inputs.imm),
                            ),
                            JoltR1CSInputs::Rd => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(inputs.rd_addr as u64),
                            ),
                            JoltR1CSInputs::RamAddress => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(inputs.ram_addr),
                                )
                            }
                            JoltR1CSInputs::NextIsNoop => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(inputs.next_is_noop),
                                )
                            }
                            JoltR1CSInputs::ShouldJump => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(inputs.should_jump),
                                )
                            }
                            JoltR1CSInputs::OpFlags(flag) => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(inputs.flags[flag as usize]),
                                )
                            }
                        }
                    };

                    let eval_lc_from_row = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                            product: Rep3PrimeFieldShare<F>,
                                            constraint_idx: usize| {
                        let mut acc = Rep3PrimeFieldShare::zero_share();
                        UNIFORM_R1CS[constraint_idx]
                            .cons
                            .b
                            .for_each_term(|input_index, coeff| {
                                let scalar = F::from_i128(coeff.to_i128());
                                acc += rep3_arithmetic::mul_public(
                                    input_share(row, product, JoltR1CSInputs::from_index(input_index)),
                                    scalar,
                                );
                            });
                        if let Some(c) = UNIFORM_R1CS[constraint_idx].cons.b.const_term() {
                            acc = rep3_arithmetic::add_public(
                                acc,
                                F::from_i128(c.to_i128()),
                                party_id,
                            );
                        }
                        acc
                    };

                    let eval_lc_from_cycle =
                        |inputs: &Rep3R1CSCycleInputs<F>, constraint_idx: usize| {
                            let mut acc = Rep3PrimeFieldShare::zero_share();
                            UNIFORM_R1CS[constraint_idx]
                                .cons
                                .b
                                .for_each_term(|input_index, coeff| {
                                    let scalar = F::from_i128(coeff.to_i128());
                                    acc += rep3_arithmetic::mul_public(
                                        input_cycle(inputs, JoltR1CSInputs::from_index(input_index)),
                                        scalar,
                                    );
                                });
                            if let Some(c) = UNIFORM_R1CS[constraint_idx].cons.b.const_term() {
                                acc = rep3_arithmetic::add_public(
                                    acc,
                                    F::from_i128(c.to_i128()),
                                    party_id,
                                );
                            }
                            acc
                        };

                    let mut out =
                        Vec::with_capacity(num_steps * (2 + 2 * target_constraints.len()));
                    for t in 0..num_steps {
                        let row = cycle_witness.row_stage1(t);
                        let fb = flags_bits[t];
                        let left_shared = (fb & mask_left_rs1) != 0;
                        let right_shared = (fb & mask_right_rs2) != 0;
                        let product = if mul_map[t] != u32::MAX {
                            mul_products[mul_map[t] as usize]
                        } else {
                            match (left_shared, right_shared) {
                                (true, false) => rep3_arithmetic::mul_public(
                                    row.rs1_value(),
                                    row.to_right_public_input(),
                                ),
                                (false, true) => rep3_arithmetic::mul_public(
                                    row.rs2_value(),
                                    row.to_left_public_input(),
                                ),
                                (false, false) => rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    row.to_left_public_input() * row.to_right_public_input(),
                                ),
                                (true, true) => unreachable!(),
                            }
                        };
                        let cycle_inputs = Rep3R1CSCycleInputs::from_trace(party_id, row, product);
                        out.push(row.to_lookup_output());
                        out.push(cycle_inputs.lookup_output);
                        for &constraint_idx in &target_constraints {
                            out.push(eval_lc_from_row(row, product, constraint_idx));
                            out.push(eval_lc_from_cycle(&cycle_inputs, constraint_idx));
                        }
                    }
                    Ok(out)
                },
            );

            let opened = rep3_arithmetic::combine_field_elements_vec(vec![
                path_shares[0].clone(),
                path_shares[1].clone(),
                path_shares[2].clone(),
            ]);
            let stride = 2 + 2 * target_constraints.len();
            for t in 0..vanilla_trace.len() {
                let row_inputs = jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                    &shared,
                    &vanilla_trace,
                    t,
                );
                let base = t * stride;
                let row_lookup = opened[base];
                let cycle_lookup = opened[base + 1];
                let vanilla_lookup = F::from_u64(row_inputs.lookup_output);
                assert_eq!(
                    row_lookup, vanilla_lookup,
                    "stage1 lookup_output row path mismatch at step {t}"
                );
                assert_eq!(
                    cycle_lookup, vanilla_lookup,
                    "stage1 lookup_output cycle-input path mismatch at step {t}"
                );
                for (offset, &constraint_idx) in target_constraints.iter().enumerate() {
                    let row_eval = opened[base + 2 + 2 * offset];
                    let cycle_eval = opened[base + 2 + 2 * offset + 1];
                    let vanilla_eval =
                        UNIFORM_R1CS[constraint_idx].cons.b.evaluate_row_with::<F>(&row_inputs);
                    assert_eq!(
                        row_eval, vanilla_eval,
                        "stage1 row-path cons.b mismatch at step {t} constraint {} ({:?})",
                        constraint_idx, UNIFORM_R1CS[constraint_idx].name
                    );
                    assert_eq!(
                        cycle_eval, vanilla_eval,
                        "stage1 cycle-input cons.b mismatch at step {t} constraint {} ({:?})",
                        constraint_idx, UNIFORM_R1CS[constraint_idx].name
                    );
                }
            }
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_PAIR_14_15").is_ok() {
            use jolt_core::field::JoltField;
            use co_jolt2::utils::types::Rep3Value;
            use co_jolt2::zkvm::r1cs::inputs::{JoltR1CSInputs, Rep3R1CSCycleInputs};
            use jolt_core::zkvm::instruction::CircuitFlags;
            use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
            use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

            let target_step = 3957usize;
            let r0: F = outer_point.r.last().copied().unwrap().into();
            let shares_stage1_pair = Arc::clone(&shares_arc);
            let io_device_stage1_pair = Arc::clone(&io_device_arc);
            let preprocessing_stage1_pair = Arc::clone(&preprocessing_arc);

            let opened = rep3_arithmetic::combine_field_elements_vec(run_rep3_test(
                15338,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_stage1_pair[party_idx].clone();
                    (
                        trace,
                        mem,
                        (*io_device_stage1_pair).clone(),
                        (*preprocessing_stage1_pair).clone(),
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
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares) = input;
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

                    let party_id = io_ctx.party_id();
                    let cycle_witness = state.get_cycle_witness();
                    let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                    let row = cycle_witness.row_stage1(target_step);
                    let fb = flags_bits[target_step];
                    let mask_left_rs1 =
                        1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                    let mask_right_rs2 =
                        1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                    let left_shared = (fb & mask_left_rs1) != 0;
                    let right_shared = (fb & mask_right_rs2) != 0;
                    let product = match (left_shared, right_shared) {
                        (true, true) => {
                            rep3_arithmetic::mul(row.rs1_value(), row.rs2_value(), io_ctx.main())?
                        }
                        (true, false) => {
                            rep3_arithmetic::mul_public(row.rs1_value(), row.to_right_public_input())
                        }
                        (false, true) => {
                            rep3_arithmetic::mul_public(row.rs2_value(), row.to_left_public_input())
                        }
                        (false, false) => rep3_arithmetic::promote_to_trivial_share(
                            party_id,
                            row.to_left_public_input() * row.to_right_public_input(),
                        ),
                    };
                    let inputs = Rep3R1CSCycleInputs::from_trace(party_id, row, product);

                    let input_as_value = |input: JoltR1CSInputs| -> Rep3Value<F> {
                        match input {
                            JoltR1CSInputs::LeftInstructionInput => Rep3Value::Shared(inputs.left_input),
                            JoltR1CSInputs::RightInstructionInput => Rep3Value::Shared(inputs.right_input),
                            JoltR1CSInputs::Product => Rep3Value::Shared(inputs.product),
                            JoltR1CSInputs::LeftLookupOperand => Rep3Value::Shared(inputs.left_lookup),
                            JoltR1CSInputs::RightLookupOperand => Rep3Value::Shared(inputs.right_lookup),
                            JoltR1CSInputs::LookupOutput => Rep3Value::Shared(inputs.lookup_output),
                            JoltR1CSInputs::Rs1Value => Rep3Value::Shared(inputs.rs1_read_value),
                            JoltR1CSInputs::Rs2Value => Rep3Value::Shared(inputs.rs2_read_value),
                            JoltR1CSInputs::RdWriteValue => Rep3Value::Shared(inputs.rd_write_value),
                            JoltR1CSInputs::RamReadValue => Rep3Value::Shared(inputs.ram_read_value),
                            JoltR1CSInputs::RamWriteValue => Rep3Value::Shared(inputs.ram_write_value),
                            JoltR1CSInputs::ShouldBranch => Rep3Value::Shared(inputs.should_branch),
                            JoltR1CSInputs::WriteLookupOutputToRD => {
                                Rep3Value::Public(F::from_u64(inputs.write_lookup_output_to_rd_addr as u64))
                            }
                            JoltR1CSInputs::WritePCtoRD => {
                                Rep3Value::Public(F::from_u64(inputs.write_pc_to_rd_addr as u64))
                            }
                            JoltR1CSInputs::PC => Rep3Value::Public(F::from_u64(inputs.pc)),
                            JoltR1CSInputs::NextPC => Rep3Value::Public(F::from_u64(inputs.next_pc)),
                            JoltR1CSInputs::UnexpandedPC => {
                                Rep3Value::Public(F::from_u64(inputs.unexpanded_pc))
                            }
                            JoltR1CSInputs::NextUnexpandedPC => {
                                Rep3Value::Public(F::from_u64(inputs.next_unexpanded_pc))
                            }
                            JoltR1CSInputs::Imm => Rep3Value::Public(F::from_i128(inputs.imm)),
                            JoltR1CSInputs::Rd => Rep3Value::Public(F::from_u64(inputs.rd_addr as u64)),
                            JoltR1CSInputs::RamAddress => Rep3Value::Public(F::from_u64(inputs.ram_addr)),
                            JoltR1CSInputs::NextIsNoop => Rep3Value::Public(F::from_bool(inputs.next_is_noop)),
                            JoltR1CSInputs::ShouldJump => Rep3Value::Public(F::from_bool(inputs.should_jump)),
                            JoltR1CSInputs::OpFlags(flag) => {
                                Rep3Value::Public(F::from_bool(inputs.flags[flag as usize]))
                            }
                        }
                    };

                    let eval_lc_rep3 = |lc: jolt_core::zkvm::r1cs::constraints::LC| {
                        let mut acc = Rep3Value::<F>::zero_public();
                        lc.for_each_term(|input_index, coeff| {
                            let scalar = F::from_i128(coeff.to_i128());
                            let val = input_as_value(JoltR1CSInputs::from_index(input_index));
                            acc.add_assign(&val.mul_public(scalar), party_id);
                        });
                        if let Some(c) = lc.const_term() {
                            acc.add_public_assign(F::from_i128(c.to_i128()), party_id);
                        }
                        acc
                    };

                    let b14 = eval_lc_rep3(UNIFORM_R1CS[14].cons.b);
                    let b15 = eval_lc_rep3(UNIFORM_R1CS[15].cons.b);
                    let bound = b14.add(&b15.sub(&b14, party_id).mul_public(r0), party_id);

                    let to_share = |value: Rep3Value<F>| -> Rep3PrimeFieldShare<F> {
                        match value {
                            Rep3Value::Public(x) => {
                                rep3_arithmetic::promote_to_trivial_share(party_id, x)
                            }
                            Rep3Value::Shared(x) => x,
                            Rep3Value::Additive(_) => unreachable!(),
                        }
                    };

                    Ok(vec![
                        inputs.right_input,
                        inputs.right_lookup,
                        inputs.lookup_output,
                        to_share(b14),
                        to_share(b15),
                        to_share(bound),
                    ])
                },
            ).to_vec());

            let row_inputs = jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                &shared,
                &vanilla_trace,
                target_step,
            );
            let vanilla_b14 = UNIFORM_R1CS[14].cons.b.evaluate_row_with::<F>(&row_inputs);
            let vanilla_b15 = UNIFORM_R1CS[15].cons.b.evaluate_row_with::<F>(&row_inputs);
            let vanilla_bound = vanilla_b14 + (vanilla_b15 - vanilla_b14) * r0;

            assert_eq!(opened[3], vanilla_b14, "step {} constraint 14 Rep3Value B mismatch", target_step);
            assert_eq!(opened[4], vanilla_b15, "step {} constraint 15 Rep3Value B mismatch", target_step);
            assert_eq!(
                opened[5], vanilla_bound,
                "step {} pair (14,15) Rep3Value round0 bound mismatch: right_input={} right_lookup={} lookup_output={}",
                target_step,
                opened[0],
                opened[1],
                opened[2],
            );
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_SHARE_SPARSE_B").is_ok() {
            use co_jolt2::zkvm::r1cs::inputs::JoltR1CSInputs;
            use jolt_core::field::JoltField;
            use jolt_core::zkvm::instruction::CircuitFlags;
            use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
            use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

            let target_constraints = [15usize, 17, 21, 22];
            let challenges_low_to_high: Vec<F> =
                outer_point.r.iter().rev().map(|r| (*r).into()).collect();
            let shares_stage1_sparse = Arc::clone(&shares_arc);
            let io_device_stage1_sparse = Arc::clone(&io_device_arc);
            let preprocessing_stage1_sparse = Arc::clone(&preprocessing_arc);

            let sparse_shares: [Vec<Rep3PrimeFieldShare<F>>; 3] = run_rep3_test(
                15333,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_stage1_sparse[party_idx].clone();
                    (
                        trace,
                        mem,
                        (*io_device_stage1_sparse).clone(),
                        (*preprocessing_stage1_sparse).clone(),
                        ram_K,
                        advice_shares,
                        challenges_low_to_high.clone(),
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                    Vec<F>,
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, challenges) =
                        input;
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

                    let party_id = io_ctx.party_id();
                    let cycle_witness = state.get_cycle_witness();
                    let num_steps = cycle_witness.len();
                    let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                    let mask_left_rs1 =
                        1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                    let mask_right_rs2 =
                        1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                    let mask_both_shared = mask_left_rs1 | mask_right_rs2;

                    let mut shared_mul_rows = Vec::new();
                    let mut mul_map = vec![u32::MAX; num_steps];
                    for (t, &fb) in flags_bits.iter().enumerate() {
                        if (fb & mask_both_shared) == mask_both_shared {
                            mul_map[t] = shared_mul_rows.len() as u32;
                            shared_mul_rows.push(t);
                        }
                    }
                    let mul_products = if shared_mul_rows.is_empty() {
                        vec![]
                    } else {
                        let lhs: Vec<_> = shared_mul_rows
                            .iter()
                            .map(|&t| cycle_witness.row_stage1(t).rs1_value())
                            .collect();
                        let rhs: Vec<_> = shared_mul_rows
                            .iter()
                            .map(|&t| cycle_witness.row_stage1(t).rs2_value())
                            .collect();
                        rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
                    };

                    let input_share = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                       product: Rep3PrimeFieldShare<F>,
                                       input: JoltR1CSInputs| {
                        match input {
                            JoltR1CSInputs::LeftInstructionInput => {
                                row.to_instruction_inputs(party_id).0
                            }
                            JoltR1CSInputs::RightInstructionInput => {
                                row.to_instruction_inputs(party_id).1
                            }
                            JoltR1CSInputs::Product => product,
                            JoltR1CSInputs::LeftLookupOperand => {
                                row.to_lookup_operands(party_id, product).0
                            }
                            JoltR1CSInputs::RightLookupOperand => {
                                row.to_lookup_operands(party_id, product).1
                            }
                            JoltR1CSInputs::LookupOutput => row.to_lookup_output(),
                            JoltR1CSInputs::Rs1Value => row.rs1_value(),
                            JoltR1CSInputs::Rs2Value => row.rs2_value(),
                            JoltR1CSInputs::RdWriteValue => row.rd_write_value(),
                            JoltR1CSInputs::RamReadValue => row.ram_read_value(),
                            JoltR1CSInputs::RamWriteValue => row.ram_write_value(),
                            JoltR1CSInputs::ShouldBranch => {
                                if row.flag(CircuitFlags::Branch) {
                                    row.to_lookup_output()
                                } else {
                                    Rep3PrimeFieldShare::zero_share()
                                }
                            }
                            JoltR1CSInputs::WriteLookupOutputToRD => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(
                                        if row.flag(CircuitFlags::WriteLookupOutputToRD) {
                                            row.rd_addr() as u64
                                        } else {
                                            0
                                        },
                                    ),
                                )
                            }
                            JoltR1CSInputs::WritePCtoRD => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(if row.flag(CircuitFlags::Jump) {
                                    row.rd_addr() as u64
                                } else {
                                    0
                                }),
                            ),
                            JoltR1CSInputs::PC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.pc_index()),
                            ),
                            JoltR1CSInputs::NextPC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.next_pc_index()),
                            ),
                            JoltR1CSInputs::UnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.unexpanded_pc()),
                                )
                            }
                            JoltR1CSInputs::NextUnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.next_unexpanded_pc()),
                                )
                            }
                            JoltR1CSInputs::Imm => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_i128(row.imm()),
                            ),
                            JoltR1CSInputs::Rd => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.rd_addr() as u64),
                            ),
                            JoltR1CSInputs::RamAddress => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.ram_addr()),
                                )
                            }
                            JoltR1CSInputs::NextIsNoop => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.next_is_noop()),
                                )
                            }
                            JoltR1CSInputs::ShouldJump => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.should_jump()),
                                )
                            }
                            JoltR1CSInputs::OpFlags(flag) => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.flag(flag)),
                                )
                            }
                        }
                    };

                    let eval_b = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                  product: Rep3PrimeFieldShare<F>,
                                  constraint_idx: usize| {
                        let mut acc = Rep3PrimeFieldShare::zero_share();
                        UNIFORM_R1CS[constraint_idx]
                            .cons
                            .b
                            .for_each_term(|input_index, coeff| {
                                let scalar = F::from_i128(coeff.to_i128());
                                acc += rep3_arithmetic::mul_public(
                                    input_share(row, product, JoltR1CSInputs::from_index(input_index)),
                                    scalar,
                                );
                            });
                        if let Some(c) = UNIFORM_R1CS[constraint_idx].cons.b.const_term() {
                            acc = rep3_arithmetic::add_public(
                                acc,
                                F::from_i128(c.to_i128()),
                                party_id,
                            );
                        }
                        acc
                    };

                    let num_pairs = UNIFORM_R1CS.len().next_power_of_two() / 2;
                    let r0 = challenges[0];
                    let mut out = Vec::with_capacity(target_constraints.len());
                    for &target_c in &target_constraints {
                        let mut sparse: Vec<(usize, Rep3PrimeFieldShare<F>)> = Vec::new();
                        for t in 0..num_steps {
                            let row = cycle_witness.row_stage1(t);
                            let fb = flags_bits[t];
                            let left_shared = (fb & mask_left_rs1) != 0;
                            let right_shared = (fb & mask_right_rs2) != 0;
                            let product = if mul_map[t] != u32::MAX {
                                mul_products[mul_map[t] as usize]
                            } else {
                                match (left_shared, right_shared) {
                                    (true, false) => rep3_arithmetic::mul_public(
                                        row.rs1_value(),
                                        row.to_right_public_input(),
                                    ),
                                    (false, true) => rep3_arithmetic::mul_public(
                                        row.rs2_value(),
                                        row.to_left_public_input(),
                                    ),
                                    (false, false) => rep3_arithmetic::promote_to_trivial_share(
                                        party_id,
                                        row.to_left_public_input() * row.to_right_public_input(),
                                    ),
                                    (true, true) => unreachable!(),
                                }
                            };
                            for pair in 0..num_pairs {
                                let c0 = pair * 2;
                                let c1 = c0 + 1;
                                let value = if target_c == c0 {
                                    let b0 = eval_b(row, product, c0);
                                    rep3_arithmetic::mul_public(b0, F::one() - r0)
                                } else if target_c == c1 && c1 < UNIFORM_R1CS.len() {
                                    let b1 = eval_b(row, product, c1);
                                    rep3_arithmetic::mul_public(b1, r0)
                                } else {
                                    continue;
                                };
                                sparse.push((3 * (t * num_pairs + pair) + 1, value));
                            }
                        }

                        for &r in challenges.iter().skip(1) {
                            let mut next = Vec::new();
                            let mut i = 0usize;
                            while i < sparse.len() {
                                let block = sparse[i].0 / 6;
                                let mut low = None;
                                let mut high = None;
                                while i < sparse.len() && sparse[i].0 / 6 == block {
                                    match sparse[i].0 % 6 {
                                        1 => low = Some(sparse[i].1),
                                        4 => high = Some(sparse[i].1),
                                        _ => {}
                                    }
                                    i += 1;
                                }
                                if low.is_some() || high.is_some() {
                                    let low = low.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                    let high =
                                        high.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                    let value = low + rep3_arithmetic::mul_public(high - low, r);
                                    next.push((3 * block + 1, value));
                                }
                            }
                            sparse = next;
                        }

                        let final_share = sparse.into_iter().fold(
                            Rep3PrimeFieldShare::zero_share(),
                            |acc, (_, value)| acc + value,
                        );
                        out.push(final_share);
                    }
                    Ok(out)
                },
            );

            let opened_sparse = rep3_arithmetic::combine_field_elements_vec(vec![
                sparse_shares[0].clone(),
                sparse_shares[1].clone(),
                sparse_shares[2].clone(),
            ]);
            for (i, &constraint_idx) in target_constraints.iter().enumerate() {
                let mut direct = F::zero();
                for (t, eq_t) in eq_cycle.iter().enumerate().take(vanilla_trace.len()) {
                    let row_inputs = jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                        &shared,
                        &vanilla_trace,
                        t,
                    );
                    direct += *eq_t
                        * eq_constr[constraint_idx]
                        * UNIFORM_R1CS[constraint_idx].cons.b.evaluate_row_with::<F>(&row_inputs);
                }
                assert_eq!(
                    opened_sparse[i], direct,
                    "share sparse B mismatch at constraint {} ({:?})",
                    constraint_idx, UNIFORM_R1CS[constraint_idx].name
                );
            }
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_FULL_B_ONLY_SPARSE").is_ok() {
            use co_jolt2::zkvm::r1cs::inputs::JoltR1CSInputs;
            use jolt_core::field::JoltField;
            use jolt_core::zkvm::instruction::CircuitFlags;
            use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
            use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

            let challenges_low_to_high: Vec<F> =
                outer_point.r.iter().rev().map(|r| (*r).into()).collect();
            let shares_stage1_full_b = Arc::clone(&shares_arc);
            let io_device_stage1_full_b = Arc::clone(&io_device_arc);
            let preprocessing_stage1_full_b = Arc::clone(&preprocessing_arc);

            let full_b_shares: [Vec<Rep3PrimeFieldShare<F>>; 3] = run_rep3_test(
                15334,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_stage1_full_b[party_idx].clone();
                    (
                        trace,
                        mem,
                        (*io_device_stage1_full_b).clone(),
                        (*preprocessing_stage1_full_b).clone(),
                        ram_K,
                        advice_shares,
                        challenges_low_to_high.clone(),
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                    Vec<F>,
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, challenges) =
                        input;
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

                    let party_id = io_ctx.party_id();
                    let cycle_witness = state.get_cycle_witness();
                    let num_steps = cycle_witness.len();
                    let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                    let mask_left_rs1 =
                        1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                    let mask_right_rs2 =
                        1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                    let mask_both_shared = mask_left_rs1 | mask_right_rs2;

                    let mut shared_mul_rows = Vec::new();
                    let mut mul_map = vec![u32::MAX; num_steps];
                    for (t, &fb) in flags_bits.iter().enumerate() {
                        if (fb & mask_both_shared) == mask_both_shared {
                            mul_map[t] = shared_mul_rows.len() as u32;
                            shared_mul_rows.push(t);
                        }
                    }
                    let mul_products = if shared_mul_rows.is_empty() {
                        vec![]
                    } else {
                        let lhs: Vec<_> = shared_mul_rows
                            .iter()
                            .map(|&t| cycle_witness.row_stage1(t).rs1_value())
                            .collect();
                        let rhs: Vec<_> = shared_mul_rows
                            .iter()
                            .map(|&t| cycle_witness.row_stage1(t).rs2_value())
                            .collect();
                        rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
                    };

                    let input_share = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                       product: Rep3PrimeFieldShare<F>,
                                       input: JoltR1CSInputs| {
                        match input {
                            JoltR1CSInputs::LeftInstructionInput => {
                                row.to_instruction_inputs(party_id).0
                            }
                            JoltR1CSInputs::RightInstructionInput => {
                                row.to_instruction_inputs(party_id).1
                            }
                            JoltR1CSInputs::Product => product,
                            JoltR1CSInputs::LeftLookupOperand => {
                                row.to_lookup_operands(party_id, product).0
                            }
                            JoltR1CSInputs::RightLookupOperand => {
                                row.to_lookup_operands(party_id, product).1
                            }
                            JoltR1CSInputs::LookupOutput => row.to_lookup_output(),
                            JoltR1CSInputs::Rs1Value => row.rs1_value(),
                            JoltR1CSInputs::Rs2Value => row.rs2_value(),
                            JoltR1CSInputs::RdWriteValue => row.rd_write_value(),
                            JoltR1CSInputs::RamReadValue => row.ram_read_value(),
                            JoltR1CSInputs::RamWriteValue => row.ram_write_value(),
                            JoltR1CSInputs::ShouldBranch => {
                                if row.flag(CircuitFlags::Branch) {
                                    row.to_lookup_output()
                                } else {
                                    Rep3PrimeFieldShare::zero_share()
                                }
                            }
                            JoltR1CSInputs::WriteLookupOutputToRD => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(
                                        if row.flag(CircuitFlags::WriteLookupOutputToRD) {
                                            row.rd_addr() as u64
                                        } else {
                                            0
                                        },
                                    ),
                                )
                            }
                            JoltR1CSInputs::WritePCtoRD => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(if row.flag(CircuitFlags::Jump) {
                                    row.rd_addr() as u64
                                } else {
                                    0
                                }),
                            ),
                            JoltR1CSInputs::PC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.pc_index()),
                            ),
                            JoltR1CSInputs::NextPC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.next_pc_index()),
                            ),
                            JoltR1CSInputs::UnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.unexpanded_pc()),
                                )
                            }
                            JoltR1CSInputs::NextUnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.next_unexpanded_pc()),
                                )
                            }
                            JoltR1CSInputs::Imm => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_i128(row.imm()),
                            ),
                            JoltR1CSInputs::Rd => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.rd_addr() as u64),
                            ),
                            JoltR1CSInputs::RamAddress => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.ram_addr()),
                                )
                            }
                            JoltR1CSInputs::NextIsNoop => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.next_is_noop()),
                                )
                            }
                            JoltR1CSInputs::ShouldJump => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.should_jump()),
                                )
                            }
                            JoltR1CSInputs::OpFlags(flag) => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.flag(flag)),
                                )
                            }
                        }
                    };

                    let eval_b = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                  product: Rep3PrimeFieldShare<F>,
                                  constraint_idx: usize| {
                        let mut acc = Rep3PrimeFieldShare::zero_share();
                        UNIFORM_R1CS[constraint_idx]
                            .cons
                            .b
                            .for_each_term(|input_index, coeff| {
                                let scalar = F::from_i128(coeff.to_i128());
                                acc += rep3_arithmetic::mul_public(
                                    input_share(row, product, JoltR1CSInputs::from_index(input_index)),
                                    scalar,
                                );
                            });
                        if let Some(c) = UNIFORM_R1CS[constraint_idx].cons.b.const_term() {
                            acc = rep3_arithmetic::add_public(
                                acc,
                                F::from_i128(c.to_i128()),
                                party_id,
                            );
                        }
                        acc
                    };

                    let num_pairs = UNIFORM_R1CS.len().next_power_of_two() / 2;
                    let r0 = challenges[0];
                    let mut sparse: Vec<(usize, Rep3PrimeFieldShare<F>)> =
                        Vec::with_capacity(num_steps * num_pairs);
                    for t in 0..num_steps {
                        let row = cycle_witness.row_stage1(t);
                        let fb = flags_bits[t];
                        let left_shared = (fb & mask_left_rs1) != 0;
                        let right_shared = (fb & mask_right_rs2) != 0;
                        let product = if mul_map[t] != u32::MAX {
                            mul_products[mul_map[t] as usize]
                        } else {
                            match (left_shared, right_shared) {
                                (true, false) => rep3_arithmetic::mul_public(
                                    row.rs1_value(),
                                    row.to_right_public_input(),
                                ),
                                (false, true) => rep3_arithmetic::mul_public(
                                    row.rs2_value(),
                                    row.to_left_public_input(),
                                ),
                                (false, false) => rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    row.to_left_public_input() * row.to_right_public_input(),
                                ),
                                (true, true) => unreachable!(),
                            }
                        };
                        for pair in 0..num_pairs {
                            let c0 = pair * 2;
                            let c1 = c0 + 1;
                            let b0 = if c0 < UNIFORM_R1CS.len() {
                                eval_b(row, product, c0)
                            } else {
                                Rep3PrimeFieldShare::zero_share()
                            };
                            let b1 = if c1 < UNIFORM_R1CS.len() {
                                eval_b(row, product, c1)
                            } else {
                                Rep3PrimeFieldShare::zero_share()
                            };
                            let value = b0 + rep3_arithmetic::mul_public(b1 - b0, r0);
                            sparse.push((3 * (t * num_pairs + pair) + 1, value));
                        }
                    }

                    for &r in challenges.iter().skip(1) {
                        let mut next = Vec::new();
                        let mut i = 0usize;
                        while i < sparse.len() {
                            let block = sparse[i].0 / 6;
                            let mut low = None;
                            let mut high = None;
                            while i < sparse.len() && sparse[i].0 / 6 == block {
                                match sparse[i].0 % 6 {
                                    1 => low = Some(sparse[i].1),
                                    4 => high = Some(sparse[i].1),
                                    _ => {}
                                }
                                i += 1;
                            }
                            if low.is_some() || high.is_some() {
                                let low = low.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                let high = high.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                let value = low + rep3_arithmetic::mul_public(high - low, r);
                                next.push((3 * block + 1, value));
                            }
                        }
                        sparse = next;
                    }

                    let final_share = sparse.into_iter().fold(
                        Rep3PrimeFieldShare::zero_share(),
                        |acc, (_, value)| acc + value,
                    );
                    Ok(vec![final_share])
                },
            );

            let opened_full_b = rep3_arithmetic::combine_field_elements_vec(vec![
                full_b_shares[0].clone(),
                full_b_shares[1].clone(),
                full_b_shares[2].clone(),
            ]);
            assert_eq!(
                opened_full_b[0], direct_bz,
                "full share B-only sparse bind mismatch"
            );
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_FULL_ABC_SHARE").is_ok() {
            use co_jolt2::zkvm::r1cs::inputs::JoltR1CSInputs;
            use jolt_core::field::JoltField;
            use jolt_core::zkvm::instruction::CircuitFlags;
            use mpc_core::protocols::additive;
            use mpc_core::protocols::rep3::arithmetic as rep3_arithmetic;
            use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

            let challenges_low_to_high: Vec<F> =
                outer_point.r.iter().rev().map(|r| (*r).into()).collect();
            let shares_stage1_full_abc = Arc::clone(&shares_arc);
            let io_device_stage1_full_abc = Arc::clone(&io_device_arc);
            let preprocessing_stage1_full_abc = Arc::clone(&preprocessing_arc);

            let full_abc_shares: [Vec<additive::AdditiveShare<F>>; 3] = run_rep3_test(
                15335,
                1,
                move |party_idx| {
                    let (trace, mem, advice_shares) = shares_stage1_full_abc[party_idx].clone();
                    (
                        trace,
                        mem,
                        (*io_device_stage1_full_abc).clone(),
                        (*preprocessing_stage1_full_abc).clone(),
                        ram_K,
                        advice_shares,
                        challenges_low_to_high.clone(),
                    )
                },
                move |input: (
                    Vec<Rep3Cycle>,
                    co_jolt2::host::memory::Rep3Memory,
                    tracer::JoltDevice,
                    JoltProverPreprocessing<F, PCS>,
                    usize,
                    co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                    Vec<F>,
                ),
                      mut io_ctx| {
                    let (trace, mem, io_device, preprocessing, ram_k, advice_shares, challenges) =
                        input;
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

                    let party_id = io_ctx.party_id();
                    let cycle_witness = state.get_cycle_witness();
                    let num_steps = cycle_witness.len();
                    let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                    let mask_left_rs1 =
                        1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                    let mask_right_rs2 =
                        1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                    let mask_both_shared = mask_left_rs1 | mask_right_rs2;

                    let mut shared_mul_rows = Vec::new();
                    let mut mul_map = vec![u32::MAX; num_steps];
                    for (t, &fb) in flags_bits.iter().enumerate() {
                        if (fb & mask_both_shared) == mask_both_shared {
                            mul_map[t] = shared_mul_rows.len() as u32;
                            shared_mul_rows.push(t);
                        }
                    }
                    let mul_products = if shared_mul_rows.is_empty() {
                        vec![]
                    } else {
                        let lhs: Vec<_> = shared_mul_rows
                            .iter()
                            .map(|&t| cycle_witness.row_stage1(t).rs1_value())
                            .collect();
                        let rhs: Vec<_> = shared_mul_rows
                            .iter()
                            .map(|&t| cycle_witness.row_stage1(t).rs2_value())
                            .collect();
                        rep3_arithmetic::mul_vec_par(&lhs, &rhs, io_ctx.main())?
                    };

                    let input_share = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                       product: Rep3PrimeFieldShare<F>,
                                       input: JoltR1CSInputs| {
                        match input {
                            JoltR1CSInputs::LeftInstructionInput => {
                                row.to_instruction_inputs(party_id).0
                            }
                            JoltR1CSInputs::RightInstructionInput => {
                                row.to_instruction_inputs(party_id).1
                            }
                            JoltR1CSInputs::Product => product,
                            JoltR1CSInputs::LeftLookupOperand => {
                                row.to_lookup_operands(party_id, product).0
                            }
                            JoltR1CSInputs::RightLookupOperand => {
                                row.to_lookup_operands(party_id, product).1
                            }
                            JoltR1CSInputs::LookupOutput => row.to_lookup_output(),
                            JoltR1CSInputs::Rs1Value => row.rs1_value(),
                            JoltR1CSInputs::Rs2Value => row.rs2_value(),
                            JoltR1CSInputs::RdWriteValue => row.rd_write_value(),
                            JoltR1CSInputs::RamReadValue => row.ram_read_value(),
                            JoltR1CSInputs::RamWriteValue => row.ram_write_value(),
                            JoltR1CSInputs::ShouldBranch => {
                                if row.flag(CircuitFlags::Branch) {
                                    row.to_lookup_output()
                                } else {
                                    Rep3PrimeFieldShare::zero_share()
                                }
                            }
                            JoltR1CSInputs::WriteLookupOutputToRD => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(
                                        if row.flag(CircuitFlags::WriteLookupOutputToRD) {
                                            row.rd_addr() as u64
                                        } else {
                                            0
                                        },
                                    ),
                                )
                            }
                            JoltR1CSInputs::WritePCtoRD => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(if row.flag(CircuitFlags::Jump) {
                                    row.rd_addr() as u64
                                } else {
                                    0
                                }),
                            ),
                            JoltR1CSInputs::PC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.pc_index()),
                            ),
                            JoltR1CSInputs::NextPC => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.next_pc_index()),
                            ),
                            JoltR1CSInputs::UnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.unexpanded_pc()),
                                )
                            }
                            JoltR1CSInputs::NextUnexpandedPC => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.next_unexpanded_pc()),
                                )
                            }
                            JoltR1CSInputs::Imm => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_i128(row.imm()),
                            ),
                            JoltR1CSInputs::Rd => rep3_arithmetic::promote_to_trivial_share(
                                party_id,
                                F::from_u64(row.rd_addr() as u64),
                            ),
                            JoltR1CSInputs::RamAddress => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_u64(row.ram_addr()),
                                )
                            }
                            JoltR1CSInputs::NextIsNoop => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.next_is_noop()),
                                )
                            }
                            JoltR1CSInputs::ShouldJump => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.should_jump()),
                                )
                            }
                            JoltR1CSInputs::OpFlags(flag) => {
                                rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    F::from_bool(row.flag(flag)),
                                )
                            }
                        }
                    };

                    let eval_lc = |row: co_jolt2::zkvm::dag::witness::Stage1RowRef<'_, F>,
                                   product: Rep3PrimeFieldShare<F>,
                                   lc: jolt_core::zkvm::r1cs::constraints::LC| {
                        let mut acc = Rep3PrimeFieldShare::zero_share();
                        lc.for_each_term(|input_index, coeff| {
                            let scalar = F::from_i128(coeff.to_i128());
                            acc += rep3_arithmetic::mul_public(
                                input_share(row, product, JoltR1CSInputs::from_index(input_index)),
                                scalar,
                            );
                        });
                        if let Some(c) = lc.const_term() {
                            acc = rep3_arithmetic::add_public(
                                acc,
                                F::from_i128(c.to_i128()),
                                party_id,
                            );
                        }
                        acc
                    };

                    let bind_sparse_interleaved = |
                        sparse: &[(usize, Rep3PrimeFieldShare<F>)],
                        r: F,
                    | {
                        let mut next = Vec::new();
                        let mut i = 0usize;
                        while i < sparse.len() {
                            let block = sparse[i].0 / 6;
                            let mut a0 = None;
                            let mut b0 = None;
                            let mut c0 = None;
                            let mut a1 = None;
                            let mut b1 = None;
                            let mut c1 = None;
                            while i < sparse.len() && sparse[i].0 / 6 == block {
                                match sparse[i].0 % 6 {
                                    0 => a0 = Some(sparse[i].1),
                                    1 => b0 = Some(sparse[i].1),
                                    2 => c0 = Some(sparse[i].1),
                                    3 => a1 = Some(sparse[i].1),
                                    4 => b1 = Some(sparse[i].1),
                                    5 => c1 = Some(sparse[i].1),
                                    _ => {}
                                }
                                i += 1;
                            }
                            let base = 3 * block;
                            if a0.is_some() || a1.is_some() {
                                let low = a0.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                let high = a1.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                next.push((base, low + rep3_arithmetic::mul_public(high - low, r)));
                            }
                            if b0.is_some() || b1.is_some() {
                                let low = b0.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                let high = b1.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                next.push((base + 1, low + rep3_arithmetic::mul_public(high - low, r)));
                            }
                            if c0.is_some() || c1.is_some() {
                                let low = c0.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                let high = c1.unwrap_or_else(Rep3PrimeFieldShare::zero_share);
                                next.push((base + 2, low + rep3_arithmetic::mul_public(high - low, r)));
                            }
                        }
                        next
                    };

                    let extract_b = |sparse: &[(usize, Rep3PrimeFieldShare<F>)]| {
                        sparse
                            .iter()
                            .filter(|(idx, _)| idx % 3 == 1)
                            .map(|(idx, value)| (*idx, *value))
                            .collect::<Vec<_>>()
                    };

                    let num_pairs = UNIFORM_R1CS.len().next_power_of_two() / 2;
                    let r0 = challenges[0];
                    let mut sparse: Vec<(usize, Rep3PrimeFieldShare<F>)> =
                        Vec::with_capacity(num_steps * num_pairs * 3);
                    let mut sparse_b_only: Vec<(usize, Rep3PrimeFieldShare<F>)> =
                        Vec::with_capacity(num_steps * num_pairs);
                    for t in 0..num_steps {
                        let row = cycle_witness.row_stage1(t);
                        let fb = flags_bits[t];
                        let left_shared = (fb & mask_left_rs1) != 0;
                        let right_shared = (fb & mask_right_rs2) != 0;
                        let product = if mul_map[t] != u32::MAX {
                            mul_products[mul_map[t] as usize]
                        } else {
                            match (left_shared, right_shared) {
                                (true, false) => rep3_arithmetic::mul_public(
                                    row.rs1_value(),
                                    row.to_right_public_input(),
                                ),
                                (false, true) => rep3_arithmetic::mul_public(
                                    row.rs2_value(),
                                    row.to_left_public_input(),
                                ),
                                (false, false) => rep3_arithmetic::promote_to_trivial_share(
                                    party_id,
                                    row.to_left_public_input() * row.to_right_public_input(),
                                ),
                                (true, true) => unreachable!(),
                            }
                        };
                        for pair in 0..num_pairs {
                            let c0 = pair * 2;
                            let c1 = c0 + 1;
                            let (a0, b0, c0v) = if c0 < UNIFORM_R1CS.len() {
                                let cons = &UNIFORM_R1CS[c0].cons;
                                (
                                    eval_lc(row, product, cons.a),
                                    eval_lc(row, product, cons.b),
                                    eval_lc(row, product, cons.c),
                                )
                            } else {
                                (
                                    Rep3PrimeFieldShare::zero_share(),
                                    Rep3PrimeFieldShare::zero_share(),
                                    Rep3PrimeFieldShare::zero_share(),
                                )
                            };
                            let (a1, b1, c1v) = if c1 < UNIFORM_R1CS.len() {
                                let cons = &UNIFORM_R1CS[c1].cons;
                                (
                                    eval_lc(row, product, cons.a),
                                    eval_lc(row, product, cons.b),
                                    eval_lc(row, product, cons.c),
                                )
                            } else {
                                (
                                    Rep3PrimeFieldShare::zero_share(),
                                    Rep3PrimeFieldShare::zero_share(),
                                    Rep3PrimeFieldShare::zero_share(),
                                )
                            };
                            let base = 3 * (t * num_pairs + pair);
                            sparse.push((base, a0 + rep3_arithmetic::mul_public(a1 - a0, r0)));
                            let b_bound = b0 + rep3_arithmetic::mul_public(b1 - b0, r0);
                            sparse.push((base + 1, b_bound));
                            sparse.push((base + 2, c0v + rep3_arithmetic::mul_public(c1v - c0v, r0)));
                            sparse_b_only.push((base + 1, b_bound));
                        }
                    }

                    assert_eq!(
                        extract_b(&sparse),
                        sparse_b_only,
                        "full ABC B projection mismatch before remaining rounds"
                    );

                    for (round_idx, &r) in challenges.iter().enumerate().skip(1) {
                        sparse = bind_sparse_interleaved(&sparse, r);
                        sparse_b_only = bind_sparse_interleaved(&sparse_b_only, r);

                        let projected_b = extract_b(&sparse);
                        assert_eq!(
                            projected_b,
                            sparse_b_only,
                            "full ABC B projection mismatch after round {round_idx}"
                        );
                    }

                    let final_b = sparse.into_iter().fold(
                        additive::AdditiveShare::zero(),
                        |acc, (idx, value)| {
                            if idx % 3 == 1 {
                                acc + value.into_additive()
                            } else {
                                acc
                            }
                        },
                    );
                    Ok(vec![final_b])
                },
            );

            let opened_full_abc = additive::combine_additive_vec(vec![
                full_abc_shares[0].clone(),
                full_abc_shares[1].clone(),
                full_abc_shares[2].clone(),
            ]);
            assert_eq!(
                opened_full_abc[0], direct_bz,
                "full ABC share sparse bind mismatch"
            );
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_DIRECT_CLAIMS_SPARSE_B").is_ok() {
            use jolt_core::zkvm::instruction::CircuitFlags;

            #[derive(Clone, Copy)]
            struct SparseCoeff<F> {
                index: usize,
                value: F,
            }

            fn bind_sparse_b_low_to_high<F: jolt_core::field::JoltField>(
                coeffs: &[SparseCoeff<F>],
                r: F,
            ) -> Vec<SparseCoeff<F>> {
                let mut out = Vec::new();
                let mut i = 0usize;
                while i < coeffs.len() {
                    let block = coeffs[i].index / 6;
                    let mut b0 = None;
                    let mut b1 = None;
                    while i < coeffs.len() && coeffs[i].index / 6 == block {
                        match coeffs[i].index % 6 {
                            1 => b0 = Some(coeffs[i].value),
                            4 => b1 = Some(coeffs[i].value),
                            _ => {}
                        }
                        i += 1;
                    }
                    if b0.is_some() || b1.is_some() {
                        let low = b0.unwrap_or(F::zero());
                        let high = b1.unwrap_or(F::zero());
                        let value = low + (high - low) * r;
                        if !value.is_zero() {
                            out.push(SparseCoeff {
                                index: 3 * block + 1,
                                value,
                            });
                        }
                    }
                }
                out
            }

            let challenges_low_to_high: Vec<F> =
                outer_point.r.iter().rev().map(|r| (*r).into()).collect();
            let padded_num_constraints = key.padded_row_constraint_per_step();
            let num_pairs = padded_num_constraints / 2;
            let mut sparse_b_prefixes = Vec::new();
            let mut sparse_b = Vec::new();
            for t in 0..vanilla_trace.len() {
                let row_inputs = jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                    &shared,
                    &vanilla_trace,
                    t,
                );
                for pair in 0..num_pairs {
                    let c0_idx = pair * 2;
                    let c1_idx = c0_idx + 1;
                    let bz0 = if c0_idx < UNIFORM_R1CS.len() {
                        UNIFORM_R1CS[c0_idx].cons.b.evaluate_row_with::<F>(&row_inputs)
                    } else {
                        F::zero()
                    };
                    let bz1 = if c1_idx < UNIFORM_R1CS.len() {
                        UNIFORM_R1CS[c1_idx].cons.b.evaluate_row_with::<F>(&row_inputs)
                    } else {
                        F::zero()
                    };
                    let value = bz0 + (bz1 - bz0) * challenges_low_to_high[0];
                    if !value.is_zero() {
                        sparse_b.push(SparseCoeff {
                            index: 3 * (t * num_pairs + pair) + 1,
                            value,
                        });
                    }
                }
            }
            sparse_b_prefixes.push(sparse_b.clone());
            for &r in challenges_low_to_high.iter().skip(1) {
                sparse_b = bind_sparse_b_low_to_high(&sparse_b, r);
                sparse_b_prefixes.push(sparse_b.clone());
            }

            let shares_stage1_prod_sparse = Arc::clone(&shares_arc);
            let io_device_stage1_prod_sparse = Arc::clone(&io_device_arc);
            let preprocessing_stage1_prod_sparse = Arc::clone(&preprocessing_arc);
            let prod_sparse_prefixes: [Vec<Vec<(usize, mpc_core::protocols::additive::AdditiveShare<F>)>>; 3] =
                run_rep3_test(
                    15336,
                    1,
                    move |party_idx| {
                        let (trace, mem, advice_shares) =
                            shares_stage1_prod_sparse[party_idx].clone();
                        (
                            trace,
                            mem,
                            (*io_device_stage1_prod_sparse).clone(),
                            (*preprocessing_stage1_prod_sparse).clone(),
                            ram_K,
                            advice_shares,
                            challenges_low_to_high.clone(),
                        )
                    },
                    move |input: (
                        Vec<Rep3Cycle>,
                        co_jolt2::host::memory::Rep3Memory,
                        tracer::JoltDevice,
                        JoltProverPreprocessing<F, PCS>,
                        usize,
                        co_jolt2::host::jolt_device::Rep3ProgramIOInput,
                        Vec<F>,
                    ),
                          mut io_ctx| {
                        let (trace, mem, io_device, preprocessing, ram_k, advice_shares, challenges) =
                            input;
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

                        let party_id = io_ctx.party_id();
                        let cycle_witness = state.get_cycle_witness();
                        let num_steps = cycle_witness.len();
                        let flags_bits = cycle_witness.pc_sumcheck_flags_bits();
                        let mask_left_rs1 =
                            1u32 << (CircuitFlags::LeftOperandIsRs1Value as usize);
                        let mask_right_rs2 =
                            1u32 << (CircuitFlags::RightOperandIsRs2Value as usize);
                        let mask_both_shared = mask_left_rs1 | mask_right_rs2;

                        let mut shared_mul_rows = Vec::new();
                        let mut mul_map = vec![u32::MAX; num_steps];
                        for (t, &fb) in flags_bits.iter().enumerate() {
                            if (fb & mask_both_shared) == mask_both_shared {
                                mul_map[t] = shared_mul_rows.len() as u32;
                                shared_mul_rows.push(t);
                            }
                        }
                        let mul_products = if shared_mul_rows.is_empty() {
                            vec![]
                        } else {
                            let lhs: Vec<_> = shared_mul_rows
                                .iter()
                                .map(|&t| cycle_witness.row_stage1(t).rs1_value())
                                .collect();
                            let rhs: Vec<_> = shared_mul_rows
                                .iter()
                                .map(|&t| cycle_witness.row_stage1(t).rs2_value())
                                .collect();
                            mpc_core::protocols::rep3::arithmetic::mul_vec_par(
                                &lhs,
                                &rhs,
                                io_ctx.main(),
                            )?
                        };

                        let cycle_inputs: Vec<_> = (0..num_steps)
                            .map(|t| {
                                let row = cycle_witness.row_stage1(t);
                                let fb = flags_bits[t];
                                let left_shared = (fb & mask_left_rs1) != 0;
                                let right_shared = (fb & mask_right_rs2) != 0;
                                let product = if mul_map[t] != u32::MAX {
                                    mul_products[mul_map[t] as usize]
                                } else {
                                    match (left_shared, right_shared) {
                                        (true, false) => mpc_core::protocols::rep3::arithmetic::mul_public(
                                            row.rs1_value(),
                                            row.to_right_public_input(),
                                        ),
                                        (false, true) => mpc_core::protocols::rep3::arithmetic::mul_public(
                                            row.rs2_value(),
                                            row.to_left_public_input(),
                                        ),
                                        (false, false) => mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                                            party_id,
                                            row.to_left_public_input() * row.to_right_public_input(),
                                        ),
                                        (true, true) => unreachable!(),
                                    }
                                };
                                co_jolt2::zkvm::r1cs::inputs::Rep3R1CSCycleInputs::from_trace(
                                    party_id, row, product,
                                )
                            })
                            .collect();

                        Ok(co_jolt2::poly::spartan_interleaved_poly::Rep3SpartanInterleavedPolynomial::<F>::debug_project_b_after_prefixes(
                            &UniformSpartanKey::<F>::new(padded_len),
                            &cycle_inputs,
                            party_id,
                            &challenges,
                        ))
                    },
                );

            for prefix_idx in 0..sparse_b_prefixes.len() {
                assert_eq!(
                    prod_sparse_prefixes[0][prefix_idx]
                        .iter()
                        .map(|(idx, _)| *idx)
                        .collect::<Vec<_>>(),
                    prod_sparse_prefixes[1][prefix_idx]
                        .iter()
                        .map(|(idx, _)| *idx)
                        .collect::<Vec<_>>(),
                    "production sparse B prefix party index mismatch after round {} between parties 0 and 1",
                    prefix_idx
                );
                assert_eq!(
                    prod_sparse_prefixes[1][prefix_idx]
                        .iter()
                        .map(|(idx, _)| *idx)
                        .collect::<Vec<_>>(),
                    prod_sparse_prefixes[2][prefix_idx]
                        .iter()
                        .map(|(idx, _)| *idx)
                        .collect::<Vec<_>>(),
                    "production sparse B prefix party index mismatch after round {} between parties 1 and 2",
                    prefix_idx
                );
                let expected = &sparse_b_prefixes[prefix_idx];
                let opened_prod: Vec<(usize, F)> = prod_sparse_prefixes[0][prefix_idx]
                    .iter()
                    .zip(prod_sparse_prefixes[1][prefix_idx].iter())
                    .zip(prod_sparse_prefixes[2][prefix_idx].iter())
                    .map(|((a, b), c)| {
                        assert_eq!(a.0, b.0);
                        assert_eq!(b.0, c.0);
                        (
                            a.0,
                            mpc_core::protocols::additive::combine_additive_share(vec![
                                a.1, b.1, c.1,
                            ]),
                        )
                    })
                    .collect();
                let opened_prod_nonzero: Vec<(usize, F)> = opened_prod
                    .into_iter()
                    .filter(|(_, value)| !value.is_zero())
                    .collect();

                let expected_pairs: Vec<(usize, F)> =
                    expected.iter().map(|c| (c.index, c.value)).collect();
                if opened_prod_nonzero != expected_pairs {
                    let max_len = opened_prod_nonzero.len().max(expected_pairs.len());
                    let first_diff = (0..max_len).find(|&i| {
                        opened_prod_nonzero.get(i) != expected_pairs.get(i)
                    });
                    let prod_entry = first_diff.and_then(|i| opened_prod_nonzero.get(i).copied());
                    let expected_entry = first_diff.and_then(|i| expected_pairs.get(i).copied());
                    let diff_index = prod_entry
                        .map(|(idx, _)| idx)
                        .or_else(|| expected_entry.map(|(idx, _)| idx));
                    let diff_loc = diff_index.map(|idx| {
                        let block = idx / 3;
                        (idx, block / num_pairs, block % num_pairs)
                    });
                    panic!(
                        "production sparse B prefix mismatch after round {}: first_diff={:?} loc={:?} prod_entry={:?} expected_entry={:?} prod_len={} expected_len={}",
                        prefix_idx,
                        first_diff,
                        diff_loc,
                        prod_entry,
                        expected_entry,
                        opened_prod_nonzero.len(),
                        expected_pairs.len()
                    );
                }
            }

            let sparse_b_final: F = sparse_b.iter().map(|c| c.value).sum();
            assert_eq!(
                sparse_b_final, *rep3_claim_bz,
                "public sparse B simulation mismatch with rep3 claim"
            );
        }

        if std::env::var("CO_JOLT2_COMPARE_STAGE1_SPARSE_B_PER_CONSTRAINT").is_ok() {
            #[derive(Clone, Copy)]
            struct SparseCoeff<F> {
                index: usize,
                value: F,
            }

            fn bind_sparse_b_low_to_high<F: jolt_core::field::JoltField>(
                coeffs: &[SparseCoeff<F>],
                r: F,
            ) -> Vec<SparseCoeff<F>> {
                let mut out = Vec::new();
                let mut i = 0usize;
                while i < coeffs.len() {
                    let block = coeffs[i].index / 6;
                    let mut b0 = None;
                    let mut b1 = None;
                    while i < coeffs.len() && coeffs[i].index / 6 == block {
                        match coeffs[i].index % 6 {
                            1 => b0 = Some(coeffs[i].value),
                            4 => b1 = Some(coeffs[i].value),
                            _ => {}
                        }
                        i += 1;
                    }
                    if b0.is_some() || b1.is_some() {
                        let low = b0.unwrap_or(F::zero());
                        let high = b1.unwrap_or(F::zero());
                        let value = low + (high - low) * r;
                        if !value.is_zero() {
                            out.push(SparseCoeff {
                                index: 3 * block + 1,
                                value,
                            });
                        }
                    }
                }
                out
            }

            let challenges_low_to_high: Vec<F> =
                outer_point.r.iter().rev().map(|r| (*r).into()).collect();
            let padded_num_constraints = key.padded_row_constraint_per_step();
            let num_pairs = padded_num_constraints / 2;

            for target_c in 0..UNIFORM_R1CS.len() {
                let mut sparse_b = Vec::new();
                let mut direct = F::zero();
                for t in 0..vanilla_trace.len() {
                    let row_inputs =
                        jolt_core::zkvm::r1cs::inputs::R1CSCycleInputs::from_trace::<F>(
                            &shared,
                            &vanilla_trace,
                            t,
                        );
                    direct += eq_cycle[t]
                        * eq_constr[target_c]
                        * UNIFORM_R1CS[target_c].cons.b.evaluate_row_with::<F>(&row_inputs);

                    for pair in 0..num_pairs {
                        let c0_idx = pair * 2;
                        let c1_idx = c0_idx + 1;
                        let mut value = F::zero();
                        if target_c == c0_idx && c0_idx < UNIFORM_R1CS.len() {
                            let bz0 = UNIFORM_R1CS[c0_idx].cons.b.evaluate_row_with::<F>(&row_inputs);
                            value += bz0 * (F::one() - challenges_low_to_high[0]);
                        }
                        if target_c == c1_idx && c1_idx < UNIFORM_R1CS.len() {
                            let bz1 = UNIFORM_R1CS[c1_idx].cons.b.evaluate_row_with::<F>(&row_inputs);
                            value += bz1 * challenges_low_to_high[0];
                        }
                        if !value.is_zero() {
                            sparse_b.push(SparseCoeff {
                                index: 3 * (t * num_pairs + pair) + 1,
                                value,
                            });
                        }
                    }
                }

                for &r in challenges_low_to_high.iter().skip(1) {
                    sparse_b = bind_sparse_b_low_to_high(&sparse_b, r);
                }
                let sparse_final: F = sparse_b.iter().map(|c| c.value).sum();
                assert_eq!(
                    sparse_final, direct,
                    "sparse B per-constraint mismatch at constraint {} ({:?})",
                    target_c, UNIFORM_R1CS[target_c].name
                );
            }
        }

        assert_eq!(
            (direct_az, direct_bz, direct_cz),
            (*rep3_claim_az, *rep3_claim_bz, *rep3_claim_cz),
            "direct stage1 claims mismatch: direct=({:?}, {:?}, {:?}) rep3=({:?}, {:?}, {:?})",
            direct_az,
            direct_bz,
            direct_cz,
            rep3_claim_az,
            rep3_claim_bz,
            rep3_claim_cz
        );
    }
    if std::env::var("CO_JOLT2_COMPARE_STAGE1_OUTER_CLAIMS").is_ok() {
        use jolt_core::poly::opening_proof::{OpeningId, SumcheckId};
        use jolt_core::zkvm::witness::VirtualPolynomial;

        for poly in [
            VirtualPolynomial::SpartanAz,
            VirtualPolynomial::SpartanBz,
            VirtualPolynomial::SpartanCz,
        ] {
            let id = OpeningId::Virtual(poly, SumcheckId::SpartanOuter);
            let rep3_claim = rep3_proof
                .opening_claims
                .0
                .get(&id)
                .unwrap_or_else(|| panic!("missing rep3 outer claim for {:?}", poly))
                .1;
            let vanilla_claim = vanilla_proof
                .opening_claims
                .0
                .get(&id)
                .unwrap_or_else(|| panic!("missing vanilla outer claim for {:?}", poly))
                .1;
            assert_eq!(
                rep3_claim, vanilla_claim,
                "stage1 outer claim mismatch for {:?}",
                poly
            );
        }
    }
    if std::env::var("CO_JOLT2_FINAL_VERIFY_ONLY").is_ok() {
        let elf_contents_owned = program.get_elf_contents();
        let elf_contents = elf_contents_owned.as_deref().expect("elf contents is None");
        let (fresh_proof, fresh_io, fresh_debug_info, _) =
            <JoltRV64IMAC as Jolt<Fr, PCS, FS>>::prove(
                &preprocessing_arc,
                elf_contents,
                &inputs,
                &[],
                &[],
                None,
            );

        // --- Compare trace lengths ---
        eprintln!(
            "trace_length: manual={} fresh={}",
            vanilla_proof.trace_length, fresh_proof.trace_length
        );

        // --- Compare commitments ---
        {
            let manual_comms = &vanilla_proof.commitments;
            let fresh_comms = &fresh_proof.commitments;
            eprintln!(
                "commitment count: manual={} fresh={}",
                manual_comms.len(),
                fresh_comms.len()
            );
            let mut comm_diffs = 0;
            for (i, (m, f)) in manual_comms.iter().zip(fresh_comms.iter()).enumerate() {
                if m != f {
                    eprintln!("  commitment[{i}] DIFFERS");
                    comm_diffs += 1;
                }
            }
            if comm_diffs == 0 {
                eprintln!("  all commitments match");
            }
        }

        // --- Compare Stage 1 proofs ---
        {
            let manual_s1 = vanilla_proof
                .proofs
                .get(&ProofKeys::Stage1Sumcheck)
                .expect("manual stage1 missing");
            let fresh_s1 = fresh_proof
                .proofs
                .get(&ProofKeys::Stage1Sumcheck)
                .expect("fresh stage1 missing");
            let mut m_bytes = Vec::new();
            let mut f_bytes = Vec::new();
            manual_s1.serialize_uncompressed(&mut m_bytes).unwrap();
            fresh_s1.serialize_uncompressed(&mut f_bytes).unwrap();
            if m_bytes == f_bytes {
                eprintln!("stage1 proofs: MATCH");
            } else {
                eprintln!(
                    "stage1 proofs: DIFFER (manual={} bytes, fresh={} bytes)",
                    m_bytes.len(),
                    f_bytes.len()
                );
                // Find first differing round
                if let (ProofData::SumcheckProof(m_sc), ProofData::SumcheckProof(f_sc)) =
                    (manual_s1, fresh_s1)
                {
                    for (i, (a, b)) in m_sc
                        .compressed_polys
                        .iter()
                        .zip(f_sc.compressed_polys.iter())
                        .enumerate()
                    {
                        let mut ab = Vec::new();
                        let mut bb = Vec::new();
                        a.serialize_uncompressed(&mut ab).unwrap();
                        b.serialize_uncompressed(&mut bb).unwrap();
                        if ab != bb {
                            eprintln!("  first differing stage1 round: {i}");
                            break;
                        }
                    }
                }
            }
        }

        // --- Compare Stage 2 proofs ---
        {
            let manual_s2 = vanilla_proof
                .proofs
                .get(&ProofKeys::Stage2Sumcheck)
                .expect("manual stage2 missing");
            let fresh_s2 = fresh_proof
                .proofs
                .get(&ProofKeys::Stage2Sumcheck)
                .expect("fresh stage2 missing");
            let mut m_bytes = Vec::new();
            let mut f_bytes = Vec::new();
            manual_s2.serialize_uncompressed(&mut m_bytes).unwrap();
            fresh_s2.serialize_uncompressed(&mut f_bytes).unwrap();
            if m_bytes == f_bytes {
                eprintln!("stage2 proofs: MATCH");
            } else {
                eprintln!(
                    "stage2 proofs: DIFFER (manual={} bytes, fresh={} bytes)",
                    m_bytes.len(),
                    f_bytes.len()
                );
                if let (ProofData::SumcheckProof(m_sc), ProofData::SumcheckProof(f_sc)) =
                    (manual_s2, fresh_s2)
                {
                    eprintln!(
                        "  stage2 rounds: manual={} fresh={}",
                        m_sc.compressed_polys.len(),
                        f_sc.compressed_polys.len()
                    );
                    for (i, (a, b)) in m_sc
                        .compressed_polys
                        .iter()
                        .zip(f_sc.compressed_polys.iter())
                        .enumerate()
                    {
                        let mut ab = Vec::new();
                        let mut bb = Vec::new();
                        a.serialize_uncompressed(&mut ab).unwrap();
                        b.serialize_uncompressed(&mut bb).unwrap();
                        if ab != bb {
                            eprintln!("  first differing stage2 round: {i}");
                            break;
                        }
                    }
                }
            }
        }

        // --- Try verify fresh proof ---
        // Build fresh verifier preprocessing from the same preprocessing
        let fresh_verifier_preprocessing = JoltVerifierPreprocessing::from(&*preprocessing_arc);
        let vanilla_verification = JoltRV64IMAC::verify(
            &fresh_verifier_preprocessing,
            fresh_proof,
            fresh_io,
            None,
            fresh_debug_info,
        );
        if vanilla_verification.is_ok() {
            eprintln!("fresh vanilla proof VERIFIED OK");
        } else {
            eprintln!(
                "fresh vanilla proof FAILED: {:?}",
                vanilla_verification.err()
            );
        }

        return;
    }
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

    let elf_contents_owned = program.get_elf_contents();
    let elf_contents = elf_contents_owned.as_deref().expect("elf contents is None");
    let (vanilla_verification_proof, vanilla_verification_io, vanilla_debug_info, _) =
        <JoltRV64IMAC as Jolt<Fr, PCS, FS>>::prove(
            &preprocessing_arc,
            elf_contents,
            &inputs,
            &[],
            &[],
            None,
        );
    let vanilla_verification = JoltRV64IMAC::verify(
        &verifier_preprocessing,
        vanilla_verification_proof,
        vanilla_verification_io,
        None,
        vanilla_debug_info,
    );
    assert!(
        vanilla_verification.is_ok(),
        "vanilla final proof verification failed: {:?}",
        vanilla_verification.err()
    );

    let rep3_verification =
        JoltRV64IMAC::verify(&verifier_preprocessing, rep3_proof, io_device, None, None);
    assert!(
        rep3_verification.is_ok(),
        "rep3 final proof verification failed: {:?}",
        rep3_verification.err()
    );
}
