use std::sync::Arc;

use ark_bn254::Fr;
use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;

use co_jolt2::host::program::Rep3Program;
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::test_utils::run_rep3_local_test_with_coordinator;
use co_jolt2::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};
use co_jolt2::zkvm::dag::Rep3DagStop;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::Rep3JoltWorker;
use co_jolt2::zkvm::{dag::coordinator::Rep3JoltDAGCoordinator, dag::worker::Rep3JoltDAGWorker};

use jolt_core::host::Program;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::transcripts::Transcript;
use jolt_core::subprotocols::sumcheck::{BatchedSumcheck, SumcheckInstance};
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

fn vanilla_up_to_stage2(
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
    let (commitments, _hints): (Vec<_>, Vec<_>) = commit_results.into_iter().unzip();
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

    // Stage 2 (incremental testing — uncomment subsystems one-by-one).
    // IMPORTANT: Instances must be created in vanilla order (spartan → registers → ram → lookups)
    // because constructors derive challenges from the transcript.
    let mut registers_dag = RegistersDag::default();
    let mut ram_dag = RamDag::new_prover::<F, FS, PCS>(&sm);
    // let mut lookups_dag = LookupsDag::<F>::default();

    let mut stage2_instances: Vec<_> = std::iter::empty()
        .chain(spartan.stage2_prover_instances(&mut sm))
        .chain(registers_dag.stage2_prover_instances(&mut sm))
        .chain(ram_dag.stage2_prover_instances(&mut sm))
        // .chain(lookups_dag.stage2_prover_instances(&mut sm))
        .collect();
    let stage2_instances_mut: Vec<&mut dyn SumcheckInstance<F, FS>> =
        stage2_instances
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

    (VanillaJoltProof::from_prover_state_manager(sm), tau)
}

#[test]
fn stage1_correct() {
    // 1) Build and trace the fibonacci program (reuse witness_batch_rep3 setup).
    let mut program = Program::new("fibonacci-guest");
    program.set_memory_size(10240);
    let inputs = postcard::to_stdvec(&9u32).unwrap();
    let (bytecode, memory_init, _) = program.decode();

    let mut rng = test_rng();
    let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);
    let (mut vanilla_trace, vanilla_memory, io_device) = program.trace(&inputs, &[], &[]);

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

    // 4) Vanilla proof up to Stage2.
    let (vanilla_proof, tau) = vanilla_up_to_stage2(
        &preprocessing,
        vanilla_trace,
        io_device.clone(),
        vanilla_memory,
    );

    // 5) Rep3 proof up to Stage2 (local MPC, no QUIC).
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
                let (trace, memory, _io) = shares_arc[party_idx].clone();
                (
                    trace,
                    memory,
                    Arc::clone(&io_device_arc),
                    Arc::clone(&preprocessing_arc),
                    ram_K,
                )
            }
        },
        {
            let verifier_preprocessing_arc = Arc::clone(&verifier_preprocessing_arc_for_coord);
            let io_device_arc = Arc::clone(&io_device_arc_for_coord);
            move || {
                (
                    Arc::clone(&verifier_preprocessing_arc),
                    Arc::clone(&io_device_arc),
                    ram_K,
                )
            }
        },
        move |input, io_ctx| {
            let (trace, final_memory_state, program_io, preprocessing, ram_K) = input;
            let mut io_ctx = io_ctx;
            let party_id = io_ctx.party_id();
            let state = StateManagerWorker::new(
                &preprocessing,
                trace,
                (*program_io).clone(),
                final_memory_state,
                party_id,
                ram_K,
            );
            Rep3JoltDAGWorker::prove_with_stop::<F, PCS, FS, _>(
                state,
                io_ctx,
                Rep3DagStop::AfterStage2,
            )
        },
        move |input, net| {
            let (verifier_preprocessing, program_io, ram_K) = input;
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
            let state: StateManagerCoordinator<'_, F, FS, PCS> = StateManagerCoordinator::new(
                &verifier_preprocessing,
                (*program_io).clone(),
                ram_K,
                twist_sumcheck_switch_index,
            );
            Rep3JoltDAGCoordinator::prove_with_stop(state, net, Rep3DagStop::AfterStage2)
        },
    );

    // 6) Compare commitments.
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
                rep3_sc.compressed_polys[i],
                vanilla_sc.compressed_polys[i],
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

    // 9) Compare opening claims bytes (stage1+stage2 openings).
    let rep3_openings_bytes = {
        let mut v = Vec::new();
        rep3_proof
            .opening_claims
            .serialize_uncompressed(&mut v)
            .unwrap();
        v
    };
    let vanilla_openings_bytes = {
        let mut v = Vec::new();
        vanilla_proof
            .opening_claims
            .serialize_uncompressed(&mut v)
            .unwrap();
        v
    };
    assert_eq!(rep3_openings_bytes, vanilla_openings_bytes);

    // 10) Metadata invariants.
    assert_eq!(rep3_proof.trace_length, vanilla_proof.trace_length);
    assert_eq!(rep3_proof.ram_K, vanilla_proof.ram_K);
    assert_eq!(rep3_proof.bytecode_d, vanilla_proof.bytecode_d);
    assert_eq!(
        rep3_proof.twist_sumcheck_switch_index,
        vanilla_proof.twist_sumcheck_switch_index
    );
}
