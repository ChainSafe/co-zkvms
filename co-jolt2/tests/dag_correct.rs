use std::collections::HashMap;
use std::sync::Arc;

use ark_bn254::Fr;
use ark_ff::Zero;
use ark_std::test_rng;

use co_jolt_coordinator::zkvm::dag::coordinator::Rep3JoltDag;
use co_jolt_coordinator::zkvm::dag::state_manager::StateManager;
use co_jolt2::host::program::Rep3Program;
use co_jolt2::poly::dense_mlpoly::combine_poly_shares_rep3;
use co_jolt2::poly::multilinear_polynomial::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::test_utils::run_rep3_local_test_with_coordinator;
use co_jolt2::utils::test_utils::run_rep3_test;
use co_jolt2::utils::tracing::init_tracing;
use co_jolt2::utils::types::Either;
use co_jolt2::zkvm::dag::state_manager::StateManagerWorker;
use co_jolt2::zkvm::instruction::LookupIndexInt;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::r1cs::inputs::{compute_claimed_witness_evals_rep3, ALL_R1CS_INPUTS};
use co_jolt2::zkvm::witness::{generate_witness_batch_rep3, populate_cycle_witness_rep3};
use co_jolt2::zkvm::Rep3JoltWorker;
use co_jolt2::zkvm::dag::worker::Rep3JoltDagWorker;

use jolt_core::host::Program;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::r1cs::constraints::UNIFORM_R1CS;
use jolt_core::zkvm::witness::CommittedPolynomial;
use jolt_core::zkvm::{
    JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing, JoltVerifierPreprocessing,
};
use tracer::instruction::Cycle;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;
type Challenge = <F as jolt_core::field::JoltField>::Challenge;

#[test]
fn dag_correct() {
    // NOTE: vanilla helper functions (vanilla_inner_sumcheck_round0, vanilla_lookup_booleanity_round0,
    // vanilla_registers_round0, vanilla_ram_round0, vanilla_ram_output_rounds,
    // vanilla_ram_valfinal_round0, vanilla_up_to_stage5) have been removed because they depended on
    // VanillaStateManager, BytecodeDag, RamDag, RegistersDag, SpartanDag, LookupsDag which were
    // stripped from jolt-core. The remaining test exercises only the MPC (rep3) code path.

    let _tracing_guard = init_tracing("dag_correct.json", std::path::Path::new("traces"));

    // 1) Build and trace the fibonacci program (reuse witness_batch_rep3 setup).
    let mut program = Program::new("fibonacci-guest");
    program.set_memory_size(10240);
    let inputs = postcard::to_stdvec(&9u32).unwrap();
    let (bytecode, memory_init, _) = program.decode();

    let mut rng = test_rng();
    let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);
    let (mut vanilla_trace, _vanilla_memory, mut io_device) = program.trace(&inputs, &[], &[]);

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
    }

    // 4) Rep3 proof (local MPC, no QUIC).
    let preprocessing_arc = Arc::new(preprocessing);
    let verifier_preprocessing_arc = Arc::new(verifier_preprocessing);
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

    // Verify the rep3 proof was produced successfully (basic sanity).
    assert!(
        rep3_proof.trace_length > 0,
        "rep3 proof should have a non-zero trace length"
    );
}
