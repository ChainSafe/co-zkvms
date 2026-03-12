use std::sync::Arc;

use ark_bn254::Fr;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use co_jolt2::host::program::Rep3Program;
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::test_utils::run_rep3_local_test_with_coordinator;
use co_jolt2::utils::tracing::init_tracing;
use co_jolt2::zkvm::dag::state_manager::StateManagerWorker;
use co_jolt2::zkvm::dag::worker::Rep3JoltDagWorker;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::{JoltArch, Rep3JoltWorker};
use co_jolt_coordinator::zkvm::dag::coordinator::Rep3JoltDag;
use co_jolt_coordinator::zkvm::dag::state_manager::StateManager;

use jolt_core::host::Program;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::dag::jolt_dag::JoltDAG;
use jolt_core::zkvm::dag::state_manager::StateManager as VanillaStateManager;
use jolt_core::zkvm::witness::DTH_ROOT_OF_K;
use jolt_core::zkvm::{JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing, JoltVerifierPreprocessing};
use tracer::instruction::Cycle;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;

#[test]
fn dag_correct() {
    let _tracing_guard = init_tracing("dag_correct.json", std::path::Path::new("traces"));

    // 1) Build and trace the fibonacci program.
    let mut program = Program::new("fibonacci-guest");
    program.set_memory_size(10240);
    let inputs = postcard::to_stdvec(&9u32).unwrap();
    let (bytecode, memory_init, _) = program.decode();

    let mut rng = ChaCha12Rng::seed_from_u64(0);
    let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);
    let (mut vanilla_trace, _vanilla_memory, mut io_device) = program.trace(&inputs, &[], &[]);

    // Truncate trailing zeros on device outputs, matching what Jolt::prove does.
    io_device.outputs.truncate(io_device.outputs.iter().rposition(|&b| b != 0).map_or(0, |pos| pos + 1));

    tracing::info!("Trace len: {}", vanilla_trace.len());
    // Pad traces to next power of 2 (+1 termination cycle).
    let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
    vanilla_trace.resize(padded_len, Cycle::NoOp);
    for (trace, _, _) in shares.iter_mut() {
        trace.resize(padded_len, Rep3Cycle::NoOp);
    }

    // 2) Preprocessing.
    let shared = JoltSharedPreprocessing {
        memory_layout: io_device.memory_layout.clone(),
        bytecode: jolt_core::zkvm::bytecode::BytecodePreprocessing::preprocess(bytecode.clone()),
        ram: jolt_core::zkvm::ram::RAMPreprocessing::preprocess(memory_init.clone()),
    };
    let preprocessing: JoltProverPreprocessing<F, PCS> = <JoltArch as Rep3JoltWorker<F, PCS, FS>>::preprocess(
        bytecode,
        io_device.memory_layout.clone(),
        memory_init,
        padded_len,
    );
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    // 3) Compute ram_K from vanilla trace (must match both sides).
    let ram_K = compute_ram_k(&vanilla_trace, &shared);

    // 4) Rep3 MPC proof.
    let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);
    let preprocessing_arc = Arc::new(preprocessing);
    let verifier_preprocessing_arc = Arc::new(verifier_preprocessing);
    let io_device_arc = Arc::new(io_device);
    let shares_arc = Arc::new(shares);

    let preprocessing_arc_for_workers = Arc::clone(&preprocessing_arc);
    let verifier_preprocessing_arc_for_coord = Arc::clone(&verifier_preprocessing_arc);
    let io_device_arc_for_coord = Arc::clone(&io_device_arc);

    let (_worker_out, rep3_proof) = run_rep3_local_test_with_coordinator(
        1,
        {
            let shares_arc = Arc::clone(&shares_arc);
            let preprocessing_arc = Arc::clone(&preprocessing_arc_for_workers);
            move |party_idx| {
                let (trace, memory, advice_shares) = shares_arc[party_idx].clone();
                (trace, memory, Arc::clone(&preprocessing_arc), ram_K, advice_shares)
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
            let (trace, final_memory_state, preprocessing, ram_K, advice_shares) = input;
            let mut io_ctx = io_ctx;
            let party_id = io_ctx.party_id();

            // Preprocessing: create EdaBits pool for B2A conversions (2 rounds).
            let mut preproc = {
                use co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget;
                use mpc_core::protocols::rep3_ring::edabits;
                let budget = compute_edabit_budget(trace.len());
                let pool_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join(format!(".preprocessing/test/party_{}", io_ctx.party_idx()));
                #[cfg(not(feature = "ring-msm"))]
                let mut pool = edabits::preprocess_pool::<F, _>(
                    &pool_dir,
                    [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                    budget.dabits,
                    budget.ring_edabits_u64,
                    budget.ring_edabits_u128,
                    &mut io_ctx,
                )?;
                #[cfg(feature = "ring-msm")]
                let mut pool = edabits::preprocess_pool::<F, _>(
                    &pool_dir,
                    [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
                    budget.dabits,
                    budget.wrap_masks,
                    budget.ring_edabits_dory,
                    budget.ring_edabits_u64,
                    budget.ring_edabits_u128,
                    &mut io_ctx,
                )?;

                // Ring MSM preprocessing (daPoints — depend on SRS, not in pool workflow)
                #[cfg(feature = "ring-msm")]
                {
                    if budget.dapoints > 0 {
                        let dory_num_columns = jolt_core::poly::commitment::dory::DoryGlobals::get_num_columns();
                        let qs = co_jolt2::poly::commitment::dory::precompute_dapoint_qs(
                            &preprocessing.generators,
                            budget.dapoints / 2,
                            dory_num_columns,
                        );
                        let lazy_dp =
                            mpc_core::protocols::rep3_ring::preprocessing::daPoint::random_dapoints(&qs, &mut io_ctx)?;
                        pool.set_dapoints(lazy_dp);
                    }
                }
                pool
            };

            let state =
                StateManagerWorker::new(&preprocessing, trace, advice_shares, final_memory_state, party_id, ram_K);
            Rep3JoltDagWorker::prove::<F, PCS, FS, _>(state, &mut io_ctx, &mut preproc)
        },
        move |input, net| {
            let (verifier_preprocessing, prover_preprocessing, program_io, ram_K) = input;
            // Match twist_sumcheck_switch_index computation in co-jolt2 zkvm/mod.rs.
            let num_chunks = rayon::current_num_threads().next_power_of_two().min(padded_len);
            let chunk_size = if num_chunks > 0 { padded_len / num_chunks } else { padded_len };
            let twist_sumcheck_switch_index = if chunk_size > 0 { chunk_size.trailing_zeros() as usize } else { 0 };
            let state: StateManager<'_, F, FS, PCS> =
                StateManager::new(&verifier_preprocessing, (*program_io).clone(), ram_K, twist_sumcheck_switch_index)
                    .with_pcs_setup(&prover_preprocessing.generators);
            Rep3JoltDag::prove(state, net)
        },
    );

    // 5) Verify the MPC-produced proof using the local jolt-core verifier.
    let verifier_preprocessing = Arc::try_unwrap(verifier_preprocessing_arc).unwrap_or_else(|arc| (*arc).clone());
    let io_device = Arc::try_unwrap(io_device_arc).unwrap_or_else(|arc| (*arc).clone());

    let verifier_sm = VanillaStateManager::from_proof(
        rep3_proof,
        // We need a reference that outlives the verify call; use a leaked box for simplicity.
        Box::leak(Box::new(verifier_preprocessing)),
        io_device,
        ram_K,
        rep3_proof_twist_switch_index(padded_len),
    );
    JoltDAG::verify::<F, FS, PCS>(verifier_sm).expect("Vanilla verification of MPC proof failed");
}

fn rep3_proof_twist_switch_index(padded_len: usize) -> usize {
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(padded_len);
    let chunk_size = if num_chunks > 0 { padded_len / num_chunks } else { padded_len };
    if chunk_size > 0 {
        chunk_size.trailing_zeros() as usize
    } else {
        0
    }
}
