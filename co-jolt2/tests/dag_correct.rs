use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use ark_bn254::Fr;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use co_jolt2::host::program::generate_trace_shares;
use co_jolt2::utils::test_utils::run_rep3_local_test_with_coordinator;
use co_jolt2::utils::tracing::init_tracing;
use co_jolt2::zkvm::dag::state_manager::StateManagerWorker;
use co_jolt2::zkvm::dag::worker::Rep3JoltDagWorker;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::{JoltArch, Rep3JoltWorker};
use co_jolt_coordinator::zkvm::dag::coordinator::Rep3JoltDag;
use co_jolt_coordinator::zkvm::dag::state_manager::StateManager;

use jolt_core::curve::Bn254Curve;
use jolt_core::field::JoltField;
use jolt_core::host::Program;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::dag::jolt_dag::JoltDAG;
use jolt_core::zkvm::dag::proof_serialization::JoltProof;
use jolt_core::zkvm::dag::state_manager::StateManager as VanillaStateManager;
use jolt_core::zkvm::dag::state_manager::{ProofData, ProofKeys};
use jolt_core::zkvm::witness::DTH_ROOT_OF_K;
use jolt_core::zkvm::{JoltProverPreprocessing, JoltRV64IMAC, JoltVerifierPreprocessing};
use tracer::JoltDevice;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;

struct DagFixture {
    proof: JoltProof<F, Bn254Curve, PCS, FS>,
    verifier_preprocessing: JoltVerifierPreprocessing<F, PCS>,
    io_device: tracer::JoltDevice,
    ram_k: usize,
}

fn dag_test_lock() -> MutexGuard<'static, ()> {
    static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    TEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn use_sha2_fixture() -> bool {
    matches!(std::env::var("TEST_SHA2").ok().as_deref().or(std::env::var("SHA2_CHAIN").ok().as_deref()), Some("1"))
}

fn build_program() -> Program {
    if use_sha2_fixture() {
        let mut program = Program::new("sha2-chain-guest");
        program.set_stack_size(65536);
        program.set_memory_size(10240);
        program
    } else {
        let mut program = Program::new("fibonacci-guest");
        program.set_memory_size(10240);
        program
    }
}

/// Returns (public_inputs, untrusted_advice).
fn build_inputs() -> (Vec<u8>, Vec<u8>) {
    if use_sha2_fixture() {
        let mut advice = postcard::to_stdvec(&[5u8; 32]).unwrap();
        advice.append(&mut postcard::to_stdvec(&1u32).unwrap());
        (vec![], advice)
    } else {
        (postcard::to_stdvec(&9u32).unwrap(), vec![])
    }
}

fn build_public_fixture(
    trace_file: &str,
) -> (
    [(Vec<Rep3Cycle>, co_jolt2::host::memory::Rep3Memory, co_jolt2::host::jolt_device::Rep3ProgramIOInput); 3],
    JoltProverPreprocessing<F, PCS>,
    JoltVerifierPreprocessing<F, PCS>,
    tracer::JoltDevice,
    usize,
    usize,
) {
    let mut program = build_program();
    let (inputs, untrusted_advice) = build_inputs();

    let mut rng = ChaCha12Rng::seed_from_u64(0);
    let (bytecode, memory_init, io_device, shares) =
        generate_trace_shares(&mut program, &inputs, &untrusted_advice, &[], &mut rng);

    // Shares are already padded to next power of 2 by generate_trace_shares.
    let padded_len = shares[0].0.len();
    tracing::info!("Padded trace len: {padded_len}");

    // 2) Preprocessing.
    let preprocessing: JoltProverPreprocessing<F, PCS> = <JoltArch as Rep3JoltWorker<F, PCS, FS>>::preprocess(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        memory_init.clone(),
        padded_len,
    );
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    // 3) Compute ram_K from shared trace (RAM addresses are public).
    let ram_K = co_jolt2::utils::compute_ram_k(&shares[0].0, &preprocessing.shared);

    (shares, preprocessing, verifier_preprocessing, io_device, ram_K, padded_len)
}

fn build_dag_fixture(trace_file: &str) -> DagFixture {
    let _test_guard = dag_test_lock();
    let _tracing_guard = init_tracing(trace_file, std::path::Path::new("traces"));

    let (shares, preprocessing, verifier_preprocessing, io_device, ram_K, padded_len) =
        build_public_fixture(trace_file);

    // 4) Rep3 MPC proof.
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
                    budget.ring_edabits_u66,
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
    // Initialize DoryGlobals here (not before proof generation) so workers can
    // use advice-sized DoryGlobals during their commit without races.
    let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);
    let verifier_preprocessing = Arc::try_unwrap(verifier_preprocessing_arc).unwrap_or_else(|arc| (*arc).clone());
    let io_device = Arc::try_unwrap(io_device_arc).unwrap_or_else(|arc| (*arc).clone());

    DagFixture { proof: rep3_proof, verifier_preprocessing, io_device, ram_k: ram_K }
}

fn verify_dag_fixture(fixture: DagFixture) -> Result<(), Box<dyn std::error::Error>> {
    let DagFixture { proof, verifier_preprocessing, io_device, ram_k } = fixture;
    let twist_sumcheck_switch_index = proof.twist_sumcheck_switch_index;
    let verifier_program_io = JoltDevice {
        inputs: io_device.inputs.clone(),
        outputs: io_device.outputs.clone(),
        panic: io_device.panic,
        memory_layout: io_device.memory_layout.clone(),
        trusted_advice: vec![],
        untrusted_advice: vec![],
    };
    let verifier_sm = VanillaStateManager::from_proof(
        proof,
        Box::leak(Box::new(verifier_preprocessing)),
        verifier_program_io,
        ram_k,
        twist_sumcheck_switch_index,
    );
    JoltDAG::verify::<F, FS, PCS>(verifier_sm).map_err(Into::into)
}

#[test]
fn dag_correct() {
    let fixture = build_dag_fixture("dag_correct.json");
    verify_dag_fixture(fixture).expect("Vanilla verification of MPC proof failed");
}

#[cfg(feature = "zk")]
#[test]
fn dag_zk_tampered_y_com_fails() {
    let mut fixture = build_dag_fixture("dag_zk_tampered_y_com.json");
    assert!(fixture.proof.blindfold_proof.is_some(), "DAG ZK proof must include BlindFold");

    let reduced_opening_proof =
        fixture.proof.proofs.get_mut(&ProofKeys::ReducedOpeningProof).expect("reduced opening proof missing");
    let reduced_opening_proof = match reduced_opening_proof {
        ProofData::ReducedOpeningProof(proof) => proof,
        _ => panic!("unexpected proof type for reduced opening proof"),
    };
    if let Some(ref mut y_com) = reduced_opening_proof.joint_opening_proof.dory_proof_data.y_com {
        *y_com = *y_com + fixture.verifier_preprocessing.generators.g1_0;
    } else if let Some(ref mut e2) = reduced_opening_proof.joint_opening_proof.dory_proof_data.e2 {
        *e2 = *e2 + fixture.verifier_preprocessing.generators.g2_0;
    } else {
        panic!("ZK reduced opening proof missing committed evaluation fields");
    }

    let err = verify_dag_fixture(fixture).expect_err("tampered y_com must fail verification");
    let err_text = format!("{err:?}");
    assert!(
        err_text.contains("Stage 5") || err_text.contains("BlindFold"),
        "unexpected verification error after tampering y_com: {err_text}"
    );
}

#[cfg(feature = "zk")]
#[test]
fn dag_zk_tampered_stage5_hidden_claim_fails() {
    let mut fixture = build_dag_fixture("dag_zk_tampered_stage5_hidden_claim.json");
    assert!(fixture.proof.blindfold_proof.is_some(), "DAG ZK proof must include BlindFold");

    let reduced_opening_proof =
        fixture.proof.proofs.get_mut(&ProofKeys::ReducedOpeningProof).expect("reduced opening proof missing");
    let reduced_opening_proof = match reduced_opening_proof {
        ProofData::ReducedOpeningProof(proof) => proof,
        _ => panic!("unexpected proof type for reduced opening proof"),
    };
    let first_claim = reduced_opening_proof
        .sumcheck_claims
        .first_mut()
        .expect("reduced opening proof must contain at least one hidden claim");
    *first_claim += F::from_u64(1);

    let err = verify_dag_fixture(fixture).expect_err("tampered stage5 hidden claim must fail verification");
    let err_text = format!("{err:?}");
    assert!(
        err_text.contains("Stage 5") || err_text.contains("BlindFold"),
        "unexpected verification error after tampering stage5 hidden claim: {err_text}"
    );
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
