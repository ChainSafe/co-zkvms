use std::path::PathBuf;

use ark_bn254::Fr;
use ark_std::test_rng;
use clap::Parser;
use color_eyre::eyre::{self, Context};
use mpc_net::config::{NetworkConfig, NetworkConfigFile};
use mpc_net::rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator};
use mpc_net::topology::{MpcStarNetCoordinator, MpcStarNetWorker};
use serde::{Deserialize, Serialize};
use tracing::{info, info_span};

use co_jolt2::host::jolt_device::Rep3ProgramIOInput;
use co_jolt2::host::memory::Rep3Memory;
use co_jolt2::host::program::Rep3Program;
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::tracing::init_tracing_bench;
use co_jolt2::zkvm::instruction::{populate_operands_casts, Rep3Cycle};
use co_jolt2::zkvm::{Rep3Jolt, Rep3JoltWorker};
use jolt_core::host::Program;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::RAMPreprocessing;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use jolt_core::zkvm::{
    JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing, JoltVerifierPreprocessing,
};
use mpc_core::protocols::rep3::network::IoContextPool;
use tracer::instruction::Cycle;
use tracer::JoltDevice;

type F = Fr;
type PCS = DoryCommitmentScheme;

#[derive(Parser)]
struct Args {
    /// Path to network config TOML
    #[clap(short = 'c', long)]
    config_file: PathBuf,

    /// Directory for trace output files
    #[clap(short = 't', long, default_value = "./.traces")]
    trace_dir: PathBuf,

    /// Number of SHA-256 iterations for guest program
    #[clap(short = 'n', long, default_value = "1")]
    num_iters: u32,
}

/// Payload sent from coordinator to each worker.
#[derive(Serialize, Deserialize)]
struct WorkerPayload {
    trace: Vec<Rep3Cycle>,
    memory: Rep3Memory,
    program_io_share: Rep3ProgramIOInput,
    io_device: JoltDevice,
    bytecode: Vec<tracer::instruction::Instruction>,
    memory_init: Vec<(u64, u8)>,
    padded_len: usize,
    ram_k: usize,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let args = Args::parse();

    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| eyre::eyre!("Could not install default rustls crypto provider"))?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .expect("set global Rayon pool");

    let config: NetworkConfigFile =
        toml::from_str(&std::fs::read_to_string(&args.config_file).context("opening config file")?)
            .context("parsing config file")?;
    let config = NetworkConfig::try_from(config).context("converting network config")?;

    if config.is_coordinator {
        run_coordinator(args, config)
    } else {
        run_worker(args, config)
    }
}

fn build_program() -> Program {
    let mut program = Program::new("sha2-chain-guest");
    // let mut program = Program::new("fibonacci-guest");
    // Match sha2-chain's #[jolt::provable(stack_size = 65536, memory_size = 10240)]
    program.set_stack_size(65536);
    program.set_memory_size(10240);
    program
}

fn build_inputs(num_iters: u32) -> Vec<u8> {
    // let mut inputs = postcard::to_stdvec(&128u8).unwrap();
    let mut inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
    inputs.append(&mut postcard::to_stdvec(&num_iters).unwrap());
    inputs
}

fn run_coordinator(args: Args, config: NetworkConfig) -> eyre::Result<()> {
    let file = format!(
        "trace_coordinator_sha2-chain-{}_{}CPU.json",
        args.num_iters,
        num_cpus::get(),
    );
    let _tracing_guard = init_tracing_bench(&file, &args.trace_dir);

    // Create coordinator network FIRST — workers connect to coordinator
    // during their Rep3QuicMpcNetWorker::new(), so we must be listening.
    info!("creating coordinator network");
    let mut network = Rep3QuicNetCoordinator::new(config, 0)?;

    // Build guest program and prepare inputs
    let mut program = build_program();
    let inputs = build_inputs(args.num_iters);
    let (bytecode, memory_init, _) = program.decode();

    // Trace to get vanilla trace and IO device
    info!("tracing guest program");
    let (mut vanilla_trace, _memory, io_device) = program.trace(&inputs, &[], &[]);

    // Pad trace
    let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
    info!(raw_len = vanilla_trace.len(), padded_len, "padding traces");
    vanilla_trace.resize(padded_len, Cycle::NoOp);

    // Build shared preprocessing for ram_K computation
    let shared = JoltSharedPreprocessing {
        memory_layout: io_device.memory_layout.clone(),
        bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
        ram: RAMPreprocessing::preprocess(memory_init.clone()),
    };
    let ram_k = compute_ram_k(&vanilla_trace, &shared);
    info!(ram_k, "computed ram_K");

    // Generate shares
    info!("generating trace shares");
    let mut rng = test_rng();
    let mut shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);
    // Pad shared traces
    // for (trace, _, _) in shares.iter_mut() {
    //     trace.resize(padded_len, Rep3Cycle::NoOp);
    // }

    // Build preprocessing (needed for verifier preprocessing)
    let preprocessing: JoltProverPreprocessing<F, PCS> =
        <JoltRV64IMAC as Rep3JoltWorker<F, PCS, _>>::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init.clone(),
            padded_len,
        );
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    // Send shares to workers
    info!("sending shares to workers");
    let worker_payloads: Vec<Vec<u8>> = shares
        .into_iter()
        .map(|(trace, memory, program_io_share)| {
            let payload = WorkerPayload {
                trace,
                memory,
                program_io_share,
                io_device: io_device.clone(),
                bytecode: bytecode.clone(),
                memory_init: memory_init.clone(),
                padded_len,
                ram_k,
            };
            bincode::serialize(&payload)
        })
        .collect::<bincode::Result<Vec<_>>>()
        .context("serializing worker payloads")?;

    network
        .send_requests_blocking(worker_payloads)
        .context("sending worker payloads")?;

    // Run coordinator prove
    info!("starting coordinator prove");
    let proof = <JoltRV64IMAC as Rep3Jolt<F, PCS, _>>::prove(
        &verifier_preprocessing,
        io_device,
        &mut network,
        ram_k,
        padded_len,
    )?;

    info!(commitments = proof.commitments.len(), "coordinator done");

    network.log_connection_stats(None);

    Ok(())
}

fn run_worker(args: Args, config: NetworkConfig) -> eyre::Result<()> {
    let my_id = config.my_id;
    let file = format!(
        "trace_party-{}_sha2-chain-{}_{}CPU.json",
        my_id,
        args.num_iters,
        num_cpus::get(),
    );
    let _tracing_guard = init_tracing_bench(&file, &args.trace_dir);

    // Create worker network
    info!("creating worker network");
    let mut network = Rep3QuicMpcNetWorker::new(config, 0)?;

    // Receive share from coordinator (includes bytecode + memory_init)
    info!("receiving share from coordinator");
    let payload_bytes: Vec<u8> = network.receive_request()?;
    let payload: WorkerPayload =
        bincode::deserialize(&payload_bytes).context("deserializing worker payload")?;

    let WorkerPayload {
        mut trace,
        memory,
        program_io_share,
        io_device,
        bytecode,
        memory_init,
        padded_len,
        ram_k,
    } = payload;

    let trace_len = trace.len();
    tracing::info!("trace length: {}", trace_len);

    // Pad trace if needed (should already be padded by coordinator)
    trace.resize(padded_len, Rep3Cycle::NoOp);

    // Build prover preprocessing
    info!("building preprocessing");
    let preprocessing: JoltProverPreprocessing<F, PCS> =
        <JoltRV64IMAC as Rep3JoltWorker<F, PCS, _>>::preprocess(
            bytecode,
            io_device.memory_layout.clone(),
            memory_init,
            padded_len,
        );

    // Init DoryGlobals (must stay alive during proving)
    let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);

    // Init AllCommittedPolynomials
    let bytecode_d = preprocessing.shared.bytecode.d;
    let ram_d = compute_d_parameter(ram_k);
    let _poly_guard = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

    // Wrap network in IoContextPool
    let num_forks = rayon::current_num_threads() as u32;

    let mut io_ctx = IoContextPool::init(network, num_forks)?;
    // populate_operands_casts: convert binary-shared operands to arithmetic
    populate_operands_casts(&mut trace, io_ctx.main())?;

    // Preprocessing: create EdaBits pool for B2A conversions
    let _span = info_span!("preprocessing", party_id = io_ctx.party_idx()).entered();
    let edabits_pool = {
        use co_jolt2::zkvm::instruction_lookups::read_raf_checking::compute_edabit_budget;
        use mpc_core::protocols::rep3_ring::edabits;
        use mpc_core::protocols::rep3_ring::pcg::edabits_pcg;
        let budget = compute_edabit_budget(trace_len);
        tracing::info!("budget: {:?}", budget);
        let lazy_u8 = edabits::random_edabits_lazy::<u8, F, _>(budget.u8, &mut io_ctx)?;
        let lazy_u16 = edabits::random_edabits_lazy::<u16, F, _>(budget.u16, &mut io_ctx)?;
        let lazy_u32 = edabits::random_edabits_lazy::<u32, F, _>(budget.u32, &mut io_ctx)?;
        let lazy_u64 = edabits::random_edabits_lazy::<u64, F, _>(budget.u64, &mut io_ctx)?;
        let lazy_u128 = edabits::random_edabits_lazy::<u128, F, _>(budget.u128, &mut io_ctx)?;
        let dabit_setup = edabits_pcg::random_pcg_dabit_setup::<F, _>(&mut io_ctx)?;
        edabits::EdaBitsPool::new(
            lazy_u8,
            lazy_u16,
            lazy_u32,
            lazy_u64,
            lazy_u128,
            dabit_setup,
            80 * trace_len,
        )
    };
    drop(_span);

    // Prove
    <JoltRV64IMAC as Rep3JoltWorker<F, PCS, _>>::prove(
        &preprocessing,
        trace,
        io_device,
        memory,
        &mut io_ctx,
        ram_k,
        Some(program_io_share),
        edabits_pool,
    )?;

    io_ctx.network().log_connection_stats();

    Ok(())
}
