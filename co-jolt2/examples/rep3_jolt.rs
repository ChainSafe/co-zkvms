use std::path::PathBuf;

#[cfg(feature = "tracy-mem")]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<tikv_jemallocator::Jemalloc> =
    tracy_client::ProfiledAllocator::new(tikv_jemallocator::Jemalloc, 0);

#[cfg(not(feature = "tracy-mem"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use ark_bn254::Fr;
use ark_std::test_rng;
use clap::Parser;
use color_eyre::eyre::{self, Context};
use mpc_net::config::{NetworkConfig, NetworkConfigFile};
use mpc_net::rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator};
use mpc_net::topology::{MpcStarNetCoordinator, MpcStarNetWorker};
use serde::{Deserialize, Serialize};
use tracing::{info, info_span, trace_span, warn};

use co_jolt_coordinator::zkvm::Rep3Jolt;
use co_jolt2::host::jolt_device::Rep3ProgramIOInput;
use co_jolt2::host::memory::Rep3Memory;
use co_jolt2::host::program::Rep3Program;
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::memory::start_jemalloc_monitor;
use co_jolt2::utils::tracing::init_tracing_bench;
use co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::JoltArch;
use co_jolt2::zkvm::Rep3JoltWorker;
use jolt_core::host::Program;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::RAMPreprocessing;
use mpc_net::rep3::quic::Rep3QuicNetCoordinator;
use mpc_net::topology::MpcStarNetCoordinator;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::RAMPreprocessing;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use jolt_core::zkvm::dag::jolt_dag::JoltDAG;
use jolt_core::zkvm::dag::state_manager::StateManager as VanillaStateManager;
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

    /// Base directory for persisted preprocessing data.
    ///
    /// Each party writes/reads from `<preproc_dir>/party_<id>/`.
    /// On first run: preprocessing runs and results are saved here.
    /// On subsequent runs (requires `--features reuse-preproc`):
    /// preprocessing is skipped and data is loaded from disk.
    #[clap(short = 'p', long)]
    preproc_dir: Option<PathBuf>,

    /// Preprocess only, without running the main computation.
    #[clap(short = 'P', long)]
    preprocess_only: Option<bool>,

    /// Number of Rayon threads to use in this process.
    #[clap(long, default_value = "4")]
    rayon_threads: usize,

    /// Number of preinitialized logical network forks to use.
    #[clap(long, default_value = "2")]
    network_forks: u32,

    /// Repeat the full proof pipeline N times in the same process.
    ///
    /// Requires `--features reuse-preproc` (or the pool will be consumed).
    #[clap(long, default_value = "1")]
    repeat_proofs: usize,
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

/// Coordinator → worker message.
///
/// For `--preprocess-only`, we only need `trace_len` (to compute the preprocessing
/// budget). Sending the full shared trace would dominate RSS and is unused.
#[derive(Serialize, Deserialize)]
enum CoordToWorkerMsg {
    Full(WorkerPayload),
    PreprocOnly(PreprocPayload),
}

#[derive(Serialize, Deserialize)]
struct PreprocPayload {
    trace_len: usize,
    edabit_counts: [usize; 5],
    dabits: usize,
}

fn log_preproc_size_estimates(counts: [usize; 5], num_dabits: usize) {
    let elem = std::mem::size_of::<F>() as u64;
    let warn_gb = std::env::var("PREPROC_WARN_GB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10);
    let warn_bytes = warn_gb * 1024 * 1024 * 1024;

    let sizes = [
        ("edabits_8.alpha2", counts[0] as u64 * 8 * elem),
        ("edabits_16.alpha2", counts[1] as u64 * 16 * elem),
        ("edabits_32.alpha2", counts[2] as u64 * 32 * elem),
        ("edabits_64.alpha2", counts[3] as u64 * 64 * elem),
        ("edabits_128.alpha2", counts[4] as u64 * 128 * elem),
        // dabits.stored size depends on party; this is the smaller (P0) bound.
        ("dabits.stored (P0)", num_dabits as u64 * elem),
        ("dabits.stored (P2)", num_dabits as u64 * 2 * elem),
    ];

    for (name, bytes) in sizes {
        if bytes >= warn_bytes {
            warn!(
                file = name,
                bytes,
                gb = (bytes as f64) / (1024.0 * 1024.0 * 1024.0),
                "large preprocessing artifact expected"
            );
        }
    }
}

/// Spawn a daemon thread that polls RSS every `interval` and reports it as a Tracy plot.
fn start_rss_monitor(interval: std::time::Duration) {
    use tracy_client::{plot_name, Client, PlotConfiguration, PlotFormat, PlotLineStyle};

    static RSS_PLOT: tracy_client::PlotName = plot_name!("RSS");
    let client = Client::running().expect("Tracy client must be running");
    client.plot_config(
        RSS_PLOT,
        PlotConfiguration::default()
            .format(PlotFormat::Memory)
            .line_style(PlotLineStyle::Smooth)
            .fill(true)
            .color(Some(0xFF6600)),
    );

    std::thread::Builder::new()
        .name("rss-monitor".into())
        .spawn(move || loop {
            let rss = get_rss_bytes();
            if let Some(c) = Client::running() {
                c.plot(RSS_PLOT, rss as f64);
            }
            std::thread::sleep(interval);
        })
        .expect("spawn rss-monitor thread");
}

#[cfg(target_os = "macos")]
fn get_rss_bytes() -> u64 {
    unsafe {
        let mut info: libc::mach_task_basic_info_data_t = std::mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        let ret = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut i32,
            &mut count,
        );
        if ret == 0 {
            info.resident_size
        } else {
            0
        }
    }
}

#[cfg(target_os = "linux")]
fn get_rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

fn main() -> eyre::Result<()> {
    // Start Tracy profiler only when TRACY=1 is set (manual-lifetime mode).
    // This lets us profile a single process (e.g. worker 0) without port conflicts.
    let _tracy = if std::env::var("TRACY").is_ok() {
        let client = tracy_client::Client::start();
        start_rss_monitor(std::time::Duration::from_millis(10));
        start_jemalloc_monitor(std::time::Duration::from_millis(50));
        Some(client)
    } else {
        None
    };

    color_eyre::install()?;

    let args = Args::parse();

    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| eyre::eyre!("Could not install default rustls crypto provider"))?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.rayon_threads)
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

    info!("creating coordinator network");
    let mut network = Rep3QuicNetCoordinator::new(config, 0)?;

    let mut program = build_program();
    let inputs = build_inputs(args.num_iters);
    let (bytecode, memory_init, _) = program.decode();

    info!("tracing guest program");
    let (mut vanilla_trace, _memory, io_device) = program.trace(&inputs, &[], &[]);

    let padded_len = (vanilla_trace.len() + 1).next_power_of_two();
    info!(raw_len = vanilla_trace.len(), padded_len, "padding traces");
    vanilla_trace.resize(padded_len, Cycle::NoOp);

    let shared = JoltSharedPreprocessing {
        memory_layout: io_device.memory_layout.clone(),
        bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
        ram: RAMPreprocessing::preprocess(memory_init.clone()),
    };
    let ram_k = compute_ram_k(&vanilla_trace, &shared);
    info!(ram_k, "computed ram_K");

    if args.preprocess_only.unwrap_or(false) {
        let budget = compute_edabit_budget(padded_len);
        let pp = PreprocPayload {
            trace_len: padded_len,
            edabit_counts: [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128],
            dabits: budget.dabits,
        };
        let msg = CoordToWorkerMsg::PreprocOnly(pp);
        let payload = bincode::serialize(&msg).context("serializing PreprocOnly")?;
        let worker_payloads = vec![payload; 3];
        info!("preprocess-only: sending PreprocOnly to workers");
        network
            .send_requests_blocking(worker_payloads)
            .context("sending PreprocOnly payloads")?;
        use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
        network.sync_with_parties()?;
        info!("preprocess-only: done");
        return Ok(());
    }

    info!("generating trace shares");
    let mut rng = test_rng();
    let shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);

    let preprocessing: JoltProverPreprocessing<F, PCS> =
        <JoltArch as Rep3JoltWorker<F, PCS, _>>::preprocess(
            bytecode.clone(),
            io_device.memory_layout.clone(),
            memory_init.clone(),
            padded_len,
        );
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

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
            let msg = CoordToWorkerMsg::Full(payload);
            bincode::serialize(&msg)
        })
        .collect::<bincode::Result<Vec<_>>>()
        .context("serializing worker payloads")?;

    network
        .send_requests_blocking(worker_payloads)
        .context("sending worker payloads")?;

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len),
        AllCommittedPolynomials::initialize(
            compute_d_parameter(ram_k),
            preprocessing.shared.bytecode.d,
        ),
    );

    let proof = <JoltArch as Rep3Jolt<F, PCS, _>>::prove(
        &verifier_preprocessing,
        &preprocessing.generators,
        io_device,
        &mut network,
        ram_k,
        padded_len,
    )?;
    info!(
        proof_size = std::mem::size_of_val(&proof),
        "coordinator proof complete"
    );

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

    if args.repeat_proofs > 1 && !cfg!(feature = "reuse-preproc") {
        return Err(eyre::eyre!(
            "--repeat-proofs > 1 requires building with --features reuse-preproc"
        ));
    }

    // Receive initial request from coordinator.
    info!("receiving request from coordinator");
    let first_payload_bytes: Vec<u8> = network.receive_request()?;
    let first_msg: CoordToWorkerMsg =
        bincode::deserialize(&first_payload_bytes).context("deserializing coordinator message")?;

    if let CoordToWorkerMsg::PreprocOnly(pp) = first_msg {
        // Wrap network in IoContextPool (required for preprocessing network rounds).
        let num_forks = args.network_forks;
        let mut io_ctx = IoContextPool::init(network, num_forks)?;
        let party_id = io_ctx.party_id();

        let _span = info_span!("preprocessing", party_id = io_ctx.party_idx()).entered();
        let counts = pp.edabit_counts;
        let num_dabits = pp.dabits;
        log_preproc_size_estimates(counts, num_dabits);

        if let Some(ref base_dir) = args.preproc_dir {
            let pool_dir = base_dir.join(format!("party_{}", my_id));
            use mpc_core::protocols::rep3_ring::edabits;

            let pool = match edabits::PreprocessingPool::load(&pool_dir, party_id) {
                Ok(mut pool) => {
                    let (rem_eda, rem_da) = pool.remaining_counts();
                    let deficit_counts: [usize; 5] =
                        std::array::from_fn(|i| counts[i].saturating_sub(rem_eda[i]));
                    let deficit_dabits = num_dabits.saturating_sub(rem_da);

                    if deficit_counts.iter().any(|&d| d > 0) || deficit_dabits > 0 {
                        info!(
                            "preprocess-only: extending pool: deficit edabits={:?}, dabits={}",
                            deficit_counts, deficit_dabits
                        );
                        edabits::extend_pool_batched(
                            &mut pool,
                            deficit_counts,
                            deficit_dabits,
                            &mut io_ctx,
                        )?;
                        match pool.save(&pool_dir) {
                            Ok(()) => info!("saved extended pool to {:?}", pool_dir),
                            Err(e) => tracing::warn!("failed to save extended pool: {e}"),
                        }
                    } else {
                        info!("preprocess-only: reusing preprocessing from {:?}", pool_dir);
                    }
                    pool
                }
                Err(e) => {
                    info!(
                        "preprocess-only: no cached preprocessing ({e}); creating pool into {:?}",
                        pool_dir
                    );
                    edabits::preprocess_pool_batched_into_dir::<F, _>(
                        &pool_dir,
                        counts,
                        num_dabits,
                        &mut io_ctx,
                    )?
                }
            };

            io_ctx.sync_with_parties()?;
            io_ctx.sync_with_coordinator()?;
            let _drop_span = trace_span!("drop_preprocessing_pool").entered();
            drop(pool);
            return Ok(());
        }

        return Err(eyre::eyre!(
            "preprocess-only requires --preproc-dir so alpha2/stored data can be persisted without OOM"
        ));
    }

    let CoordToWorkerMsg::Full(first_payload) = first_msg else {
        unreachable!("handled PreprocOnly above");
    };

    let WorkerPayload {
        trace: first_trace,
        memory: first_memory,
        program_io_share: first_program_io_share,
        io_device,
        bytecode,
        memory_init,
        padded_len,
        ram_k,
    } = first_payload;
    let mut first_trace = Some(first_trace);
    let mut first_memory = Some(first_memory);
    let mut first_program_io_share = Some(first_program_io_share);

    let trace_len = first_trace.as_ref().expect("worker trace present").len();
    tracing::info!("trace length: {}", trace_len);

    // Build prover preprocessing
    info!("building preprocessing");
    let preprocessing: JoltProverPreprocessing<F, PCS> =
        <JoltArch as Rep3JoltWorker<F, PCS, _>>::preprocess(
            bytecode,
            io_device.memory_layout.clone(),
            memory_init,
            padded_len,
        );

    // Init DoryGlobals (must stay alive during proving)
    let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);
    let dory_num_columns = DoryGlobals::get_num_columns();

    // Init AllCommittedPolynomials
    let bytecode_d = preprocessing.shared.bytecode.d;
    let ram_d = compute_d_parameter(ram_k);
    let _poly_guard = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

    // Wrap network in IoContextPool
    let num_forks = args.network_forks;

    let mut io_ctx = IoContextPool::init(network, num_forks)?;

    // Preprocessing: create EdaBits pool for B2A conversions.
    //
    // If `--preproc-dir` is provided, we attempt to load a previously saved pool
    // from `<preproc_dir>/party_<id>/`.  On a cache miss (files absent or corrupt)
    // we fall back to running preprocessing and saving the result.
    //
    // NOTE: all three parties must make the same load-vs-preprocess decision.
    // The run script achieves this by ensuring either all parties have their
    // preproc files (reuse run) or none do (fresh run).  Build with
    // `--features reuse-preproc` so consumed data is NOT zeroed on disk.
    let party_id = io_ctx.party_id();
    let _span = info_span!("preprocessing", party_id = io_ctx.party_idx()).entered();
    let budget = {
        use co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget;
        let b = compute_edabit_budget(trace_len);
        tracing::info!("budget: {:?}", b);
        b
    };
    let mut preproc = {
        use mpc_core::protocols::rep3_ring::edabits;
        let counts = [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128];
        let num_dabits = budget.dabits;
        log_preproc_size_estimates(counts, num_dabits);

        if let Some(ref base_dir) = args.preproc_dir {
            let pool_dir = base_dir.join(format!("party_{}", my_id));
            match edabits::PreprocessingPool::load(&pool_dir, party_id) {
                Ok(mut pool) => {
                    let (rem_eda, rem_da) = pool.remaining_counts();
                    let deficit_counts: [usize; 5] =
                        std::array::from_fn(|i| counts[i].saturating_sub(rem_eda[i]));
                    let deficit_dabits = num_dabits.saturating_sub(rem_da);

                    if deficit_counts.iter().any(|&d| d > 0) || deficit_dabits > 0 {
                        info!(
                            "extending pool: deficit edabits={:?}, dabits={}",
                            deficit_counts, deficit_dabits
                        );
                        edabits::extend_pool_batched(
                            &mut pool,
                            deficit_counts,
                            deficit_dabits,
                            &mut io_ctx,
                        )?;
                        match pool.save(&pool_dir) {
                            Ok(()) => info!("saved extended pool to {:?}", pool_dir),
                            Err(e) => {
                                tracing::warn!("failed to save extended pool: {e}")
                            }
                        }
                    } else {
                        info!("reusing preprocessing from {:?}", pool_dir);
                    }
                    pool
                }
                Err(e) => {
                    info!("no cached preprocessing ({e}); running preprocessing...");
                    edabits::preprocess_pool_batched_into_dir::<F, _>(
                        &pool_dir,
                        counts,
                        num_dabits,
                        &mut io_ctx,
                    )?
                }
            }
        } else {
            edabits::preprocess_pool_batched::<F, _>(counts, num_dabits, &mut io_ctx)?
        }
    };

    // daPoints for Dory U64Scalars wrap correction (offline, not persisted)
    if budget.dapoints > 0 {
        let qs = co_jolt2::poly::commitment::dory::precompute_dapoint_qs(
            &preprocessing.generators,
            budget.dapoints / 2,
            dory_num_columns,
        );
        let lazy_dp = mpc_core::protocols::rep3_ring::preprocessing::daPoint::random_dapoints(
            &qs,
            &mut io_ctx,
        )?;
        preproc.set_dapoints(lazy_dp);
    }
    drop(_span);

    if args.preprocess_only.unwrap_or(false) {
        io_ctx.sync_with_parties()?;
        let _drop_span = trace_span!("drop_preprocessing_pool").entered();
        drop(preproc);
        return Ok(());
    }

    for iter in 0..args.repeat_proofs {
        let (mut trace, memory, program_io_share) = if iter == 0 {
            (
                first_trace.take().expect("missing first trace"),
                first_memory.take().expect("missing first memory"),
                first_program_io_share
                    .take()
                    .expect("missing first program_io_share"),
            )
        } else {
            let payload_bytes: Vec<u8> = io_ctx.network().receive_request()?;
            let msg: CoordToWorkerMsg = bincode::deserialize(&payload_bytes)
                .context("deserializing coordinator message")?;
            let CoordToWorkerMsg::Full(payload) = msg else {
                return Err(eyre::eyre!("unexpected PreprocOnly message during proving"));
            };
            (payload.trace, payload.memory, payload.program_io_share)
        };

        // Pad trace if needed (should already be padded by coordinator)
        trace.resize(padded_len, Rep3Cycle::NoOp);

        if iter > 0 {
            #[cfg(feature = "reuse-preproc")]
            preproc.reset_cursors_for_reuse();
        }

        info!(iter, total = args.repeat_proofs, "starting worker prove");
        <JoltArch as Rep3JoltWorker<F, PCS, _>>::prove(
            &preprocessing,
            trace,
            io_device.clone(),
            memory,
            &mut io_ctx,
            ram_k,
            Some(program_io_share),
            &mut preproc,
        )?;

        let (rem_eda, rem_da) = preproc.remaining_counts();
        info!(
            iter,
            u8 = rem_eda[0],
            u16 = rem_eda[1],
            u32 = rem_eda[2],
            u64 = rem_eda[3],
            u128 = rem_eda[4],
            dabits = rem_da,
            "remaining preprocessing"
        );

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    io_ctx.network().log_connection_stats();

    Ok(())
}

fn print_used_instructions(instruction_trace: &[Rep3Cycle]) {
    use itertools::Itertools;
    use rayon::prelude::*;
    let opcodes_used = instruction_trace
        .par_iter()
        .filter_map(|cycle| match cycle {
            Rep3Cycle::NoOp => None,
            _ => {
                let name: &'static str = cycle.instruction().into();
                Some(name)
            }
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .sorted()
        .collect::<Vec<_>>();
    tracing::info!("opcodes_used: {:?}", opcodes_used);
}

fn proof_twist_sumcheck_switch_index(padded_len: usize) -> usize {
    let num_chunks = rayon::current_num_threads()
        .next_power_of_two()
        .min(padded_len);
    let chunk_size = if num_chunks > 0 {
        padded_len / num_chunks
    } else {
        padded_len
    };
    if chunk_size > 0 {
        chunk_size.trailing_zeros() as usize
    } else {
        0
    }
}
