use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[cfg(feature = "tracy-mem")]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<tikv_jemallocator::Jemalloc> =
    tracy_client::ProfiledAllocator::new(tikv_jemallocator::Jemalloc, 0);

#[cfg(not(feature = "tracy-mem"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use ark_bn254::Fr;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use color_eyre::eyre::{self, Context};
use mpc_net::config::{NetworkConfig, NetworkConfigFile};
use mpc_net::rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator};
use mpc_net::rep3::tls::worker_listener::TlsWorkerListener;
use mpc_net::topology::{MpcStarNetCoordinator, MpcStarNetWorker};
use serde::{Deserialize, Serialize};
use tracing::{info, info_span, trace_span, warn};

#[path = "../../jolt-sdk/src/client.rs"]
mod proving_client;

use co_jolt2::host::jolt_device::Rep3ProgramIOInput;
use co_jolt2::host::memory::Rep3Memory;
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::memory::start_jemalloc_monitor;
use co_jolt2::utils::tracing::init_tracing_bench;
use co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::JoltArch;
use co_jolt2::zkvm::Rep3JoltWorker;
use co_jolt_coordinator::proving::coordinate_once;
use co_jolt_coordinator::transport::ephemeral_identity::EphemeralIdentity;
use co_jolt_coordinator::transport::tcp_tls::TcpTlsCoordinator;
use co_jolt_coordinator::types::ProofRequest;
use jolt_core::curve::Bn254Curve;
use jolt_core::host::Program;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::dag::jolt_dag::JoltDAG;
use jolt_core::zkvm::dag::state_manager::StateManager as VanillaStateManager;
use jolt_core::zkvm::ram::RAMPreprocessing;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use jolt_core::zkvm::Jolt;
use jolt_core::zkvm::{JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing, JoltVerifierPreprocessing};
use mpc_core::protocols::rep3::network::IoContextPool;
use tracer::instruction::Cycle;
use tracer::JoltDevice;

use proving_client::Client as ProvingClient;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;

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
    preproc_dir: PathBuf,

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
    let warn_gb = std::env::var("PREPROC_WARN_GB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(10);
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

    rayon::ThreadPoolBuilder::new().num_threads(args.rayon_threads).build_global().expect("set global Rayon pool");

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

fn worker_addrs() -> [SocketAddr; 3] {
    let base_port: u16 =
        std::env::var("USER_LISTEN_BASE_PORT").ok().and_then(|value| value.parse().ok()).unwrap_or(30000);

    [
        SocketAddr::from(([127, 0, 0, 1], base_port)),
        SocketAddr::from(([127, 0, 0, 1], base_port + 1)),
        SocketAddr::from(([127, 0, 0, 1], base_port + 2)),
    ]
}

fn connect_client_with_retry(worker_addrs: [SocketAddr; 3]) -> eyre::Result<ProvingClient> {
    let attempts = 50;
    let delay = Duration::from_millis(200);
    let mut last_err = None;

    for _ in 0..attempts {
        match ProvingClient::connect(worker_addrs) {
            Ok(client) => return Ok(client),
            Err(err) => {
                last_err = Some(err);
                std::thread::sleep(delay);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| eyre::eyre!("failed to connect to workers")))
}

fn run_coordinator(args: Args, config: NetworkConfig) -> eyre::Result<()> {
    let file = format!("trace_coordinator_sha2-chain-{}_{}CPU.json", args.num_iters, num_cpus::get(),);
    let _tracing_guard = init_tracing_bench(&file, &args.trace_dir);

    if args.preprocess_only.unwrap_or(false) {
        return Err(eyre::eyre!("--preprocess-only is not supported by rep3_jolt.rs after the client/worker split"));
    }
    if args.repeat_proofs > 1 {
        return Err(eyre::eyre!("--repeat-proofs is not supported by rep3_jolt.rs after the client/worker split"));
    }

    let worker_addrs = worker_addrs();
    info!(?worker_addrs, "connecting proving client to workers");
    let num_iters = args.num_iters;
    let proof_thread = std::thread::spawn(move || -> eyre::Result<Vec<u8>> {
        let mut client = connect_client_with_retry(worker_addrs)?;
        let mut program = build_program();
        let inputs = build_inputs(num_iters);
        client.delegate(&mut program, &[], &inputs, &[])
    });

    let coordinator_protocol = config.coordinator.as_ref().map(|coordinator| coordinator.protocol).unwrap_or_default();
    match coordinator_protocol {
        mpc_net::config::CoordinatorProtocol::Quic => {
            info!("creating QUIC coordinator network");
            let mut network = Rep3QuicNetCoordinator::new(config, 0)?;
            coordinate_once(&mut network)?;
        }
        mpc_net::config::CoordinatorProtocol::Tls => {
            info!("creating TLS coordinator network");
            let identity = EphemeralIdentity::generate().context("generating coordinator TLS identity")?;
            let mut network = TcpTlsCoordinator::accept(config.bind_addr, &identity, None)
                .context("accepting TLS coordinator connections")?;
            coordinate_once(&mut network)?;
        }
    }
    let proof_bytes = proof_thread.join().map_err(|_| eyre::eyre!("proving client thread panicked"))??;
    info!(proof_len = proof_bytes.len(), "received proof bytes from worker relay");

    let mut program = build_program();
    let inputs = build_inputs(args.num_iters);
    let (bytecode, memory_init, _) = program.decode();

    info!("tracing guest program");
    let (mut vanilla_trace, _memory, mut io_device) = program.trace(&[], &inputs, &[]);
    // Truncate trailing zeros from outputs, matching what Jolt::prove does.
    io_device.outputs.truncate(io_device.outputs.iter().rposition(|&b| b != 0).map_or(0, |pos| pos + 1));

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

    let preprocessing: JoltProverPreprocessing<F, PCS> = <JoltArch as Jolt<F, PCS, FS>>::prover_preprocess(
        bytecode,
        io_device.memory_layout.clone(),
        memory_init,
        padded_len,
    );
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    let proof: jolt_core::zkvm::dag::proof_serialization::JoltProof<F, Bn254Curve, PCS, FS> =
        CanonicalDeserialize::deserialize_compressed(&proof_bytes[..]).context("deserializing proof")?;

    let twist_switch = proof.twist_sumcheck_switch_index;
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
        twist_switch,
    );
    JoltDAG::verify::<F, FS, PCS>(verifier_sm).map_err(|e| eyre::eyre!("{e:#}"))?;
    info!("coordinator proof verified successfully");

    Ok(())
}

fn run_worker(args: Args, config: NetworkConfig) -> eyre::Result<()> {
    let my_id = config.my_id;
    let file = format!("trace_party-{}_sha2-chain-{}_{}CPU.json", my_id, args.num_iters, num_cpus::get(),);
    let _tracing_guard = init_tracing_bench(&file, &args.trace_dir);

    if args.preprocess_only.unwrap_or(false) {
        return Err(eyre::eyre!("--preprocess-only is not supported by rep3_jolt.rs after the client/worker split"));
    }
    if args.repeat_proofs > 1 {
        return Err(eyre::eyre!("--repeat-proofs is not supported by rep3_jolt.rs after the client/worker split"));
    }

    let user_listen_addr =
        config.user_listen_addr.ok_or_else(|| eyre::eyre!("worker config must have user_listen_addr"))?;
    let user_listener =
        TlsWorkerListener::bind(user_listen_addr, config.parties[my_id].cert.clone(), config.key.clone_key())?;
    info!(%user_listen_addr, "TLS user listener started");

    info!("creating worker network");
    let network = Rep3QuicMpcNetWorker::new(config, 0)?;
    let num_forks = args.network_forks;
    let mut io_ctx = IoContextPool::init(network, num_forks)?;

    info!("waiting for user connection...");
    let mut user_conn = user_listener.accept()?;
    info!(peer = %user_conn.peer_addr(), "accepted user connection");

    let payload_bytes = user_conn.recv()?;
    let payload: WorkerPayload = bincode::deserialize(&payload_bytes).context("deserializing WorkerPayload")?;

    let WorkerPayload { mut trace, memory, program_io_share, bytecode, memory_init, padded_len, ram_k } = payload;
    let trace_len = trace.len();
    tracing::info!("trace length: {}", trace_len);

    io_ctx.sync_with_coordinator()?;

    let proof_request = ProofRequest {
        bytecode: bytecode.clone(),
        memory_init: memory_init.clone(),
        padded_len,
        ram_k,
        memory_layout: program_io_share.memory_layout.clone(),
        inputs: program_io_share.inputs.clone(),
        outputs: program_io_share.outputs.clone(),
        panic: program_io_share.panic,
    };
    let request_bytes = bincode::serialize(&proof_request).context("serializing ProofRequest")?;
    io_ctx.network().send_response(request_bytes)?;

    // Build prover preprocessing
    info!("building preprocessing");
    let preprocessing: JoltProverPreprocessing<F, PCS> = <JoltArch as Rep3JoltWorker<F, PCS, _>>::preprocess(
        bytecode,
        program_io_share.memory_layout.clone(),
        memory_init,
        padded_len,
    );

    // Init DoryGlobals (must stay alive during proving)
    let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);
    #[cfg(feature = "ring-msm")]
    let dory_num_columns = DoryGlobals::get_num_columns();

    // Init AllCommittedPolynomials
    let bytecode_d = preprocessing.shared.bytecode.d;
    let ram_d = compute_d_parameter(ram_k);
    let _poly_guard = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

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

        let pool_dir = args.preproc_dir.join(format!("party_{}", my_id));
        match edabits::PreprocessingPool::load(&pool_dir, party_id) {
            Ok(mut pool) => {
                let (rem_eda, rem_da) = pool.remaining_counts();
                let deficit_counts: [usize; 5] = std::array::from_fn(|i| counts[i].saturating_sub(rem_eda[i]));
                let deficit_dabits = num_dabits.saturating_sub(rem_da);
                let deficit_re64 = budget.ring_edabits_u64.saturating_sub(pool.remaining_ring_edabits_u64());
                let deficit_re128 = budget.ring_edabits_u128.saturating_sub(pool.remaining_ring_edabits_u128());
                #[cfg(feature = "ring-msm")]
                let (deficit_wm, deficit_re) = (
                    budget.wrap_masks.saturating_sub(pool.remaining_wrap_masks()),
                    budget.ring_edabits_u66.saturating_sub(pool.remaining_ring_edabits_u66()),
                );

                let need_extend = deficit_counts.iter().any(|&d| d > 0)
                    || deficit_dabits > 0
                    || deficit_re64 > 0
                    || deficit_re128 > 0;
                #[cfg(feature = "ring-msm")]
                let need_extend = need_extend || deficit_wm > 0 || deficit_re > 0;

                if need_extend {
                    info!("extending pool: deficit edabits={:?}, dabits={}", deficit_counts, deficit_dabits);
                    #[cfg(not(feature = "ring-msm"))]
                    edabits::extend_pool_batched(
                        &mut pool,
                        deficit_counts,
                        deficit_dabits,
                        deficit_re64,
                        deficit_re128,
                        &mut io_ctx,
                    )?;
                    #[cfg(feature = "ring-msm")]
                    edabits::extend_pool_batched(
                        &mut pool,
                        deficit_counts,
                        deficit_dabits,
                        deficit_wm,
                        deficit_re,
                        deficit_re64,
                        deficit_re128,
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
                #[cfg(not(feature = "ring-msm"))]
                {
                    edabits::preprocess_pool::<F, _>(
                        &pool_dir,
                        counts,
                        num_dabits,
                        budget.ring_edabits_u64,
                        budget.ring_edabits_u128,
                        &mut io_ctx,
                    )?
                }
                #[cfg(feature = "ring-msm")]
                {
                    edabits::preprocess_pool::<F, _>(
                        &pool_dir,
                        counts,
                        num_dabits,
                        budget.wrap_masks,
                        budget.ring_edabits_u66,
                        budget.ring_edabits_u64,
                        budget.ring_edabits_u128,
                        &mut io_ctx,
                    )?
                }
            }
        }
    };

    // Ring MSM preprocessing (daPoints only — wrap masks and ring edaBits are in the pool)
    #[cfg(feature = "ring-msm")]
    {
        if budget.dapoints > 0 {
            let qs = co_jolt2::poly::commitment::dory::precompute_dapoint_qs(
                &preprocessing.generators,
                budget.dapoints / 2,
                dory_num_columns,
            );
            let lazy_dp = mpc_core::protocols::rep3_ring::preprocessing::daPoint::random_dapoints(&qs, &mut io_ctx)?;
            preproc.set_dapoints(lazy_dp);
        }
    }
    drop(_span);

    trace.resize(padded_len, Rep3Cycle::NoOp);

    info!("starting worker prove");
    <JoltArch as Rep3JoltWorker<F, PCS, _>>::prove(
        &preprocessing,
        trace,
        program_io_share,
        memory,
        &mut io_ctx,
        ram_k,
        &mut preproc,
    )?;

    let (rem_eda, rem_da) = preproc.remaining_counts();
    info!(
        u8 = rem_eda[0],
        u16 = rem_eda[1],
        u32 = rem_eda[2],
        u64 = rem_eda[3],
        u128 = rem_eda[4],
        dabits = rem_da,
        "remaining preprocessing"
    );

    if my_id == 0 {
        let proof_bytes: Vec<u8> = io_ctx.network().receive_request()?;
        info!(proof_len = proof_bytes.len(), "received proof from coordinator");
        user_conn.send(&proof_bytes)?;
    }

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
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(padded_len);
    let chunk_size = if num_chunks > 0 { padded_len / num_chunks } else { padded_len };
    if chunk_size > 0 {
        chunk_size.trailing_zeros() as usize
    } else {
        0
    }
}
