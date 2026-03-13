//! TEE worker binary — stand-by loop proving service.
//!
//! Each worker:
//! 1. Loads network config, connects to peers (QUIC) and coordinator
//! 2. Starts a TLS listener for user connections (`user_listen_addr`)
//! 3. Loops:
//!    a. Accept user connection, receive WorkerPayload
//!    b. Sync with coordinator (barrier)
//!    c. Send ProofRequest (public data) to coordinator
//!    d. Preprocess + prove (MPC)
//!    e. [worker 0] Receive proof from coordinator, relay to user

use std::path::PathBuf;
#[cfg(feature = "test-utils")]
use std::sync::OnceLock;
use std::time::Duration;

#[cfg(feature = "tracy-mem")]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<tikv_jemallocator::Jemalloc> =
    tracy_client::ProfiledAllocator::new(tikv_jemallocator::Jemalloc, 0);

#[cfg(not(feature = "tracy-mem"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use ark_bn254::Fr;
use clap::Parser;
use eyre::Context;
use tracing::{info, info_span};

use co_jolt2::host::jolt_device::Rep3ProgramIOInput;
use co_jolt2::host::memory::Rep3Memory;
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::memory::start_jemalloc_monitor;
#[cfg(feature = "test-utils")]
use co_jolt2::utils::tracing::start_rss_monitor;
use co_jolt2::utils::tracing::{init_tracing_bench, worker_trace_file};
use co_jolt2::zkvm::dag::preproc_budget::compute_edabit_budget;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::JoltArch;
use co_jolt2::zkvm::Rep3JoltWorker;
use co_jolt_coordinator::types::ProofRequest;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use jolt_core::zkvm::JoltProverPreprocessing;
use mpc_core::protocols::rep3::network::IoContextPool;
use mpc_net::config::{NetworkConfig, NetworkConfigFile};
use mpc_net::rep3::quic::Rep3QuicMpcNetWorker;
use mpc_net::rep3::tls::worker_listener::TlsWorkerListener;
use mpc_net::topology::MpcStarNetWorker;
use serde::{Deserialize, Serialize};

type F = Fr;
type PCS = DoryCommitmentScheme;

#[derive(Parser)]
struct Args {
    /// Path to network config TOML
    #[clap(short = 'c', long)]
    config_file: PathBuf,

    /// Directory for trace output files
    #[cfg(feature = "test-utils")]
    #[clap(short = 't', long, default_value = "./.traces")]
    trace_dir: PathBuf,

    /// Number of preinitialized logical network forks
    #[clap(long, default_value = "4")]
    network_forks: u32,

    /// Number of Rayon threads
    #[clap(long, default_value = "4")]
    rayon_threads: usize,

    /// Base directory for persisted preprocessing data.
    /// Each party writes/reads from `<preproc_dir>/party_<id>/`.
    #[clap(short = 'p', long)]
    preproc_dir: PathBuf,
}

/// Payload sent by user's ProvingClient to each worker.
///
/// Contains the worker's secret share plus public data needed for proving.
/// NOTE: No plaintext advice or io_device — only shares and public metadata.
/// Workers compute `padded_len` (= trace.len()) and `ram_k` locally.
#[derive(Serialize, Deserialize)]
struct WorkerPayload {
    trace: Vec<Rep3Cycle>,
    memory: Rep3Memory,
    program_io_share: Rep3ProgramIOInput,
    bytecode: Vec<tracer::instruction::Instruction>,
    memory_init: Vec<(u64, u8)>,
    program_id: String,
    preprocess_trace_len: usize,
}

fn main() -> eyre::Result<()> {
    #[cfg(feature = "test-utils")]
    let _tracy = if std::env::var("TRACY").is_ok() {
        let client = tracy_client::Client::start();
        start_rss_monitor(Duration::from_millis(10));
        start_jemalloc_monitor(Duration::from_millis(50));
        Some(client)
    } else {
        None
    };

    let args = Args::parse();

    rayon::ThreadPoolBuilder::new().num_threads(args.rayon_threads).build_global().ok();

    let config: NetworkConfigFile =
        toml::from_str(&std::fs::read_to_string(&args.config_file).context("opening config file")?)
            .context("parsing config file")?;
    let config = NetworkConfig::try_from(config).context("converting network config")?;

    let my_id = config.my_id;

    rustls::crypto::aws_lc_rs::default_provider().install_default().ok();

    // Start TLS listener for user connections
    let user_listen_addr =
        config.user_listen_addr.ok_or_else(|| eyre::eyre!("worker config must have user_listen_addr"))?;
    let user_listener =
        TlsWorkerListener::bind(user_listen_addr, config.parties[my_id].cert.clone(), config.key.clone_key())?;
    info!(%user_listen_addr, "TLS user listener started");

    // Create worker network (QUIC ring + coordinator connection)
    info!("creating worker network");
    let network = Rep3QuicMpcNetWorker::new(config, 0)?;

    // Wrap in IoContextPool (lives for the lifetime of the process)
    let num_forks = args.network_forks;
    let mut io_ctx = IoContextPool::init(network, num_forks)?;
    info!("network initialized");

    prove_loop(&args, my_id, &mut io_ctx, &user_listener)
}

fn prove_loop(
    args: &Args,
    my_id: usize,
    io_ctx: &mut IoContextPool<Rep3QuicMpcNetWorker>,
    user_listener: &TlsWorkerListener,
) -> eyre::Result<()> {
    loop {
        // 1. Accept user connection, receive payload
        info!("waiting for user connection...");
        let mut user_conn = user_listener.accept()?;
        info!(peer = %user_conn.peer_addr(), "accepted user connection");

        let payload_bytes = user_conn.recv()?;
        let payload: WorkerPayload = bincode::deserialize(&payload_bytes).context("deserializing WorkerPayload")?;

        let WorkerPayload {
            trace,
            memory,
            program_io_share,
            bytecode,
            memory_init,
            program_id,
            preprocess_trace_len,
        } = payload;

        // Trace is already padded to next power of 2 by the client.
        let padded_len = trace.len();

        #[cfg(feature = "test-utils")]
        {
            static TRACING_INIT: OnceLock<()> = OnceLock::new();
            let file = worker_trace_file(my_id, &program_id);
            let _ = TRACING_INIT.get_or_init(|| {
                let guard = init_tracing_bench(&file, &args.trace_dir);
                let _ = Box::leak(Box::new(guard));
            });
        }

        // 2. Sync with coordinator (barrier: "we have shares, ready to prove")
        io_ctx.sync_with_coordinator()?;

        // 3. Build prover preprocessing (needed for ram_k computation and proving)
        let preprocessing: JoltProverPreprocessing<F, PCS> = <JoltArch as Rep3JoltWorker<F, PCS, _>>::preprocess(
            bytecode.clone(),
            program_io_share.memory_layout.clone(),
            memory_init.clone(),
            preprocess_trace_len,
        );

        // Compute ram_k from the shared trace (RAM addresses are public).
        let ram_k = compute_ram_k(&trace, &preprocessing.shared);
        info!(padded_len, ram_k, trace_len = trace.len(), "received payload from user");

        // 4. Send ProofRequest (public data) to coordinator
        let proof_request = ProofRequest {
            bytecode,
            memory_init,
            program_id,
            preprocess_trace_len,
            padded_len,
            ram_k,
            memory_layout: program_io_share.memory_layout.clone(),
            inputs: program_io_share.inputs.clone(),
            outputs: program_io_share.outputs.clone(),
            panic: program_io_share.panic,
        };
        let request_bytes = bincode::serialize(&proof_request).context("serializing ProofRequest")?;
        io_ctx.network().send_response(request_bytes)?;

        let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);
        #[cfg(feature = "ring-msm")]
        let dory_num_columns = DoryGlobals::get_num_columns();
        let bytecode_d = preprocessing.shared.bytecode.d;
        let ram_d = compute_d_parameter(ram_k);
        let _poly_guard = AllCommittedPolynomials::initialize(ram_d, bytecode_d);

        // 5. Preprocessing (edaBits + daBits + ring-MSM material)
        let party_id = io_ctx.party_id();
        let _span = info_span!("preprocessing", party_id = io_ctx.party_idx()).entered();

        let budget = compute_edabit_budget(trace.len());
        info!(?budget, "edabit budget");

        let mut preproc = {
            use mpc_core::protocols::rep3_ring::edabits;
            let counts = [budget.u8, budget.u16, budget.u32, budget.u64, budget.u128];
            let num_dabits = budget.dabits;

            let pool_dir = args.preproc_dir.join(format!("party_{}", my_id));
            match edabits::PreprocessingPool::load(&pool_dir, party_id) {
                Ok(mut pool) => {
                    let (rem_eda, rem_da) = pool.remaining_counts();
                    let deficit_counts: [usize; 5] = std::array::from_fn(|i| counts[i].saturating_sub(rem_eda[i]));
                    let deficit_dabits = num_dabits.saturating_sub(rem_da);
                    let deficit_re64 = budget.ring_edabits_u64.saturating_sub(pool.remaining_ring_edabits_u64());
                    let deficit_re128 = budget.ring_edabits_u128.saturating_sub(pool.remaining_ring_edabits_u128());
                    #[cfg(feature = "ring-msm")]
                    let (deficit_wm, deficit_re66) = (
                        budget.wrap_masks.saturating_sub(pool.remaining_wrap_masks()),
                        budget.ring_edabits_u66.saturating_sub(pool.remaining_ring_edabits_u66()),
                    );

                    let need_extend = deficit_counts.iter().any(|&d| d > 0)
                        || deficit_dabits > 0
                        || deficit_re64 > 0
                        || deficit_re128 > 0;
                    #[cfg(feature = "ring-msm")]
                    let need_extend = need_extend || deficit_wm > 0 || deficit_re66 > 0;

                    if need_extend {
                        info!(
                            ?deficit_counts,
                            deficit_dabits, deficit_re64, deficit_re128, "extending preprocessing pool"
                        );
                        #[cfg(not(feature = "ring-msm"))]
                        edabits::extend_pool_batched(
                            &mut pool,
                            deficit_counts,
                            deficit_dabits,
                            deficit_re64,
                            deficit_re128,
                            io_ctx,
                        )?;
                        #[cfg(feature = "ring-msm")]
                        edabits::extend_pool_batched(
                            &mut pool,
                            deficit_counts,
                            deficit_dabits,
                            deficit_wm,
                            deficit_re66,
                            deficit_re64,
                            deficit_re128,
                            io_ctx,
                        )?;
                        pool.save(&pool_dir).ok();
                    } else {
                        info!("reusing cached preprocessing");
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
                            io_ctx,
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
                            io_ctx,
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
                let lazy_dp = mpc_core::protocols::rep3_ring::preprocessing::daPoint::random_dapoints(&qs, io_ctx)?;
                preproc.set_dapoints(lazy_dp);
            }
        }
        drop(_span);

        // 6. Prove
        <JoltArch as Rep3JoltWorker<F, PCS, _>>::prove(
            &preprocessing,
            trace,
            program_io_share,
            memory,
            io_ctx,
            ram_k,
            &mut preproc,
        )?;

        // 7. [worker 0] Receive proof from coordinator, relay to user
        if my_id == 0 {
            let proof_bytes: Vec<u8> = io_ctx.network().receive_request()?;
            info!(proof_len = proof_bytes.len(), "received proof from coordinator");
            user_conn.send(&proof_bytes)?;
            info!("relayed proof to user");
        }

        info!("returning to stand-by");
    }
}
