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
use mpc_net::rep3::quic::Rep3QuicNetCoordinator;
use mpc_net::topology::MpcStarNetCoordinator;
use serde::{Deserialize, Serialize};
use tracing::info;

use co_jolt_coordinator::zkvm::Rep3Jolt;
use co_jolt2::host::jolt_device::Rep3ProgramIOInput;
use co_jolt2::host::memory::Rep3Memory;
use co_jolt2::host::program::Rep3Program;
use co_jolt2::utils::compute_ram_k;
use co_jolt2::utils::memory::start_jemalloc_monitor;
use co_jolt2::utils::tracing::init_tracing_bench;
use co_jolt2::zkvm::instruction::Rep3Cycle;
use co_jolt2::zkvm::Rep3JoltWorker;
use jolt_core::host::Program;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::ram::RAMPreprocessing;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use jolt_core::zkvm::{
    JoltProverPreprocessing, JoltRV64IMAC, JoltSharedPreprocessing, JoltVerifierPreprocessing,
};
use tracer::instruction::Cycle;
use tracer::JoltDevice;

type F = Fr;
type PCS = DoryCommitmentScheme;

#[derive(Parser)]
struct Args {
    #[clap(short = 'c', long)]
    config_file: PathBuf,
    #[clap(short = 't', long, default_value = "./.traces")]
    trace_dir: PathBuf,
    #[clap(short = 'n', long, default_value = "1")]
    num_iters: u32,
    #[clap(short = 'p', long)]
    preproc_dir: Option<PathBuf>,
    #[clap(short = 'P', long)]
    preprocess_only: Option<bool>,
    #[clap(long, default_value = "4")]
    rayon_threads: usize,
    #[clap(long, default_value = "1")]
    repeat_proofs: usize,
}

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

fn build_program() -> Program {
    let mut program = Program::new("sha2-chain-guest");
    program.set_stack_size(65536);
    program.set_memory_size(10240);
    program
}

fn build_inputs(num_iters: u32) -> Vec<u8> {
    let mut inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
    inputs.append(&mut postcard::to_stdvec(&num_iters).unwrap());
    inputs
}

fn main() -> eyre::Result<()> {
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
    eyre::ensure!(config.is_coordinator, "coordinator example requires coordinator config");

    run_coordinator(args, config)
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

    info!("generating trace shares");
    let mut rng = test_rng();
    let shares = program.generate_trace_shares(&inputs, &[], &[], &mut rng);

    let preprocessing: JoltProverPreprocessing<F, PCS> =
        <JoltRV64IMAC as Rep3JoltWorker<F, PCS, _>>::preprocess(
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
            bincode::serialize(&payload)
        })
        .collect::<bincode::Result<Vec<_>>>()
        .context("serializing worker payloads")?;

    if args.preprocess_only.unwrap_or(false) {
        info!("preprocess-only: sending worker payload once and exiting");
    }
    network
        .send_requests_blocking(worker_payloads)
        .context("sending worker payloads")?;

    let _guard = (
        DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len),
        AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), preprocessing.shared.bytecode.d),
    );

    let proof = <JoltRV64IMAC as Rep3Jolt<F, PCS, _>>::prove(
        &verifier_preprocessing,
        &preprocessing.generators,
        io_device,
        &mut network,
        ram_k,
        padded_len,
    )?;
    info!(proof_size = std::mem::size_of_val(&proof), "coordinator proof complete");

    Ok(())
}
