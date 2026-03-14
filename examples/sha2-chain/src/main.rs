use std::net::SocketAddr;
use std::path::PathBuf;

use ::guest::{
    build_delegate_sha2_chain, build_verifier_sha2_chain, compile_sha2_chain, memory_config_sha2_chain, sha2_chain,
};
use ark_bn254::Fr;
use clap::Parser;
use eyre::Context;
use serde::Deserialize;
use tracing::info;
use tracing_forest::ForestLayer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::{EnvFilter, Layer};

use jolt_sdk::*;

type F = Fr;
type PCS = jolt_sdk::PCS;

#[derive(Deserialize)]
struct DelegatorConfig {
    workers: Vec<String>,
}

#[derive(Parser)]
struct Args {
    /// Path to delegator config TOML.
    #[clap(long, default_value = ".artifacts/config_delegator.toml")]
    config_path: PathBuf,

    /// Number of SHA-256 iterations
    #[clap(long, default_value = "1")]
    num_iters: u32,
}

fn init_tracing() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("rustls=off".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap());

    let _ = tracing::subscriber::set_global_default(
        Registry::default().with(env_filter).with(ForestLayer::default().with_filter(LevelFilter::INFO)),
    );
}

fn main() -> eyre::Result<()> {
    init_tracing();

    let args = Args::parse();
    let config: DelegatorConfig =
        toml::from_str(&std::fs::read_to_string(&args.config_path).context("reading delegator config")?)
            .context("parsing delegator config")?;

    // Parse worker addresses
    let addrs: Vec<SocketAddr> = config
        .workers
        .iter()
        .map(|s| s.trim().parse::<SocketAddr>())
        .collect::<Result<_, _>>()
        .context("parsing worker addresses")?;
    let worker_addrs: [SocketAddr; 3] =
        addrs.try_into().map_err(|v: Vec<_>| eyre::eyre!("expected 3 worker addresses, got {}", v.len()))?;

    // Connect to workers
    info!(?worker_addrs, "connecting to workers");
    let mut client = Client::connect(worker_addrs)?;
    info!("connected to all 3 workers");

    // Compile guest program
    let target_dir = "/tmp/jolt-guest-targets";
    let mut preprocessing_program = compile_sha2_chain(target_dir);
    let delegate = build_delegate_sha2_chain(compile_sha2_chain(target_dir));
    let input = [5u8; 32];
    let native_output = sha2_chain(input, args.num_iters);

    // Delegate proof to workers
    info!("delegating proof...");
    let program_id = format!("sha2-chain-{}", args.num_iters);
    let (output, proof, program_io) = delegate(&mut client, input, args.num_iters, &program_id)?;

    // Verify the proof
    let (bytecode, memory_init, program_size) = preprocessing_program.decode();
    let mut memory_config = memory_config_sha2_chain();
    memory_config.program_size = Some(program_size);
    let memory_layout = MemoryLayout::new(&memory_config);
    let prover_preprocessing: JoltProverPreprocessing<F, PCS> =
        JoltRVArch::prover_preprocess(bytecode, memory_layout, memory_init, proof.trace_length);
    let verifier = build_verifier_sha2_chain(JoltVerifierPreprocessing::from(&prover_preprocessing));
    info!("verifying proof...");
    let is_valid = verifier(input, args.num_iters, output, program_io.panic, proof);

    if !is_valid {
        return Err(eyre::eyre!("proof verification failed"));
    }
    if output != native_output {
        return Err(eyre::eyre!("native output mismatch"));
    }

    info!("proof verified successfully!");
    Ok(())
}
