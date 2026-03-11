//! TEE proving client binary — traces guest program, distributes shares,
//! receives proof, and verifies it.

use std::net::SocketAddr;
use std::path::PathBuf;

use ark_bn254::Fr;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use eyre::Context;
use tracing::info;

use co_jolt2::client::ProvingClient;
use co_jolt2::utils::tracing::init_tracing_bench;
use co_jolt2::zkvm::JoltArch;
use jolt_core::curve::Bn254Curve;
use jolt_core::host::Program;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::bytecode::BytecodePreprocessing;
use jolt_core::zkvm::dag::jolt_dag::JoltDAG;
use jolt_core::zkvm::dag::proof_serialization::JoltProof;
use jolt_core::zkvm::dag::state_manager::StateManager as VanillaStateManager;
use jolt_core::zkvm::ram::RAMPreprocessing;
use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};
use jolt_core::zkvm::{Jolt, JoltSharedPreprocessing, JoltVerifierPreprocessing};
use tracer::JoltDevice;

type F = Fr;
type PCS = DoryCommitmentScheme;
type FS = Blake2bTranscript;

#[derive(Parser)]
struct Args {
    /// Worker TLS addresses (comma-separated: host:port,host:port,host:port)
    #[clap(short = 'w', long)]
    workers: String,

    /// Directory for trace output files
    #[clap(short = 't', long, default_value = "./.traces")]
    trace_dir: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();
    let _tracing_guard = init_tracing_bench("trace_client.json", &args.trace_dir);

    // Parse worker addresses
    let addrs: Vec<SocketAddr> = args
        .workers
        .split(',')
        .map(|s| s.trim().parse::<SocketAddr>())
        .collect::<Result<_, _>>()
        .context("parsing worker addresses")?;
    let worker_addrs: [SocketAddr; 3] =
        addrs.try_into().map_err(|v: Vec<_>| eyre::eyre!("expected 3 worker addresses, got {}", v.len()))?;

    info!(?worker_addrs, "connecting to workers");
    let mut client = ProvingClient::connect(worker_addrs)?;
    info!("connected to all 3 workers");

    // Build + trace + delegate
    let mut program = Program::new("fibonacci-guest");
    program.set_memory_size(10240);
    let inputs = postcard::to_stdvec(&9u32).context("encoding inputs")?;

    info!("tracing guest program and delegating proof...");
    let proof_bytes = client.delegate(&mut program, &inputs, &[], &[])?;
    info!(proof_len = proof_bytes.len(), "received proof from workers");

    // Verify the proof
    info!("verifying proof...");

    let (bytecode, memory_init, _) = program.decode();
    let (vanilla_trace, _, io_device) = program.trace(&inputs, &[], &[]);
    let memory_layout = io_device.memory_layout.clone();

    // Compute padded_len the same way as delegate()
    let padded_len = (vanilla_trace.len() + 1).next_power_of_two();

    // Compute ram_k
    let shared = JoltSharedPreprocessing {
        memory_layout: memory_layout.clone(),
        bytecode: BytecodePreprocessing::preprocess(bytecode.clone()),
        ram: RAMPreprocessing::preprocess(memory_init.clone()),
    };
    let ram_k = co_jolt2::utils::compute_ram_k(
        &{
            let mut t = vanilla_trace;
            t.resize(padded_len, tracer::instruction::Cycle::NoOp);
            t
        },
        &shared,
    );

    info!(padded_len, ram_k, "computed verification parameters");

    // Initialize globals before deserialization (Dory types need these)
    let preprocessing =
        <JoltArch as Jolt<F, PCS, FS>>::prover_preprocess(bytecode, memory_layout.clone(), memory_init, padded_len);
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);
    let _poly_guard = AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), preprocessing.shared.bytecode.d);

    // Deserialize the proof
    let proof: JoltProof<F, Bn254Curve, PCS, FS> =
        CanonicalDeserialize::deserialize_compressed(&proof_bytes[..]).context("deserializing proof")?;

    let twist_switch = proof.twist_sumcheck_switch_index;
    info!(
        proof_padded_len = proof.trace_length,
        proof_ram_k = proof.ram_K,
        proof_twist_switch = twist_switch,
        "proof metadata"
    );

    let program_io = JoltDevice {
        inputs: io_device.inputs.clone(),
        outputs: io_device.outputs.clone(),
        panic: io_device.panic,
        memory_layout,
        trusted_advice: vec![],
        untrusted_advice: vec![],
    };

    let verifier_sm = VanillaStateManager::from_proof(
        proof,
        Box::leak(Box::new(verifier_preprocessing)),
        program_io,
        ram_k,
        twist_switch,
    );
    JoltDAG::verify::<F, Bn254Curve, FS, PCS>(verifier_sm)
        .map_err(|e| eyre::eyre!("{e:#}"))?;

    info!("proof verified successfully!");
    Ok(())
}
