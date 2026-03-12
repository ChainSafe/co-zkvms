use std::net::SocketAddr;

use ark_bn254::Fr;
use ark_serialize::CanonicalDeserialize;
use clap::Parser;
use eyre::Context;
use ::guest::{compile_sha2_chain, delegate_sha2_chain};
use tracing::info;

use co_jolt2::utils::compute_ram_k;
use co_jolt2::zkvm::JoltArch;
use jolt_sdk::host::Program;
use jolt_sdk::*;

type F = Fr;
type PCS = jolt_sdk::PCS;
type FS = jolt_core::transcripts::Blake2bTranscript;

#[derive(Parser)]
struct Args {
    /// Worker TLS addresses (comma-separated: host:port,host:port,host:port)
    #[clap(short = 'w', long)]
    workers: String,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Parse worker addresses
    let addrs: Vec<SocketAddr> = args
        .workers
        .split(',')
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
    let program = compile_sha2_chain(target_dir);

    // Delegate proof to workers
    let input = [5u8; 32];
    let num_iters = 10u32;

    info!("delegating proof...");
    let proof_bytes = delegate_sha2_chain(&mut client, program, input, num_iters)?;
    info!(proof_len = proof_bytes.len(), "received proof from workers");

    // Verify the proof
    info!("verifying proof...");

    let mut program = compile_sha2_chain(target_dir);
    let inputs = {
        let mut v = vec![];
        v.extend_from_slice(&jolt_sdk::postcard::to_stdvec(&input).unwrap());
        v.extend_from_slice(&jolt_sdk::postcard::to_stdvec(&num_iters).unwrap());
        v
    };

    let (bytecode, memory_init, _) = program.decode();
    let (vanilla_trace, _, io_device) = program.trace(&inputs, &[], &[]);
    let memory_layout = io_device.memory_layout.clone();

    let padded_len = (vanilla_trace.len() + 1).next_power_of_two();

    let shared = jolt_core::zkvm::JoltSharedPreprocessing {
        memory_layout: memory_layout.clone(),
        bytecode: jolt_core::zkvm::bytecode::BytecodePreprocessing::preprocess(bytecode.clone()),
        ram: jolt_core::zkvm::ram::RAMPreprocessing::preprocess(memory_init.clone()),
    };
    let ram_k = compute_ram_k(
        &{
            let mut t = vanilla_trace;
            t.resize(padded_len, tracer::instruction::Cycle::NoOp);
            t
        },
        &shared,
    );

    info!(padded_len, ram_k, "computed verification parameters");

    let preprocessing =
        <JoltArch as Jolt<F, PCS, FS>>::prover_preprocess(bytecode, memory_layout.clone(), memory_init, padded_len);
    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);

    use jolt_core::poly::commitment::dory::DoryGlobals;
    use jolt_core::zkvm::witness::{compute_d_parameter, AllCommittedPolynomials, DTH_ROOT_OF_K};

    let _dory_guard = DoryGlobals::initialize(DTH_ROOT_OF_K, padded_len);
    let _poly_guard = AllCommittedPolynomials::initialize(compute_d_parameter(ram_k), preprocessing.shared.bytecode.d);

    let proof: jolt_core::zkvm::dag::proof_serialization::JoltProof<F, PCS, FS> =
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

    use jolt_core::zkvm::dag::jolt_dag::JoltDAG;
    use jolt_core::zkvm::dag::state_manager::StateManager as VanillaStateManager;

    let verifier_sm = VanillaStateManager::from_proof(
        proof,
        Box::leak(Box::new(verifier_preprocessing)),
        program_io,
        ram_k,
        twist_switch,
    );
    JoltDAG::verify::<F, FS, PCS>(verifier_sm).map_err(|e| eyre::eyre!("{e:#}"))?;

    info!("proof verified successfully!");
    Ok(())
}
