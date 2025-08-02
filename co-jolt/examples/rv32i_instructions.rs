use ark_ff::UniformRand;
use ark_std::test_rng;
use clap::Parser;
use co_jolt::utils::math::Math;
use co_jolt::{
    host,
    jolt::{
        instruction::JoltInstructionSet,
        vm::{
            coordinator::JoltRep3,
            rv32i_vm::{RV32IJoltRep3Prover, RV32IJoltVM, RV32I},
            witness::{Rep3JoltPolynomials, Rep3Polynomials},
            worker::JoltRep3Prover,
            Jolt, JoltTraceStep,
        },
    },
    poly::{
        commitment::mock::MockCommitScheme,
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    utils::transcript::{KeccakTranscript, Transcript},
};
use co_jolt::{lasso::memory_checking::StructuredPolynomialData, poly::commitment::pst13::PST13};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use itertools::Itertools;
use jolt_core::{field::JoltField, jolt::vm::JoltProverPreprocessing, msm::icicle_init};
use jolt_tracer::JoltDevice;
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
};
use mpc_net::{
    config::{NetworkConfig, NetworkConfigFile},
    mpc_star::MpcStarNetWorker,
};
use mpc_net::{
    mpc_star::MpcStarNetCoordinator,
    rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator},
};
use std::path::PathBuf;

use clap::Subcommand;
use tracing_forest::ForestLayer;
use tracing_subscriber::{prelude::*, util::SubscriberInitExt, EnvFilter, Registry};

const C: usize = co_jolt::jolt::vm::rv32i_vm::C;
const M: usize = co_jolt::jolt::vm::rv32i_vm::M;
type F = ark_bn254::Fr;
type E = ark_bn254::Bn254;

// type CommitmentScheme = PST13<E>;
type CommitmentScheme = MockCommitScheme<F, KeccakTranscript>;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Parser)]
pub struct Args {
    /// The config file path
    #[clap(short, long, value_name = "FILE")]
    pub config_file: PathBuf,

    #[clap(short, long, value_name = "NUM_INPUTS", default_value = "8")]
    pub log_num_inputs: usize,

    #[arg(short, long, value_name = "SOLVE_WITNESS", env = "SOLVE_WITNESS")]
    pub solve_witness: bool,

    #[clap(short, long, value_name = "DEBUG", env = "DEBUG")]
    pub debug: bool,

    #[clap(short, long, value_name = "TRACE_PARTIES", env = "TRACE_PARTIES")]
    pub trace_parties: bool,

    #[clap(
        short,
        long,
        value_name = "NUM_WORKERS_PER_PARTY",
        default_value = "1",
        env = "NUM_WORKERS_PER_PARTY"
    )]
    pub num_workers_per_party: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| eyre!("Could not install default rustls crypto provider"))?;

    let config: NetworkConfigFile =
        toml::from_str(&std::fs::read_to_string(&args.config_file).context("opening config file")?)
            .context("parsing config file")?;
    let config = NetworkConfig::try_from(config).context("converting network config")?;

    if config.is_coordinator {
        // init_tracing();
    }

    let mut program = host::Program::new("sha3-guest");
    // let mut program = host::Program::new("fibonacci-guest");
    program.build(co_jolt::host::DEFAULT_TARGET_DIR);

    // let inputs = postcard::to_stdvec(&50u32).unwrap();
    let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();

    if config.is_coordinator {
        run_coordinator(args, config, program, inputs)?;
    } else {
        run_party(args, config, program, inputs)?;
    }

    Ok(())
}

pub fn run_party(
    args: Args,
    config: NetworkConfig,
    mut program: host::Program,
    inputs: Vec<u8>,
) -> Result<()> {
    let (bytecode, memory_init) = program.decode();
    let (program_io, trace) = program.trace::<F>(&inputs);

    let my_id = config.my_id;

    if args.trace_parties && my_id == 0 {
        init_tracing();
    }

    // let span = tracing::info_span!("run_party", id = my_id);
    // let _enter = span.enter();

    // if args.debug {
    //     return Ok(());
    // }

    if args.debug {
        return Ok(());
    }
    icicle_init();

    let network = Rep3QuicMpcNetWorker::new(
        config.clone(),
        args.num_workers_per_party.log_2(),
    )
    .unwrap();

    let preprocessing = RV32IJoltVM::prover_preprocess(
        bytecode,
        program_io.memory_layout,
        memory_init,
        trace.len().next_power_of_two(),
        trace.len().next_power_of_two(),
        trace.len().next_power_of_two(),
    );

    let mut prover = RV32IJoltRep3Prover::<F, CommitmentScheme, KeccakTranscript, _>::init(
        None,
        preprocessing,
        network,
    )?;

    prover.prove()?;

    prover.io_ctx.network().log_connection_stats();
    // drop(_enter);
    Ok(())
}

#[tracing::instrument(skip_all)]
pub fn run_coordinator(
    args: Args,
    config: NetworkConfig,
    mut program: host::Program,
    inputs: Vec<u8>,
) -> Result<()> {
    let (bytecode, memory_init) = program.decode();
    let (program_io, trace) = program.trace::<F>(&inputs);

    if config.is_coordinator {
        print_used_instructions(&trace);
    }

    let num_inputs = trace.len();
    if args.solve_witness {
        tracing::info!("Witness solving enabled");
    } else {
        tracing::warn!("Witness solving disabled");
    }

    // use jolt_core::poly::commitment::mock::MockCommitScheme;

    let preprocessing: JoltProverPreprocessing<C, F, CommitmentScheme, KeccakTranscript> =
        RV32IJoltVM::prover_preprocess(
            bytecode,
            program_io.memory_layout.clone(),
            memory_init,
            num_inputs.next_power_of_two(),
            num_inputs.next_power_of_two(),
            num_inputs.next_power_of_two(),
        );

    if args.debug {
        let (proof_check, commitments_check) =
            RV32IJoltVM::prove(program_io.clone(), trace.clone(), preprocessing.clone());

        RV32IJoltVM::verify(
            preprocessing.shared.clone(),
            proof_check,
            commitments_check,
            program_io.clone(),
        )
        .context("while verifying Lasso proof")?;
        return Ok(());
    }

    let mut network =
        Rep3QuicNetCoordinator::new(config.extend_with_workers(args.num_workers_per_party), args.num_workers_per_party.log_2()).unwrap();
    network.trim_subnets(1).unwrap();
    let (spartan_key, meta) = RV32IJoltVM::init_rep3(
        &preprocessing.shared,
        Some((trace.clone(), program_io.clone())),
        &mut network,
    )?;

    network.log_connection_stats(Some("Coordinator send witness communication"));
    network.reset_stats();

    let (proof, commitments) = RV32IJoltVM::prove_rep3(
        meta,
        &program_io,
        &spartan_key,
        &preprocessing.shared,
        &mut network,
    )?;

    RV32IJoltVM::verify(preprocessing.shared, proof, commitments, program_io)
        .context("while verifying Lasso (rep3) proof")?;

    network.log_connection_stats(None);

    Ok(())
}

fn print_used_instructions<F: JoltField, Instructions: JoltInstructionSet<F>>(
    instruction_trace: &[JoltTraceStep<F, Instructions>],
) {
    let opcodes_used = instruction_trace
        .par_iter()
        .filter_map(|step| match &step.instruction_lookup {
            Some(op) => Some(op.name()),
            None => None,
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .unique()
        .sorted()
        .collect::<Vec<_>>();
    tracing::info!("opcodes_used: {:?}", opcodes_used);
}

pub fn init_tracing() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();
    // .add_directive("jolt_core=trace".parse().unwrap());

    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default());

    let _ = tracing::subscriber::set_global_default(subscriber);
}
