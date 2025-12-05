#[cfg(feature = "debug")]
mod debug;

use co_jolt::jolt::vm::witness::Rep3Polynomials as _;
#[cfg(feature = "debug")]
use debug::*;

use ark_std::test_rng;
use clap::Parser;
use co_jolt::jolt::vm::read_write_memory::witness::Rep3ProgramIOInput;
use co_jolt::poly::commitment::pst13::PST13;
use co_jolt::utils::math::Math;
use co_jolt::{
    host,
    jolt::{
        instruction::JoltInstructionSet,
        vm::{
            coordinator::JoltRep3,
            rv32i_vm::{RV32IJoltRep3Prover, RV32IJoltVM, RV32I},
            Jolt, JoltTraceStep,
        },
    },
    utils::transcript::KeccakTranscript,
};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use itertools::{izip, Itertools};
use jolt_core::jolt::vm::JoltProverPreprocessing;

use mpc_core::protocols::rep3::network::{IoContext, IoContextPool};
use mpc_net::rep3::PartyWorkerID;
use mpc_net::{
    config::{NetworkConfig, NetworkConfigFile},
    topology::MpcStarNetWorker,
};
use mpc_net::{
    rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator},
    topology::MpcStarNetCoordinator,
};

use std::path::{Path, PathBuf};
use tracing_chrome::{ChromeLayerBuilder, FlushGuard};
use tracing_forest::util::LevelFilter;

use tracing_forest::ForestLayer;
use tracing_subscriber::{prelude::*, EnvFilter, Registry};

const C: usize = co_jolt::jolt::vm::rv32i_vm::C;
type F = ark_bn254::Fr;
type E = ark_bn254::Bn254;

type CommitmentScheme = PST13<E>;
// type CommitmentScheme = MockCommitScheme<F, KeccakTranscript>;

use rayon::prelude::*;

#[derive(Parser)]
pub struct Args {
    /// The config file path
    #[clap(short, long, value_name = "FILE")]
    pub config_file: PathBuf,

    #[arg(
        short,
        long,
        value_name = "SOLVE_WITNESS",
        env = "SOLVE_WITNESS",
        default_value = "false"
    )]
    pub solve_witness: bool,

    #[clap(short, long, value_name = "DEBUG", env = "DEBUG")]
    pub debug: bool,

    #[clap(
        short,
        long,
        value_name = "TRACE_PARTIES",
        env = "TRACE_PARTIES",
        default_value = "true"
    )]
    pub trace_parties: TraceParties,

    #[clap(
        short,
        long,
        value_name = "TRACE_DIR",
        env = "TRACE_DIR",
        default_value = "./traces"
    )]
    pub trace_dir: PathBuf,

    #[clap(
        short,
        long,
        value_name = "NUM_WORKERS_PER_PARTY",
        default_value = "1",
        env = "NUM_WORKERS_PER_PARTY"
    )]
    pub num_workers_per_party: usize,

    #[clap(
        short,
        long,
        value_name = "NUM_ITERATIONS",
        default_value = "1",
        env = "NUM_ITERATIONS"
    )]
    pub num_iterations: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| eyre!("Could not install default rustls crypto provider"))?;

    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .expect("set global Rayon pool");

    let config: NetworkConfigFile =
        toml::from_str(&std::fs::read_to_string(&args.config_file).context("opening config file")?)
            .context("parsing config file")?;
    let config = NetworkConfig::try_from(config).context("converting network config")?;

    let mut program = host::Program::new("sha2-chain-guest");
    program.build(co_jolt::host::DEFAULT_TARGET_DIR);

    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&args.num_iterations).unwrap());
    // let inputs = postcard::to_stdvec(&1u32).unwrap();
    // let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();

    if config.is_coordinator {
        run_coordinator(args, config, program, inputs)?;
    } else {
        run_party(args, config, program)?;
    }

    Ok(())
}

pub fn run_party(args: Args, config: NetworkConfig, mut program: host::Program) -> Result<()> {
    let (bytecode, memory_init) = program.decode();

    let my_id = config.my_id;
    let file = format!(
        "trace_party-{}_sha2-chain-{}_{}CPU.json",
        my_id,
        args.num_iterations,
        num_cpus::get(),
    );

    let tracing_guard = match args.trace_parties {
        TraceParties::All(true) => init_tracing(&file, &args.trace_dir),
        TraceParties::Party(parties) => {
            if parties.contains(&my_id) {
                init_tracing(&file, &args.trace_dir)
            } else {
                None
            }
        }
        _ => None,
    };

    if args.debug {
        return Ok(());
    }

    let mut network =
        Rep3QuicMpcNetWorker::new(config.clone(), args.num_workers_per_party.log_2()).unwrap();

    // let program_io: Rep3ProgramIOInput = network.receive_request()?;
    let (program_io, trace): (Rep3ProgramIOInput, Vec<JoltTraceStep<RV32I>>) =
        bincode::deserialize(&network.receive_request::<Vec<u8>>()?)?;
    tracing::info!("trace len: {}", trace.len());

    let max_bytecode_size = bytecode.len().next_power_of_two();

    let preprocessing = RV32IJoltVM::prover_preprocess(
        bytecode,
        program_io.memory_layout,
        memory_init,
        max_bytecode_size,
        trace.len().next_power_of_two(),
        trace.len().next_power_of_two(),
    );

    let mut prover = RV32IJoltRep3Prover::<F, CommitmentScheme, KeccakTranscript, _>::init(
        trace,
        program_io,
        preprocessing,
        network,
    )?;

    // prover.io_ctx.network().send_response(prover.program_io)?;

    // let mut io_ctx = IoContextPool::init(network, rayon::current_num_threads() as u32)?;

    // println!(
    //     "worker {} party {} got {}",
    //     prover.io_ctx.network().worker_idx(),
    //     prover.io_ctx.network().party_id() as usize,
    //     prover.io_ctx.network().receive_request::<usize>()?
    // );

    // let gid = PartyWorkerID::new(
    //     prover.io_ctx.party_idx(),
    //     prover.io_ctx.network().worker_idx(),
    // )
    // .global_worker_id();
    // prover.io_ctx.network().send_response(gid)?;

    // prover.io_ctx.network().send_response(prover.polynomials)?;

    prover.prove()?;

    prover.io_ctx.log_connection_stats();
    drop(tracing_guard);
    Ok(())
}

#[tracing::instrument(skip_all)]
pub fn run_coordinator(
    args: Args,
    config: NetworkConfig,
    mut program: host::Program,
    inputs: Vec<u8>,
) -> Result<()> {
    let file = format!(
        "trace_coordinator_sha2-chain-{}_{}CPU.json",
        args.num_iterations,
        num_cpus::get(),
    );

    let _tracing_guard = init_tracing(&file, &args.trace_dir);

    let (bytecode, memory_init) = program.decode();
    let (program_io, trace) = program.trace(&inputs);

    if config.is_coordinator {
        print_used_instructions(&trace);
    }

    let num_inputs = trace.len();
    if args.solve_witness {
        tracing::info!("Witness solving enabled");
        unimplemented!();
    } else {
        tracing::warn!("Witness solving disabled");
    }

    let max_bytecode_size = bytecode.len().next_power_of_two();

    let preprocessing: JoltProverPreprocessing<C, F, CommitmentScheme, KeccakTranscript> =
        RV32IJoltVM::prover_preprocess(
            bytecode,
            program_io.memory_layout,
            memory_init,
            max_bytecode_size,
            num_inputs.next_power_of_two(),
            num_inputs.next_power_of_two(),
        );

    // if args.debug {
    //     let (proof_check, commitments_check) =
    //         RV32IJoltVM::prove(program_io.clone(), trace.clone(), preprocessing.clone());

    //     RV32IJoltVM::verify(
    //         preprocessing.shared.clone(),
    //         proof_check,
    //         commitments_check,
    //         program_io.clone(),
    //     )
    //     .context("while verifying Lasso proof")?;
    //     return Ok(());
    // }
    //

    let mut rng = test_rng();
    let (program_io_shares, trace_shares) =
        program.generate_trace_shares::<F, _>(&inputs, &mut rng);

    let mut network =
        Rep3QuicNetCoordinator::new(config, args.num_workers_per_party.log_2()).unwrap();
    // network.trim_subnets(1).unwrap();
    let worker_shares = izip!(program_io_shares, trace_shares)
        .map(|s| bincode::serialize(&s))
        .cycle()
        .take(3 * args.num_workers_per_party)
        .collect::<bincode::Result<Vec<_>>>()
        .context("while serializing trace shares")?;
    network.send_requests_blocking(worker_shares)?;

    let (spartan_key, meta) = RV32IJoltVM::init_rep3(&preprocessing.shared, &mut network)?;

    network.log_connection_stats(Some("IO witness: "));
    // network.reset_stats();

    // network.send_requests(vec![0usize, 1, 2, 3, 4, 5]).unwrap();
    // println!("ids: {:?}", network.receive_responses::<usize>().unwrap());

    // let worker_polys: Vec<_> = network.receive_responses().unwrap();

    // let worker_polys = worker_polys
    //     .into_iter()
    //     .chunks(3)
    //     .into_iter()
    //     .map(|chunk| {
    //         co_jolt::jolt::vm::witness::Rep3JoltPolynomials::combine_polynomials(
    //             &preprocessing.shared,
    //             chunk.collect(),
    //         )
    //     })
    //     .collect::<Vec<_>>();

    // JoltTraceStep::pad(&mut trace);
    // let mut check = RV32IJoltVM::generate_witness(&preprocessing.shared, trace, &program_io);
    // let r1cs_builder: jolt_core::r1cs::builder::CombinedUniformBuilder<
    //     C,
    //     F,
    //     co_jolt::r1cs::inputs::JoltR1CSInputs,
    // > = <co_jolt::r1cs::constraints::JoltRV32IMConstraints as jolt_core::r1cs::constraints::R1CSConstraints>::construct_constraints(
    //     meta.padded_trace_length,
    //     program_io.memory_layout.input_start,
    // );
    // r1cs_builder.compute_aux(&mut check);

    // check_instruction_polys(
    //     worker_polys
    //         .iter()
    //         .map(|p| &p.instruction_lookups)
    //         .collect_vec(),
    //     &check.instruction_lookups,
    // );

    // println!("CORRECT");

    let (proof, commitments) = RV32IJoltVM::prove_rep3(
        meta,
        // &program_io,
        &spartan_key,
        &preprocessing.shared,
        &mut network,
    )?;

    RV32IJoltVM::verify(preprocessing.shared, proof, commitments, program_io)
        .context("while verifying Lasso (rep3) proof")?;

    tracing::info!("VERIFIED!");

    network.log_connection_stats(None);

    Ok(())
}

fn print_used_instructions<Instructions: JoltInstructionSet>(
    instruction_trace: &[JoltTraceStep<Instructions>],
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

pub fn init_tracing(file: &str, trace_dir: &Path) -> Option<TracingGuard> {
    std::fs::create_dir_all(trace_dir).unwrap();
    let trace_path = trace_dir.join(file);
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy()
        .add_directive("jolt_core=info".parse().unwrap())
        .add_directive("co-snarks=info".parse().unwrap())
        .add_directive("mpc_net=info".parse().unwrap())
        .add_directive("quinn=off".parse().unwrap());

    let current_level = env_filter.max_level_hint().unwrap_or(LevelFilter::INFO);
    let subscriber = Registry::default().with(env_filter);

    if current_level == LevelFilter::TRACE {
        let (chrome_layer, _guard) = ChromeLayerBuilder::new().file(trace_path).build();
        let _ = tracing::subscriber::set_global_default(
            subscriber
                .with(chrome_layer)
                .with(ForestLayer::default().with_filter(LevelFilter::INFO)),
        );
        tracing::info!("tracing_chrome writes to file: {}", file);
        Some(TracingGuard {
            _guard: Some(_guard),
            file: file.to_string(),
        })
    } else {
        let _ = tracing::subscriber::set_global_default(subscriber.with(ForestLayer::default()));
        // let _ = tracing::subscriber::set_global_default(
        //     subscriber.with(fmt::layer().with_writer(std::io::stderr)),
        // );

        None
    }
}

pub struct TracingGuard {
    _guard: Option<FlushGuard>,
    file: String,
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        tracing::info!("tracing_chrome available at: {}", self.file);
        if let Some(guard) = self._guard.take() {
            drop(guard);
        }
    }
}

#[derive(Clone, Debug)]
pub enum TraceParties {
    All(bool),
    Party(Vec<usize>),
}

impl Default for TraceParties {
    fn default() -> Self {
        TraceParties::All(true)
    }
}

impl std::str::FromStr for TraceParties {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(b) = s.parse::<bool>() {
            Ok(TraceParties::All(b))
        } else if let Ok(nums) = s
            .split(',')
            .map(|n| n.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(TraceParties::Party(nums))
        } else {
            Err(format!("Invalid trace parties: {}", s))
        }
    }
}
