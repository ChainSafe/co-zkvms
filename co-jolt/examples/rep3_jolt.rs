use ark_ff::{Field, UniformRand};
use ark_std::test_rng;
use clap::Parser;
use co_jolt::field::JoltField;
use co_jolt::jolt::instruction::Rep3JoltInstructionSet;
use co_jolt::jolt::vm::instruction_lookups::witness::Rep3InstructionLookupPolynomials;
use co_jolt::jolt::vm::instruction_lookups::{
    InstructionLookupPolynomials, InstructionLookupsProof,
};
use co_jolt::jolt::vm::read_write_memory::witness::{Rep3ProgramIO, Rep3ProgramIOInput};
use co_jolt::poly::Rep3MultilinearPolynomial;
use co_jolt::r1cs::constraints::JoltRV32IMConstraints;
use co_jolt::r1cs::inputs::JoltR1CSInputs;
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
use itertools::{izip, Itertools};
use jolt_core::jolt::vm::bytecode::BytecodePolynomials;
use jolt_core::jolt::vm::read_write_memory::{
    memory_address_to_witness_index, ReadWriteMemoryPolynomials,
};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::r1cs::builder::CombinedUniformBuilder;
use jolt_core::r1cs::constraints::R1CSConstraints as _;
use jolt_core::r1cs::inputs::R1CSPolynomials;
use jolt_core::r1cs::key::UniformSpartanKey;
use jolt_core::{jolt::vm::JoltProverPreprocessing, msm::icicle_init};
use jolt_tracer::JoltDevice;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
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
use std::env;
use std::iter::Inspect;
use std::path::{Path, PathBuf};
use tracing_chrome::{ChromeLayerBuilder, FlushGuard};
use tracing_forest::util::LevelFilter;
use tracing_subscriber::fmt;

use clap::Subcommand;
use tracing_forest::ForestLayer;
use tracing_subscriber::{prelude::*, util::SubscriberInitExt, EnvFilter, Registry};

const C: usize = co_jolt::jolt::vm::rv32i_vm::C;
type F = ark_bn254::Fr;
type E = ark_bn254::Bn254;

type CommitmentScheme = PST13<E>;
// type CommitmentScheme = MockCommitScheme<F, KeccakTranscript>;

#[cfg(feature = "parallel")]
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
        .num_threads(num_cpus::get())
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

    // let span = tracing::info_span!("run_party", id = my_id);
    // let _enter = span.enter();

    // if args.debug {
    //     return Ok(());
    // }

    if args.debug {
        return Ok(());
    }
    // icicle_init();

    let mut network =
        Rep3QuicMpcNetWorker::new(config.clone(), args.num_workers_per_party.log_2(), 1 << 10)
            .unwrap();

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
        Some((trace, program_io)),
        preprocessing,
        network,
    )?;

    prover.io_ctx.network().send_response(prover.program_io)?;
    prover.io_ctx.network().send_response(prover.polynomials)?;

    // prover.prove()?;

    prover.io_ctx.log_connection_stats();
    // drop(_enter);
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
        // std::time::SystemTime::now()
        //     .duration_since(std::time::UNIX_EPOCH)
        //     .unwrap()
        //     .as_secs()
    );

    let _tracing_guard = init_tracing(&file, &args.trace_dir);

    let (bytecode, memory_init) = program.decode();
    let (program_io, mut trace) = program.trace(&inputs);

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

    // use jolt_core::poly::commitment::mock::MockCommitScheme;
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

    let mut network = Rep3QuicNetCoordinator::new(
        config.extend_with_workers(args.num_workers_per_party),
        args.num_workers_per_party.log_2(),
    )
    .unwrap();
    network.trim_subnets(1).unwrap();
    network.send_requests_blocking(
        program_io_shares
            .into_iter()
            .zip(trace_shares)
            .map(|s| bincode::serialize(&s))
            .collect::<bincode::Result<Vec<_>>>()
            .context("while serializing trace shares")?,
    )?;

    let (spartan_key, meta) = RV32IJoltVM::init_rep3(
        &preprocessing.shared,
        None, // Some((trace, program_io.clone())),
        &mut network,
    )?;

    network.log_connection_stats(Some("IO witness: "));
    network.reset_stats();

    check_program_io::<F>(network.receive_responses()?, &program_io);

    let polys = Rep3JoltPolynomials::combine_polynomials(
        &preprocessing.shared,
        network.receive_responses()?,
    );
    JoltTraceStep::pad(&mut trace);
    let mut check = RV32IJoltVM::generate_witness(&preprocessing.shared, trace, &program_io);
    let r1cs_builder: CombinedUniformBuilder<C, F, JoltR1CSInputs> =
        JoltRV32IMConstraints::construct_constraints(
            meta.padded_trace_length,
            program_io.memory_layout.input_start,
        );
    r1cs_builder.compute_aux(&mut check);

    check_instruction_polys(&polys.instruction_lookups, &check.instruction_lookups);
    check_read_write_polys(&polys.read_write_memory, &check.read_write_memory);
    check_bytecode(&polys.bytecode, &check.bytecode);
    check_r1cs(&polys.r1cs, &check.r1cs);

    // let (proof, commitments) = RV32IJoltVM::prove_rep3(
    //     meta,
    //     // &program_io,
    //     &spartan_key,
    //     &preprocessing.shared,
    //     &mut network,
    // )?;

    // RV32IJoltVM::verify(preprocessing.shared, proof, commitments, program_io)
    //     .context("while verifying Lasso (rep3) proof")?;

    // network.log_connection_stats(None);

    Ok(())
}

fn check_instruction_polys<F: JoltField>(
    polys: &InstructionLookupPolynomials<F>,
    check: &InstructionLookupPolynomials<F>,
) {
    check_poly(
        &polys.lookup_outputs,
        &check.lookup_outputs,
        "lookup_outputs",
    );
    check_polys(&polys.dim, &check.dim, "dim");
    check_polys(&polys.final_cts, &check.final_cts, "final_cts");
    check_polys(&polys.read_cts, &check.read_cts, "read_cts");
    check_polys(&polys.E_polys, &check.E_polys, "E_polys");
}

fn check_read_write_polys(
    polys: &ReadWriteMemoryPolynomials<F>,
    check: &ReadWriteMemoryPolynomials<F>,
) {
    check_poly(
        polys.v_init.as_ref().unwrap(),
        check.v_init.as_ref().unwrap(),
        "v_init",
    );
    check_poly(&polys.v_final, &check.v_final, "v_final");
    check_poly(&polys.v_read_rd, &check.v_read_rd, "read_rd");
    check_poly(&polys.v_read_rs1, &check.v_read_rs1, "read_rs1");
    check_poly(&polys.v_read_rs2, &check.v_read_rs2, "read_rs2");
    check_poly(&polys.v_read_ram, &check.v_read_ram, "read_ram");
    check_poly(&polys.v_write_rd, &check.v_write_rd, "write_rd");
    check_poly(&polys.v_write_ram, &check.v_write_ram, "write_ram");

    assert_eq!(polys.a_ram, check.a_ram);
    check_poly(&polys.t_read_rd, &check.t_read_rd, "t_read_rd");
    check_poly(&polys.t_read_rs1, &check.t_read_rs1, "t_read_rs1");
    check_poly(&polys.t_read_rs2, &check.t_read_rs2, "t_read_rs2");
    check_poly(&polys.t_read_ram, &check.t_read_ram, "t_read_ram");
    check_poly(&polys.t_final, &check.t_final, "t_final");
}

fn check_program_io<F: JoltField>(polys: Vec<Rep3ProgramIO<F>>, program_io: &JoltDevice) {
    let v_io: Vec<F> = Rep3MultilinearPolynomial::combine_shares(vec![
        polys[0].v_io.clone(),
        polys[1].v_io.clone(),
        polys[2].v_io.clone(),
    ])
    .coeffs_as_field_elements();

    let memory_size = v_io.len();

    let mut v_io_check: Vec<_> = vec![F::zero(); memory_size];
    let mut input_index = memory_address_to_witness_index(
        program_io.memory_layout.input_start,
        &program_io.memory_layout,
    );
    // Convert input bytes into words and populate `v_io`
    for chunk in program_io.inputs.chunks(4) {
        let mut word = [0u8; 4];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        let word = F::from_u32(u32::from_le_bytes(word));
        v_io_check[input_index] = word;
        input_index += 1;
    }
    let mut output_index = memory_address_to_witness_index(
        program_io.memory_layout.output_start,
        &program_io.memory_layout,
    );
    // Convert output bytes into words and populate `v_io`
    for chunk in program_io.outputs.chunks(4) {
        let mut word = [0u8; 4];
        for (i, byte) in chunk.iter().enumerate() {
            word[i] = *byte;
        }
        let word = u32::from_le_bytes(word);
        v_io_check[output_index] = F::from_u32(word);
        output_index += 1;
    }

    // Copy panic bit
    v_io_check[memory_address_to_witness_index(
        program_io.memory_layout.panic,
        &program_io.memory_layout,
    )] = F::from_u32(program_io.panic as u32);
    if !program_io.panic {
        // Set termination bit
        v_io_check[memory_address_to_witness_index(
            program_io.memory_layout.termination,
            &program_io.memory_layout,
        )] = F::one();
    }

    assert_eq!(v_io, v_io_check);
}

fn check_bytecode<F: JoltField>(polys: &BytecodePolynomials<F>, check: &BytecodePolynomials<F>) {
    check_poly(&polys.a_read_write, &check.a_read_write, "a_read_write");
    check_polys(&polys.v_read_write, &check.v_read_write, "v_read_write");
    check_poly(&polys.t_read, &check.t_read, "t_read");
    check_poly(&polys.t_final, &check.t_final, "t_final");
}

fn check_r1cs<F: JoltField>(polys: &R1CSPolynomials<F>, check: &R1CSPolynomials<F>) {
    check_polys(&polys.chunks_x, &check.chunks_x, "chunks_x");
    check_polys(&polys.chunks_y, &check.chunks_y, "chunks_y");
    check_polys(&polys.circuit_flags, &check.circuit_flags, "circuit_flags");

    check_poly(
        &polys.aux.left_lookup_operand,
        &check.aux.left_lookup_operand,
        "left_lookup_operand",
    );
    check_poly(
        &polys.aux.right_lookup_operand,
        &check.aux.right_lookup_operand,
        "right_lookup_operand",
    );
    check_poly(&polys.aux.product, &check.aux.product, "product");
    check_polys(
        &polys.aux.relevant_y_chunks,
        &check.aux.relevant_y_chunks,
        "relevant_y_chunks",
    );
    check_poly(
        &polys.aux.write_lookup_output_to_rd,
        &check.aux.write_lookup_output_to_rd,
        "write_lookup_output_to_rd",
    );
    check_poly(
        &polys.aux.write_pc_to_rd,
        &check.aux.write_pc_to_rd,
        "write_pc_to_rd",
    );
    check_poly(
        &polys.aux.next_pc_jump,
        &check.aux.next_pc_jump,
        "next_pc_jump",
    );
    check_poly(
        &polys.aux.should_branch,
        &check.aux.should_branch,
        "should_branch",
    );
    check_poly(&polys.aux.next_pc, &check.aux.next_pc, "next_pc");
}

fn check_polys<F: JoltField>(
    polys: &[MultilinearPolynomial<F>],
    check: &[MultilinearPolynomial<F>],
    label: &str,
) {
    assert_eq!(polys.len(), check.len(), "len mismatch {}", label);
    for (i, (poly, check)) in izip!(polys, check).enumerate() {
        check_poly(poly, check, &(label.to_owned() + &format!("_{}", i)));
    }
}

fn check_poly<F: JoltField>(
    poly: &MultilinearPolynomial<F>,
    check: &MultilinearPolynomial<F>,
    label: &str,
) {
    assert_eq!(poly.len(), check.len(), "len mismatch {}", label);
    let poly = poly.coeffs_as_field_elements();
    let check = check.coeffs_as_field_elements();
    let p = izip!(&poly, &check).position(|(i, check)| *i != *check);
    if let Some(pos) = p {
        panic!(
            "{label} mismatch at position {} {:?} != {:?}",
            pos,
            &poly[pos..pos + 5],
            &check[pos..pos + 5]
        );
    }
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
