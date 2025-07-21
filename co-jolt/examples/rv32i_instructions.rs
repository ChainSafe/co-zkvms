mod three_party;

use ark_std::test_rng;
use clap::Parser;
use co_jolt::{
    host,
    jolt::{
        instruction::JoltInstructionSet,
        vm::{
            coordinator::JoltRep3,
            rv32i_vm::{RV32IJoltVM, RV32I},
            witness::{Rep3JoltPolynomials, Rep3Polynomials},
            worker::JoltRep3Prover,
            Jolt, JoltTraceStep,
        },
    },
    poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    utils::transcript::{KeccakTranscript, Transcript},
};
use co_lasso::{memory_checking::StructuredPolynomialData, subprotocols::commitment::PST13};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use itertools::Itertools;
use jolt_core::{field::JoltField, jolt::vm::JoltProverPreprocessing};
use mpc_core::protocols::rep3::{self, network::Rep3Network};
use mpc_net::{
    config::{NetworkConfig, NetworkConfigFile},
    mpc_star::MpcStarNetWorker,
};
use mpc_net::{
    mpc_star::MpcStarNetCoordinator,
    rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator},
};

use crate::three_party::{init_tracing, Args};

type Instructions = co_jolt::jolt::vm::rv32i_vm::RV32I<F>;
type Subtables = co_jolt::jolt::vm::rv32i_vm::RV32ISubtables<F>;

const C: usize = co_jolt::jolt::vm::rv32i_vm::C;
const M: usize = co_jolt::jolt::vm::rv32i_vm::M;
type F = ark_bn254::Fr;
type E = ark_bn254::Bn254;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
        init_tracing();
    }

    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&9u32).unwrap();
    let (_, instruction_trace) = program.trace::<F>(&inputs);

    if config.is_coordinator {
        print_used_instructions(&instruction_trace);
    }

    let trace = instruction_trace
        .into_iter()
        .map(|op| JoltTraceStep::<F, Instructions>::from_instruction_lookup(op))
        .collect_vec();

    if config.is_coordinator {
        run_coordinator(args, config, trace, 1, 1)?;
    } else {
        run_party(args, config, trace, 1, 1)?;
    }

    Ok(())
}

pub fn run_party(
    args: Args,
    config: NetworkConfig,
    trace: Vec<JoltTraceStep<F, Instructions>>,
    log_num_workers_per_party: usize,
    log_num_pub_workers: usize,
) -> Result<()> {
    let my_id = config.my_id;

    if args.trace_parties && my_id == 0 {
        init_tracing();
    }

    let span = tracing::info_span!("run_party", id = my_id);
    let _enter = span.enter();

    // if args.debug {
    //     return Ok(());
    // }

    let network = Rep3QuicMpcNetWorker::new_with_coordinator(
        config,
        log_num_workers_per_party,
        log_num_pub_workers,
    )
    .unwrap();

    let preprocessing = RV32IJoltVM::prover_preprocess(trace.len());

    let mut prover = JoltRep3Prover::<F, C, M, Instructions, Subtables, _>::new(network)?;

    prover.prove::<PST13<E>, KeccakTranscript>(None, preprocessing)?;

    // prover.io_ctx.network.log_connection_stats();
    drop(_enter);
    Ok(())
}

#[tracing::instrument(skip_all)]
pub fn run_coordinator(
    args: Args,
    config: NetworkConfig,
    trace: Vec<JoltTraceStep<F, Instructions>>,
    log_num_workers_per_party: usize,
    log_num_pub_workers: usize,
) -> Result<()> {
    let num_inputs = trace.len();
    if args.solve_witness {
        tracing::info!("Witness solving enabled");
    } else {
        tracing::warn!("Witness solving disabled");
    }

    let preprocessing: JoltProverPreprocessing<C, F, PST13<E>, KeccakTranscript> =
        RV32IJoltVM::prover_preprocess(1 << num_inputs.next_power_of_two());

    let mut rep3_net =
        Rep3QuicNetCoordinator::new(config, log_num_workers_per_party, log_num_pub_workers)
            .unwrap();

    let (proof, commitments) =
        RV32IJoltVM::prove_rep3(Some(trace.clone()), &preprocessing, &mut rep3_net)?;

    if args.debug {
        let (proof_check, commitments_check) =
            RV32IJoltVM::prove(trace.clone(), preprocessing.clone());

        for (i, (commitment, commitment_check)) in commitments
            .instruction_lookups
            .dim
            .iter()
            .zip(commitments_check.instruction_lookups.dim.iter())
            .enumerate()
        {
            assert_eq!(commitment, commitment_check, "at index {}", i);
        }

        for (i, (commitment, commitment_check)) in commitments
            .instruction_lookups
            .read_cts
            .iter()
            .zip(commitments_check.instruction_lookups.read_cts.iter())
            .enumerate()
        {
            assert_eq!(commitment, commitment_check, "at index {}", i);
        }

        for (i, (commitment, commitment_check)) in commitments
            .instruction_lookups
            .E_polys
            .iter()
            .zip(commitments_check.instruction_lookups.E_polys.iter())
            .enumerate()
        {
            assert_eq!(commitment, commitment_check, "at index {}", i);
        }

        for (i, (commitment, commitment_check)) in commitments
            .instruction_lookups
            .instruction_flags
            .iter()
            .zip(
                commitments_check
                    .instruction_lookups
                    .instruction_flags
                    .iter(),
            )
            .enumerate()
        {
            assert_eq!(commitment, commitment_check, "at index {}", i);
        }

        for (i, (commitment, commitment_check)) in commitments
            .instruction_lookups
            .final_cts
            .iter()
            .zip(commitments_check.instruction_lookups.final_cts.iter())
            .enumerate()
        {
            assert_eq!(commitment, commitment_check, "at index {}", i);
        }

        assert_eq!(
            commitments.instruction_lookups.lookup_outputs,
            commitments_check.instruction_lookups.lookup_outputs,
            "lookup_outputs"
        );

        RV32IJoltVM::verify(preprocessing.shared.clone(), proof_check, commitments_check)
            .context("while verifying Lasso proof")?;
    }

    RV32IJoltVM::verify(preprocessing.shared, proof, commitments)
        .context("while verifying Lasso (rep3) proof")?;

    Ok(())
}

fn print_used_instructions<F: JoltField>(instruction_trace: &[Option<RV32I<F>>]) {
    let opcodes_used = instruction_trace
        .par_iter()
        .filter_map(|op| match op {
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
