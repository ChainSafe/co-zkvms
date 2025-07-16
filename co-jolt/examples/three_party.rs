use ark_ff::Zero;
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_std::test_rng;
use clap::Parser;
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use itertools::Itertools;
use jolt_core::poly::structured_poly::StructuredCommitment;
use jolt_core::utils::transcript::ProofTranscript;
use mpc_core::protocols::rep3;
use mpc_net::{
    config::{NetworkConfig, NetworkConfigFile},
    mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker},
    rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator},
};
use rand::Rng;
use std::{iter, path::PathBuf};
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

use co_jolt::jolt::{
    instruction::{
        self,
        range_check::RangeLookup,
        xor::{self, XORInstruction},
        JoltInstruction, JoltInstructionSet, Rep3JoltInstructionSet,
    },
    subtable::{self, JoltSubtableSet},
    vm::instruction_lookups::{
        witness::{Rep3InstructionPolynomials, Rep3InstructionWitnessSolver},
        worker::Rep3InstructionLookupsProver,
        InstructionCommitment, InstructionLookupsPreprocessing, InstructionLookupsProof,
    },
};
use co_lasso::subprotocols::commitment::PST13;

type Instructions = instruction::TestInstructions<F>;
type Subtables = subtable::TestInstructionSubtables<F>;

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
}

const C: usize = 2; // num chunks
const M: usize = 1 << 8;
type F = ark_bn254::Fr;
type E = ark_bn254::Bn254;

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let num_inputs = 1 << args.log_num_inputs;
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| eyre!("Could not install default rustls crypto provider"))?;

    let config: NetworkConfigFile =
        toml::from_str(&std::fs::read_to_string(&args.config_file).context("opening config file")?)
            .context("parsing config file")?;
    let config = NetworkConfig::try_from(config).context("converting network config")?;

    let lookups = {
        let mut rng = test_rng();
        iter::repeat_with(|| {
            let a = rng.gen_range::<u64, _>(0..256);
            let b = rng.gen_range::<u64, _>(0..256);
            Some(Instructions::XOR(XORInstruction::<F>::public(a, b)))
        })
        .take(num_inputs)
        .collect_vec()
    };

    // let lookups = chain!(
    //     iter::repeat_with(|| {
    //         let value = F::from(rng.gen_range(0..256));
    //         Lookups::Range256(RangeLookup::shared(
    //             rep3::arithmetic::promote_to_trivial_share(party_id, value),
    //         ))
    //     })
    //     .take(num_inputs / 2)
    //     .collect_vec(),
    //     iter::repeat_with(|| {
    //         let value = F::from(rng.gen_range(0..320));
    //         Lookups::Range320(RangeLookup::shared(
    //             rep3::arithmetic::promote_to_trivial_share(party_id, value),
    //         ))
    //     })
    //     .take(num_inputs / 2)
    //     .collect_vec(),
    // )
    // .collect_vec();

    if config.is_coordinator {
        run_coordinator::<C, M, Instructions, Subtables>(args, config, lookups, 1, 1)?;
    } else {
        run_party::<C, M, Instructions, Subtables>(args, config, lookups, 1, 1)?;
    }

    Ok(())
}

pub fn run_party<
    const C: usize,
    const M: usize,
    Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
>(
    args: Args,
    config: NetworkConfig,
    lookups: Vec<Option<Instructions>>,
    log_num_workers_per_party: usize,
    log_num_pub_workers: usize,
) -> Result<()> {
    type LassoProof<const C: usize, const M: usize, Instructions, Subtables> =
        InstructionLookupsProof<C, M, F, PST13<E>, Instructions, Subtables>;

    type LassoWitnessSolver<
        const C: usize,
        const M: usize,
        Instructions,
        Subtables,
    > = Rep3InstructionWitnessSolver<
        C,
        M,
        F,
        PST13<E>,
        Instructions,
        Subtables,
        Rep3QuicMpcNetWorker,
    >;

    let my_id = config.my_id;

    let span = tracing::info_span!("run_party", id = my_id);
    let _enter = span.enter();

    let mut rep3_net = Rep3QuicMpcNetWorker::new_with_coordinator(
        config,
        log_num_workers_per_party,
        log_num_pub_workers,
    )
    .unwrap();

    let preprocessing =
        InstructionLookupsPreprocessing::preprocess::<C, M, Instructions, Subtables>();

    let setup = {
        let commitment_shapes = LassoProof::<C, M, Instructions, Subtables>::commitment_shapes(
            &preprocessing,
            lookups.len(),
        );
        let mut rng = test_rng();
        PST13::setup(&commitment_shapes, &mut rng)
    };

    let polynomials = if args.solve_witness {
        let mut witness_solver =
            LassoWitnessSolver::<C, M, Instructions, Subtables>::new(rep3_net).unwrap();
        let polynomials = witness_solver.polynomialize(&preprocessing, lookups)?;

        rep3_net = witness_solver.io_ctx0.network;
        polynomials
    } else {
        let polynomials = rep3_net.receive_request()?;
        polynomials
    };

    polynomials.commit::<PST13<E>>(&setup, &mut rep3_net)?;

    if args.debug && args.solve_witness {
        rep3_net.send_response(polynomials.clone())?;
    }

    let mut prover =
        Rep3InstructionLookupsProver::<C, M, F, PST13<E>, Instructions, Subtables, _>::new(
            rep3_net, setup,
        )?;

    prover.prove(&preprocessing, &polynomials)?;

    prover.io_ctx.network.log_connection_stats();
    drop(_enter);
    Ok(())
}

#[tracing::instrument(skip_all)]
pub fn run_coordinator<
    const C: usize,
    const M: usize,
    Instructions: JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
>(
    args: Args,
    config: NetworkConfig,
    lookups: Vec<Option<Instructions>>,
    log_num_workers_per_party: usize,
    log_num_pub_workers: usize,
) -> Result<()> {
    type LassoProof<
        const C: usize,
        const M: usize,
        Instructions,
        Subtables,
    > = InstructionLookupsProof<C, M, F, PST13<E>, Instructions, Subtables>;

    let num_inputs = lookups.len();
    if args.solve_witness {
        tracing::info!("Witness solving enabled");
    } else {
        tracing::warn!("Witness solving disabled");
    }

    // init_tracing();
    let mut rep3_net =
        Rep3QuicNetCoordinator::new(config, log_num_workers_per_party, log_num_pub_workers)
            .unwrap();

    let preprocessing =
        InstructionLookupsPreprocessing::preprocess::<C, M, Instructions, Subtables>();

    let commitment_shapes =
        LassoProof::<C, M, Instructions, Subtables>::commitment_shapes(&preprocessing, num_inputs);
    let setup = {
        let mut rng = test_rng();
        PST13::setup(&commitment_shapes, &mut rng)
    };

    if !args.solve_witness {
        let mut rng = test_rng();
        let polynomials = LassoProof::<C, M, _, Subtables>::polynomialize(&preprocessing, &lookups);
        let polynomials_shares = polynomials.into_secret_shares_rep3(&mut rng)?;
        rep3_net.send_requests(polynomials_shares.to_vec())?;
    }

    let commitments = Rep3InstructionPolynomials::receive_commitments::<PST13<E>>(&mut rep3_net)?;
    if args.debug {
        let polynomials_check =
            LassoProof::<C, M, _, Subtables>::polynomialize(&preprocessing, &lookups);

        if args.solve_witness {
            let polynomials_shares = rep3_net.receive_responses(Default::default())?;
            let polynomials =
                Rep3InstructionPolynomials::combine_polynomials::<PST13<E>>(polynomials_shares);

            assert_eq!(polynomials.dim, polynomials_check.dim);
            assert_eq!(polynomials.read_cts, polynomials_check.read_cts);
            assert_eq!(polynomials.final_cts, polynomials_check.final_cts);
            assert_eq!(polynomials.E_polys, polynomials_check.E_polys);
            assert_eq!(polynomials.lookup_outputs, polynomials_check.lookup_outputs);
        }

        let commitment_check: InstructionCommitment<PST13<ark_bn254::Bn254>> =
            polynomials_check.commit(&setup);

        assert_eq!(
            commitment_check.trace_commitment,
            commitments.trace_commitment
        );
        assert_eq!(
            commitment_check.final_commitment,
            commitments.final_commitment
        );

        let mut transcript = ProofTranscript::new(b"Lasso");
        let proof = LassoProof::<C, M, Instructions, Subtables>::prove(
            &polynomials_check,
            &preprocessing,
            &setup,
            &mut transcript,
        );

        let mut verifier_transcript = ProofTranscript::new(b"Lasso");
        LassoProof::<C, M, Instructions, Subtables>::verify(
            &preprocessing,
            &setup,
            proof,
            &commitments,
            &mut verifier_transcript,
        )
        .context("while verifying Lasso (check) proof")?;
    }

    let mut transcript: ProofTranscript = ProofTranscript::new(b"Lasso");

    let proof = LassoProof::<C, M, Instructions, Subtables>::prove_rep3(
        num_inputs,
        &preprocessing,
        &mut rep3_net,
        &mut transcript,
    )?;

    let mut verifier_transcript = ProofTranscript::new(b"Lasso");
    LassoProof::verify(
        &preprocessing,
        &setup,
        proof,
        &commitments,
        &mut verifier_transcript,
    )
    .context("while verifying Lasso (rep3) proof")?;

    Ok(())
}

pub fn init_tracing() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();

    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default());

    let _ = tracing::subscriber::set_global_default(subscriber);
}
