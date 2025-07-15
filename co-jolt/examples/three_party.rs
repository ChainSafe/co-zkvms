use ark_ff::Zero;
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_std::test_rng;
use clap::Parser;
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use itertools::{chain, Itertools};
use jolt_core::poly::structured_poly::StructuredCommitment;
use jolt_core::utils::{math::Math, transcript::ProofTranscript};
use mpc_core::protocols::rep3;
use mpc_net::{
    config::{NetworkConfig, NetworkConfigFile},
    mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker},
    rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator},
};
use num_bigint::BigUint;
use rand::Rng;
use std::{iter, path::PathBuf};
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

use co_jolt::jolt::{
    instruction::{self, range_check::RangeLookup, xor::XORInstruction},
    subtable,
    vm::instruction_lookups::{
        coordinator,
        witness::{Rep3InstructionPolynomials, Rep3LassoWitnessSolver},
        worker::Rep3InstructionLookupsProver,
        InstructionCommitment, InstructionLookupsPreprocessing, InstructionLookupsProof,
    },
};
use co_lasso::subprotocols::commitment::PST13;

type Lookups = instruction::TestLookups<F>;
type Subtables = subtable::TestSubtables<F>;
type TestLassoProof = InstructionLookupsProof<C, M, F, PST13<ark_bn254::Bn254>, Lookups, Subtables>;
type TestLassoWitnessSolver<Network> =
    Rep3LassoWitnessSolver<C, M, F, PST13<ark_bn254::Bn254>, Lookups, Subtables, Network>;

#[derive(Parser)]
struct Args {
    /// The config file path
    #[clap(short, long, value_name = "FILE")]
    config_file: PathBuf,

    #[clap(short, long, value_name = "NUM_INPUTS")]
    log_num_inputs: usize,

    #[arg(short, long, value_name = "SOLVE_WITNESS", env = "SOLVE_WITNESS")]
    solve_witness: bool,

    #[clap(short, long, value_name = "DEBUG", env = "DEBUG")]
    debug: bool,
}

const C: usize = 2; // num chunks
const M: usize = 1 << 8;
type F = ark_bn254::Fr;

// #[tokio::main]
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

    if config.is_coordinator {
        run_coordinator(args, config, 1, 1, num_inputs)?;
    } else {
        run_party(args, config, num_inputs, 1, 1)?;
    }

    Ok(())
}

fn run_party(
    args: Args,
    config: NetworkConfig,
    num_inputs: usize,
    log_num_workers_per_party: usize,
    log_num_pub_workers: usize,
) -> Result<()> {
    let my_id = config.my_id;

    let span = tracing::info_span!("run_party", id = my_id);
    let _enter = span.enter();

    let mut rep3_net = Rep3QuicMpcNetWorker::new_with_coordinator(
        config,
        log_num_workers_per_party,
        log_num_pub_workers,
    )
    .unwrap();

    let preprocessing = InstructionLookupsPreprocessing::preprocess::<C, M, Lookups, Subtables>();

    let setup = {
        let commitment_shapes = TestLassoProof::commitment_shapes(&preprocessing, num_inputs);
        let mut rng = test_rng();
        PST13::setup(&commitment_shapes, &mut rng)
    };

    let party_id = rep3_net.party_id();

    let mut rng = test_rng();

    // let lookups = iter::repeat_with(|| {
    //     let a = F::from(rng.gen_range::<u64, _>(0..256));
    //     let b = F::from(rng.gen_range::<u64, _>(0..256));
    //     Lookups::XOR(XORInstruction::shared_binary(
    //         rep3::binary::promote_to_trivial_share(party_id, &a.into()),
    //         rep3::binary::promote_to_trivial_share(party_id, &b.into()),
    //     ))
    // })
    // .take(num_inputs)
    // .collect_vec();

    let polynomials = if args.solve_witness {
        let lookups = chain!(
            iter::repeat_with(|| {
                let value = F::from(rng.gen_range(0..256));
                Lookups::Range256(RangeLookup::shared(
                    rep3::arithmetic::promote_to_trivial_share(party_id, value),
                ))
            })
            .take(num_inputs / 2)
            .collect_vec(),
            iter::repeat_with(|| {
                let value = F::from(rng.gen_range(0..320));
                Lookups::Range320(RangeLookup::shared(
                    rep3::arithmetic::promote_to_trivial_share(party_id, value),
                ))
            })
            .take(num_inputs / 2)
            .collect_vec(),
        )
        .collect_vec();
        let mut witness_solver =
            TestLassoWitnessSolver::<Rep3QuicMpcNetWorker>::new(rep3_net).unwrap();
        let polynomials = witness_solver.polynomialize(&preprocessing, lookups)?;

        rep3_net = witness_solver.io_ctx0.network;
        polynomials
    } else {
        tracing::info!("Receiving witness share");
        rep3_net.receive_request()?
    };

    polynomials.commit::<PST13<ark_bn254::Bn254>>(&setup, &mut rep3_net)?;

    if args.debug && args.solve_witness {
        rep3_net.send_response(polynomials.clone())?;
    }

    let mut prover = Rep3InstructionLookupsProver::<
        C,
        M,
        F,
        PST13<ark_bn254::Bn254>,
        Lookups,
        Subtables,
        _,
    >::new(rep3_net, setup)?;

    prover.prove(&preprocessing, &polynomials)?;

    prover.io_ctx.network.log_connection_stats();
    drop(_enter);
    Ok(())
}

#[tracing::instrument(skip_all)]
fn run_coordinator(
    args: Args,
    config: NetworkConfig,
    log_num_workers_per_party: usize,
    log_num_pub_workers: usize,
    num_inputs: usize,
) -> Result<()> {
    if args.solve_witness {
        tracing::info!("Witness solving enabled");
    } else {
        tracing::warn!("Witness solving disabled");
    }

    // init_tracing();
    let mut rep3_net =
        Rep3QuicNetCoordinator::new(config, log_num_workers_per_party, log_num_pub_workers)
            .unwrap();

    let preprocessing = InstructionLookupsPreprocessing::preprocess::<C, M, Lookups, Subtables>();

    let commitment_shapes = TestLassoProof::commitment_shapes(&preprocessing, num_inputs);
    let setup = {
        let mut rng = test_rng();
        PST13::setup(&commitment_shapes, &mut rng)
    };

    let lookups = {
        let mut rng = test_rng();
        chain!(
            iter::repeat_with(|| Some(Lookups::Range256(RangeLookup::public(F::from(
                rng.gen_range(0..256)
            )))))
            .take(num_inputs / 2)
            .collect_vec(),
            iter::repeat_with(|| Some(Lookups::Range320(RangeLookup::public(F::from(
                rng.gen_range(0..320)
            )))))
            .take(num_inputs / 2)
            .collect_vec(),
        )
        .collect_vec()
    };

    if !args.solve_witness {
        let mut rng = test_rng();
        let polynomials = TestLassoProof::polynomialize(&preprocessing, &lookups);
        let polynomials_shares = polynomials.into_secret_shares_rep3(&mut rng)?;
        rep3_net.send_requests(polynomials_shares.to_vec())?;
    }

    let commitments =
        Rep3InstructionPolynomials::receive_commitments::<PST13<ark_bn254::Bn254>>(&mut rep3_net)?;

    if args.debug {
        let polynomials_check = TestLassoProof::polynomialize(&preprocessing, &lookups);

        if args.solve_witness {
            let polynomials_shares = rep3_net.receive_responses(Default::default())?;
            let polynomials = TestLassoWitnessSolver::<Rep3QuicMpcNetWorker>::combine_polynomials(
                polynomials_shares,
            );

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
        let proof =
            TestLassoProof::prove(&polynomials_check, &preprocessing, &setup, &mut transcript);

        let mut verifier_transcript = ProofTranscript::new(b"Lasso");
        TestLassoProof::verify(
            &preprocessing,
            &setup,
            proof,
            &commitments,
            &mut verifier_transcript,
        )
        .context("while verifying Lasso (check) proof")?;
    }

    let mut transcript: ProofTranscript = ProofTranscript::new(b"Lasso");

    let proof =
        InstructionLookupsProof::<C, M, F, PST13<ark_bn254::Bn254>, Lookups, Subtables>::prove_rep3(
            num_inputs,
            &preprocessing,
            &mut rep3_net,
            &mut transcript,
        )?;

    // assert_eq!(proof.primary_sumcheck.opening_proof.proofs, proof_check.primary_sumcheck.opening_proof.proofs);

    let mut verifier_transcript = ProofTranscript::new(b"Lasso");
    TestLassoProof::verify(
        &preprocessing,
        &setup,
        proof,
        &commitments,
        &mut verifier_transcript,
    )
    .context("while verifying Lasso (rep3) proof")?;

    Ok(())
}

fn init_tracing() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();

    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default());

    let _ = tracing::subscriber::set_global_default(subscriber);
}
