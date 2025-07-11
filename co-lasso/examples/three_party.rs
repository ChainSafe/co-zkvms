use ark_std::test_rng;
use clap::Parser;
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use itertools::{chain, Itertools};
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

use co_lasso::{
    instructions::{self, range_check::RangeLookup, xor::XORInstruction},
    lasso, memory_checking, subtables, Rep3LassoWitnessSolver,
};

type Lookups = instructions::TestLookups<F>;
type Subtables = subtables::TestSubtables<F>;
type TestLassoProver = lasso::LassoProver<C, M, F, Lookups, Subtables>;
type TestLassoWitnessSolver<Network> = Rep3LassoWitnessSolver<C, M, F, Lookups, Subtables, Network>;

#[derive(Parser)]
struct Args {
    /// The config file path
    #[clap(short, long, value_name = "FILE")]
    config_file: PathBuf,

    #[clap(short, long, value_name = "NUM_INPUTS")]
    log_num_inputs: usize,
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
        toml::from_str(&std::fs::read_to_string(args.config_file).context("opening config file")?)
            .context("parsing config file")?;
    let config = NetworkConfig::try_from(config).context("converting network config")?;
    if config.is_coordinator {
        run_coordinator(config, 1, 1, num_inputs)?;
    } else {
        run_party(config, num_inputs, 1, 1)?;
    }

    Ok(())
}

fn run_party(
    config: NetworkConfig,
    num_inputs: usize,
    log_num_workers_per_party: usize,
    log_num_pub_workers: usize,
) -> Result<()> {
    let my_id = config.my_id;

    let span = tracing::info_span!("run_party", id = my_id);
    let _enter = span.enter();

    let rep3_net = Rep3QuicMpcNetWorker::new_with_coordinator(
        config,
        log_num_workers_per_party,
        log_num_pub_workers,
    )
    .unwrap();

    let preprocessing = lasso::LassoPreprocessing::preprocess::<C, M, Lookups, Subtables>();

    let party_id = rep3_net.party_id();

    let mut rng = test_rng();
    // let lookups = iter::repeat_with(|| {
    //     let a = F::from(rng.gen_range(0..256));
    //     let b = F::from(rng.gen_range(0..256));
    //     Lookups::XOR(XORInstruction::shared(
    //         rep3::binary::promote_to_trivial_share(party_id, &a.into()),
    //         rep3::binary::promote_to_trivial_share(party_id, &b.into()),
    //     ))
    // })
    // .take(num_inputs)
    // .collect_vec();
    let lookups = chain!(
        iter::repeat_with(|| {
            let value = F::from(rng.gen_range(0..256));
            Lookups::Range256(RangeLookup::shared(
                rep3::arithmetic::promote_to_trivial_share(party_id, value),
                rep3::binary::promote_to_trivial_share(party_id, &value.into()),
            ))
        })
        .take(num_inputs / 2)
        .collect_vec(),
        iter::repeat_with(|| {
            let value = F::from(rng.gen_range(0..320));
            Lookups::Range320(RangeLookup::shared(
                rep3::arithmetic::promote_to_trivial_share(party_id, value),
                rep3::binary::promote_to_trivial_share(party_id, &value.into()),
            ))
        })
        .take(num_inputs / 2)
        .collect_vec(),
    )
    .collect_vec();
    let mut witness_solver = TestLassoWitnessSolver::<Rep3QuicMpcNetWorker>::new(rep3_net).unwrap();
    let polynomials = witness_solver.polynomialize(&preprocessing, &lookups)?;

    let mut rep3_net = witness_solver.io_ctx0.network;

    rep3_net.send_response(polynomials.clone())?;

    let mut prover =
        memory_checking::worker::Rep3MemoryCheckingProver::<C, M, F, Lookups, Subtables, _>::new(
            rep3_net,
        )?;
    prover.prove(&preprocessing, &polynomials)?;

    prover.io_ctx.network.log_connection_stats();
    drop(_enter);
    Ok(())
}

#[tracing::instrument(skip_all)]
fn run_coordinator(
    config: NetworkConfig,
    log_num_workers_per_party: usize,
    log_num_pub_workers: usize,
    num_inputs: usize,
) -> Result<()> {
    // init_tracing();
    let mut rep3_net =
        Rep3QuicNetCoordinator::new(config, log_num_workers_per_party, log_num_pub_workers)
            .unwrap();
    let preprocessing = lasso::LassoPreprocessing::preprocess::<C, M, Lookups, Subtables>();
    let mut transcript = ProofTranscript::new(b"Memory checking");

    let polynomials_shares = rep3_net.receive_responses(Default::default())?;
    if std::env::var("SANITY_CHECK").is_ok() {
        let polynomials =
            TestLassoWitnessSolver::<Rep3QuicMpcNetWorker>::combine_polynomials(polynomials_shares);

        let polynomials_check = {
            let mut rng = test_rng();

            // let lookups = chain!(iter::repeat_with(|| {
            //     let a = rng.gen_range(0..256);
            //     let b = rng.gen_range(0..256);
            //     Lookups::XOR(XORInstruction::public(a, b))
            // })
            // .take(num_inputs)
            // .collect_vec())
            // .collect_vec();
            let lookups = chain!(
                iter::repeat_with(|| Lookups::Range256(RangeLookup::public(F::from(
                    rng.gen_range(0..256)
                ))))
                .take(num_inputs / 2)
                .collect_vec(),
                iter::repeat_with(|| Lookups::Range320(RangeLookup::public(F::from(
                    rng.gen_range(0..320)
                ))))
                .take(num_inputs / 2)
                .collect_vec(),
            )
            .collect_vec();

            lasso::LassoProver::<C, M, F, Lookups, Subtables>::polynomialize(
                &preprocessing,
                &lookups,
            )
        };
        assert_eq!(polynomials.dims, polynomials_check.dims);
        assert_eq!(polynomials.read_cts, polynomials_check.read_cts);
        assert_eq!(polynomials.final_cts, polynomials_check.final_cts);
        assert_eq!(polynomials.e_polys, polynomials_check.e_polys);
        assert_eq!(
            polynomials.lookup_flag_polys,
            polynomials_check.lookup_flag_polys
        );
        assert_eq!(
            polynomials.lookup_flag_bitvectors,
            polynomials_check.lookup_flag_bitvectors
        );
        assert_eq!(polynomials.lookup_outputs, polynomials_check.lookup_outputs);

        {
            let mut transcript = ProofTranscript::new(b"Memory checking");
            let proof = TestLassoProver::prove(&preprocessing, &polynomials_check, &mut transcript);

            let mut verifier_transcript = ProofTranscript::new(b"Memory checking");
            TestLassoProver::verify(&preprocessing, proof, &mut verifier_transcript)?;
        }
    }

    let proof =
        memory_checking::coordinator::Rep3MemoryCheckingProver::<C, M, F, Subtables, _>::prove(
            &preprocessing,
            &mut rep3_net,
            &mut transcript,
        )?;

    let mut verifier_transcript = ProofTranscript::new(b"Memory checking");
    TestLassoProver::verify(&preprocessing, proof, &mut verifier_transcript)?;

    Ok(())
}

// /// Promotes a public field element to a replicated share by setting the additive share of the party with id=0 and leaving all other shares to be 0. Thus, the replicated shares of party 0 and party 1 are set.
// pub fn promote_from_trivial<F: PrimeField>(val: &F, id: PartyID) -> Rep3BigUintShare<F> {
//     let val: BigUint = val.clone().into();
//     match id {
//         PartyID::ID0 => Rep3BigUintShare::new(val, BigUint::zero()),
//         PartyID::ID1 => Rep3BigUintShare::new(BigUint::zero(), val),
//         PartyID::ID2 => Rep3BigUintShare::zero_share(),
//     }
// }

fn init_tracing() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing::Level::INFO.into())
        .from_env_lossy();

    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default());

    let _ = tracing::subscriber::set_global_default(subscriber);
}
