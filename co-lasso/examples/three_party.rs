use std::{iter, path::PathBuf};

use ark_ff::{PrimeField, Zero};
use clap::Parser;
use co_lasso::{subtables::range_check::RangeLookup, LassoPreprocessing, Rep3LassoWitnessSolver};
use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use futures::{SinkExt, StreamExt};
use itertools::Itertools;
use jolt_core::{poly::field::JoltField, utils::transcript::ProofTranscript};
use mpc_core::protocols::rep3::{network::Rep3MpcNet, PartyID, Rep3BigUintShare};
use mpc_net::{
    config::{NetworkConfig, NetworkConfigFile},
    rep3::quic::{Rep3QuicMpcNetWorker, Rep3QuicNetCoordinator},
    MpcNetworkHandler,
};
use num_bigint::BigUint;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

#[derive(Parser)]
struct Args {
    /// The config file path
    #[clap(short, long, value_name = "FILE")]
    config_file: PathBuf,

    #[clap(short, long, value_name = "NUM_INPUTS")]
    log_num_inputs: usize,
}

const LIMB_BITS: usize = 8;
const C: usize = 1;
const M: usize = 1 << LIMB_BITS;
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
        run_coordinator(config, 1, 1)?;
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

    let rep3_net =
        Rep3QuicMpcNetWorker::new_with_coordinator(config, log_num_workers_per_party, log_num_pub_workers).unwrap();

    let preprocessing = LassoPreprocessing::preprocess::<C, M>([RangeLookup::<F>::new_boxed(256)]);

    let mut transcript = ProofTranscript::new(b"Memory checking");
    let inputs = iter::repeat_with(|| F::from(rand::random::<u64>() % 256))
        .take(num_inputs)
        .collect_vec();
    let inputs_shares = promote_to_trivial_shares(&inputs, my_id.try_into().unwrap());
    let mut witness_solver = Rep3LassoWitnessSolver::new(rep3_net).unwrap();
    let polynomials = witness_solver.polynomialize(
        &preprocessing,
        &inputs_shares,
        &iter::repeat(RangeLookup::<F>::id_for(256))
            .take(num_inputs)
            .collect_vec(),
        M,
        C,
    );

    witness_solver.io_context0.network.log_connection_stats();
    drop(_enter);
    Ok(())
}

#[tracing::instrument(skip_all)]
fn run_coordinator(config: NetworkConfig, log_num_workers_per_party: usize, log_num_pub_workers: usize) -> Result<()> {
    let rep3_net = Rep3QuicNetCoordinator::new(config, log_num_workers_per_party, log_num_pub_workers).unwrap();
    Ok(())
}

fn promote_to_trivial_shares(public_values: &[F], id: PartyID) -> Vec<Rep3BigUintShare<F>> {
    public_values
        .iter()
        .map(|value| promote_from_trivial(value, id))
        .collect()
}

/// Promotes a public field element to a replicated share by setting the additive share of the party with id=0 and leaving all other shares to be 0. Thus, the replicated shares of party 0 and party 1 are set.
pub fn promote_from_trivial<F: PrimeField>(val: &F, id: PartyID) -> Rep3BigUintShare<F> {
    let val: BigUint = val.clone().into();
    match id {
        PartyID::ID0 => Rep3BigUintShare::new(val, BigUint::zero()),
        PartyID::ID1 => Rep3BigUintShare::new(BigUint::zero(), val),
        PartyID::ID2 => Rep3BigUintShare::zero_share(),
    }
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
