mod setup;
mod work;

use std::path::PathBuf;

use ark_bn254::Bn254;
use clap::{Parser, Subcommand};
use mimalloc::MiMalloc;
use setup::setup;
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};
use work::work;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Setup {
        #[clap(long, value_name = "DIR")]
        r1cs_noir_scheme_path: PathBuf,

        #[clap(long, value_name = "NUM")]
        log_num_workers_per_party: usize,

        #[clap(long, value_name = "NUM")]
        log_num_public_workers: Option<usize>,

        #[clap(long, value_name = "DIR", default_value = "./artifacts")]
        artifacts_dir: PathBuf,
    },

    Work {
        #[clap(long, value_name = "DIR")]
        r1cs_noir_scheme_path: PathBuf,

        #[clap(long, value_name = "DIR")]
        r1cs_input_path: PathBuf,

        /// The number of workers who will do the committing and proving. Each worker has 1 core.
        #[clap(long, value_name = "NUM")]
        log_num_workers_per_party: usize,

        #[clap(long, value_name = "NUM")]
        log_num_public_workers: Option<usize>,

        #[clap(long, value_name = "NUM")]
        worker_id: Option<usize>,

        #[clap(long, value_name = "NUM", default_value = "true")]
        local: bool,

        #[clap(long, value_name = "DIR", default_value = "./artifacts")]
        artifacts_dir: PathBuf,
    },
}

fn main() {
    init_tracing();
    let args = Args::parse();

    match args.command {
        Command::Setup {
            r1cs_noir_scheme_path,
            log_num_workers_per_party,
            log_num_public_workers,
            artifacts_dir,
        } => setup::<Bn254>(
            artifacts_dir,
            r1cs_noir_scheme_path,
            log_num_workers_per_party,
            log_num_public_workers,
        ),
        Command::Work {
            r1cs_noir_scheme_path,
            r1cs_input_path,
            artifacts_dir,
            log_num_workers_per_party,
            log_num_public_workers,
            worker_id,
            local,
        } => {
            work::<Bn254>(
                artifacts_dir,
                r1cs_noir_scheme_path,
                r1cs_input_path,
                log_num_workers_per_party,
                log_num_public_workers,
                local,
                worker_id,
            );
        }
    }
}

#[cfg(feature = "parallel")]
pub use rayon::current_num_threads;

#[cfg(not(feature = "parallel"))]
pub fn current_num_threads() -> usize {
    1
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
