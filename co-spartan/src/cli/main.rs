mod setup;
mod work;

use ark_bn254::Bn254;
use clap::{Parser, Subcommand};
use mimalloc::MiMalloc;
use setup::setup;
use std::path::PathBuf;
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
        r1cs_noir_instance_path: PathBuf,

        #[clap(long, value_name = "DIR")]
        r1cs_input_path: PathBuf,

        // #[clap(long, value_name = "NUM")]
        // log_instance_size: usize,
        #[clap(long, value_name = "NUM")]
        log_num_workers_per_party: usize,

        #[clap(long, value_name = "NUM")]
        log_num_public_workers: Option<usize>,

        #[clap(long, value_name = "DIR", default_value = "./artifacts")]
        artifacts_dir: PathBuf,
    },

    Work {
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
    let args = Args::parse();

    match args.command {
        Command::Setup {
            // log_instance_size,
            r1cs_noir_instance_path,
            r1cs_input_path,
            log_num_workers_per_party,
            log_num_public_workers,
            artifacts_dir: key_out,
        } => setup::<Bn254>(
            key_out,
            r1cs_noir_instance_path,
            r1cs_input_path,
            log_num_workers_per_party,
            log_num_public_workers,
        ),
        Command::Work {
            artifacts_dir,
            log_num_workers_per_party,
            log_num_public_workers,
            worker_id,
            local,
        } => {
            work::<Bn254>(
                artifacts_dir,
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
