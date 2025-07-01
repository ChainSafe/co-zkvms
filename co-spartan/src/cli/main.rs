#![allow(warnings)]
use ark_std::{cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use clap::{Parser, Subcommand};
use co_spartan_noir::DistributedRootKey;
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::Process,
    Count,
};
use mpi::{request, traits::*};
use noir_r1cs::{FieldElement, NoirProofScheme};
// use mpi_snark::worker::WorkerState;

use std::{
    fs::File,
    io::{Read, Write},
    num::NonZeroUsize,
    path::PathBuf,
};

use mimalloc::MiMalloc;

use crossbeam::thread;
use itertools::{merge, Itertools};
use rayon::prelude::*;

use ark_bn254::Config;
use ark_bn254::{Bn254, Fr};
use ark_ff::UniformRand;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use core::num;
use std::mem;
use std::{collections::HashMap, rc::Rc};

// use ark_ec::bn::Bls12;
use ark_ec::pairing::Pairing;
use ark_ff::{Field, One, Zero};
use ark_linear_sumcheck::ml_sumcheck::protocol::verifier::SubClaim;
use ark_linear_sumcheck::ml_sumcheck::protocol::IPForMLSumcheck;
use ark_linear_sumcheck::ml_sumcheck::protocol::PolynomialInfo;
use ark_linear_sumcheck::ml_sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};

use ark_poly_commit::multilinear_pc::{
    data_structures::Commitment, data_structures::CommitterKey, data_structures::VerifierKey,
    MultilinearPC,
};
use ark_std::fs;
use dfs::utils::generate_dumb_sponge;
use dfs::{logup::append_sumcheck_polys, mpc_snark::coordinator};
use dfs::{logup::LogLookupProof, SparseMatEntry, SparseMatPolynomial};
use dfs::{mpc::rss::RssPoly, mpi_snark::worker::PublicProver};
use dfs::{mpc::utils::generate_rss_share_randomness, mpc_snark::worker::DistributedRSSProverKey};
use dfs::{mpi_snark::coordinator::mpi_batch_open_poly_coordinator, utils::produce_test_r1cs};
use dfs::{mpi_utils::send_responses, utils::split_ipk};
use dfs::{snark::indexer::Indexer, R1CSInstance};
use dfs::{snark::zk::SRS, utils::split_r1cs};

use dfs::utils::{
    aggregate_comm, aggregate_eval, aggregate_poly, boost_degree, combine_comm, distributed_open,
    generate_eq, map_poly, merge_list_of_distributed_poly, merge_proof, normalized_multiplicities,
    partial_generate_eq, split_ck, split_poly, split_vec, two_pow_n,
};
use dfs::{end_timer_buf, start_timer_buf};
use dfs::{VerificationError, VerificationResult};
use rand::RngCore;

use dfs::mpc::rss::SSRandom;
use dfs::mpc::utils::generate_poly_shares_rss;
use dfs::mpc_snark::coordinator::PrivateProver;
use dfs::mpi_snark::coordinator::{mpi_poly_commit_coordinator, mpi_sumcheck_coordinator};
use dfs::mpi_snark::worker::{mpi_batch_open_poly_worker, mpi_sumcheck_worker};
use dfs::mpi_utils::{gather_responses, receive_requests, scatter_requests};
use dfs::snark::indexer::IndexProverKey;
use dfs::snark::indexer::IndexVerifierKey;
use dfs::snark::prover::ProverMessage;
use dfs::snark::verifier::DFSVerifier;
use dfs::snark::verifier::VerifierState;
use dfs::snark::{batch_open_poly, batch_verify_poly, BatchOracleEval, OracleEval};
use dfs::transcript::{get_scalar_challenge, get_vector_challenge};

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
        log_num_public_workers: usize,

        #[clap(long, value_name = "DIR")]
        key_out: PathBuf,
    },

    Work {
        /// Path to the coordinator key package
        #[clap(long, value_name = "DIR")]
        key_file: PathBuf,

        /// The number of workers who will do the committing and proving. Each worker has 1 core.
        #[clap(long, value_name = "NUM")]
        log_num_workers_per_party: usize,

        #[clap(long, value_name = "NUM")]
        log_num_public_workers: usize,

        #[clap(long, value_name = "NUM")]
        worker_id: usize,

        #[clap(long, value_name = "NUM")]
        local: usize,
    },
}

fn main() {
    println!(
        "Rayon num threads: {}",
        co_spartan_noir::current_num_threads()
    );

    let args = Args::parse();

    match args.command {
        Command::Setup {
            // log_instance_size,
            r1cs_noir_instance_path,
            r1cs_input_path,
            log_num_workers_per_party,
            log_num_public_workers,
            key_out,
        } => co_spartan_noir::setup(
            key_out,
            r1cs_noir_instance_path,
            r1cs_input_path,
            log_num_workers_per_party,
            log_num_public_workers,
        ),
        Command::Work {
            key_file,
            log_num_workers_per_party,
            log_num_public_workers,
            worker_id,
            local,
        } => {
            co_spartan_noir::work::<Bn254>(
                log_num_workers_per_party,
                key_file,
                worker_id,
                log_num_public_workers,
                local,
            );
        }
    }
}
