#![allow(warnings)]
use ark_std::{cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use clap::{Parser, Subcommand};
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::Process,
    Count,
};
use mpi::{request, traits::*};
use noir_r1cs::NoirProofScheme;
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

use ark_ec::bls12::Bls12;
use ark_ec::pairing::Pairing;
use ark_ff::{Field, One, Zero};
use ark_linear_sumcheck::ml_sumcheck::protocol::verifier::SubClaim;
use ark_linear_sumcheck::ml_sumcheck::protocol::IPForMLSumcheck;
use ark_linear_sumcheck::ml_sumcheck::protocol::PolynomialInfo;
use ark_linear_sumcheck::ml_sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_poly_commit::challenge::ChallengeGenerator;

use ark_poly_commit::multilinear_pc::{
    data_structures::Commitment, data_structures::CommitterKey, data_structures::VerifierKey,
    MultilinearPC,
};
use dfs::mpi_snark::worker::PublicProver;
use dfs::subprotocols::loglookup::sumcheck_polynomial_list;
use dfs::subprotocols::loglookup::LogLookupProof;
use dfs::utils::generate_dumb_sponge;
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

use dfs::mpi_snark::coordinator::{mpi_poly_commit_coordinator, mpi_sumcheck_coordinator};
use dfs::mpi_snark::worker::DistributedProverKey;
use dfs::mpi_snark::worker::{mpi_batch_open_poly_worker, mpi_sumcheck_worker};
use dfs::mpi_utils::{gather_responses, receive_requests, scatter_requests};
use dfs::snark::indexer::IndexProverKey;
use dfs::snark::indexer::IndexVerifierKey;
use dfs::snark::prover::ProverMessage;
use dfs::snark::verifier::DFSVerifier;
use dfs::snark::verifier::VerifierState;
use dfs::snark::R1CSProof;
use dfs::snark::{batch_open_poly, batch_verify_poly, BatchOracleEval, OracleEval};
use dfs::transcript::{get_scalar_challenge, get_vector_challenge};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistributedRootKey<E: Pairing> {
    pub ipk: IndexProverKey<E>,
    pub ivk: IndexVerifierKey<E>,
}

// #[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
// pub struct Round1Msg<E: Pairing> {
//     pub comm_0: Commitment<E>,
//     pub comm_1: Commitment<E>,
// }

#[derive(Parser)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Work {
        #[clap(long, value_name = "DIR")]
        r1cs_noir_instance_path: PathBuf,

        #[clap(long, value_name = "DIR")]
        r1cs_input_path: PathBuf,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Work {
            r1cs_noir_instance_path,
            r1cs_input_path,
        } => {
            work(r1cs_noir_instance_path, r1cs_input_path);
        }
    }
}

fn work(r1cs_noir_instance_path: PathBuf, r1cs_input_path: PathBuf) {
    type E = Bn254;

    let mut proof_scheme: NoirProofScheme = noir_r1cs::read(&r1cs_noir_instance_path).unwrap();
    let mut z = proof_scheme.solve_witness(&r1cs_input_path).unwrap();
    println!("z: {:?}", z.len());
    let r1cs_noir = proof_scheme.r1cs;
    let log_instance_size = r1cs_noir.log2_instance_size();
    println!("log_instance_size: {:?}", log_instance_size);
    let num_repeatition = 1;
    let num_vars = log_instance_size;
    let num_cons = 1 << num_vars;

    r1cs_noir.verify_witness(&z).unwrap();

    let mut rng = StdRng::seed_from_u64(12);
    let mut za = r1cs_noir.a() * &z;
    let mut zb = r1cs_noir.b() * &z;
    let mut zc = r1cs_noir.c() * &z;
    let r1cs = co_spartan_noir::from_noir_r1cs(&r1cs_noir);

    co_spartan_noir::utils::pad_to_power_of_two(&mut z, log_instance_size);
    co_spartan_noir::utils::pad_to_power_of_two(&mut za, log_instance_size);
    co_spartan_noir::utils::pad_to_power_of_two(&mut zb, log_instance_size);
    co_spartan_noir::utils::pad_to_power_of_two(&mut zc, log_instance_size);

    // let log_instance_size = 14;
    // let mut rng = StdRng::seed_from_u64(12);
    // let (r1cs, z, za, zb, zc) = produce_test_r1cs(14, &mut rng);

   
    // R1CSInstance::<$bench_field>::produce_synthetic_r1cs(num_cons, num_z, num_wit, rng);
    // let num_vars = 16 + 3;
    let num_vars = log_instance_size;
    println!("num_vars: {:?}", num_vars);
    let srs = SRS::<E, _>::generate_srs(num_vars + 2, 4, &mut rng);
    let (pk, vk) = Indexer::index_for_prover_and_verifier(&r1cs, &srs);

    let start = ark_std::time::Instant::now();

    let _ = R1CSProof::new(&r1cs, &pk, &vk, &z, &Vec::new());

    let prove_final_time = start.elapsed();

    let proof = R1CSProof::new(&r1cs, &pk, &vk, &z, &Vec::new());

    let start = ark_std::time::Instant::now();

    // println!("proof.second_sumcheck_msgs.sumcheck_proof[0]: {:?}", proof.second_sumcheck_msgs.sumcheck_proof[0]);
    let _ = proof.verify(&vk, &Vec::new());

    let verify_final_time = start.elapsed();

    println!(
        "proving time for 2^{:?} num_cons: {:?}",
        num_vars,
        prove_final_time
    );

    println!(
        "verification time for 2^{:?} num_cons: {:?}",
        num_vars,
        verify_final_time
    );
}
