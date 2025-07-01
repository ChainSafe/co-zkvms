#![allow(warnings)]
use ark_std::{cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter};

use crate::{utils::current_num_threads, DistributedRootKey};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use clap::{Parser, Subcommand};
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
use dfs::{network::{NetworkCoordinator, NetworkWorker, Rep3CoordinatorMPI, Rep3WorkerMPI}, utils::generate_dumb_sponge};
use dfs::{logup::append_sumcheck_polys, mpc_snark::coordinator};
use dfs::{
    logup::LogLookupProof, transcript::TranscriptMerlin, SparseMatEntry, SparseMatPolynomial,
};
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

pub fn work<E: Pairing>(
    log_num_workers_per_party: usize,
    key_file: PathBuf,
    mut worker_id: usize,
    log_num_public_workers: usize,
    local: usize,
) {
    let mut send_size = 0;
    let mut recv_size = 0;

    let (universe, _) = mpi::initialize_with_threading(mpi::Threading::Funneled).unwrap();
    let world = universe.world();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    let rank = world.rank();
    let size = world.size();
    println!("SIZE: {:?}", size);

    let mut log = Vec::new();
    let very_start = start_timer_buf!(log, || format!("Node {rank}: Beginning work"));

    // if rank == root_rank {
    // Deserialize the proving keys

    println!("running mode: {}", local);

    if local == 0 {
        println!("worker id: {}", worker_id);
    } else {
        println!("worker id: {}", rank);
    }

    if (local == 0 && worker_id == 0) || (local == 1 && rank == root_rank) {
        let proving_keys = {
            let mut buf = Vec::new();

            let mut file_name = key_file.clone();
            file_name.set_extension("root");

            let mut f =
                File::open(&file_name).expect(&format!("couldn't open file {:?}", key_file));
            let _ = f.read_to_end(&mut buf);
            let buf_slice = buf.as_slice();

            let pk =
                DistributedRootKey::<E>::deserialize_uncompressed_unchecked(buf_slice).unwrap();
            pk
        };

        let log_instance_size = proving_keys.log_instance_size;

        // Initial proof
        let start = start_timer_buf!(log, || format!("Coord: construct coordinator proof"));

        let mut network = Rep3CoordinatorMPI::new(
            root_process,
            &mut log,
            log_num_workers_per_party,
            log_num_public_workers,
            size,
        );

        let responses_chunked: Vec<_> = network.receive_responses("ready".to_string());

        use ark_std::time::Instant;
        let time = Instant::now();


        let mut transcript = TranscriptMerlin::new(b"dfs");

        let (proof, coordinator_time) = dfs::co_spartan::SpartanProverCoordinator::prove(
            &proving_keys.ipk,
            &proving_keys.pub_ipk,
            &proving_keys.ivk,
            &mut transcript,
            &mut network,
        );

        let final_time = time.elapsed();
        println!("proving time: {:?}", final_time);
        println!("coordinator time: {:?}", coordinator_time);

        end_timer_buf!(log, start);

        /***************************************************************************/
        /***************************************************************************/

        {
            let time = Instant::now();
            assert!(proof.verify(&proving_keys.ivk, &Vec::new()).is_ok());
            let final_time = time.elapsed();
            println!("verification time: {:?}", final_time);

            let bytes = proof.serialized_size(ark_serialize::Compress::Yes);
            println!("proof size: {:?}", bytes);
        }
    } else {
        // Deserialize the proving keys

        // Worker code

        if local == 1 {
            worker_id = (rank as usize) - 1;
        } else {
            worker_id = worker_id - 1;
        }

        let proving_keys = {
            let mut buf = Vec::new();

            let mut file_name = key_file.clone();

            file_name.set_extension(worker_id.to_string());

            let mut f =
                File::open(&file_name).expect(&format!("couldn't open file {:?}", file_name));
            let _ = f.read_to_end(&mut buf);
            let buf_slice = buf.as_slice();

            let pk = dfs::co_spartan::Rep3ProverKey::<E>::deserialize_uncompressed_unchecked(buf_slice)
                .unwrap();

            pk
        };

        let current_num_threads = current_num_threads();
        println!(
            "Rayon num threads in worker {rank}: {}",
            current_num_threads
        );

        // let worker_id = (rank as usize) - 1;

        let log_chunk_size = proving_keys.num_variables - log_num_workers_per_party;
        let start_eq = (1 << log_chunk_size) * (worker_id / 3);

        let pub_log_chunk_size = proving_keys.num_variables - log_num_public_workers;
        println!(
            "proving_keys.num_variables: {:?}",
            proving_keys.num_variables
        );

        let pub_start_eq = (1 << pub_log_chunk_size) * worker_id;
        let active = (worker_id < (1 << log_num_public_workers));

        let mut seed_0 = Blake2s512Rng::setup();
        seed_0.feed(&proving_keys.seed_0.as_bytes());
        let mut seed_1 = Blake2s512Rng::setup();
        seed_1.feed(&proving_keys.seed_1.as_bytes());
        let mut random = SSRandom::<Blake2s512Rng>::new(seed_0, seed_1);

        let mut network = Rep3WorkerMPI::new(
            root_process,
            &mut log,
            log_num_workers_per_party,
            log_num_public_workers,
            size,
            rank as usize,
        );

        network.send_response("ready".to_string());



        let (size1, size2) = dfs::co_spartan::SpartanProverWorker::new(
            log_chunk_size,
            start_eq,
            pub_log_chunk_size,
            pub_start_eq,
        ).prove(
            &proving_keys,
            &mut random,
            active,
            &mut network,
        );
        send_size = send_size + size1;
        recv_size = recv_size + size2;
        /***************************************************************************/
        /***************************************************************************/
    }

    end_timer_buf!(log, very_start);

    println!(
        "Rank {rank} log: {}\nsend_msg_size: {}\nrecv_msg_size: {}\n",
        log.join(";"),
        send_size,
        recv_size
    );
}

#[cfg(feature = "parallel")]
fn execute_in_pool<T: Send>(f: impl FnOnce() -> T + Send, num_threads: usize) -> T {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    pool.install(f)
}
#[cfg(not(feature = "parallel"))]
fn execute_in_pool<T: Send>(f: impl FnOnce() -> T + Send, num_threads: usize) -> T {
    f()
}
#[cfg(not(feature = "parallel"))]
fn execute_in_pool_with_all_threads<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    f()
}

fn pool_and_chunk_size(num_threads: usize, num_requests: usize) -> (usize, usize) {
    let pool_size = if num_threads > num_requests {
        num_threads / num_requests
    } else {
        1
    };
    let chunk_size = if num_requests >= num_threads {
        num_requests / num_threads
    } else {
        num_threads
    };
    dbg!((pool_size, chunk_size))
}

fn compute_responses<'a, R, W, U, F>(
    requests: &'a [R],
    worker_states: impl IntoIterator<Item = W>,
    stage_fn: F,
) -> Vec<U>
where
    R: 'a + Send + Sync,
    W: Send + Sync,
    U: Send + Sync,
    F: Send + Sync + Fn(&'a R, W) -> U,
{
    let (pool_size, chunk_size) = pool_and_chunk_size(current_num_threads(), requests.len());
    thread::scope(|s| {
        let mut thread_results = Vec::new();
        let worker_state_chunks: Vec<Vec<_>> = worker_states
            .into_iter()
            .chunks(chunk_size)
            .into_iter()
            .map(|c| c.into_iter().collect())
            .collect();
        for (reqs, states) in requests.chunks(chunk_size).zip(worker_state_chunks) {
            let result = s.spawn(|_| {
                execute_in_pool(
                    || {
                        reqs.into_iter()
                            .zip(states)
                            .map(|(req, state)| stage_fn(req, state))
                            .collect::<Vec<_>>()
                    },
                    pool_size,
                )
            });
            thread_results.push(result);
        }
        thread_results
            .into_iter()
            .map(|t| t.join().unwrap())
            .flatten()
            .collect::<Vec<_>>()
    })
    .unwrap()
}
