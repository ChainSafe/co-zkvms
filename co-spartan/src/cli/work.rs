#![allow(warnings)]
use core::num;
use std::{collections::HashMap, mem, rc::Rc};
// use mpi_snark::worker::WorkerState;
use std::{
    fs::File,
    io::{Read, Write},
    num::NonZeroUsize,
    path::PathBuf,
};

use ark_bn254::{Bn254, Config, Fr};
// use ark_ec::bn::Bls12;
use crate::current_num_threads;
use ark_ec::pairing::Pairing;
use ark_ff::{Field, One, UniformRand, Zero};
use ark_linear_sumcheck::{
    ml_sumcheck::{
        protocol::{
            verifier::SubClaim, IPForMLSumcheck, ListOfProductsOfPolynomials, PolynomialInfo,
        },
        MLSumcheck,
    },
    rng::{Blake2s512Rng, FeedableRNG},
};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, CommitterKey, VerifierKey},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter, fs};
use clap::{Parser, Subcommand};
use co_spartan::{
    mpc::{rep3::RssPoly, SSRandom},
    network::{
        mpi::{Rep3CoordinatorMPI, Rep3WorkerMPI},
        NetworkCoordinator, NetworkWorker,
    },
};
use crossbeam::thread;
use itertools::{merge, Itertools};
use mpi::{
    datatype::{Partition, PartitionMut},
    request,
    topology::Process,
    traits::*,
    Count,
};
use rayon::prelude::*;
use spartan::{transcript::TranscriptMerlin, IndexProverKey, IndexVerifierKey, Indexer, SRS};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistributedRootKey<E: Pairing> {
    pub log_instance_size: usize,
    pub ipk: IndexProverKey<E>,
    pub pub_ipk: IndexProverKey<E>,
    pub ivk: IndexVerifierKey<E>,
}

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

        let (proof, coordinator_time) = co_spartan::SpartanProverCoordinator::prove(
            &proving_keys.ipk,
            &proving_keys.pub_ipk,
            &proving_keys.ivk,
            &mut transcript,
            &mut network,
        );

        let final_time = time.elapsed();
        println!("proving time: {:?}", final_time);
        println!("coordinator time: {:?}", coordinator_time);

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

            let pk = co_spartan::Rep3ProverKey::<E>::deserialize_uncompressed_unchecked(buf_slice)
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

        let (size1, size2) = co_spartan::SpartanProverWorker::new(
            log_chunk_size,
            start_eq,
            pub_log_chunk_size,
            pub_start_eq,
        )
        .prove(&proving_keys, &mut random, active, &mut network);
        send_size = send_size + size1;
        recv_size = recv_size + size2;
        /***************************************************************************/
        /***************************************************************************/
    }

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
