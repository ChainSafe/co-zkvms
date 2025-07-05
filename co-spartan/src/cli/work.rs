#![allow(warnings)]
use core::num;
use std::{collections::HashMap, mem, rc::Rc};
use std::{
    fs::File,
    io::{Read, Write},
    num::NonZeroUsize,
    path::PathBuf,
};

use ark_bn254::{Bn254, Config, Fr};
use ark_ec::pairing::Pairing;
use ark_ff::{BigInt, Field, One, PrimeField, UniformRand, Zero};
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
use ark_std::{cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter, fs, time::Instant};
use bytesize::ByteSize;
use clap::{Parser, Subcommand};
use co_spartan::{
    mpc::{rep3::Rep3Poly, SSRandom},
    setup::CoordinatorKey,
};
use crossbeam::thread;
use itertools::{merge, Itertools};
use mpc_net::{
    mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker},
    rep3::mpi::{Rep3CoordinatorMPI, Rep3WorkerMPI},
};
use noir_r1cs::NoirProofScheme;
use rand::RngCore;
use rayon::prelude::*;
use spartan::{transcript::TranscriptMerlin, IndexProverKey, IndexVerifierKey, Indexer, SRS};

use crate::current_num_threads;

pub fn work<E: Pairing>(
    artifacts_dir: PathBuf,
    r1cs_noir_scheme_path: PathBuf,
    r1cs_input_path: PathBuf,
    log_num_workers_per_party: usize,
    log_num_public_workers: Option<usize>,
    local: bool,
    worker_id: Option<usize>,
) where
    E::ScalarField: PrimeField<BigInt = BigInt<4>>,
{
    let log_num_public_workers = log_num_public_workers
        .unwrap_or(((1 << log_num_workers_per_party) * 3 as u64).ilog2() as usize);

    #[cfg(feature = "mpi")]
    let mpi_ctx = mpc_net::rep3::mpi::initialize_mpi();
    #[cfg(feature = "mpi")]
    let is_coordinator = mpi_ctx.is_root();

    let keys_dir = artifacts_dir.join(format!(
        "keys_{}_{}",
        log_num_workers_per_party, log_num_public_workers
    ));

    if !keys_dir.exists() {
        eprintln!(
            "keys directory for this configuration does not exist: {:?}",
            keys_dir
        );
        std::process::exit(1);
    }

    #[cfg(feature = "mpi")]
    let is_coordinator = mpi_ctx.is_root();
    #[cfg(not(feature = "mpi"))]
    let is_coordinator = todo!();

    if is_coordinator {
        #[cfg(feature = "mpi")]
        let mut network =
            Rep3CoordinatorMPI::new(log_num_workers_per_party, log_num_public_workers, &mpi_ctx);
        coordinator_work::<E, _>(
            keys_dir,
            r1cs_noir_scheme_path,
            r1cs_input_path,
            log_num_workers_per_party,
            log_num_public_workers,
            &mut network,
        );
    } else {
        #[cfg(feature = "mpi")]
        let (worker_id, mut network) = {
            let worker_id = if local {
                mpi_ctx.rank as usize - 1
            } else {
                worker_id.map(|x| x - 1).unwrap_or(0) // 0 worker is coordinator
            };

            let network = Rep3WorkerMPI::new(
                log_num_public_workers,
                log_num_workers_per_party,
                &mpi_ctx,
            );

            (worker_id, network)
        };

        worker_work::<E, _>(
            keys_dir,
            log_num_workers_per_party,
            log_num_public_workers,
            &mut network,
            worker_id,
        );
    }
}

#[tracing::instrument(skip_all, name = "coordinator_work")]
fn coordinator_work<E: Pairing, N: MpcStarNetCoordinator>(
    keys_dir: PathBuf,
    r1cs_noir_scheme_path: PathBuf,
    r1cs_input_path: PathBuf,
    log_num_workers_per_party: usize,
    log_num_public_workers: usize,
    network: &mut N,
) where
    E::ScalarField: PrimeField<BigInt = BigInt<4>>,
{
    let pk = {
        let mut buf = Vec::new();

        let mut file_name = keys_dir.join("coordinator.key");

        let mut f = File::open(&file_name).expect(&format!("couldn't open file {:?}", keys_dir));
        let _ = f.read_to_end(&mut buf);
        let buf_slice = buf.as_slice();

        CoordinatorKey::<E>::deserialize_uncompressed_unchecked(buf_slice).unwrap()
    };

    let mut proof_scheme: NoirProofScheme = noir_r1cs::read(&r1cs_noir_scheme_path).unwrap();
    let mut z: Vec<E::ScalarField> = proof_scheme
        .solve_witness(&r1cs_input_path)
        .unwrap()
        .iter()
        .map(|v| E::ScalarField::from_bigint(v.into_bigint()).unwrap())
        .collect();
    let r1cs: spartan::R1CS<E::ScalarField> = proof_scheme.r1cs.into();

    let mut rng = Blake2s512Rng::setup();
    let witness_shares = co_spartan::split_witness::<E>(
        z,
        r1cs.log2_instance_size(),
        log_num_workers_per_party,
        &mut rng,
    );

    let log_instance_size = pk.log_instance_size;

    let _: Vec<_> = network.receive_responses("ready".to_string()).unwrap();

    let witness_shares = witness_shares
        .into_iter()
        .flatten()
        .sorted_by_key(|(worker_id, _)| *worker_id)
        .map(|(_, z)| z)
        .collect();

    // todo: send witness shares to workers
    network.send_requests(witness_shares);
    let (send_bytes, _) = network.total_bandwidth_used();
    tracing::info!(
        "bandwidth to send witness shares: {}",
        ByteSize(send_bytes as u64)
    );

    let mut transcript = TranscriptMerlin::new(b"dfs");

    let (proof, coordinator_time) = co_spartan::SpartanProverCoordinator::prove(
        &pk.ipk,
        &pk.pub_ipk,
        &pk.ivk,
        &mut transcript,
        network,
    ).unwrap();

    let mut verifier_transcript = TranscriptMerlin::new(b"dfs");
    if let Err(e) = proof.verify(&pk.ivk, &Vec::new(), &mut verifier_transcript) {
        eprintln!("proof verification failed: {:?}", e);
        std::process::exit(1);
    }

    tracing::info!("coordinator time: {:?}", coordinator_time);
    tracing::info_span!("proof size").in_scope(|| {
        proof.log_size_report();
        tracing::info!("total size: {}", ByteSize(proof.compressed_size() as u64));
    });

    let (send_bytes, recv_bytes) = network.total_bandwidth_used();
    tracing::info!(
        "bandwidth used: send {}, recv {}",
        ByteSize(send_bytes as u64),
        ByteSize(recv_bytes as u64)
    );
}

#[tracing::instrument(skip_all, name = "worker_work", fields(worker_id = %worker_id))]
fn worker_work<E: Pairing, N: MpcStarNetWorker>(
    keys_dir: PathBuf,
    log_num_workers_per_party: usize,
    log_num_public_workers: usize,
    network: &mut N,
    worker_id: usize,
) where
    E::ScalarField: PrimeField<BigInt = BigInt<4>>,
{
    let pk = {
        let mut buf = Vec::new();

        let mut file_name = keys_dir.join(format!("worker_{}.key", worker_id));

        let mut f = File::open(&file_name).expect(&format!("couldn't open file {:?}", file_name));
        let _ = f.read_to_end(&mut buf);
        let buf_slice = buf.as_slice();

        co_spartan::Rep3ProverKey::<E>::deserialize_uncompressed_unchecked(buf_slice).unwrap()
    };

    let current_num_threads = current_num_threads();
    tracing::info!(
        "Rayon num threads in worker {worker_id}: {}",
        current_num_threads
    );

    let log_chunk_size = pk.num_variables - log_num_workers_per_party;
    let start_eq = (1 << log_chunk_size) * (worker_id / 3);

    let pub_log_chunk_size = pk.num_variables - log_num_public_workers;

    let pub_start_eq = (1 << pub_log_chunk_size) * worker_id;
    let active = (worker_id < (1 << log_num_public_workers));

    let mut seed_0 = Blake2s512Rng::setup();
    seed_0.feed(&pk.seed_0.as_bytes());
    let mut seed_1 = Blake2s512Rng::setup();
    seed_1.feed(&pk.seed_1.as_bytes());
    let mut random = SSRandom::<Blake2s512Rng>::new(seed_0, seed_1);

    network.send_response("ready".to_string());

    let witness_share = network.receive_request();

    co_spartan::SpartanProverWorker::new(
        log_chunk_size,
        start_eq,
        pub_log_chunk_size,
        pub_start_eq,
    )
    .prove(&pk, witness_share, &mut random, active, network);

    let (send_bytes, recv_bytes) = network.total_bandwidth_used();
    tracing::info!(
        "bandwidth used: send {}, recv {}",
        ByteSize(send_bytes as u64),
        ByteSize(recv_bytes as u64)
    );
}
