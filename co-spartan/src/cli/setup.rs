#![allow(warnings)]
use core::num;
use std::fmt::format;
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
use ark_ec::pairing::Pairing;
use ark_ff::{BigInt, Field, One, PrimeField, UniformRand, Zero};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, CommitterKey, VerifierKey},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter, fs};
use clap::{Parser, Subcommand};
use co_spartan::{
    mpc::{
        rep3::{generate_poly_shares_rss, RssPoly},
        SSRandom,
    },
    utils::{pad_to_power_of_two, split_vec},
    Rep3ProverKey,
};
use crossbeam::thread;
use itertools::{merge, Itertools};
use mimalloc::MiMalloc;
use mpi::{
    datatype::{Partition, PartitionMut},
    request,
    topology::Process,
    traits::*,
    Count,
};
use noir_r1cs::{FieldElement, NoirProofScheme};
use rand::{rngs::StdRng, seq::SliceRandom, RngCore, SeedableRng};
use rayon::prelude::*;
use spartan::math::{SparseMatEntry, SparseMatPolynomial};
use spartan::{
    transcript::{get_scalar_challenge, get_vector_challenge},
    IndexProverKey, IndexVerifierKey, Indexer, SRS,
};

// use dfs::mpc_snark::worker::DistributedRSSProverKey as Rep3ProverKey;
// use dfs::{snark::zk::SRS, utils::split_r1cs};
// use dfs::snark::indexer::{IndexProverKey, IndexVerifierKey, Indexer};
// use dfs::utils::split_ipk;

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistributedRootKey<E: Pairing> {
    pub log_instance_size: usize,
    pub ipk: IndexProverKey<E>,
    pub pub_ipk: IndexProverKey<E>,
    pub ivk: IndexVerifierKey<E>,
}

pub fn setup<E: Pairing>(
    artifacts_dir_path: PathBuf,
    // log_instance_size: usize,
    r1cs_noir_instance_path: PathBuf,
    r1cs_input_path: PathBuf,
    log_num_workers_per_party: usize,
    log_num_public_workers: Option<usize>,
) where
    E::ScalarField: PrimeField<BigInt = BigInt<4>>,
{
    let log_num_public_workers = log_num_public_workers
        .unwrap_or(((1 << log_num_workers_per_party) * 3 as u64).ilog2() as usize);

    let mut rng = StdRng::seed_from_u64(12);

    let mut proof_scheme: NoirProofScheme = noir_r1cs::read(&r1cs_noir_instance_path).unwrap();
    let mut z: Vec<E::ScalarField> = proof_scheme
        .solve_witness(&r1cs_input_path)
        .unwrap()
        .iter()
        .map(|v| E::ScalarField::from_bigint(v.into_bigint()).unwrap())
        .collect();
    let r1cs: spartan::R1CS<E::ScalarField> = proof_scheme.r1cs.into();

    let (coordinator_key, prover_keys) = co_spartan::setup_rep3::<E>(
        &r1cs,
        z,
        log_num_workers_per_party,
        log_num_public_workers,
        &mut rng,
    );

    let log_instance_size = coordinator_key.log_instance_size;

    let key_out_path_dir = artifacts_dir_path.join(format!(
        "keys_{}_{}",
        log_num_workers_per_party, log_num_public_workers
    ));
    fs::create_dir_all(&key_out_path_dir).unwrap();

    for i in 0..1 << log_num_workers_per_party {
        for j in 0..3 {
            let mut buf = Vec::new();
            prover_keys[i][j].serialize_uncompressed(&mut buf).unwrap();

            let mut file_name = key_out_path_dir.join(format!("worker_{}.key", 3 * i + j));
            let mut f =
                File::create(&file_name).expect(&format!("could not create file {:?}", file_name));
            f.write_all(&buf).unwrap();
        }
    }

    let mut buf = Vec::new();
    coordinator_key.serialize_uncompressed(&mut buf).unwrap();
    let mut file_name = key_out_path_dir.join("coordinator.key");
    let mut f = File::create(&file_name).expect(&format!("could not create file {:?}", file_name));
    f.write_all(&buf).unwrap();
}
