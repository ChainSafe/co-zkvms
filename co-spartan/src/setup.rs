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
use ark_poly_commit::challenge::ChallengeGenerator;

use ark_poly_commit::multilinear_pc::{
    data_structures::Commitment, data_structures::CommitterKey, data_structures::VerifierKey,
    MultilinearPC,
};
use ark_std::fs;
use dfs::utils::generate_dumb_sponge;
use dfs::{mpc::rss::RssPoly, mpi_snark::worker::PublicProver};
use dfs::{mpc::utils::generate_rss_share_randomness, mpc_snark::worker::DistributedRSSProverKey};
use dfs::{mpc_snark::coordinator, subprotocols::loglookup::sumcheck_polynomial_list};
use dfs::{mpi_snark::coordinator::mpi_batch_open_poly_coordinator, utils::produce_test_r1cs};
use dfs::{mpi_utils::send_responses, utils::split_ipk};
use dfs::{snark::indexer::Indexer, R1CSInstance};
use dfs::{snark::zk::SRS, utils::split_r1cs};
use dfs::{subprotocols::loglookup::LogLookupProof, SparseMatEntry, SparseMatPolynomial};

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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistributedRootKey<E: Pairing> {
    pub log_instance_size: usize,
    pub ipk: IndexProverKey<E>,
    pub pub_ipk: IndexProverKey<E>,
    pub ivk: IndexVerifierKey<E>,
}

pub fn setup(
    origin_key_out_path: PathBuf,
    // log_instance_size: usize,
    r1cs_noir_instance_path: PathBuf,
    r1cs_input_path: PathBuf,
    log_num_workers_per_party: usize,
    log_num_public_workers: usize,
) {
    type E = Bn254;

    // let log_instance_size = 14;
    // let mut rng = StdRng::seed_from_u64(12);
    // let (r1cs, z, za, zb, zc) = produce_test_r1cs(14, &mut rng);

    let mut proof_scheme: NoirProofScheme = noir_r1cs::read(&r1cs_noir_instance_path).unwrap();
    let mut z = proof_scheme.solve_witness(&r1cs_input_path).unwrap();
    println!("z: {:?}", z.len());
    let r1cs_noir = proof_scheme.r1cs;
    let log_instance_size = r1cs_noir.log2_instance_size();
    println!("log_instance_size: {:?}", log_instance_size);

    r1cs_noir.verify_witness(&z).unwrap();

    let mut rng = StdRng::seed_from_u64(12);
    let mut za = r1cs_noir.a() * &z;
    let mut zb = r1cs_noir.b() * &z;
    let mut zc = r1cs_noir.c() * &z;
    let r1cs = crate::from_noir_r1cs(&r1cs_noir);

    crate::utils::pad_to_power_of_two(&mut z, log_instance_size);
    crate::utils::pad_to_power_of_two(&mut za, log_instance_size);
    crate::utils::pad_to_power_of_two(&mut zb, log_instance_size);
    crate::utils::pad_to_power_of_two(&mut zc, log_instance_size);

    println!("z: {:?}", z.len());
    println!("za: {:?}", za.len());
    println!("zb: {:?}", zb.len());
    println!("zc: {:?}", zc.len());

    let za_poly = DenseMultilinearExtension::from_evaluations_vec(log_instance_size, za.clone());
    let zb_poly = DenseMultilinearExtension::from_evaluations_vec(log_instance_size, zb.clone());
    let zc_poly = DenseMultilinearExtension::from_evaluations_vec(log_instance_size, zc.clone());

    // let (r1cs, z, za, zb, zc) = produce_test_r1cs(log_instance_size, &mut rng);

    let srs = SRS::<E, _>::generate_srs(log_instance_size + 2, 4, &mut rng);

    let (pk, vk) = Indexer::index_for_prover_and_verifier(&r1cs, &srs);
    println!("pk.real_len_val: {:?}", pk.real_len_val);

    // let P: usize = log_num_public_workers;
    // let P_W: usize = log_num_workers_per_party;
    for P_W in 1..log_num_workers_per_party {
        let P: usize = P_W;

        let mut key_out_path = origin_key_out_path.clone();

        fs::create_dir(
            "inst_folder/inst_".to_owned()
                + &log_instance_size.to_string()
                + "_"
                + &P_W.to_string()
                + "_"
                + &P.to_string(),
        );
        key_out_path.push(
            "inst_".to_owned()
                + &log_instance_size.to_string()
                + "_"
                + &P_W.to_string()
                + "_"
                + &P.to_string()
                + "/test",
        );

        println!(
            "{:?}",
            "inst_".to_owned()
                + &log_instance_size.to_string()
                + "_"
                + &P_W.to_string()
                + "_"
                + &P.to_string()
                + "/test"
        );

        println!("pk.num_variables_val: {:?}", pk.num_variables_val);
        println!("P_W: {:?}", P_W);
        println!("P: {:?}", P);

        let (ipk_vec, root_ipk) = split_ipk(&pk, P_W);
        println!(
            "ipk_vec.len: {:?}",
            ipk_vec
                .iter()
                .map(|ipk| ipk.num_variables_val)
                .collect::<Vec<_>>()
        );

        // println!("pk.row: {:?}", ipk_vec[1].row_reindex.len());
        // println!("pk.col: {:?}", ipk_vec[1].col_reindex.len());
        // println!("pk.row_reindex: {:?}", ipk_vec[1].row_reindex);
        // println!("pk.val_a: {:?}", ipk_vec[0].val_a.evaluations.iter().map(|x| {let mut x = x.to_string(); x.truncate(4); x}).collect::<Vec<_>>());
        // println!("pk.val_c: {:?}", ipk_vec[0].val_c.evaluations.iter().map(|x| {let mut x = x.to_string(); x.truncate(4); x}).collect::<Vec<_>>());
        // println!("pk.row: {:?}", ipk_vec[1].row);
        // println!("pk.row_reindex: {:?}", ipk_vec[1].row_reindex);

        // assert_eq!(ipk_vec[0].col, ipk_vec[0].col_reindex);
        // assert_eq!(ipk_vec[1].col, ipk_vec[1].col_reindex);
        // assert_eq!(ipk_vec[0].row, ipk_vec[0].row_reindex);
        // assert_eq!(ipk_vec[1].row, ipk_vec[1].row_reindex);

        let (pub_ipk_vec, pub_root_ipk) = split_ipk(&pk, P);
        println!(
            "pub_ipk_vec.len: {:?}",
            pub_ipk_vec
                .iter()
                .map(|ipk| ipk.num_variables_val)
                .collect::<Vec<_>>()
        );
        

        // let r1cs_vec = split_r1cs(&r1cs, P);
        let z_vec = split_vec(&z, P_W);
        let za_vec = split_vec(&za, P_W);
        let zb_vec = split_vec(&zb, P_W);
        let zc_vec = split_vec(&zc, P_W);

        let num_vars = log_instance_size - P_W;

        let mut cnt = 0;

        for i in 0..1 << P_W {
            let z = DenseMultilinearExtension::from_evaluations_vec(num_vars, z_vec[i].clone());
            let za = DenseMultilinearExtension::from_evaluations_vec(num_vars, za_vec[i].clone());
            let zb = DenseMultilinearExtension::from_evaluations_vec(num_vars, zb_vec[i].clone());
            let zc = DenseMultilinearExtension::from_evaluations_vec(num_vars, zc_vec[i].clone());

            let (z_share_0, z_share_1, z_share_2) = generate_poly_shares_rss(&z, &mut rng);
            let (za_share_0, za_share_1, za_share_2) = generate_poly_shares_rss(&za, &mut rng);
            let (zb_share_0, zb_share_1, zb_share_2) = generate_poly_shares_rss(&zb, &mut rng);
            let (zc_share_0, zc_share_1, zc_share_2) = generate_poly_shares_rss(&zc, &mut rng);

            let z_0 = RssPoly::<E>::new(0, z_share_0.clone(), z_share_1.clone());
            let za_0 = RssPoly::<E>::new(0, za_share_0.clone(), za_share_1.clone());
            let zb_0 = RssPoly::<E>::new(0, zb_share_0.clone(), zb_share_1.clone());
            let zc_0 = RssPoly::<E>::new(0, zc_share_0.clone(), zc_share_1.clone());

            let z_1 = RssPoly::<E>::new(1, z_share_1.clone(), z_share_2.clone());
            let za_1 = RssPoly::<E>::new(1, za_share_1.clone(), za_share_2.clone());
            let zb_1 = RssPoly::<E>::new(1, zb_share_1.clone(), zb_share_2.clone());
            let zc_1 = RssPoly::<E>::new(1, zc_share_1.clone(), zc_share_2.clone());

            let z_2 = RssPoly::<E>::new(2, z_share_2.clone(), z_share_0.clone());
            let za_2 = RssPoly::<E>::new(2, za_share_2.clone(), za_share_0.clone());
            let zb_2 = RssPoly::<E>::new(2, zb_share_2.clone(), zb_share_0.clone());
            let zc_2 = RssPoly::<E>::new(2, zc_share_2.clone(), zc_share_0.clone());

            let pk_0 = DistributedRSSProverKey {
                party_id: i,
                num_parties: P,
                ipk: ipk_vec[i].clone(),
                pub_ipk: pub_ipk_vec[cnt].clone(),
                row: pk.row.clone(),
                col: pk.col.clone(),
                val_a: pk.val_a.clone(),
                val_b: pk.val_b.clone(),
                val_c: pk.val_c.clone(),
                z: z_0,
                za: za_0,
                zb: zb_0,
                zc: zc_0,
                num_variables: r1cs.num_vars,
                seed_0: "seed 0".to_string(),
                seed_1: "seed 1".to_string(),
                // todo: remove
                za_poly: Some(za_poly.clone()),
                zb_poly: Some(zb_poly.clone()),
                zc_poly: Some(zc_poly.clone()),
            };
            if cnt < (1 << P) - 1 {
                cnt = cnt + 1;
            }

            let pk_1 = DistributedRSSProverKey {
                party_id: i,
                num_parties: P,
                ipk: ipk_vec[i].clone(),
                pub_ipk: pub_ipk_vec[cnt].clone(),
                row: pk.row.clone(),
                col: pk.col.clone(),
                val_a: pk.val_a.clone(),
                val_b: pk.val_b.clone(),
                val_c: pk.val_c.clone(),
                z: z_1,
                za: za_1,
                zb: zb_1,
                zc: zc_1,
                num_variables: r1cs.num_vars,
                seed_0: "seed 1".to_string(),
                seed_1: "seed 2".to_string(),

                // todo: remove
                za_poly: Some(za_poly.clone()),
                zb_poly: Some(zb_poly.clone()),
                zc_poly: Some(zc_poly.clone()),
            };
            if cnt < (1 << P) - 1 {
                cnt = cnt + 1;
            }

            let pk_2 = DistributedRSSProverKey {
                party_id: i,
                num_parties: P,
                ipk: ipk_vec[i].clone(),
                pub_ipk: pub_ipk_vec[cnt].clone(),
                row: pk.row.clone(),
                col: pk.col.clone(),
                val_a: pk.val_a.clone(),
                val_b: pk.val_b.clone(),
                val_c: pk.val_c.clone(),
                z: z_2,
                za: za_2,
                zb: zb_2,
                zc: zc_2,
                num_variables: r1cs.num_vars,
                seed_0: "seed 2".to_string(),
                seed_1: "seed 0".to_string(),

                // todo: remove
                za_poly: Some(za_poly.clone()),
                zb_poly: Some(zb_poly.clone()),
                zc_poly: Some(zc_poly.clone()),
            };
            if cnt < (1 << P) - 1 {
                cnt = cnt + 1;
            }

            let pk_vec = vec![pk_0, pk_1, pk_2];

            for j in 0..3 {
                let mut buf = Vec::new();
                pk_vec[j].serialize_uncompressed(&mut buf).unwrap();

                let mut file_name = key_out_path.clone();
                file_name.set_extension((3 * i + j).to_string());

                // create the directory if it doesn't exist
                let dir = file_name.parent().unwrap();
                if !dir.exists() {
                    fs::create_dir_all(dir).unwrap();
                }

                let mut f = File::create(&file_name)
                    .expect(&format!("could not create file {:?}", file_name));
                f.write_all(&buf).unwrap();
            }
        }

        {
            let mut buf = Vec::new();
            let pk = DistributedRootKey {
                log_instance_size,
                ipk: root_ipk,
                pub_ipk: pub_root_ipk,
                ivk: vk.clone(),
            };
            pk.serialize_uncompressed(&mut buf).unwrap();
            let mut file_name = key_out_path.clone();
            file_name.set_extension("root");
            let mut f =
                File::create(&file_name).expect(&format!("could not create file {:?}", file_name));
            f.write_all(&buf).unwrap();
        }
    }
}
