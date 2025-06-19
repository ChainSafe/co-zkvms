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
use dfs::{subprotocols::loglookup::LogLookupProof, SparseMatEntry, SparseMatPolynomial};
use dfs::utils::generate_dumb_sponge;
use dfs::{mpc::rss::RssPoly, mpi_snark::worker::PublicProver};
use dfs::{mpc::utils::generate_rss_share_randomness, mpc_snark::worker::DistributedRSSProverKey};
use dfs::{mpc_snark::coordinator, subprotocols::loglookup::sumcheck_polynomial_list};
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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistributedRootKey<E: Pairing> {
    pub ipk: IndexProverKey<E>,
    pub pub_ipk: IndexProverKey<E>,
    pub ivk: IndexVerifierKey<E>,
}

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

        #[clap(long, value_name = "NUM")]
        log_instance_size: usize,

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
    println!("Rayon num threads: {}", current_num_threads());

    let args = Args::parse();

    match args.command {
        Command::Setup {
            // log_instance_size,
            r1cs_noir_instance_path,
            r1cs_input_path,
            log_num_workers_per_party,
            log_num_public_workers,
            key_out,
        } => setup(
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
            log_instance_size,
            worker_id,
            local,
        } => {
            work::<Bn254>(
                log_num_workers_per_party,
                log_instance_size,
                key_file,
                worker_id,
                log_num_public_workers,
                local,
            );
        }
    }
}

fn setup(
    origin_key_out_path: PathBuf,
    // log_instance_size: usize,
    r1cs_noir_instance_path: PathBuf,
    r1cs_input_path: PathBuf,
    log_num_workers_per_party: usize,
    _log_num_public_workers: usize,
) {
    type E = Bn254;

    let mut proof_scheme: NoirProofScheme = noir_r1cs::read(&r1cs_noir_instance_path).unwrap();
    let z = proof_scheme.solve_witness(&r1cs_input_path).unwrap();
    let r1cs_noir = proof_scheme.r1cs;
    let log_instance_size = r1cs_noir.log2_instance_size();

    let mut rng = StdRng::seed_from_u64(12);
    let za = r1cs_noir.a() * &z;
    let zb = r1cs_noir.b() * &z;
    let zc = r1cs_noir.c() * &z;
    let r1cs = from_noir_r1cs(&r1cs_noir);

    // let (r1cs, z, za, zb, zc) = produce_test_r1cs(log_instance_size, &mut rng);

    let srs = SRS::<E, _>::generate_srs(log_instance_size + 2, 4, &mut rng);

    let (pk, vk) = Indexer::index_for_prover_and_verifier(&r1cs, &srs);

    // let P: usize = log_num_public_workers;
    // let P_W: usize = log_num_workers_per_party;
    for P_W in 0..log_num_workers_per_party {
        let P: usize = P_W + 1;

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

        let (ipk_vec, root_ipk) = split_ipk(&pk, P_W);

        let (pub_ipk_vec, pub_root_ipk) = split_ipk(&pk, P);
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
                ipk: ipk_vec[i].clone(),
                pub_ipk: pub_ipk_vec[cnt].clone(),
                z: z_0,
                za: za_0,
                zb: zb_0,
                zc: zc_0,
                num_variables: r1cs.num_vars,
                seed_0: "seed 0".to_string(),
                seed_1: "seed 1".to_string(),
            };
            if cnt < (1 << P) - 1 {
                cnt = cnt + 1;
            }

            let pk_1 = DistributedRSSProverKey {
                ipk: ipk_vec[i].clone(),
                pub_ipk: pub_ipk_vec[cnt].clone(),
                z: z_1,
                za: za_1,
                zb: zb_1,
                zc: zc_1,
                num_variables: r1cs.num_vars,
                seed_0: "seed 1".to_string(),
                seed_1: "seed 2".to_string(),
            };
            if cnt < (1 << P) - 1 {
                cnt = cnt + 1;
            }

            let pk_2 = DistributedRSSProverKey {
                ipk: ipk_vec[i].clone(),
                pub_ipk: pub_ipk_vec[cnt].clone(),
                z: z_2,
                za: za_2,
                zb: zb_2,
                zc: zc_2,
                num_variables: r1cs.num_vars,
                seed_0: "seed 2".to_string(),
                seed_1: "seed 0".to_string(),
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

                let mut f = File::create(&file_name)
                    .expect(&format!("could not create file {:?}", file_name));
                f.write_all(&buf).unwrap();
            }
        }

        {
            let mut buf = Vec::new();
            let pk = DistributedRootKey {
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

fn work<E: Pairing>(
    log_num_workers_per_party: usize,
    log_instance_size: usize,
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

        // Initial proof
        let start = start_timer_buf!(log, || format!("Coord: construct coordinator proof"));

        let default_response = "ready".to_string();
        let responses_chunked: Vec<_> = gather_responses(
            &mut log,
            &("mpi_snark_coordinator_".to_owned() + "_start"),
            size,
            &root_process,
            default_response,
        );

        use ark_std::time::Instant;
        let time = Instant::now();

        let (proof, coordinator_time) = dfs::mpc_snark::coordinator::PrivateProver::new(
            &mut log,
            "mpi_snark_coordinator_",
            size,
            &root_process,
            log_num_workers_per_party,
            log_num_public_workers,
            &proving_keys.ipk,
            &proving_keys.pub_ipk,
            &proving_keys.ivk,
            dfs::mpc::utils::ShareType::Rss,
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
                File::open(&file_name).expect(&format!("couldn't open file {:?}", key_file));
            let _ = f.read_to_end(&mut buf);
            let buf_slice = buf.as_slice();

            let pk = DistributedRSSProverKey::<E>::deserialize_uncompressed_unchecked(buf_slice)
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
        let pub_start_eq = (1 << pub_log_chunk_size) * worker_id;
        let active = (worker_id < (1 << log_num_public_workers));

        let mut seed_0 = Blake2s512Rng::setup();
        seed_0.feed(&proving_keys.seed_0.as_bytes());
        let mut seed_1 = Blake2s512Rng::setup();
        seed_1.feed(&proving_keys.seed_1.as_bytes());
        let mut random = SSRandom::<Blake2s512Rng>::new(seed_0, seed_1);

        let responses = "ready".to_string();
        send_responses(&mut log, rank, "worker ready", &root_process, &responses, 1);

        let (size1, size2) = dfs::mpc_snark::worker::RssPrivateProver::prove(
            &proving_keys,
            &mut log,
            "mpc_rss_worker_",
            size,
            rank,
            &root_process,
            log_num_workers_per_party,
            log_num_public_workers,
            start_eq,
            log_chunk_size,
            &mut random,
            active,
            pub_start_eq,
            pub_log_chunk_size,
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

/// Convert from a noir-r1cs R1CS struct to this R1CSInstance struct.
/// This method converts the sparse matrix format used by noir-r1cs to the
/// SparseMatPolynomial format used by this implementation.
pub fn from_noir_r1cs(noir_r1cs: &noir_r1cs::R1CS) -> R1CSInstance<FieldElement> {
    // Convert sparse matrix entries from noir-r1cs format to our format
    let convert_matrix = |matrix: &noir_r1cs::SparseMatrix| -> Vec<SparseMatEntry<FieldElement>> {
        let hydrated = matrix.hydrate(&noir_r1cs.interner);
        let mut entries = Vec::new();

        for ((row, col), value) in hydrated.iter() {
            entries.push(SparseMatEntry::new(row, col, value));
        }

        entries
    };

    // Convert the three matrices
    let a_entries = convert_matrix(&noir_r1cs.a);
    let b_entries = convert_matrix(&noir_r1cs.b);
    let c_entries = convert_matrix(&noir_r1cs.c);

    // Create SparseMatPolynomials
    let poly_a = SparseMatPolynomial::new(noir_r1cs.constraints, noir_r1cs.witnesses, a_entries);
    let poly_b = SparseMatPolynomial::new(noir_r1cs.constraints, noir_r1cs.witnesses, b_entries);
    let poly_c = SparseMatPolynomial::new(noir_r1cs.constraints, noir_r1cs.witnesses, c_entries);

    R1CSInstance {
        num_cons: noir_r1cs.constraints,
        num_vars: noir_r1cs.witnesses,
        num_inputs: noir_r1cs.public_inputs,
        A: poly_a,
        B: poly_b,
        C: poly_c,
    }
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

#[cfg(feature = "parallel")]
use rayon::current_num_threads;

#[cfg(not(feature = "parallel"))]
fn current_num_threads() -> usize {
    1
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
