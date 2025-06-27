use ark_ff::Field;
use ark_linear_sumcheck::ml_sumcheck::data_structures::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::data_structures::PolynomialInfo;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverState as SumcheckProverState;
use ark_poly::DenseMultilinearExtension;

use crate::mpi_snark::coordinator::PartialProof;
use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter, rc::Rc};
use mpi::point_to_point::send_receive;
use mpi::point_to_point::send_receive_into;
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::Process,
    Count,
};
use mpi::{request, traits::*};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Prover State
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct DistrbutedSumcheckProverState<F: Field> {
    /// Stores the list of products that is meant to be added together. Each multiplicand is represented by
    /// the index in flattened_ml_extensions
    pub list_of_products: Vec<(F, Vec<F>)>,
}

pub fn merge_list_of_distributed_poly<F: Field>(
    prover_states: &Vec<DistrbutedSumcheckProverState<F>>,
    poly_info: &PolynomialInfo,
    log_num_parties: usize,
) -> ListOfProductsOfPolynomials<F> {
    let mut merge_poly = ListOfProductsOfPolynomials::new(log_num_parties);
    merge_poly.max_multiplicands = poly_info.max_multiplicands;
    for j in 0..prover_states[0].list_of_products.len() {
        let mut evals: Vec<Vec<F>> = vec![Vec::new(); prover_states[0].list_of_products[j].1.len()];
        for i in 0..(1 << log_num_parties) {
            let (_, prods) = &prover_states[i].list_of_products[j];
            for k in 0..prods.len() {
                evals[k].push(prods[k]);
            }
        }
        let mut prod: Vec<Rc<DenseMultilinearExtension<F>>> = Vec::new();
        for e in &evals {
            prod.push(Rc::new(DenseMultilinearExtension::from_evaluations_vec(
                log_num_parties,
                e.clone(),
            )))
        }

        merge_poly.add_product(prod.into_iter(), prover_states[0].list_of_products[j].0);
    }

    merge_poly
}

pub fn obtain_distrbuted_sumcheck_prover_state<F: Field>(
    prover_state: &SumcheckProverState<F>,
) -> DistrbutedSumcheckProverState<F> {
    let mut distributed_state = DistrbutedSumcheckProverState {
        list_of_products: Vec::new(),
    };
    for i in 0..prover_state.list_of_products.len() {
        let coeff = prover_state.list_of_products[i].0;
        let mut prod = Vec::new();
        for j in &prover_state.list_of_products[i].1 {
            prod.push(prover_state.flattened_ml_extensions[*j].evaluations[0]);
        }
        distributed_state.list_of_products.push((coeff, prod));
    }
    distributed_state
}

pub fn hash_tuple<F: Field>(v: &[usize], eq: &DenseMultilinearExtension<F>, v_msg: &F) -> Vec<F> {
    let mut result = cfg_iter!(v)
        .enumerate()
        .map(|(idx, v_i)| F::from_random_bytes(&v_i.to_le_bytes()).unwrap() + *v_msg * eq[idx])
        .collect::<Vec<_>>();

    for _ in 0..result.len().next_power_of_two() - result.len() {
        result.push(result[0]);
    }
    result
}

pub fn hash_tuple_custom_pad<F: Field>(v: &[usize], eq: &DenseMultilinearExtension<F>, v_msg: &F, pad_to_log_size: usize) -> Vec<F> {
    let mut result = cfg_iter!(v)
        .enumerate()
        .map(|(idx, v_i)| F::from_random_bytes(&v_i.to_le_bytes()).unwrap() + *v_msg * eq[idx])
        .collect::<Vec<_>>();

    for _ in 0..(1 << pad_to_log_size) - result.len() {
        result.push(result[0]);
    }
    result
}

pub fn combine_partial_proof<E: Pairing>(pfs: &[PartialProof<E>]) -> PartialProof<E> {
    let mut res = pfs[0].clone();
    for i in 1..pfs.len() {
        for j in 0..res.proofs.proofs.len() {
            res.proofs.proofs[j] = (res.proofs.proofs[j] + pfs[i].proofs.proofs[j]).into();
        }
        res.val = res.val + pfs[i].val;
        for j in 0..res.evals.len() {
            res.evals[j] = res.evals[j] + pfs[i].evals[j];
        }
    }
    res
}

macro_rules! start_timer_buf {
    ($buf:ident, $msg:expr) => {{
        let _log = &$buf;
        let _ = $msg;
        // use std::time::Instant;

        // let msg = $msg();
        // let start_info = "Start:";

        // $buf.push(format!("{:8} {}", start_info, msg));
        // (msg.to_string(), Instant::now())
    }};
}

macro_rules! end_timer_buf {
    ($buf:ident, $time:expr) => {{
        let _log = &$buf;
        let _ = $time;
        // let time = $time.1;
        // let final_time = time.elapsed();

        // let end_info = "End:";
        // let message = format!("{}", $time.0);

        // $buf.push(format!(
        //     "{:8} {} {}μs",
        //     end_info,
        //     message,
        //     final_time.as_micros()
        // ));
    }};
}

#[macro_export]
macro_rules! construct_partitioned_buffer_for_scatter {
    ($items:expr, $flattened_item_bytes: expr) => {{
        let item_bytes = ($items).iter().map(serialize_to_vec).collect::<Vec<_>>();
        let counts = std::iter::once(&vec![])
            .chain(item_bytes.iter())
            .map(|bytes| bytes.len() as Count)
            .collect::<Vec<_>>();
        let displacements: Vec<Count> = counts
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect();
        *$flattened_item_bytes = item_bytes.concat();
        Partition::new(&*$flattened_item_bytes, counts, displacements)
    }};
}

#[macro_export]
macro_rules! construct_partitioned_mut_buffer_for_gather {
    ($size: expr, $default: expr, $flattened_item_bytes: expr) => {{
        let item_size = $default.uncompressed_size();
        let item_bytes = std::iter::once(vec![])
            .chain(std::iter::repeat(vec![0u8; item_size]))
            .take($size as usize)
            .collect::<Vec<_>>();
        let counts = item_bytes
            .iter()
            .map(|bytes| bytes.len() as Count)
            .collect::<Vec<_>>();
        let displacements: Vec<Count> = counts
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect();
        *$flattened_item_bytes = item_bytes.concat();
        PartitionMut::new(&mut $flattened_item_bytes[..], counts, displacements)
    }};
}

#[macro_export]
macro_rules! deserialize_flattened_bytes {
    ($flattened_item_bytes: expr, $default: expr, $item_type: ty) => {{
        let item_size = $default.uncompressed_size();
        $flattened_item_bytes
            .chunks_exact(item_size)
            .map(<$item_type>::deserialize_uncompressed_unchecked)
            .collect::<Result<Vec<_>, _>>()
    }};
}

pub fn serialize_to_vec(item: &impl CanonicalSerialize) -> Vec<u8> {
    let mut bytes = vec![];
    (*item).serialize_uncompressed(&mut bytes).unwrap();
    bytes
}

pub fn scatter_requests<'a, C: 'a + Communicator>(
    log: &mut Vec<String>,
    stage: &str,
    root_process: &Process<'a, C>,
    requests: &[impl CanonicalSerialize + Send],
) {
    let start = start_timer_buf!(log, || format!("Coord: Serializing {stage} requests"));
    let mut request_bytes = vec![];
    let request_bytes_buf = construct_partitioned_buffer_for_scatter!(requests, &mut request_bytes);
    end_timer_buf!(log, start);

    let counts = request_bytes_buf.counts().to_vec();
    root_process.scatter_into_root(&counts, &mut 0i32);
    let mut _recv_buf: Vec<u8> = vec![];

    let start = start_timer_buf!(log, || format!("Coord: Scattering {stage} requests"));
    root_process.scatter_varcount_into_root(&request_bytes_buf, &mut _recv_buf);
    end_timer_buf!(log, start);
}

pub fn gather_responses<'a, C, T>(
    log: &mut Vec<String>,
    stage: &str,
    size: Count,
    root_process: &Process<'a, C>,
    default_response: T,
) -> Vec<T>
where
    C: 'a + Communicator,
    T: CanonicalSerialize + CanonicalDeserialize,
{
    let mut response_bytes = vec![];
    let mut response_bytes_buf =
        construct_partitioned_mut_buffer_for_gather!(size, default_response, &mut response_bytes);
    // Root does not send anything, it only receives.
    let start = start_timer_buf!(log, || format!("Coord: Gathering {stage} responses"));
    root_process.gather_varcount_into_root(&[0u8; 0], &mut response_bytes_buf);
    end_timer_buf!(log, start);

    let start = start_timer_buf!(log, || format!("Coord: Deserializing {stage} responses"));
    let ret = deserialize_flattened_bytes!(response_bytes, default_response, T).unwrap();
    end_timer_buf!(log, start);

    ret
}

pub fn send_responses<'a, C: 'a + Communicator, T: CanonicalSerialize>(
    log: &mut Vec<String>,
    rank: i32,
    stage: &str,
    root_process: &Process<'a, C>,
    responses: &T,
    dummy: usize,
) -> usize {
    // Send Stage 1 response
    let start = start_timer_buf!(log, || format!(
        "Worker {rank}: Serializing {stage} {}, responses",
        1,
    ));
    let responses_bytes = serialize_to_vec(responses);
    end_timer_buf!(log, start);

    let start = start_timer_buf!(log, || format!(
        "Worker {rank}: Gathering {stage} response, each of size {}",
        responses_bytes.len() / 1
    ));
    root_process.gather_varcount_into(&responses_bytes);
    end_timer_buf!(log, start);

    if dummy == 1 {
        return 0;
    }
    responses_bytes.len()
}

pub fn receive_requests<'a, C: 'a + Communicator, T: CanonicalDeserialize>(
    log: &mut Vec<String>,
    rank: i32,
    stage: &str,
    root_process: &Process<'a, C>,
    dummy: usize,
) -> (T, usize) {
    let mut size = 0 as Count;
    root_process.scatter_into(&mut size);

    let start = start_timer_buf!(log, || format!(
        "Worker {rank}: Receiving scattered {stage} request of size {size}"
    ));
    let mut request_bytes = vec![0u8; size as usize];
    root_process.scatter_varcount_into(&mut request_bytes);
    end_timer_buf!(log, start);

    let start = start_timer_buf!(log, || format!(
        "Worker {rank}: Deserializing {stage} request"
    ));
    let ret = T::deserialize_uncompressed_unchecked(&request_bytes[..]).unwrap();
    end_timer_buf!(log, start);

    if dummy == 1 {
        return (ret, 0);
    }

    (ret, request_bytes.len())
}

pub fn send_and_receive<'a, C: 'a + Communicator, T: CanonicalSerialize + Clone>(
    log: &mut Vec<String>,
    rank: i32,
    stage: &str,
    size: Count,
    root_process: &Process<'a, C>,
    peer_process: &Process<'a, C>,
    send_msg: &Vec<T>,
    dummy: usize,
) -> (Vec<T>, usize, usize)
where
    T: CanonicalDeserialize,
{
    // Send Stage 1 response
    // let msg_bytes = serialize_to_vec(msg);
    let send_bytes = serialize_to_vec(send_msg);
    let mut response_bytes_buf = send_bytes.clone();

    let status = send_receive_into(
        &send_bytes,
        peer_process,
        &mut response_bytes_buf,
        peer_process,
    );

    let ret = Vec::<T>::deserialize_uncompressed_unchecked(&response_bytes_buf[..]).unwrap();

    if dummy == 1 {
        return (ret, 0, 0);
    }

    (ret, send_bytes.len(), response_bytes_buf.len())
}
