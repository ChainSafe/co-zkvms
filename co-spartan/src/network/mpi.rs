use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use mpi::{
    datatype::{Partition, PartitionMut},
    topology::Process,
    traits::{Communicator, Partitioned, Root},
    Count,
};

use super::{NetworkCoordinator, NetworkWorker};
use crate::{
    construct_partitioned_buffer_for_scatter, construct_partitioned_mut_buffer_for_gather,
    deserialize_flattened_bytes,
};

pub struct Rep3CoordinatorMPI<'a, C: 'a + Communicator> {
    pub root_process: Process<'a, C>,
    pub log: &'a mut Vec<String>,
    pub log_num_workers_per_party: usize,
    pub log_num_public_workers: usize,
    pub size: Count,
    pub total_send_bytes: usize,
    pub total_recv_bytes: usize,
}

impl<'a, C: 'a + Communicator> Rep3CoordinatorMPI<'a, C> {
    pub fn new(
        root_process: Process<'a, C>,
        log: &'a mut Vec<String>,
        log_num_workers_per_party: usize,
        log_num_public_workers: usize,
        size: Count,
    ) -> Self {
        Self {
            root_process,
            log,
            log_num_workers_per_party,
            log_num_public_workers,
            size,
            total_send_bytes: 0,
            total_recv_bytes: 0,
        }
    }
}

impl<'a, C: 'a + Communicator> NetworkCoordinator for Rep3CoordinatorMPI<'a, C> {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        default_response: T,
    ) -> Vec<T> {
        let mut response_bytes = vec![];
        let mut response_bytes_buf = construct_partitioned_mut_buffer_for_gather!(
            self.size,
            default_response,
            &mut response_bytes
        );
        // Root does not send anything, it only receives.
        self.root_process
            .gather_varcount_into_root(&[0u8; 0], &mut response_bytes_buf);

        let ret = deserialize_flattened_bytes!(response_bytes, default_response, T).unwrap();
        self.total_recv_bytes += response_bytes.len();
        ret
    }

    fn broadcast_request<T: CanonicalSerialize + CanonicalDeserialize + Clone>(&mut self, data: T) {
        let requests_chunked = vec![data; (1 << self.log_num_workers_per_party) * 3];
        let mut request_bytes = vec![];
        let request_bytes_buf =
            construct_partitioned_buffer_for_scatter!(requests_chunked, &mut request_bytes);

        let counts = request_bytes_buf.counts().to_vec();
        self.root_process.scatter_into_root(&counts, &mut 0i32);
        let mut _recv_buf: Vec<u8> = vec![];
        self.root_process
            .scatter_varcount_into_root(&request_bytes_buf, &mut _recv_buf);
        self.total_send_bytes += request_bytes.len();
    }

    fn send_requests<T: CanonicalSerialize + CanonicalDeserialize + Clone>(
        &mut self,
        data: Vec<T>,
    ) {
        let mut request_bytes = vec![];
        let request_bytes_buf = construct_partitioned_buffer_for_scatter!(data, &mut request_bytes);

        let counts = request_bytes_buf.counts().to_vec();
        self.root_process.scatter_into_root(&counts, &mut 0i32);
        let mut _recv_buf: Vec<u8> = vec![];
        self.root_process
            .scatter_varcount_into_root(&request_bytes_buf, &mut _recv_buf);
        self.total_send_bytes += request_bytes.len();
    }

    fn log_num_workers_per_party(&self) -> usize {
        self.log_num_workers_per_party
    }

    fn log_num_pub_workers(&self) -> usize {
        self.log_num_public_workers
    }

    fn total_bandwidth_used(&self) -> (usize, usize) {
        (self.total_send_bytes, self.total_recv_bytes)
    }
}

pub struct Rep3WorkerMPI<'a, C: 'a + Communicator> {
    pub root_process: Process<'a, C>,
    pub log: &'a mut Vec<String>,
    pub log_num_workers_per_party: usize,
    pub log_num_public_workers: usize,
    pub size: Count,
    pub rank: usize,
    pub total_send_bytes: usize,
    pub total_recv_bytes: usize,
}

impl<'a, C: 'a + Communicator> Rep3WorkerMPI<'a, C> {
    pub fn new(
        root_process: Process<'a, C>,
        log: &'a mut Vec<String>,
        log_num_workers_per_party: usize,
        log_num_public_workers: usize,
        size: Count,
        rank: usize,
    ) -> Self {
        Self {
            root_process,
            log,
            log_num_workers_per_party,
            log_num_public_workers,
            size,
            rank,
            total_send_bytes: 0,
            total_recv_bytes: 0,
        }
    }
}

impl<'a, C: 'a + Communicator> NetworkWorker for Rep3WorkerMPI<'a, C> {
    fn send_response<T: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: T) {
        let responses_bytes = serialize_to_vec(&data);
        self.root_process.gather_varcount_into(&responses_bytes);
        self.total_send_bytes += responses_bytes.len();
    }

    fn receive_request<T: CanonicalSerialize + CanonicalDeserialize>(&mut self) -> T {
        let mut size = 0 as Count;
        self.root_process.scatter_into(&mut size);

        let mut request_bytes = vec![0u8; size as usize];
        self.root_process.scatter_varcount_into(&mut request_bytes);

        let ret = T::deserialize_uncompressed_unchecked(&request_bytes[..]).unwrap();
        self.total_recv_bytes += request_bytes.len();
        ret
    }

    fn log_num_pub_workers(&self) -> usize {
        self.log_num_public_workers
    }

    fn log_num_workers_per_party(&self) -> usize {
        self.log_num_workers_per_party
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn total_bandwidth_used(&self) -> (usize, usize) {
        (self.total_send_bytes, self.total_recv_bytes)
    }
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
