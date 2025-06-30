use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use mpi::{
    topology::Process,
    traits::{Communicator, Root}, Count,
};

use crate::{construct_partitioned_buffer_for_scatter, construct_partitioned_mut_buffer_for_gather, deserialize_flattened_bytes, mpi_utils::scatter_requests};

pub trait NetworkCoordinator {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &self,
        default_response: T,
    ) -> Vec<T>;
    fn broadcast_requests<T: CanonicalSerialize + CanonicalDeserialize + Clone>(&self, data: T);

    fn log_num_pub_workers(&self) -> usize;
    fn log_num_workers_per_party(&self) -> usize;
}

pub trait NetworkWorker {
    fn send_response<T: CanonicalSerialize + CanonicalDeserialize>(&self, data: T);
    fn receive_request<T: CanonicalSerialize + CanonicalDeserialize>(&self) -> T;

    fn log_num_pub_workers(&self) -> usize;
    fn log_num_workers_per_party(&self) -> usize;
    fn rank(&self) -> usize;
}

pub struct Rep3CoordinatorMPI<'a, C: 'a + Communicator> {
    pub root_process: Process<'a, C>,
    pub log: &'a mut Vec<String>,
    pub log_num_workers_per_party: usize,
    pub size: Count,
}

impl<'a, C: 'a + Communicator> Rep3CoordinatorMPI<'a, C> {
    pub fn new(root_process: Process<'a, C>, log: &'a mut Vec<String>, log_num_workers_per_party: usize, size: Count) -> Self {
        Self {
            root_process,
            log,
            log_num_workers_per_party,
            size,
        }
    }
}

impl<'a, C: 'a + Communicator> Rep3CoordinatorMPI<'a, C> {
    pub fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &self,
        default_response: T,
    ) -> Vec<T> {
        let mut response_bytes = vec![];
        let mut response_bytes_buf = construct_partitioned_mut_buffer_for_gather!(
            self.size,
            default_response,
            &mut response_bytes
        );
        // Root does not send anything, it only receives.
        let start = start_timer_buf!(log, || format!("Coord: Gathering {stage} responses"));
        self.root_process.gather_varcount_into_root(&[0u8; 0], &mut response_bytes_buf);
        end_timer_buf!(log, start);

        let start = start_timer_buf!(log, || format!("Coord: Deserializing {stage} responses"));
        let ret = deserialize_flattened_bytes!(response_bytes, default_response, T).unwrap();
        end_timer_buf!(log, start);

        ret
    }

    pub fn broadcast_requests<T: CanonicalSerialize + CanonicalDeserialize + Clone>(&self, data: T) {
        let requests_chunked = vec![data; (1 << self.log_num_workers_per_party) * 3];
        let start = start_timer_buf!(log, || format!("Coord: Serializing {stage} requests"));
        let mut request_bytes = vec![];
        let request_bytes_buf =
            construct_partitioned_buffer_for_scatter!(requests_chunked, &mut request_bytes);
        end_timer_buf!(log, start);

        let counts = request_bytes_buf.counts().to_vec();
        self.root_process.scatter_into_root(&counts, &mut 0i32);
        let mut _recv_buf: Vec<u8> = vec![];

        let start = start_timer_buf!(log, || format!("Coord: Scattering {stage} requests"));
        self.root_process
            .scatter_varcount_into_root(&request_bytes_buf, &mut _recv_buf);
        end_timer_buf!(log, start);
    }

    fn log_num_workers_per_party(&self) -> usize {
        self.log_num_workers_per_party
    }
}
