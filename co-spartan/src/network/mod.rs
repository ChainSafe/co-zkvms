pub mod mpi;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub trait NetworkCoordinator {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &self,
        default_response: T,
    ) -> Vec<T>;
    fn broadcast_request<T: CanonicalSerialize + CanonicalDeserialize + Clone>(&self, data: T);
    fn send_requests<T: CanonicalSerialize + CanonicalDeserialize + Clone>(&self, data: Vec<T>);

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
