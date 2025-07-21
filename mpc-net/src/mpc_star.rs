use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use mpc_types::protocols::rep3::id::PartyID;
use crate::Result;

pub trait MpcStarNetCoordinator {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        default_response: T,
    ) -> Result<Vec<T>>;
    fn receive_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
        default_response: T,
    ) -> Result<T>;
    fn broadcast_request<T: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: T) -> Result<()>;
    fn send_requests<T: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: Vec<T>) -> Result<()>;

    fn log_num_pub_workers(&self) -> usize;
    fn log_num_workers_per_party(&self) -> usize;
    fn total_bandwidth_used(&self) -> (u64, u64);
}

pub trait MpcStarNetWorker {
    fn send_response<T: CanonicalSerialize + CanonicalDeserialize>(&mut self, data: T) -> Result<()>;
    fn receive_request<T: CanonicalSerialize + CanonicalDeserialize>(&mut self) -> Result<T>;

    fn log_num_pub_workers(&self) -> usize;
    fn log_num_workers_per_party(&self) -> usize;
    // fn rank(&self) -> usize;

    fn total_bandwidth_used(&self) -> (u64, u64);

    fn party_id(&self) -> PartyID;
}
