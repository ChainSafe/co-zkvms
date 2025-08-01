use crate::Result;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use mpc_types::protocols::rep3::id::PartyID;

pub trait MpcStarNetCoordinator: Sized {
    fn receive_responses<T: CanonicalSerialize + CanonicalDeserialize>(&mut self)
        -> Result<Vec<T>>;
    fn receive_responses_from_subnets<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
    ) -> Result<Vec<Vec<T>>>;
    fn receive_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
    ) -> Result<T>;
    fn broadcast_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: T,
    ) -> Result<()>;
    fn send_requests<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> Result<()>;

    fn send_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
        data: T,
    ) -> Result<()>;

    fn log_num_pub_workers(&self) -> usize;
    fn log_num_workers_per_party(&self) -> usize;
    fn total_bandwidth_used(&self) -> (u64, u64);

    /// Print the connection stats of the network
    fn log_connection_stats(&self, label: Option<&str>);
    fn reset_stats(&mut self);

    fn fork(&mut self) -> Result<Self>;
    fn extend_with_worker_subnets(&mut self, num_workers: usize) -> Result<()>;
    fn trim_subnets(&mut self, num_workers: usize) -> Result<()>;
}

pub trait MpcStarNetWorker: Sized {
    fn send_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: T,
    ) -> Result<()>;
    fn receive_request<T: CanonicalSerialize + CanonicalDeserialize>(&mut self) -> Result<T>;

    fn log_num_pub_workers(&self) -> usize;
    fn log_num_workers_per_party(&self) -> usize;
    // fn rank(&self) -> usize;

    fn total_bandwidth_used(&self) -> (u64, u64);

    fn party_id(&self) -> PartyID;
    fn worker_idx(&self) -> usize;

    fn fork_with_coordinator(&mut self) -> Result<Self>;
    fn get_worker_subnets(&self, num_workers: usize) -> Result<Vec<Self>>;
}
