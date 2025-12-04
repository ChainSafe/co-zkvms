use std::collections::BTreeMap;

use crate::Result;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use mpc_types::protocols::rep3::id::PartyID;
use serde::{de::DeserializeOwned, Serialize};

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
    fn receive_response_from_workers<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
    ) -> Result<Vec<T>>;
    fn broadcast_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: T,
    ) -> Result<()>;
    fn send_requests<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> Result<()>;

    fn send_requests_to_workers<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> Result<()>;

    fn send_requests_blocking<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: Vec<T>,
    ) -> Result<()>;

    fn send_request<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        party_id: PartyID,
        worker_id: usize,
        data: T,
    ) -> Result<()>;

    fn log_num_workers(&self) -> usize;
    fn total_bandwidth_used(&self) -> (u64, u64);
    fn is_distributed(&self) -> bool {
        self.log_num_workers() > 0
    }

    /// Print the connection stats of the network
    fn log_connection_stats(&self, label: Option<&str>);
    fn reset_stats(&mut self);

    fn fork(&mut self) -> Result<Self>;
    fn set_worker_subnets(&mut self, num_workers: usize);
    fn reset_worker_subnets(&mut self);
}

pub trait MpcStarNetWorker: Sized + Clone {
    fn send_response<T: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: T,
    ) -> Result<()>;
    fn receive_request<T: CanonicalSerialize + CanonicalDeserialize>(&mut self) -> Result<T>;

    fn log_num_workers(&self) -> usize;
    fn is_distributed(&self) -> bool {
        self.log_num_workers() > 0
    }

    fn io_stats_total(&self) -> (u64, u64);
    fn io_stats_per_party(&self) -> BTreeMap<usize, (u64, u64)>;

    fn party_id(&self) -> PartyID;
    fn worker_idx(&self) -> usize;

    fn fork(&self) -> Self;
    fn fork_with_coordinator(&mut self) -> Result<Self>;
    fn get_worker_subnets(&self, num_workers: usize) -> Result<Vec<Self>>;
}

pub trait MpcRingNetWorkerExt: Sized + Clone {
    fn send_next_link<T: Serialize + DeserializeOwned>(&mut self, data: T) -> Result<()>;
    fn send_prev_link<T: Serialize + DeserializeOwned>(&mut self, data: T) -> Result<()>;
    fn resv_prev_link<T: Serialize + DeserializeOwned>(&mut self) -> Result<T>;
    fn resv_next_link<T: Serialize + DeserializeOwned>(&mut self) -> Result<T>;
}
