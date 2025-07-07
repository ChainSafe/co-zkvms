// mod network;

use mpc_types::protocols::rep3::id::PartyID;

#[cfg(feature = "mpi")]
pub mod mpi;

#[cfg(feature = "quic")]
pub mod quic;

pub type WorkerID = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartyWorkerID(PartyID, WorkerID);

impl PartyWorkerID {
    pub fn new(party: usize, worker: usize) -> Self {
        Self(PartyID::try_from(party).unwrap(), worker)
    }

    pub fn party_id(&self) -> PartyID {
        self.0
    }

    pub fn worker_idx(&self) -> WorkerID {
        self.1
    }

    pub fn global_worker_id(&self) -> usize {
        let party_id: usize = self.party_id().into();
        (self.worker_idx() * 3) + party_id
    }
}

