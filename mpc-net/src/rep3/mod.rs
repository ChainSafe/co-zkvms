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

    pub fn from_global_worker_id(global_worker_id: usize) -> Self {
        let party_id = global_worker_id % 3;
        let worker_id = global_worker_id / 3;
        Self::new(party_id, worker_id)
    }
}

#[test]
fn test_global_worker_idn() {
    assert_eq!(PartyWorkerID::new(0, 0).global_worker_id(), 0);
    assert_eq!(PartyWorkerID::new(1, 0).global_worker_id(), 1);
    assert_eq!(PartyWorkerID::new(2, 0).global_worker_id(), 2);
    assert_eq!(PartyWorkerID::new(0, 1).global_worker_id(), 3);
    assert_eq!(PartyWorkerID::new(1, 1).global_worker_id(), 4);
    assert_eq!(PartyWorkerID::new(2, 1).global_worker_id(), 5);

    assert_eq!(
        PartyWorkerID::from_global_worker_id(0),
        PartyWorkerID::new(0, 0)
    );
    assert_eq!(
        PartyWorkerID::from_global_worker_id(1),
        PartyWorkerID::new(1, 0)
    );
    assert_eq!(
        PartyWorkerID::from_global_worker_id(2),
        PartyWorkerID::new(2, 0)
    );
    assert_eq!(
        PartyWorkerID::from_global_worker_id(3),
        PartyWorkerID::new(0, 1)
    );
    assert_eq!(
        PartyWorkerID::from_global_worker_id(4),
        PartyWorkerID::new(1, 1)
    );
    assert_eq!(
        PartyWorkerID::from_global_worker_id(5),
        PartyWorkerID::new(2, 1)
    );
}
