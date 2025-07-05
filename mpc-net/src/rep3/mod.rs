#[cfg(feature = "mpi")]
pub mod mpi;

#[cfg(feature = "quic")]
pub mod quic;

pub struct PartyWorkerID(usize, usize);

impl PartyWorkerID {
    pub fn new(party: usize, worker: usize) -> Self {
        Self(party, worker)
    }

    pub fn party_id(&self) -> usize {
        self.0
    }

    pub fn worker_idx(&self) -> usize {
        self.1
    }

    pub fn global_worker_id(&self) -> usize {
        self.party_id() * 3 + self.worker_idx()
    }
}
