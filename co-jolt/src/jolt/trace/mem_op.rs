use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rand::RngCore;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryOp {
    Read(u64),                      // (address)
    Write(u64, Rep3RingShare<u64>), // (address, new_value)
}

impl MemoryOp {
    pub fn generate_shares_rep3<R: RngCore>(
        op: jolt_common::rv_trace::MemoryOp,
        rng: &mut R,
    ) -> Vec<Self> {
        match op {
            jolt_common::rv_trace::MemoryOp::Read(addr) => vec![MemoryOp::Read(addr); 3],
            jolt_common::rv_trace::MemoryOp::Write(addr, new_val) => {
                rep3_ring::binary::generate_shares_rep3(new_val.into(), rng)
                    .into_iter()
                    .map(|share| MemoryOp::Write(addr, share))
                    .collect()
            }
        }
    }
}
