use jolt_common::constants::MEMORY_OPS_PER_INSTRUCTION;
use jolt_tracer::RVTraceRow;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::utils::types::Either;

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryOp {
    Read(u64),                                   // (address)
    Write(u64, Either<u64, Rep3RingShare<u64>>), // (address, new_value)
}

impl MemoryOp {
    pub fn generate_shares_rep3<R: RngCore>(op: Self, rng: &mut R) -> Vec<Self> {
        match op {
            MemoryOp::Read(addr) => vec![MemoryOp::Read(addr); 3],
            MemoryOp::Write(addr, Either::Public(new_val)) => {
                rep3_ring::binary::generate_shares_rep3(new_val.into(), rng)
                    .into_iter()
                    .map(|share| MemoryOp::Write(addr, Either::Shared(share)))
                    .collect()
            }
            _ => unreachable!(),
        }
    }

    pub fn from_trace_row(row: &RVTraceRow) -> [MemoryOp; MEMORY_OPS_PER_INSTRUCTION] {
        let native: [jolt_common::rv_trace::MemoryOp; MEMORY_OPS_PER_INSTRUCTION] = row.into();
        native.map(|op| op.into())
    }

    pub fn noop_read() -> Self {
        Self::Read(0)
    }

    pub fn noop_write() -> Self {
        Self::Write(0, 0.into())
    }
}

impl Default for MemoryOp {
    fn default() -> Self {
        Self::noop_read()
    }
}

impl From<jolt_common::rv_trace::MemoryOp> for MemoryOp {
    fn from(row: jolt_common::rv_trace::MemoryOp) -> Self {
        match row {
            jolt_common::rv_trace::MemoryOp::Read(addr) => MemoryOp::Read(addr),
            jolt_common::rv_trace::MemoryOp::Write(addr, new_val) => {
                MemoryOp::Write(addr, Either::Public(new_val))
            }
        }
    }
}

impl Into<jolt_common::rv_trace::MemoryOp> for MemoryOp {
    fn into(self) -> jolt_common::rv_trace::MemoryOp {
        match self {
            MemoryOp::Read(addr) => jolt_common::rv_trace::MemoryOp::Read(addr),
            MemoryOp::Write(addr, Either::Public(new_val)) => {
                jolt_common::rv_trace::MemoryOp::Write(addr, new_val)
            }
            _ => unreachable!(),
        }
    }
}
