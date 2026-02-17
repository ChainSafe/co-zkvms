use mpc_core::protocols::rep3::PartyID;
use serde::{Deserialize, Serialize};
use tracer::instruction::{RAMAccess, RAMRead, RAMWrite};

use super::rep3_operand::{promote_operand_to_share, Rep3Operand};

/// Rep3 version of vanilla `RAMRead`. Address is public, value is shared.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rep3RAMRead {
    pub address: u64,
    pub value: Rep3Operand,
}

/// Rep3 version of vanilla `RAMWrite`. Address is public, pre/post values are shared.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rep3RAMWrite {
    pub address: u64,
    pub pre_value: Rep3Operand,
    pub post_value: Rep3Operand,
}

/// Rep3 version of vanilla `RAMAccess`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Rep3RAMAccess {
    Read(Rep3RAMRead),
    Write(Rep3RAMWrite),
    NoOp,
}

/// Static NoOp for returning references from Rep3Cycle::ram_access() on NoOp variants.
pub static REP3_RAM_NOOP: Rep3RAMAccess = Rep3RAMAccess::NoOp;

impl Rep3RAMAccess {
    pub fn address(&self) -> u64 {
        match self {
            Rep3RAMAccess::Read(read) => read.address,
            Rep3RAMAccess::Write(write) => write.address,
            Rep3RAMAccess::NoOp => 0,
        }
    }

    pub fn promote_to_shares(&mut self, party_id: PartyID) {
        match self {
            Rep3RAMAccess::Read(read) => {
                read.value = promote_operand_to_share(&read.value, party_id);
            }
            Rep3RAMAccess::Write(write) => {
                write.pre_value = promote_operand_to_share(&write.pre_value, party_id);
                write.post_value = promote_operand_to_share(&write.post_value, party_id);
            }
            Rep3RAMAccess::NoOp => {}
        }
    }

    pub fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand> {
        match self {
            Rep3RAMAccess::Read(read) => vec![&mut read.value],
            Rep3RAMAccess::Write(write) => vec![&mut write.pre_value, &mut write.post_value],
            Rep3RAMAccess::NoOp => vec![],
        }
    }

    /// Extract operand values from vanilla RAMAccess in the same order as `shared_operands_mut`.
    pub fn operand_values(access: &RAMAccess) -> Vec<u64> {
        match access {
            RAMAccess::Read(read) => vec![read.value],
            RAMAccess::Write(write) => vec![write.pre_value, write.post_value],
            RAMAccess::NoOp => vec![],
        }
    }

    /// Build from vanilla RAMAccess using pre-generated shares.
    /// Consumes operands from `shares` in the same order as `shared_operands_mut`.
    pub fn from_shared(access: RAMAccess, shares: &mut impl Iterator<Item = Rep3Operand>) -> Self {
        match access {
            RAMAccess::Read(read) => Rep3RAMAccess::Read(Rep3RAMRead {
                address: read.address,
                value: shares.next().unwrap(),
            }),
            RAMAccess::Write(write) => Rep3RAMAccess::Write(Rep3RAMWrite {
                address: write.address,
                pre_value: shares.next().unwrap(),
                post_value: shares.next().unwrap(),
            }),
            RAMAccess::NoOp => Rep3RAMAccess::NoOp,
        }
    }
}

impl From<RAMRead> for Rep3RAMAccess {
    fn from(read: RAMRead) -> Self {
        Rep3RAMAccess::Read(Rep3RAMRead {
            address: read.address,
            value: Rep3Operand::Public(read.value),
        })
    }
}

impl From<RAMWrite> for Rep3RAMAccess {
    fn from(write: RAMWrite) -> Self {
        Rep3RAMAccess::Write(Rep3RAMWrite {
            address: write.address,
            pre_value: Rep3Operand::Public(write.pre_value),
            post_value: Rep3Operand::Public(write.post_value),
        })
    }
}

impl From<()> for Rep3RAMAccess {
    fn from(_: ()) -> Self {
        Rep3RAMAccess::NoOp
    }
}

impl From<RAMAccess> for Rep3RAMAccess {
    fn from(access: RAMAccess) -> Self {
        match access {
            RAMAccess::Read(read) => read.into(),
            RAMAccess::Write(write) => write.into(),
            RAMAccess::NoOp => Rep3RAMAccess::NoOp,
        }
    }
}
