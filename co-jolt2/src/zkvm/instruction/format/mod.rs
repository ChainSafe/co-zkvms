pub mod format_assert_align;
pub mod format_b;
pub mod format_i;
pub mod format_inline;
pub mod format_j;
pub mod format_load;
pub mod format_r;
pub mod format_s;
pub mod format_u;
pub mod format_virtual_right_shift_i;
pub mod format_virtual_right_shift_r;

use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::PartyID;
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
use tracer::instruction::format::{InstructionFormat, InstructionRegisterState};

use crate::zkvm::instruction::Rep3Operand;

/// Trait for Rep3 register state types holding Rep3Operand values.
/// Mirrors vanilla `InstructionRegisterState` but with `Rep3Operand` instead of `u64`.
pub trait Rep3RegisterState:
    Default + Clone + Serialize + DeserializeOwned + Debug + Send + Sync
{
    /// Get source register 1 operand
    fn rs1_operand(&self) -> Rep3Operand;

    /// Get source register 2 operand
    fn rs2_operand(&self) -> Rep3Operand;

    /// Get destination register operands (old_value, new_value)
    fn rd_operands(&self) -> (Rep3Operand, Rep3Operand);

    /// Convert from vanilla RegisterState (u64 -> Rep3Operand::Public)
    fn from_public<T: InstructionRegisterState>(public_state: &T) -> Self;

    /// Promote public operands to trivial shares
    fn promote_to_shares(&mut self, party_id: PartyID);

    /// Populate arithmetic representations via network
    fn populate_arithmetic<N: Rep3Network>(
        &mut self,
        io_ctx: &mut IoContext<N>,
    ) -> std::io::Result<()>;
}

/// Maps vanilla InstructionFormat to its Rep3 register state equivalent.
/// One impl per format type (not per instruction).
pub trait Rep3InstructionFormat: InstructionFormat {
    type Rep3RegisterState: Rep3RegisterState;
}
