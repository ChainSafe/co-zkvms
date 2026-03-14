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

use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
use tracer::instruction::format::{InstructionFormat, InstructionRegisterState};

use crate::zkvm::instruction::Rep3Operand;

/// Trait for Rep3 register state types holding Rep3Operand values.
/// Mirrors vanilla `InstructionRegisterState` but with `Rep3Operand` instead of `u64`.
pub trait Rep3RegisterState: Default + Clone + Serialize + DeserializeOwned + Debug + Send + Sync {
    /// Get source register 1 operand
    fn rs1_operand(&self) -> Rep3Operand;

    /// Get source register 2 operand
    fn rs2_operand(&self) -> Rep3Operand;

    /// Get destination register operands (old_value, new_value)
    fn rd_operands(&self) -> (Rep3Operand, Rep3Operand);

    /// Convert from vanilla RegisterState (u64 -> Rep3Operand::Public)
    fn from_public<T: InstructionRegisterState>(public_state: &T) -> Self;

    /// Convert from vanilla RegisterState using pre-generated binary shares.
    /// Consumes operands from `shares` in the same order as `shared_operands_mut`.
    fn from_shared<T: InstructionRegisterState>(
        public_state: &T,
        shares: &mut impl Iterator<Item = Rep3Operand>,
    ) -> Self;

    /// Returns mutable references to all shared operand fields.
    /// Used by batched `populate_operands_casts` to collect binary shares
    /// across the entire trace in one pass.
    fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand>;

    /// Visit all shared operand fields without allocating an intermediate vec.
    fn for_each_shared_operand_mut(&mut self, f: impl FnMut(&mut Rep3Operand));

    /// Extract operand values from a vanilla register state in the same order
    /// as `shared_operands_mut` returns them. Used by `share_cycle` to know
    /// which values to generate binary shares for.
    fn operand_values<T: InstructionRegisterState>(state: &T) -> Vec<u64>;
}

/// Maps vanilla InstructionFormat to its Rep3 register state equivalent.
/// One impl per format type (not per instruction).
pub trait Rep3InstructionFormat: InstructionFormat {
    type Rep3RegisterState: Rep3RegisterState;
}
