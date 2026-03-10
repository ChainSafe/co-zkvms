use serde::{Deserialize, Serialize};
use tracer::instruction::format::format_assert_align::AssertAlignFormat;
use tracer::instruction::format::InstructionRegisterState;

use super::{Rep3InstructionFormat, Rep3RegisterState};
use crate::zkvm::instruction::{Rep3Operand, PUBLIC_ZERO};

/// Rep3 register state for AssertAlign-format instructions.
/// Only rs1, no rd or rs2.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rep3AssertAlignRegisterState {
    pub rs1: Rep3Operand,
}

impl Rep3RegisterState for Rep3AssertAlignRegisterState {
    fn rs1_operand(&self) -> Rep3Operand {
        self.rs1
    }

    fn rs2_operand(&self) -> Rep3Operand {
        PUBLIC_ZERO
    }

    fn rd_operands(&self) -> (Rep3Operand, Rep3Operand) {
        (PUBLIC_ZERO, PUBLIC_ZERO)
    }

    fn from_public<T: InstructionRegisterState>(public_state: &T) -> Self {
        Self {
            rs1: Rep3Operand::Public(public_state.rs1_value()),
        }
    }

    fn from_shared<T: InstructionRegisterState>(
        _public_state: &T,
        shares: &mut impl Iterator<Item = Rep3Operand>,
    ) -> Self {
        Self {
            rs1: shares.next().unwrap(),
        }
    }

    fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand> {
        vec![&mut self.rs1]
    }

    fn operand_values<T: InstructionRegisterState>(state: &T) -> Vec<u64> {
        vec![state.rs1_value()]
    }
}

impl Rep3InstructionFormat for AssertAlignFormat {
    type Rep3RegisterState = Rep3AssertAlignRegisterState;
}
