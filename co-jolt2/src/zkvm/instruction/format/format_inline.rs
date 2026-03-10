use serde::{Deserialize, Serialize};
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::format::InstructionRegisterState;

use super::{Rep3InstructionFormat, Rep3RegisterState};
use crate::zkvm::instruction::{Rep3Operand, PUBLIC_ZERO};

/// Rep3 register state for Inline-format instructions.
/// Three source registers (rs1, rs2, rs3), no register writes.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rep3RegisterStateFormatInline {
    pub rs1: Rep3Operand,
    pub rs2: Rep3Operand,
    pub rs3: Rep3Operand,
}

impl Rep3RegisterState for Rep3RegisterStateFormatInline {
    fn rs1_operand(&self) -> Rep3Operand {
        self.rs1
    }

    fn rs2_operand(&self) -> Rep3Operand {
        self.rs2
    }

    fn rd_operands(&self) -> (Rep3Operand, Rep3Operand) {
        (PUBLIC_ZERO, PUBLIC_ZERO)
    }

    fn from_public<T: InstructionRegisterState>(public_state: &T) -> Self {
        Self {
            rs1: Rep3Operand::Public(public_state.rs1_value()),
            rs2: Rep3Operand::Public(public_state.rs2_value()),
            // rs3 is not exposed by InstructionRegisterState trait, default to 0
            rs3: Rep3Operand::Public(0),
        }
    }

    fn from_shared<T: InstructionRegisterState>(
        _public_state: &T,
        shares: &mut impl Iterator<Item = Rep3Operand>,
    ) -> Self {
        Self {
            rs1: shares.next().unwrap(),
            rs2: shares.next().unwrap(),
            rs3: shares.next().unwrap(),
        }
    }

    fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand> {
        vec![&mut self.rs1, &mut self.rs2, &mut self.rs3]
    }

    fn operand_values<T: InstructionRegisterState>(state: &T) -> Vec<u64> {
        // rs3 is not exposed by InstructionRegisterState trait, default to 0
        vec![state.rs1_value(), state.rs2_value(), 0]
    }
}

impl Rep3InstructionFormat for FormatInline {
    type Rep3RegisterState = Rep3RegisterStateFormatInline;
}
