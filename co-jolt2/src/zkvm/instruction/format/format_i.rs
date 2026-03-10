use serde::{Deserialize, Serialize};
use tracer::instruction::format::format_i::FormatI;
use tracer::instruction::format::InstructionRegisterState;

use super::{Rep3InstructionFormat, Rep3RegisterState};
use crate::zkvm::instruction::{Rep3Operand, PUBLIC_ZERO};

/// Rep3 register state for I-format instructions (ADDI, ANDI, ORI, JALR, etc.)
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rep3RegisterStateFormatI {
    pub rd: (Rep3Operand, Rep3Operand),
    pub rs1: Rep3Operand,
}

impl Rep3RegisterState for Rep3RegisterStateFormatI {
    fn rs1_operand(&self) -> Rep3Operand {
        self.rs1
    }

    fn rs2_operand(&self) -> Rep3Operand {
        PUBLIC_ZERO
    }

    fn rd_operands(&self) -> (Rep3Operand, Rep3Operand) {
        (self.rd.0, self.rd.1)
    }

    fn from_public<T: InstructionRegisterState>(public_state: &T) -> Self {
        let (old, new) = public_state.rd_values();
        Self {
            rs1: Rep3Operand::Public(public_state.rs1_value()),
            rd: (Rep3Operand::Public(old), Rep3Operand::Public(new)),
        }
    }

    fn from_shared<T: InstructionRegisterState>(
        _public_state: &T,
        shares: &mut impl Iterator<Item = Rep3Operand>,
    ) -> Self {
        Self {
            rs1: shares.next().unwrap(),
            rd: (shares.next().unwrap(), shares.next().unwrap()),
        }
    }

    fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand> {
        vec![&mut self.rs1, &mut self.rd.0, &mut self.rd.1]
    }

    fn operand_values<T: InstructionRegisterState>(state: &T) -> Vec<u64> {
        let (old, new) = state.rd_values();
        vec![state.rs1_value(), old, new]
    }
}

impl Rep3InstructionFormat for FormatI {
    type Rep3RegisterState = Rep3RegisterStateFormatI;
}
