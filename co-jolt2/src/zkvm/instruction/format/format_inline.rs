use mpc_core::protocols::rep3::PartyID;
use serde::{Deserialize, Serialize};
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::format::InstructionRegisterState;

use super::{Rep3InstructionFormat, Rep3RegisterState};
use crate::zkvm::instruction::{promote_operand_to_share, Rep3Operand, PUBLIC_ZERO};

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

    fn promote_to_shares(&mut self, party_id: PartyID) {
        self.rs1 = promote_operand_to_share(&self.rs1, party_id);
        self.rs2 = promote_operand_to_share(&self.rs2, party_id);
        self.rs3 = promote_operand_to_share(&self.rs3, party_id);
    }

    fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand> {
        vec![&mut self.rs1, &mut self.rs2, &mut self.rs3]
    }
}

impl Rep3InstructionFormat for FormatInline {
    type Rep3RegisterState = Rep3RegisterStateFormatInline;
}
