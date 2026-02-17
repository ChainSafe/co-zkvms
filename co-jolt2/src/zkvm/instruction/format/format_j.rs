use mpc_core::protocols::rep3::PartyID;
use serde::{Deserialize, Serialize};
use tracer::instruction::format::format_j::FormatJ;
use tracer::instruction::format::InstructionRegisterState;

use super::{Rep3InstructionFormat, Rep3RegisterState};
use crate::zkvm::instruction::{promote_operand_to_share, Rep3Operand, PUBLIC_ZERO};

/// Rep3 register state for J-format instructions (JAL, VirtualAdvice, etc.)
/// Only rd (old/new), no source registers.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rep3RegisterStateFormatJ {
    pub rd: (Rep3Operand, Rep3Operand),
}

impl Rep3RegisterState for Rep3RegisterStateFormatJ {
    fn rs1_operand(&self) -> Rep3Operand {
        PUBLIC_ZERO
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
            rd: (Rep3Operand::Public(old), Rep3Operand::Public(new)),
        }
    }

    fn from_shared<T: InstructionRegisterState>(
        _public_state: &T,
        shares: &mut impl Iterator<Item = Rep3Operand>,
    ) -> Self {
        Self {
            rd: (shares.next().unwrap(), shares.next().unwrap()),
        }
    }

    fn promote_to_shares(&mut self, party_id: PartyID) {
        self.rd.0 = promote_operand_to_share(&self.rd.0, party_id);
        self.rd.1 = promote_operand_to_share(&self.rd.1, party_id);
    }

    fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand> {
        vec![&mut self.rd.0, &mut self.rd.1]
    }

    fn operand_values<T: InstructionRegisterState>(state: &T) -> Vec<u64> {
        let (old, new) = state.rd_values();
        vec![old, new]
    }
}

impl Rep3InstructionFormat for FormatJ {
    type Rep3RegisterState = Rep3RegisterStateFormatJ;
}
