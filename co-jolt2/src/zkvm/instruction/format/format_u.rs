use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::casts::upcast_many_from_binary;
use serde::{Deserialize, Serialize};
use tracer::instruction::format::format_u::FormatU;
use tracer::instruction::format::InstructionRegisterState;

use super::{Rep3InstructionFormat, Rep3RegisterState};
use crate::zkvm::instruction::{promote_operand_to_share, Rep3Operand, PUBLIC_ZERO};

/// Rep3 register state for U-format instructions (LUI, AUIPC)
/// Only rd (old/new), no source registers.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rep3RegisterStateFormatU {
    pub rd: (Rep3Operand, Rep3Operand),
}

impl Rep3RegisterState for Rep3RegisterStateFormatU {
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

    fn promote_to_shares(&mut self, party_id: PartyID) {
        self.rd.0 = promote_operand_to_share(&self.rd.0, party_id);
        self.rd.1 = promote_operand_to_share(&self.rd.1, party_id);
    }

    fn populate_arithmetic<N: Rep3Network>(
        &mut self,
        io_ctx: &mut IoContext<N>,
    ) -> std::io::Result<()> {
        let binary_shares = vec![self.rd.0.as_binary(), self.rd.1.as_binary()];

        let arithmetic_shares: Vec<_> = upcast_many_from_binary(&binary_shares, io_ctx)?;

        self.rd.0 = Rep3Operand::from_arithmetic(binary_shares[0], arithmetic_shares[0]);
        self.rd.1 = Rep3Operand::from_arithmetic(binary_shares[1], arithmetic_shares[1]);

        Ok(())
    }
}

impl Rep3InstructionFormat for FormatU {
    type Rep3RegisterState = Rep3RegisterStateFormatU;
}
