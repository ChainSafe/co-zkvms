use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::casts::upcast_many_from_binary;
use serde::{Deserialize, Serialize};
use tracer::instruction::format::format_assert_align::AssertAlignFormat;
use tracer::instruction::format::InstructionRegisterState;

use super::{Rep3InstructionFormat, Rep3RegisterState};
use crate::zkvm::instruction::{promote_operand_to_share, Rep3Operand, PUBLIC_ZERO};

/// Rep3 register state for AssertAlign-format instructions.
/// Only rs1, no rd or rs2.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rep3AssertAlignRegisterState {
    pub rs1: Rep3Operand,
}

impl Rep3RegisterState for Rep3AssertAlignRegisterState {
    fn rs1_operand(&self) -> &Rep3Operand {
        &self.rs1
    }

    fn rs2_operand(&self) -> &Rep3Operand {
        &PUBLIC_ZERO
    }

    fn rd_operands(&self) -> (&Rep3Operand, &Rep3Operand) {
        (&PUBLIC_ZERO, &PUBLIC_ZERO)
    }

    fn from_public<T: InstructionRegisterState>(public_state: &T) -> Self {
        Self {
            rs1: Rep3Operand::Public(public_state.rs1_value()),
        }
    }

    fn promote_to_shares(&mut self, party_id: PartyID) {
        self.rs1 = promote_operand_to_share(&self.rs1, party_id);
    }

    fn populate_arithmetic<N: Rep3Network>(
        &mut self,
        io_ctx: &mut IoContext<N>,
    ) -> std::io::Result<()> {
        let binary_shares = vec![self.rs1.as_binary()];

        let arithmetic_shares: Vec<_> = upcast_many_from_binary(&binary_shares, io_ctx)?;

        self.rs1 = Rep3Operand::from_arithmetic(binary_shares[0], arithmetic_shares[0]);

        Ok(())
    }
}

impl Rep3InstructionFormat for AssertAlignFormat {
    type Rep3RegisterState = Rep3AssertAlignRegisterState;
}
