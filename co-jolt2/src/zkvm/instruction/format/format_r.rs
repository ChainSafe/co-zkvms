use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::casts::upcast_many_from_binary;
use serde::{Deserialize, Serialize};
use tracer::instruction::format::format_r::FormatR;
use tracer::instruction::format::InstructionRegisterState;

use super::{Rep3InstructionFormat, Rep3RegisterState};
use crate::zkvm::instruction::{promote_operand_to_share, Rep3Operand};

/// Rep3 register state for R-format instructions (ADD, SUB, AND, OR, etc.)
/// Parallel to vanilla `RegisterStateFormatR` but with Rep3Operand values.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rep3RegisterStateFormatR {
    pub rd: (Rep3Operand, Rep3Operand), // (old_value, new_value)
    pub rs1: Rep3Operand,
    pub rs2: Rep3Operand,
}

impl Rep3RegisterState for Rep3RegisterStateFormatR {
    fn rs1_operand(&self) -> Rep3Operand {
        self.rs1
    }

    fn rs2_operand(&self) -> Rep3Operand {
        self.rs2
    }

    fn rd_operands(&self) -> (Rep3Operand, Rep3Operand) {
        (self.rd.0, self.rd.1)
    }

    fn from_public<T: InstructionRegisterState>(public_state: &T) -> Self {
        let (old, new) = public_state.rd_values();
        Self {
            rs1: Rep3Operand::Public(public_state.rs1_value()),
            rs2: Rep3Operand::Public(public_state.rs2_value()),
            rd: (Rep3Operand::Public(old), Rep3Operand::Public(new)),
        }
    }

    fn promote_to_shares(&mut self, party_id: PartyID) {
        self.rs1 = promote_operand_to_share(&self.rs1, party_id);
        self.rs2 = promote_operand_to_share(&self.rs2, party_id);
        self.rd.0 = promote_operand_to_share(&self.rd.0, party_id);
        self.rd.1 = promote_operand_to_share(&self.rd.1, party_id);
    }

    fn populate_arithmetic<N: Rep3Network>(
        &mut self,
        io_ctx: &mut IoContext<N>,
    ) -> std::io::Result<()> {
        let binary_shares = vec![
            self.rs1.as_binary(),
            self.rs2.as_binary(),
            self.rd.0.as_binary(),
            self.rd.1.as_binary(),
        ];

        let arithmetic_shares: Vec<_> = upcast_many_from_binary(&binary_shares, io_ctx)?;

        self.rs1 = Rep3Operand::from_arithmetic(binary_shares[0], arithmetic_shares[0]);
        self.rs2 = Rep3Operand::from_arithmetic(binary_shares[1], arithmetic_shares[1]);
        self.rd.0 = Rep3Operand::from_arithmetic(binary_shares[2], arithmetic_shares[2]);
        self.rd.1 = Rep3Operand::from_arithmetic(binary_shares[3], arithmetic_shares[3]);

        Ok(())
    }
}

impl Rep3InstructionFormat for FormatR {
    type Rep3RegisterState = Rep3RegisterStateFormatR;
}
