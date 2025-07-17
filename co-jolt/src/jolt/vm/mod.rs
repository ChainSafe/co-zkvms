pub mod bytecode;
pub mod instruction_lookups;
pub mod rv32i_vm;

use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use jolt_common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{MemoryOp, NUM_CIRCUIT_FLAGS},
};

use self::bytecode::BytecodeRow;
use crate::jolt::{instruction::JoltInstructionSet};
use crate::poly::field::JoltField;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltTraceStep<F: JoltField, InstructionSet: JoltInstructionSet<F>> {
    pub instruction_lookup: Option<InstructionSet>,
    pub bytecode_row: BytecodeRow,
    pub memory_ops: [MemoryOp; MEMORY_OPS_PER_INSTRUCTION],
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
    _field: PhantomData<F>,
}

impl<F: JoltField, InstructionSet: JoltInstructionSet<F>> JoltTraceStep<F, InstructionSet> {
    fn no_op() -> Self {
        JoltTraceStep {
            instruction_lookup: None,
            bytecode_row: BytecodeRow::no_op(0),
            memory_ops: [
                MemoryOp::noop_read(),  // rs1
                MemoryOp::noop_read(),  // rs2
                MemoryOp::noop_write(), // rd is write-only
                MemoryOp::noop_read(),  // RAM
            ],
            circuit_flags: [false; NUM_CIRCUIT_FLAGS],
            _field: PhantomData,
        }
    }

    fn pad(trace: &mut Vec<Self>) {
        let unpadded_length = trace.len();
        let padded_length = unpadded_length.next_power_of_two();
        trace.resize(padded_length, Self::no_op());
    }
}
