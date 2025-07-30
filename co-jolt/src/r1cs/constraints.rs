use jolt_common::{constants::REGISTER_COUNT, rv_trace::CircuitFlags};
use strum::IntoEnumIterator;

use jolt_core::{
    field::JoltField,
    r1cs::{
        builder::{CombinedUniformBuilder, OffsetEqConstraint, R1CSBuilder},
        constraints::{R1CSConstraints, LOG_M, OPERAND_SIZE, PC_NOOP_SHIFT, PC_START_ADDRESS},
        inputs::{AuxVariable, ConstraintInput},
        ops::Variable,
    },
};

use crate::{
    jolt::{
        instruction::{
            add::ADDInstruction,
            mul::MULInstruction,
            mulhu::MULHUInstruction,
            mulu::MULUInstruction,
            sll::SLLInstruction,
            sra::SRAInstruction,
            srl::SRLInstruction,
            sub::SUBInstruction,
            virtual_assert_halfword_alignment::AssertHalfwordAlignmentInstruction,
            virtual_move::MOVEInstruction,
            virtual_movsign::MOVSIGNInstruction, // mul::MULInstruction, mulhu::MULHUInstruction, mulu::MULUInstruction,
                                                 // virtual_assert_halfword_alignment::AssertHalfwordAlignmentInstruction,
                                                 // virtual_move::MOVEInstruction, virtual_movsign::MOVSIGNInstruction,
        },
        vm::rv32i_vm::RV32I,
    },
    r1cs::inputs::JoltR1CSInputs,
};

pub struct JoltRV32IMConstraints;
impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for JoltRV32IMConstraints {
    type Inputs = JoltR1CSInputs<F>;

    fn uniform_constraints(cs: &mut R1CSBuilder<C, F, Self::Inputs>, memory_start: u64) {
        for flag in RV32I::<F>::iter() {
            cs.constrain_binary(Self::Inputs::InstructionFlags(flag));
        }
        for flag in CircuitFlags::iter() {
            cs.constrain_binary(JoltR1CSInputs::<F>::OpFlags(flag));
        }

        let flags = CircuitFlags::iter()
            .map(|flag| JoltR1CSInputs::<F>::OpFlags(flag).into())
            .chain(
                RV32I::<F>::iter().map(|flag| JoltR1CSInputs::<F>::InstructionFlags(flag).into()),
            )
            .collect();
        cs.constrain_pack_be(flags, JoltR1CSInputs::<F>::Bytecode_Bitflags, 1);

        let real_pc =
            4i64 * JoltR1CSInputs::<F>::Bytecode_ELFAddress + (PC_START_ADDRESS - PC_NOOP_SHIFT);
        let x = cs.allocate_if_else(
            JoltR1CSInputs::<F>::Aux(AuxVariable::LeftLookupOperand),
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::LeftOperandIsPC),
            real_pc,
            JoltR1CSInputs::<F>::RS1_Read,
        );
        let y = cs.allocate_if_else(
            JoltR1CSInputs::<F>::Aux(AuxVariable::RightLookupOperand),
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::RightOperandIsImm),
            JoltR1CSInputs::<F>::Bytecode_Imm,
            JoltR1CSInputs::<F>::RS2_Read,
        );

        let is_load_or_store = JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Load)
            + JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Store);
        let memory_start: i64 = memory_start.try_into().unwrap();
        cs.constrain_eq_conditional(
            is_load_or_store,
            JoltR1CSInputs::<F>::RS1_Read + JoltR1CSInputs::<F>::Bytecode_Imm,
            4 * JoltR1CSInputs::<F>::RAM_Address + memory_start - 4 * REGISTER_COUNT as i64,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::<F>::RAM_Read,
            JoltR1CSInputs::<F>::RAM_Write,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Load),
            JoltR1CSInputs::<F>::RAM_Read,
            JoltR1CSInputs::<F>::RD_Write,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Store),
            JoltR1CSInputs::<F>::RS2_Read,
            JoltR1CSInputs::<F>::RAM_Write,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Lui),
            JoltR1CSInputs::<F>::RD_Write,
            JoltR1CSInputs::<F>::Bytecode_Imm,
        );

        let query_chunks: Vec<Variable> = (0..C)
            .map(|i| Variable::Input(JoltR1CSInputs::<F>::ChunksQuery(i).to_index::<C>()))
            .collect();
        let packed_query =
            R1CSBuilder::<C, F, JoltR1CSInputs<F>>::pack_be(query_chunks.clone(), LOG_M);

        // For the `AssertHalfwordAlignmentInstruction` lookups, we add the `rs1` and `imm` values
        // to obtain the memory address being accessed.
        let add_operands = JoltR1CSInputs::<F>::InstructionFlags(ADDInstruction::default().into())
            + JoltR1CSInputs::InstructionFlags(
                AssertHalfwordAlignmentInstruction::<32, F>::default().into(),
            );
        cs.constrain_eq_conditional(add_operands, packed_query.clone(), x + y);
        // Converts from unsigned to twos-complement representation
        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::InstructionFlags(SUBInstruction::default().into()),
            packed_query.clone(),
            x - y + (0xffffffffi64 + 1),
        );
        let is_mul = JoltR1CSInputs::<F>::InstructionFlags(MULInstruction::default().into())
            + JoltR1CSInputs::<F>::InstructionFlags(MULUInstruction::default().into())
            + JoltR1CSInputs::<F>::InstructionFlags(MULHUInstruction::default().into());
        let product = cs.allocate_prod(
            JoltR1CSInputs::<F>::Aux(AuxVariable::Product),
            JoltR1CSInputs::<F>::RS1_Read,
            JoltR1CSInputs::<F>::RS2_Read,
        );
        cs.constrain_eq_conditional(is_mul, packed_query.clone(), product);
        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::InstructionFlags(MOVSIGNInstruction::default().into())
                + JoltR1CSInputs::<F>::InstructionFlags(MOVEInstruction::default().into()),
            packed_query.clone(),
            x,
        );

        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Assert),
            JoltR1CSInputs::<F>::LookupOutput,
            1,
        );

        let x_chunks: Vec<Variable> = (0..C)
            .map(|i| Variable::Input(JoltR1CSInputs::<F>::ChunksX(i).to_index::<C>()))
            .collect();
        let y_chunks: Vec<Variable> = (0..C)
            .map(|i| Variable::Input(JoltR1CSInputs::<F>::ChunksY(i).to_index::<C>()))
            .collect();
        let x_concat =
            R1CSBuilder::<C, F, JoltR1CSInputs<F>>::pack_be(x_chunks.clone(), OPERAND_SIZE);
        let y_concat =
            R1CSBuilder::<C, F, JoltR1CSInputs<F>>::pack_be(y_chunks.clone(), OPERAND_SIZE);
        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::ConcatLookupQueryChunks),
            x_concat,
            x,
        );
        cs.constrain_eq_conditional(
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::ConcatLookupQueryChunks),
            y_concat,
            y,
        );

        // if is_shift ? chunks_query[i] == zip(chunks_x[i], chunks_y[C-1]) : chunks_query[i] == zip(chunks_x[i], chunks_y[i])
        let is_shift = JoltR1CSInputs::<F>::InstructionFlags(SLLInstruction::default().into())
            + JoltR1CSInputs::<F>::InstructionFlags(SRLInstruction::default().into())
            + JoltR1CSInputs::<F>::InstructionFlags(SRAInstruction::default().into());
        for i in 0..C {
            let relevant_chunk_y = cs.allocate_if_else(
                JoltR1CSInputs::<F>::Aux(AuxVariable::RelevantYChunk(i)),
                is_shift.clone(),
                y_chunks[C - 1],
                y_chunks[i],
            );
            cs.constrain_eq_conditional(
                JoltR1CSInputs::<F>::OpFlags(CircuitFlags::ConcatLookupQueryChunks),
                query_chunks[i],
                x_chunks[i] * (1i64 << 8) + relevant_chunk_y,
            );
        }

        // if (rd != 0 && update_rd_with_lookup_output == 1) constrain(rd_val == LookupOutput)
        let rd_nonzero_and_lookup_to_rd = cs.allocate_prod(
            JoltR1CSInputs::<F>::Aux(AuxVariable::WriteLookupOutputToRD),
            JoltR1CSInputs::<F>::Bytecode_RD,
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::WriteLookupOutputToRD),
        );
        cs.constrain_eq_conditional(
            rd_nonzero_and_lookup_to_rd,
            JoltR1CSInputs::<F>::RD_Write,
            JoltR1CSInputs::<F>::LookupOutput,
        );
        // if (rd != 0 && is_jump_instr == 1) constrain(rd_val == 4 * PC)
        let rd_nonzero_and_jmp = cs.allocate_prod(
            JoltR1CSInputs::<F>::Aux(AuxVariable::WritePCtoRD),
            JoltR1CSInputs::<F>::Bytecode_RD,
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Jump),
        );
        cs.constrain_eq_conditional(
            rd_nonzero_and_jmp,
            4 * JoltR1CSInputs::<F>::Bytecode_ELFAddress + PC_START_ADDRESS,
            JoltR1CSInputs::<F>::RD_Write,
        );

        let next_pc_jump = cs.allocate_if_else(
            JoltR1CSInputs::<F>::Aux(AuxVariable::NextPCJump),
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Jump),
            JoltR1CSInputs::<F>::LookupOutput + 4,
            4 * JoltR1CSInputs::<F>::Bytecode_ELFAddress + PC_START_ADDRESS + 4
                - 4 * JoltR1CSInputs::<F>::OpFlags(CircuitFlags::DoNotUpdatePC),
        );

        let should_branch = cs.allocate_prod(
            JoltR1CSInputs::<F>::Aux(AuxVariable::ShouldBranch),
            JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Branch),
            JoltR1CSInputs::<F>::LookupOutput,
        );
        let _next_pc = cs.allocate_if_else(
            JoltR1CSInputs::<F>::Aux(AuxVariable::NextPC),
            should_branch,
            4 * JoltR1CSInputs::<F>::Bytecode_ELFAddress
                + PC_START_ADDRESS
                + JoltR1CSInputs::<F>::Bytecode_Imm,
            next_pc_jump,
        );
    }

    fn cross_step_constraints() -> Vec<OffsetEqConstraint> {
        // If the next instruction's ELF address is not zero (i.e. it's
        // not padding), then check the PC update.
        let pc_constraint = OffsetEqConstraint::new(
            (JoltR1CSInputs::<F>::Bytecode_ELFAddress, true),
            (JoltR1CSInputs::<F>::Aux(AuxVariable::NextPC), false),
            (
                4 * JoltR1CSInputs::<F>::Bytecode_ELFAddress + PC_START_ADDRESS,
                true,
            ),
        );

        // If the current instruction is virtual, check that the next instruction
        // in the trace is the next instruction in bytecode. Virtual sequences
        // do not involve jumps or branches, so this should always hold,
        // EXCEPT if we encounter a virtual instruction followed by a padding
        // instruction. But that should never happen because the execution
        // trace should always end with some return handling, which shouldn't involve
        // any virtual sequences.
        let virtual_sequence_constraint = OffsetEqConstraint::new(
            (JoltR1CSInputs::<F>::OpFlags(CircuitFlags::Virtual), false),
            (JoltR1CSInputs::<F>::Bytecode_A, true),
            (JoltR1CSInputs::<F>::Bytecode_A + 1, false),
        );

        vec![pc_constraint, virtual_sequence_constraint]
        // vec![]
    }
}
