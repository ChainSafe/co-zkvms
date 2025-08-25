use crate::field::JoltField;
use crate::jolt::vm::worker::JoltRep3Prover;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::inputs::JoltR1CSInputs;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use enum_dispatch::enum_dispatch;
use jolt_core::utils::transcript::Transcript;
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;
use strum_macros::{AsRefStr, EnumCount, EnumIter};

use crate::jolt::instruction::{
    add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
    bgeu::BGEUInstruction, bne::BNEInstruction, mul::MULInstruction, mulhu::MULHUInstruction,
    mulu::MULUInstruction, or::ORInstruction, sll::SLLInstruction, slt::SLTInstruction,
    sltu::SLTUInstruction, sra::SRAInstruction, srl::SRLInstruction, sub::SUBInstruction,
    virtual_advice::ADVICEInstruction,
    virtual_assert_halfword_alignment::AssertHalfwordAlignmentInstruction,
    virtual_assert_lte::ASSERTLTEInstruction,
    virtual_assert_valid_div0::AssertValidDiv0Instruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
    virtual_assert_valid_unsigned_remainder::AssertValidUnsignedRemainderInstruction,
    virtual_move::MOVEInstruction, virtual_movsign::MOVSIGNInstruction,
    virtual_pow2::POW2Instruction, virtual_right_shift_padding::RightShiftPaddingInstruction,
    xor::XORInstruction, JoltInstruction, JoltInstructionSet, Rep3JoltInstruction,
    Rep3JoltInstructionSet, Rep3Operand, SubtableIndices,
};
use crate::jolt::vm::{Jolt, JoltProof};
use crate::r1cs::constraints::JoltRV32IMConstraints;
use crate::utils::future::FutureVal;
use jolt_core::jolt::subtable::LassoSubtable;
use jolt_core::jolt::vm::rv32i_vm::RV32ISubtables;
use paste::paste;

use mpc_core::protocols::rep3::{
    network::{IoContext, Rep3Network},
    Rep3BigUintShare, Rep3PrimeFieldShare,
};

const WORD_SIZE: usize = 32;

crate::instruction_set!(
  RV32I,
  ADD: ADDInstruction<WORD_SIZE, F>,
  SUB: SUBInstruction<WORD_SIZE, F>,
  AND: ANDInstruction<F>,
  OR: ORInstruction<F>,
  XOR: XORInstruction<F>,
  BEQ: BEQInstruction<F>,
  BGE: BGEInstruction<F>,
  BGEU: BGEUInstruction<F>,
  BNE: BNEInstruction<F>,
  SLT: SLTInstruction<F>,
  SLTU: SLTUInstruction<F>,
  SLL: SLLInstruction<WORD_SIZE, F>,
  SRA: SRAInstruction<WORD_SIZE, F>,
  SRL: SRLInstruction<WORD_SIZE, F>,
  MOVSIGN: MOVSIGNInstruction<WORD_SIZE, F>,
  MUL: MULInstruction<WORD_SIZE, F>,
  MULU: MULUInstruction<WORD_SIZE, F>,
  MULHU: MULHUInstruction<WORD_SIZE, F>,
  VIRTUAL_ADVICE: ADVICEInstruction<WORD_SIZE, F>,
  VIRTUAL_MOVE: MOVEInstruction<WORD_SIZE, F>,
  VIRTUAL_ASSERT_LTE: ASSERTLTEInstruction<WORD_SIZE, F>,
  VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER: AssertValidSignedRemainderInstruction<WORD_SIZE, F>,
  VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER: AssertValidUnsignedRemainderInstruction<WORD_SIZE, F>,
  VIRTUAL_ASSERT_VALID_DIV0: AssertValidDiv0Instruction<WORD_SIZE, F>,
  VIRTUAL_ASSERT_HALFWORD_ALIGNMENT: AssertHalfwordAlignmentInstruction<WORD_SIZE, F>,
  VIRTUAL_POW2: POW2Instruction<WORD_SIZE, F>,
  VIRTUAL_SRA_PADDING: RightShiftPaddingInstruction<WORD_SIZE, F>
);

// ==================== JOLT ====================

pub enum RV32IJoltVM {}

pub const C: usize = 4;
pub const M: usize = 1 << 16;

impl<F, PCS, ProofTranscript> Jolt<F, PCS, C, M, ProofTranscript> for RV32IJoltVM
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
{
    type InstructionSet = RV32I<F>;
    type Subtables = RV32ISubtables<F>;

    type Constraints = JoltRV32IMConstraints;
}

pub type RV32IJoltProof<F, PCS, ProofTranscript> =
    JoltProof<C, M, JoltR1CSInputs<F>, F, PCS, RV32I<F>, RV32ISubtables<F>, ProofTranscript>;

pub type RV32IJoltRep3Prover<F, PCS, ProofTranscript, Network> = JoltRep3Prover<
    F,
    C,
    M,
    RV32I<F>,
    RV32ISubtables<F>,
    JoltRV32IMConstraints,
    PCS,
    ProofTranscript,
    Network,
>;

impl<F: JoltField> CanonicalSerialize for RV32I<F> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (JoltInstructionSet::enum_index(self) as u8).serialize_with_mode(&mut writer, compress)?;
        match self {
            RV32I::ADD(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::SUB(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::AND(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::OR(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::XOR(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::BEQ(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::BGE(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::BGEU(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::BNE(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::SLT(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::SLTU(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::SLL(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::SRA(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::SRL(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::MOVSIGN(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::MUL(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::MULU(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::MULHU(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::VIRTUAL_ADVICE(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::VIRTUAL_MOVE(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::VIRTUAL_ASSERT_LTE(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER(op) => {
                op.serialize_with_mode(&mut writer, compress)?
            }
            RV32I::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER(op) => {
                op.serialize_with_mode(&mut writer, compress)?
            }
            RV32I::VIRTUAL_ASSERT_VALID_DIV0(op) => {
                op.serialize_with_mode(&mut writer, compress)?
            }
            RV32I::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT(op) => {
                op.serialize_with_mode(&mut writer, compress)?
            }
            RV32I::VIRTUAL_POW2(op) => op.serialize_with_mode(&mut writer, compress)?,
            RV32I::VIRTUAL_SRA_PADDING(op) => op.serialize_with_mode(&mut writer, compress)?,
        };
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let size = match self {
            RV32I::ADD(op) => op.serialized_size(compress),
            RV32I::SUB(op) => op.serialized_size(compress),
            RV32I::AND(op) => op.serialized_size(compress),
            RV32I::OR(op) => op.serialized_size(compress),
            RV32I::XOR(op) => op.serialized_size(compress),
            RV32I::BEQ(op) => op.serialized_size(compress),
            RV32I::BGE(op) => op.serialized_size(compress),
            RV32I::BGEU(op) => op.serialized_size(compress),
            RV32I::BNE(op) => op.serialized_size(compress),
            RV32I::SLT(op) => op.serialized_size(compress),
            RV32I::SLTU(op) => op.serialized_size(compress),
            RV32I::SLL(op) => op.serialized_size(compress),
            RV32I::SRA(op) => op.serialized_size(compress),
            RV32I::SRL(op) => op.serialized_size(compress),
            RV32I::MOVSIGN(op) => op.serialized_size(compress),
            RV32I::MUL(op) => op.serialized_size(compress),
            RV32I::MULU(op) => op.serialized_size(compress),
            RV32I::MULHU(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_ADVICE(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_MOVE(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_ASSERT_LTE(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_ASSERT_VALID_DIV0(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_POW2(op) => op.serialized_size(compress),
            RV32I::VIRTUAL_SRA_PADDING(op) => op.serialized_size(compress),
        };
        JoltInstructionSet::enum_index(self).serialized_size(compress) + size
    }
}

impl<F: JoltField> CanonicalDeserialize for RV32I<F> {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        // TODO: Can we use strum for this?
        let discriminant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let res = match discriminant {
            0 => RV32I::ADD(ADDInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            1 => RV32I::SUB(SUBInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            2 => RV32I::AND(ANDInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            3 => RV32I::OR(ORInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            4 => RV32I::XOR(XORInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            5 => RV32I::BEQ(BEQInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            6 => RV32I::BGE(BGEInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            7 => RV32I::BGEU(BGEUInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            8 => RV32I::BNE(BNEInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            9 => RV32I::SLT(SLTInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            10 => RV32I::SLTU(SLTUInstruction::<F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            11 => RV32I::SLL(SLLInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            12 => RV32I::SRA(SRAInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            13 => RV32I::SRL(SRLInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            14 => RV32I::MOVSIGN(MOVSIGNInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            15 => RV32I::MUL(MULInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            16 => RV32I::MULU(MULUInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            17 => RV32I::MULHU(MULHUInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            18 => RV32I::VIRTUAL_ADVICE(ADVICEInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            19 => RV32I::VIRTUAL_MOVE(MOVEInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            20 => RV32I::VIRTUAL_ASSERT_LTE(
                ASSERTLTEInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                    reader, compress, validate,
                )?,
            ),
            21 => RV32I::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER(
                AssertValidSignedRemainderInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                    reader, compress, validate,
                )?,
            ),
            22 => RV32I::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER(
                AssertValidUnsignedRemainderInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                    reader, compress, validate,
                )?,
            ),
            23 => RV32I::VIRTUAL_ASSERT_VALID_DIV0(
                AssertValidDiv0Instruction::<WORD_SIZE, F>::deserialize_with_mode(
                    reader, compress, validate,
                )?,
            ),
            24 => RV32I::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT(AssertHalfwordAlignmentInstruction::<
                WORD_SIZE,
                F,
            >::deserialize_with_mode(
                reader, compress, validate
            )?),
            25 => RV32I::VIRTUAL_POW2(POW2Instruction::<WORD_SIZE, F>::deserialize_with_mode(
                reader, compress, validate,
            )?),
            26 => RV32I::VIRTUAL_SRA_PADDING(
                RightShiftPaddingInstruction::<WORD_SIZE, F>::deserialize_with_mode(
                    reader, compress, validate,
                )?,
            ),
            _ => Err(SerializationError::InvalidData)?,
        };
        Ok(res)
    }
}

impl<F: JoltField> Valid for RV32I<F> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            RV32I::ADD(instruction) => instruction.check(),
            RV32I::SUB(instruction) => instruction.check(),
            RV32I::AND(instruction) => instruction.check(),
            RV32I::OR(instruction) => instruction.check(),
            RV32I::XOR(instruction) => instruction.check(),
            RV32I::BEQ(instruction) => instruction.check(),
            RV32I::BGE(instruction) => instruction.check(),
            RV32I::BGEU(instruction) => instruction.check(),
            RV32I::BNE(instruction) => instruction.check(),
            RV32I::SLT(instruction) => instruction.check(),
            RV32I::SLTU(instruction) => instruction.check(),
            RV32I::SLL(instruction) => instruction.check(),
            RV32I::SRA(instruction) => instruction.check(),
            RV32I::SRL(instruction) => instruction.check(),
            RV32I::MOVSIGN(instruction) => instruction.check(),
            RV32I::MUL(instruction) => instruction.check(),
            RV32I::MULU(instruction) => instruction.check(),
            RV32I::MULHU(instruction) => instruction.check(),
            RV32I::VIRTUAL_ADVICE(instruction) => instruction.check(),
            RV32I::VIRTUAL_MOVE(instruction) => instruction.check(),
            RV32I::VIRTUAL_ASSERT_LTE(instruction) => instruction.check(),
            RV32I::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER(instruction) => instruction.check(),
            RV32I::VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER(instruction) => instruction.check(),
            RV32I::VIRTUAL_ASSERT_VALID_DIV0(instruction) => instruction.check(),
            RV32I::VIRTUAL_ASSERT_HALFWORD_ALIGNMENT(instruction) => instruction.check(),
            RV32I::VIRTUAL_POW2(instruction) => instruction.check(),
            RV32I::VIRTUAL_SRA_PADDING(instruction) => instruction.check(),
        }
    }
}
