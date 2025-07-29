use crate::jolt::vm::worker::JoltRep3Prover;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::inputs::JoltR1CSInputs;
use enum_dispatch::enum_dispatch;
use jolt_core::field::JoltField;
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
