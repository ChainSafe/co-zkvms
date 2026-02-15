pub mod format;

use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::PartyID;
use mpc_core::protocols::rep3_ring::casts::downcast;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
use mpc_core::protocols::rep3_ring::{self, Rep3RingShare};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use tracer::instruction::format::NormalizedOperands;
use tracer::instruction::{Cycle, Instruction, RISCVCycle, RISCVInstruction};

// Import all instruction types for Rep3Cycle enum
use tracer::instruction::add::ADD;
use tracer::instruction::addi::ADDI;
use tracer::instruction::addiw::ADDIW;
use tracer::instruction::addw::ADDW;
use tracer::instruction::amoaddd::AMOADDD;
use tracer::instruction::amoaddw::AMOADDW;
use tracer::instruction::amoandd::AMOANDD;
use tracer::instruction::amoandw::AMOANDW;
use tracer::instruction::amomaxd::AMOMAXD;
use tracer::instruction::amomaxud::AMOMAXUD;
use tracer::instruction::amomaxuw::AMOMAXUW;
use tracer::instruction::amomaxw::AMOMAXW;
use tracer::instruction::amomind::AMOMIND;
use tracer::instruction::amominud::AMOMINUD;
use tracer::instruction::amominuw::AMOMINUW;
use tracer::instruction::amominw::AMOMINW;
use tracer::instruction::amoord::AMOORD;
use tracer::instruction::amoorw::AMOORW;
use tracer::instruction::amoswapd::AMOSWAPD;
use tracer::instruction::amoswapw::AMOSWAPW;
use tracer::instruction::amoxord::AMOXORD;
use tracer::instruction::amoxorw::AMOXORW;
use tracer::instruction::and::AND;
use tracer::instruction::andi::ANDI;
use tracer::instruction::andn::ANDN;
use tracer::instruction::auipc::AUIPC;
use tracer::instruction::beq::BEQ;
use tracer::instruction::bge::BGE;
use tracer::instruction::bgeu::BGEU;
use tracer::instruction::blt::BLT;
use tracer::instruction::bltu::BLTU;
use tracer::instruction::bne::BNE;
use tracer::instruction::div::DIV;
use tracer::instruction::divu::DIVU;
use tracer::instruction::divuw::DIVUW;
use tracer::instruction::divw::DIVW;
use tracer::instruction::ecall::ECALL;
use tracer::instruction::fence::FENCE;
use tracer::instruction::inline::INLINE;
use tracer::instruction::jal::JAL;
use tracer::instruction::jalr::JALR;
use tracer::instruction::lb::LB;
use tracer::instruction::lbu::LBU;
use tracer::instruction::ld::LD;
use tracer::instruction::lh::LH;
use tracer::instruction::lhu::LHU;
use tracer::instruction::lrd::LRD;
use tracer::instruction::lrw::LRW;
use tracer::instruction::lui::LUI;
use tracer::instruction::lw::LW;
use tracer::instruction::lwu::LWU;
use tracer::instruction::mul::MUL;
use tracer::instruction::mulh::MULH;
use tracer::instruction::mulhsu::MULHSU;
use tracer::instruction::mulhu::MULHU;
use tracer::instruction::mulw::MULW;
use tracer::instruction::or::OR;
use tracer::instruction::ori::ORI;
use tracer::instruction::rem::REM;
use tracer::instruction::remu::REMU;
use tracer::instruction::remuw::REMUW;
use tracer::instruction::remw::REMW;
use tracer::instruction::sb::SB;
use tracer::instruction::scd::SCD;
use tracer::instruction::scw::SCW;
use tracer::instruction::sd::SD;
use tracer::instruction::sh::SH;
use tracer::instruction::sll::SLL;
use tracer::instruction::slli::SLLI;
use tracer::instruction::slliw::SLLIW;
use tracer::instruction::sllw::SLLW;
use tracer::instruction::slt::SLT;
use tracer::instruction::slti::SLTI;
use tracer::instruction::sltiu::SLTIU;
use tracer::instruction::sltu::SLTU;
use tracer::instruction::sra::SRA;
use tracer::instruction::srai::SRAI;
use tracer::instruction::sraiw::SRAIW;
use tracer::instruction::sraw::SRAW;
use tracer::instruction::srl::SRL;
use tracer::instruction::srli::SRLI;
use tracer::instruction::srliw::SRLIW;
use tracer::instruction::srlw::SRLW;
use tracer::instruction::sub::SUB;
use tracer::instruction::subw::SUBW;
use tracer::instruction::sw::SW;
use tracer::instruction::virtual_advice::VirtualAdvice;
use tracer::instruction::virtual_assert_eq::VirtualAssertEQ;
use tracer::instruction::virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment;
use tracer::instruction::virtual_assert_lte::VirtualAssertLTE;
use tracer::instruction::virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow;
use tracer::instruction::virtual_assert_valid_div0::VirtualAssertValidDiv0;
use tracer::instruction::virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder;
use tracer::instruction::virtual_assert_word_alignment::VirtualAssertWordAlignment;
use tracer::instruction::virtual_change_divisor::VirtualChangeDivisor;
use tracer::instruction::virtual_change_divisor_w::VirtualChangeDivisorW;
use tracer::instruction::virtual_lw::VirtualLW;
use tracer::instruction::virtual_move::VirtualMove;
use tracer::instruction::virtual_movsign::VirtualMovsign;
use tracer::instruction::virtual_muli::VirtualMULI;
use tracer::instruction::virtual_pow2::VirtualPow2;
use tracer::instruction::virtual_pow2_w::VirtualPow2W;
use tracer::instruction::virtual_pow2i::VirtualPow2I;
use tracer::instruction::virtual_pow2i_w::VirtualPow2IW;
use tracer::instruction::virtual_rev8w::VirtualRev8W;
use tracer::instruction::virtual_rotri::VirtualROTRI;
use tracer::instruction::virtual_rotriw::VirtualROTRIW;
use tracer::instruction::virtual_shift_right_bitmask::VirtualShiftRightBitmask;
use tracer::instruction::virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI;
use tracer::instruction::virtual_sign_extend_word::VirtualSignExtendWord;
use tracer::instruction::virtual_sra::VirtualSRA;
use tracer::instruction::virtual_srai::VirtualSRAI;
use tracer::instruction::virtual_srl::VirtualSRL;
use tracer::instruction::virtual_srli::VirtualSRLI;
use tracer::instruction::virtual_sw::VirtualSW;
use tracer::instruction::virtual_xor_rot::{
    VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63,
};
use tracer::instruction::virtual_xor_rotw::{
    VirtualXORROTW12, VirtualXORROTW16, VirtualXORROTW7, VirtualXORROTW8,
};
use tracer::instruction::virtual_zero_extend_word::VirtualZeroExtendWord;
use tracer::instruction::xor::XOR;
use tracer::instruction::xori::XORI;

use self::format::{Rep3InstructionFormat, Rep3RegisterState};

// ── Rep3Operand ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Rep3Operand {
    Shared {
        binary: Rep3RingShare<u32>,
        arithmetic: Option<Rep3RingShare<u128>>,
        public: Option<u64>, // Some for trivial shares
    },
    Public(u64),
}

impl Rep3Operand {
    pub fn from_binary(share: Rep3RingShare<u32>) -> Self {
        Rep3Operand::Shared {
            binary: share,
            arithmetic: None,
            public: None,
        }
    }

    pub fn from_arithmetic(binary: Rep3RingShare<u32>, arithmetic: Rep3RingShare<u128>) -> Self {
        Rep3Operand::Shared {
            binary,
            arithmetic: Some(arithmetic),
            public: None,
        }
    }

    pub fn as_public(&self) -> u64 {
        match self {
            Rep3Operand::Public(x)
            | Rep3Operand::Shared {
                public: Some(x), ..
            } => *x,
            _ => panic!("Not a public operand"),
        }
    }

    pub fn as_arithmetic<T: IntRing2k>(&self) -> Rep3RingShare<T>
    where
        u128: AsPrimitive<T>,
    {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u32(&self) -> Rep3RingShare<u32> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u64(&self) -> Rep3RingShare<u64> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_arithmetic_u128(&self) -> Rep3RingShare<u128> {
        match self {
            Rep3Operand::Shared { arithmetic, .. } => downcast(arithmetic.unwrap()),
            _ => panic!("Not an arithmetic operand"),
        }
    }

    pub fn as_binary(&self) -> Rep3RingShare<u32> {
        match self {
            Rep3Operand::Shared { binary, .. } => binary.clone(),
            _ => panic!("Not a binary operand"),
        }
    }

    pub fn as_binary_or_trivial(&self, id: PartyID) -> Rep3RingShare<u32> {
        match *self {
            Rep3Operand::Shared { binary, .. } => binary,
            Rep3Operand::Public(value) => {
                rep3_ring::binary::promote_to_trivial_share(id, &(value as u32).into())
            }
        }
    }
}

/// Static zero operand for returning references from formats without certain registers.
pub static PUBLIC_ZERO: Rep3Operand = Rep3Operand::Public(0);

impl Default for Rep3Operand {
    fn default() -> Self {
        Rep3Operand::Public(0)
    }
}

impl From<u64> for Rep3Operand {
    fn from(value: u64) -> Self {
        Rep3Operand::Public(value)
    }
}

impl From<u32> for Rep3Operand {
    fn from(value: u32) -> Self {
        Rep3Operand::Public(value as u64)
    }
}

impl From<Rep3Operand> for u64 {
    fn from(value: Rep3Operand) -> u64 {
        match value {
            Rep3Operand::Public(x) => x,
            _ => panic!("Cannot convert Rep3Operand to u64"),
        }
    }
}

impl From<Rep3Operand> for u32 {
    fn from(value: Rep3Operand) -> u32 {
        match value {
            Rep3Operand::Public(x) => x as u32,
            _ => panic!("Cannot convert Rep3Operand to u32"),
        }
    }
}

/// Convert a public Rep3Operand to a shared operand with trivial shares.
pub fn promote_operand_to_share(operand: &Rep3Operand, party_id: PartyID) -> Rep3Operand {
    match operand {
        Rep3Operand::Public(x) => Rep3Operand::Shared {
            binary: rep3_ring::binary::promote_to_trivial_share(party_id, &RingElement(*x as u32)),
            arithmetic: Some(rep3_ring::arithmetic::promote_to_trivial_share(
                party_id,
                RingElement(*x as u128),
            )),
            public: Some(*x),
        },
        already_shared @ Rep3Operand::Shared { .. } => already_shared.clone(),
    }
}

// ── Rep3RISCVCycle ──────────────────────────────────────────────────────────

/// Shorthand: the Rep3RegisterState type for an instruction T
pub type Rep3RegState<T> =
    <<T as RISCVInstruction>::Format as Rep3InstructionFormat>::Rep3RegisterState;

/// Rep3 version of RISCVCycle.
/// Register state type derived from instruction's Format (same pattern as vanilla).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize, T::RAMAccess: Serialize, Rep3RegState<T>: Serialize",
    deserialize = "T: Deserialize<'de>, T::RAMAccess: Deserialize<'de>, Rep3RegState<T>: Deserialize<'de>"
))]
pub struct Rep3RISCVCycle<T: RISCVInstruction>
where
    T::Format: Rep3InstructionFormat,
{
    pub instruction: T,
    pub register_state: Rep3RegState<T>,
    pub ram_access: T::RAMAccess,
}

impl<T: RISCVInstruction> Rep3RISCVCycle<T>
where
    T::Format: Rep3InstructionFormat,
{
    /// Convert from vanilla RISCVCycle with public values
    pub fn from_public_cycle(cycle: &RISCVCycle<T>) -> Self {
        Self {
            instruction: cycle.instruction,
            register_state: Rep3RegState::<T>::from_public(&cycle.register_state),
            ram_access: cycle.ram_access,
        }
    }

    /// Promote all public operands to trivial shares
    pub fn promote_to_shares(&mut self, party_id: PartyID) {
        self.register_state.promote_to_shares(party_id);
    }

    /// Populate arithmetic representations via network
    pub fn populate_arithmetic<N: Rep3Network>(
        &mut self,
        io_ctx: &mut IoContext<N>,
    ) -> std::io::Result<()> {
        self.register_state.populate_arithmetic(io_ctx)
    }
}

/// Convert vanilla trace to Rep3 trace (public values)
pub fn convert_public_trace_to_rep3<T: RISCVInstruction>(
    public_trace: &[RISCVCycle<T>],
) -> Vec<Rep3RISCVCycle<T>>
where
    T::Format: Rep3InstructionFormat,
{
    public_trace
        .iter()
        .map(|cycle| Rep3RISCVCycle::from_public_cycle(cycle))
        .collect()
}

/// Promote all trace operands to trivial shares (parallel)
pub fn promote_trace_to_shares<T: RISCVInstruction + Send + Sync>(
    trace: &mut [Rep3RISCVCycle<T>],
    party_id: PartyID,
) where
    T::Format: Rep3InstructionFormat,
    T::RAMAccess: Send + Sync,
    Rep3RegState<T>: Send + Sync,
{
    use rayon::prelude::*;
    trace
        .par_iter_mut()
        .for_each(|step: &mut Rep3RISCVCycle<T>| {
            step.promote_to_shares(party_id);
        });
}

/// Populate arithmetic representations for all trace operands (sequential - needs network)
pub fn populate_trace_arithmetic<T: RISCVInstruction, N: Rep3Network>(
    trace: &mut [Rep3RISCVCycle<T>],
    io_ctx: &mut IoContext<N>,
) -> std::io::Result<()>
where
    T::Format: Rep3InstructionFormat,
{
    for step in trace.iter_mut() {
        step.populate_arithmetic(io_ctx)?;
    }
    Ok(())
}

// ── Rep3LookupQuery ─────────────────────────────────────────────────────────

/// Rep3 version of LookupQuery trait.
/// Returns shared operands instead of plaintext values.
pub trait Rep3LookupQuery<const XLEN: usize> {
    fn to_instruction_inputs_rep3(&self) -> (Rep3Operand, Rep3Operand);
    fn to_lookup_index_rep3(&self) -> Rep3RingShare<u128> {
        todo!("to_lookup_index_rep3 deferred to Lasso/Shout phase")
    }
    fn to_lookup_output_rep3(&self) -> Rep3Operand {
        todo!("to_lookup_output_rep3 deferred to Lasso/Shout phase")
    }
}

// ── Rep3Cycle enum ──────────────────────────────────────────────────────────

/// Macro to generate the Rep3Cycle enum and its method implementations.
/// Mirrors vanilla `define_rv32im_enums!` but uses `Rep3RISCVCycle<T>` instead of `RISCVCycle<T>`.
macro_rules! define_rep3_cycle {
    (
        instructions: [$($instr:ident),* $(,)?]
    ) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub enum Rep3Cycle {
            NoOp,
            $(
                $instr(Rep3RISCVCycle<$instr>),
            )*
            INLINE(Rep3RISCVCycle<INLINE>),
        }

        impl Rep3Cycle {
            /// Get the instruction encoding (public).
            pub fn instruction(&self) -> Instruction {
                match self {
                    Rep3Cycle::NoOp => Instruction::NoOp,
                    $(
                        Rep3Cycle::$instr(cycle) => cycle.instruction.into(),
                    )*
                    Rep3Cycle::INLINE(cycle) => cycle.instruction.into(),
                }
            }

            /// Get rs1 register read: (register_index, shared_value).
            /// Register index is public, value is shared.
            pub fn rs1_read(&self) -> (u8, &Rep3Operand) {
                match self {
                    Rep3Cycle::NoOp => (0, &PUBLIC_ZERO),
                    $(
                        Rep3Cycle::$instr(cycle) => (
                            NormalizedOperands::from(cycle.instruction.operands).rs1,
                            cycle.register_state.rs1_operand(),
                        ),
                    )*
                    Rep3Cycle::INLINE(cycle) => (
                        cycle.instruction.operands.rs1,
                        cycle.register_state.rs1_operand(),
                    ),
                }
            }

            /// Get rs2 register read: (register_index, shared_value).
            pub fn rs2_read(&self) -> (u8, &Rep3Operand) {
                match self {
                    Rep3Cycle::NoOp => (0, &PUBLIC_ZERO),
                    $(
                        Rep3Cycle::$instr(cycle) => (
                            NormalizedOperands::from(cycle.instruction.operands).rs2,
                            cycle.register_state.rs2_operand(),
                        ),
                    )*
                    Rep3Cycle::INLINE(cycle) => (
                        cycle.instruction.operands.rs2,
                        cycle.register_state.rs2_operand(),
                    ),
                }
            }

            /// Get rd register write: (register_index, pre_value, post_value).
            /// Register index is public, pre/post values are shared.
            pub fn rd_write(&self) -> (u8, &Rep3Operand, &Rep3Operand) {
                match self {
                    Rep3Cycle::NoOp => (0, &PUBLIC_ZERO, &PUBLIC_ZERO),
                    $(
                        Rep3Cycle::$instr(cycle) => {
                            let (pre, post) = cycle.register_state.rd_operands();
                            (
                                NormalizedOperands::from(cycle.instruction.operands).rd,
                                pre,
                                post,
                            )
                        }
                    )*
                    Rep3Cycle::INLINE(cycle) => {
                        let (pre, post) = cycle.register_state.rd_operands();
                        (cycle.instruction.operands.rs3, pre, post)
                    }
                }
            }

            /// Convert from vanilla Cycle (all values become public Rep3Operands).
            pub fn from_public_cycle(cycle: &Cycle) -> Self {
                match cycle {
                    Cycle::NoOp => Rep3Cycle::NoOp,
                    $(
                        Cycle::$instr(c) => Rep3Cycle::$instr(Rep3RISCVCycle::from_public_cycle(c)),
                    )*
                    Cycle::INLINE(c) => Rep3Cycle::INLINE(Rep3RISCVCycle::from_public_cycle(c)),
                }
            }

            /// Promote all public operands to trivial shares.
            pub fn promote_to_shares(&mut self, party_id: PartyID) {
                match self {
                    Rep3Cycle::NoOp => {}
                    $(
                        Rep3Cycle::$instr(cycle) => cycle.promote_to_shares(party_id),
                    )*
                    Rep3Cycle::INLINE(cycle) => cycle.promote_to_shares(party_id),
                }
            }

            /// Populate arithmetic representations via network.
            pub fn populate_arithmetic<N: Rep3Network>(
                &mut self,
                io_ctx: &mut IoContext<N>,
            ) -> std::io::Result<()> {
                match self {
                    Rep3Cycle::NoOp => Ok(()),
                    $(
                        Rep3Cycle::$instr(cycle) => cycle.populate_arithmetic(io_ctx),
                    )*
                    Rep3Cycle::INLINE(cycle) => cycle.populate_arithmetic(io_ctx),
                }
            }
        }
    };
}

define_rep3_cycle! {
    instructions: [
        ADD, ADDI, AND, ANDI, ANDN, AUIPC, BEQ, BGE, BGEU, BLT, BLTU, BNE, DIV, DIVU,
        ECALL, FENCE, JAL, JALR, LB, LBU, LD, LH, LHU, LUI, LW, MUL, MULH, MULHSU,
        MULHU, OR, ORI, REM, REMU, SB, SD, SH, SLL, SLLI, SLT, SLTI, SLTIU, SLTU,
        SRA, SRAI, SRL, SRLI, SUB, SW, XOR, XORI,
        // RV64I
        ADDIW, SLLIW, SRLIW, SRAIW, ADDW, SUBW, SLLW, SRLW, SRAW, LWU,
        // RV64M
        DIVUW, DIVW, MULW, REMUW, REMW,
        // RV32A
        LRW, SCW, AMOSWAPW, AMOADDW, AMOANDW, AMOORW, AMOXORW, AMOMINW, AMOMAXW, AMOMINUW, AMOMAXUW,
        // RV64A
        LRD, SCD, AMOSWAPD, AMOADDD, AMOANDD, AMOORD, AMOXORD, AMOMIND, AMOMAXD, AMOMINUD, AMOMAXUD,
        // Virtual
        VirtualAdvice, VirtualAssertEQ, VirtualAssertHalfwordAlignment, VirtualAssertWordAlignment,
        VirtualAssertLTE, VirtualAssertValidDiv0, VirtualAssertValidUnsignedRemainder,
        VirtualAssertMulUNoOverflow, VirtualChangeDivisor, VirtualChangeDivisorW,
        VirtualLW, VirtualSW, VirtualZeroExtendWord, VirtualSignExtendWord,
        VirtualPow2W, VirtualPow2IW, VirtualMove, VirtualMovsign, VirtualMULI,
        VirtualPow2, VirtualPow2I, VirtualRev8W, VirtualROTRI, VirtualROTRIW,
        VirtualShiftRightBitmask, VirtualShiftRightBitmaskI,
        VirtualSRA, VirtualSRAI, VirtualSRL, VirtualSRLI,
        VirtualXORROT32, VirtualXORROT24, VirtualXORROT16, VirtualXORROT63,
        VirtualXORROTW16, VirtualXORROTW12, VirtualXORROTW8, VirtualXORROTW7,
    ]
}

/// Convert a full vanilla trace to Rep3 (public values).
pub fn convert_trace_to_rep3(trace: &[Cycle]) -> Vec<Rep3Cycle> {
    trace.iter().map(Rep3Cycle::from_public_cycle).collect()
}

/// Promote all trace operands to trivial shares.
pub fn promote_rep3_trace_to_shares(trace: &mut [Rep3Cycle], party_id: PartyID) {
    for cycle in trace.iter_mut() {
        cycle.promote_to_shares(party_id);
    }
}

/// Populate arithmetic representations for all trace operands.
pub fn populate_rep3_trace_arithmetic<N: Rep3Network>(
    trace: &mut [Rep3Cycle],
    io_ctx: &mut IoContext<N>,
) -> std::io::Result<()> {
    for cycle in trace.iter_mut() {
        cycle.populate_arithmetic(io_ctx)?;
    }
    Ok(())
}
