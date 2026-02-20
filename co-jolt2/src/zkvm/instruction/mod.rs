pub mod format;
pub mod types;

pub use types::rep3_operand::{promote_operand_to_share, Rep3Operand, PUBLIC_ZERO};
pub use types::rep3_ram::{Rep3RAMAccess, Rep3RAMRead, Rep3RAMWrite, REP3_RAM_NOOP};

use jolt2_common::constants::XLEN;
use jolt_core::zkvm::instruction::InstructionLookup;
use jolt_core::zkvm::lookup_table::LookupTables;
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::casts::upcast_many_from_binary;
// Re-exported for child instruction modules (used via `use super::*`)
pub use mpc_core::protocols::rep3_ring::casts::downcast;
pub use mpc_core::protocols::rep3_ring::ring::bit::Bit;
pub use mpc_core::protocols::rep3_ring::ring::ring_impl::RingElement;
pub use mpc_core::protocols::rep3_ring::{self as rep3_ring, Rep3RingShare};
use serde::{Deserialize, Serialize};
use tracer::instruction::format::NormalizedOperands;
use tracer::instruction::{Cycle, Instruction, RAMAccess, RISCVCycle, RISCVInstruction};

use crate::field::JoltField;
use crate::utils::future_ring::FutureRep3Ring;
pub use crate::utils::instruction_utils::bit_to_ring32;
pub use crate::utils::instruction_utils::bit_to_ring64;
use crate::utils::instruction_utils::{interleave_bits_shared, operand_to_binary_u128};

use self::format::{Rep3InstructionFormat, Rep3RegisterState};

// ── Rep3RISCVCycle ──────────────────────────────────────────────────────────

/// Shorthand: the Rep3RegisterState type for an instruction T
pub type Rep3RegState<T> =
    <<T as RISCVInstruction>::Format as Rep3InstructionFormat>::Rep3RegisterState;

/// Rep3 version of RISCVCycle.
/// Register state type derived from instruction's Format (same pattern as vanilla).
/// RAM access uses Rep3RAMAccess (shared values, public addresses).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "T: Serialize, Rep3RegState<T>: Serialize",
    deserialize = "T: Deserialize<'de>, Rep3RegState<T>: Deserialize<'de>"
))]
pub struct Rep3RISCVCycle<T: RISCVInstruction>
where
    T::Format: Rep3InstructionFormat,
{
    pub instruction: T,
    pub register_state: Rep3RegState<T>,
    pub ram_access: Rep3RAMAccess,
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
            ram_access: Rep3RAMAccess::from(Into::<RAMAccess>::into(cycle.ram_access)),
        }
    }

    /// Promote all public operands to trivial shares
    pub fn promote_to_shares(&mut self, party_id: PartyID) {
        self.register_state.promote_to_shares(party_id);
        self.ram_access.promote_to_shares(party_id);
    }

    /// Build from vanilla RISCVCycle using pre-generated binary shares.
    /// `shares` must yield operands in the same order as `shared_operands_mut`:
    /// register state operands first, then RAM operands.
    pub fn from_cycle_shared(
        cycle: &RISCVCycle<T>,
        shares: &mut impl Iterator<Item = Rep3Operand>,
    ) -> Self {
        Self {
            instruction: cycle.instruction,
            register_state: Rep3RegState::<T>::from_shared(&cycle.register_state, shares),
            ram_access: Rep3RAMAccess::from_shared(
                Into::<RAMAccess>::into(cycle.ram_access),
                shares,
            ),
        }
    }

    /// Extract operand values from a vanilla RISCVCycle in the same order
    /// as `shared_operands_mut` returns them: register state operands first, then RAM.
    pub fn extract_operand_values(cycle: &RISCVCycle<T>) -> Vec<u64> {
        let mut values = Rep3RegState::<T>::operand_values(&cycle.register_state);
        let ram: RAMAccess = cycle.ram_access.into();
        values.extend(Rep3RAMAccess::operand_values(&ram));
        values
    }

    /// Returns mutable references to all shared operands (register state + RAM).
    pub fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand> {
        let mut ops = self.register_state.shared_operands_mut();
        ops.extend(self.ram_access.shared_operands_mut());
        ops
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
    Rep3RegState<T>: Send + Sync,
{
    use rayon::prelude::*;
    trace
        .par_iter_mut()
        .for_each(|step: &mut Rep3RISCVCycle<T>| {
            step.promote_to_shares(party_id);
        });
}

// ── Rep3LookupQuery ─────────────────────────────────────────────────────────

/// Rep3 version of LookupQuery trait.
/// Returns shared operands instead of plaintext values.
pub trait Rep3LookupQuery<const XLEN: usize> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand);

    /// Returns a FutureRep3Ring representing the lookup index.
    /// - Interleave instructions: Ready(interleaved) — no network needed.
    /// - Add/Sub-index: Pending(RingA2B(sum)) — needs batch A2B.
    /// - Mul-index: Pending(RingMulA2B(a, b)) — needs batch mul + A2B.
    ///
    /// Default: computes interleave from binary operands (Ready, no comms).
    fn to_lookup_index(&self, party_id: PartyID) -> FutureRep3Ring<u128, Rep3RingShare<u128>> {
        let (left, right) = self.to_instruction_inputs();
        let left = operand_to_binary_u128(&left, party_id);
        let right = operand_to_binary_u128(&right, party_id);
        FutureRep3Ring::Ready(interleave_bits_shared(left, right))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()>;
}

// ── Rep3Cycle enum ──────────────────────────────────────────────────────────

// Import all instruction types for Rep3Cycle enum
use tracer::instruction::add::ADD;
use tracer::instruction::addi::ADDI;
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
use tracer::instruction::lui::LUI;
use tracer::instruction::lw::LW;
use tracer::instruction::mul::MUL;
use tracer::instruction::mulh::MULH;
use tracer::instruction::mulhsu::MULHSU;
use tracer::instruction::mulhu::MULHU;
use tracer::instruction::or::OR;
use tracer::instruction::ori::ORI;
use tracer::instruction::rem::REM;
use tracer::instruction::remu::REMU;
use tracer::instruction::sb::SB;
use tracer::instruction::sd::SD;
use tracer::instruction::sh::SH;
use tracer::instruction::sll::SLL;
use tracer::instruction::slli::SLLI;
use tracer::instruction::slt::SLT;
use tracer::instruction::slti::SLTI;
use tracer::instruction::sltiu::SLTIU;
use tracer::instruction::sltu::SLTU;
use tracer::instruction::sra::SRA;
use tracer::instruction::srai::SRAI;
use tracer::instruction::srl::SRL;
use tracer::instruction::srli::SRLI;
use tracer::instruction::sub::SUB;
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
            pub fn rs1_read(&self) -> (u8, Rep3Operand) {
                match self {
                    Rep3Cycle::NoOp => (0, PUBLIC_ZERO),
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
            pub fn rs2_read(&self) -> (u8, Rep3Operand) {
                match self {
                    Rep3Cycle::NoOp => (0, PUBLIC_ZERO),
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
            pub fn rd_write(&self) -> (u8, Rep3Operand, Rep3Operand) {
                match self {
                    Rep3Cycle::NoOp => (0, PUBLIC_ZERO, PUBLIC_ZERO),
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

            /// Get the RAM access for this cycle.
            pub fn ram_access(&self) -> &Rep3RAMAccess {
                match self {
                    Rep3Cycle::NoOp => &REP3_RAM_NOOP,
                    $(
                        Rep3Cycle::$instr(cycle) => &cycle.ram_access,
                    )*
                    Rep3Cycle::INLINE(cycle) => &cycle.ram_access,
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
                    other => panic!("Unsupported instruction for Rep3: {:?}", other.instruction()),
                }
            }

            /// Build from vanilla Cycle using pre-generated binary shares.
            /// `shares` must yield operands in the same order as `extract_operand_values`.
            pub fn from_cycle_shared(
                cycle: &Cycle,
                shares: &mut impl Iterator<Item = Rep3Operand>,
            ) -> Self {
                match cycle {
                    Cycle::NoOp => Rep3Cycle::NoOp,
                    $(
                        Cycle::$instr(c) => Rep3Cycle::$instr(Rep3RISCVCycle::from_cycle_shared(c, shares)),
                    )*
                    Cycle::INLINE(c) => Rep3Cycle::INLINE(Rep3RISCVCycle::from_cycle_shared(c, shares)),
                    other => panic!("Unsupported instruction for Rep3: {:?}", other.instruction()),
                }
            }

            /// Extract operand values from a vanilla Cycle in the same order
            /// as `shared_operands_mut` returns them.
            pub fn extract_operand_values(cycle: &Cycle) -> Vec<u64> {
                match cycle {
                    Cycle::NoOp => vec![],
                    $(
                        Cycle::$instr(c) => Rep3RISCVCycle::extract_operand_values(c),
                    )*
                    Cycle::INLINE(c) => Rep3RISCVCycle::extract_operand_values(c),
                    other => panic!("Unsupported instruction for Rep3: {:?}", other.instruction()),
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

            /// Returns mutable references to all shared operands in this cycle.
            pub fn shared_operands_mut(&mut self) -> Vec<&mut Rep3Operand> {
                match self {
                    Rep3Cycle::NoOp => vec![],
                    $(
                        Rep3Cycle::$instr(cycle) => cycle.shared_operands_mut(),
                    )*
                    Rep3Cycle::INLINE(cycle) => cycle.shared_operands_mut(),
                }
            }

            pub fn get_pc(&self, preprocessing: &BytecodePreprocessing) -> usize {
                if matches!(self, Rep3Cycle::NoOp) {
                    return 0;
                }
                let instr = self.instruction().normalize();
                preprocessing.pc_map
                    .get_pc(instr.address, instr.inline_sequence_remaining.unwrap_or(0))
            }
        }
    };
}

use jolt_core::zkvm::bytecode::BytecodePreprocessing;
define_rep3_cycle! {
    instructions: [
        ADD, ADDI, AND, ANDI, ANDN, AUIPC, BEQ, BGE, BGEU, BLT, BLTU, BNE, DIV, DIVU,
        ECALL, FENCE, JAL, JALR, LB, LBU, LD, LH, LHU, LUI, LW, MUL, MULH, MULHSU,
        MULHU, OR, ORI, REM, REMU, SB, SD, SH, SLL, SLLI, SLT, SLTI, SLTIU, SLTU,
        SRA, SRAI, SRL, SRLI, SUB, SW, XOR, XORI,
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

// ── InstructionLookup for Rep3Cycle ──────────────────────────────────────────

impl InstructionLookup<XLEN> for Rep3Cycle {
    fn lookup_table(&self) -> Option<LookupTables<XLEN>> {
        self.instruction().lookup_table()
    }
}

// ── Rep3LookupQuery for Rep3Cycle ───────────────────────────────────────────

/// Macro to implement `Rep3LookupQuery` for `Rep3Cycle`, dispatching to
/// per-instruction `Rep3RISCVCycle<X>` implementations (mirrors vanilla
/// `define_rv32im_trait_impls!`).
macro_rules! impl_rep3_lookup_query {
    (instructions: [$($instr:ident),* $(,)?]) => {
        impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3Cycle {
            fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
                match self {
                    Rep3Cycle::NoOp => (Rep3Operand::Public(0), Rep3Operand::Public(0)),
                    $(
                        Rep3Cycle::$instr(cycle) => Rep3LookupQuery::<XLEN>::to_instruction_inputs(cycle),
                    )*
                    _ => panic!(
                        "Unexpected instruction for Rep3LookupQuery: {:?}",
                        self.instruction()
                    ),
                }
            }

            fn to_lookup_index(
                &self,
                party_id: PartyID,
            ) -> FutureRep3Ring<u128, Rep3RingShare<u128>> {
                match self {
                    Rep3Cycle::NoOp => FutureRep3Ring::Ready(Rep3RingShare::default()),
                    $(
                        Rep3Cycle::$instr(cycle) => Rep3LookupQuery::<XLEN>::to_lookup_index(cycle, party_id),
                    )*
                    _ => panic!(
                        "Unexpected instruction for Rep3LookupQuery: {:?}",
                        self.instruction()
                    ),
                }
            }

            fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
                &self,
                steps: &[&impl Rep3LookupQuery<XLEN>],
                io_ctx: &mut IoContext<N>,
                out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
            ) -> eyre::Result<()> {
                match self {
                    Rep3Cycle::NoOp => {
                        Ok(())
                    },
                    $(
                        Rep3Cycle::$instr(cycle) => Rep3LookupQuery::<XLEN>::to_lookup_output_batched(cycle, steps, io_ctx, out),
                    )*
                    _ => panic!(
                        "Unexpected instruction for Rep3LookupQuery: {:?}",
                        self.instruction()
                    ),
                }
            }
        }
    };
}

impl_rep3_lookup_query! {
    instructions: [
        ADD, ADDI, AND, ANDI, ANDN, AUIPC, BEQ, BGE, BGEU, BLT, BLTU, BNE,
        ECALL, FENCE, JAL, JALR, LUI, LD, MUL, MULHU, OR, ORI,
        SLT, SLTI, SLTIU, SLTU, SUB, SD, XOR, XORI,
        VirtualAdvice, VirtualAssertEQ, VirtualAssertHalfwordAlignment,
        VirtualAssertWordAlignment, VirtualAssertLTE,
        VirtualAssertValidDiv0, VirtualAssertValidUnsignedRemainder,
        VirtualChangeDivisor, VirtualChangeDivisorW, VirtualAssertMulUNoOverflow,
        VirtualZeroExtendWord, VirtualSignExtendWord, VirtualMove, VirtualMovsign, VirtualMULI, VirtualPow2,
        VirtualPow2I, VirtualPow2W, VirtualPow2IW, VirtualRev8W, VirtualShiftRightBitmask, VirtualShiftRightBitmaskI,
        VirtualROTRI, VirtualROTRIW,
        VirtualSRA, VirtualSRAI, VirtualSRL, VirtualSRLI,
        VirtualXORROT32, VirtualXORROT24, VirtualXORROT16, VirtualXORROT63,
        VirtualXORROTW16, VirtualXORROTW12, VirtualXORROTW8, VirtualXORROTW7
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

/// Populate arithmetic representations for all shared operands across the trace
/// in a single batched `upcast_many_from_binary` call.
///
/// Mirrors v1's `Rep3JoltInstructionSet::populate_operands_casts`:
/// 1. Collect all `Shared { arithmetic: None }` operands and their binary shares
/// 2. Single batched `upcast_many_from_binary`
/// 3. Write arithmetic shares back
#[tracing::instrument(skip_all, name = "populate_operands_casts")]
pub fn populate_operands_casts<N: Rep3Network>(
    trace: &mut [Rep3Cycle],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<()> {
    let (binary, operands): (Vec<Rep3RingShare<u64>>, Vec<&mut Rep3Operand>) = trace
        .iter_mut()
        .flat_map(|cycle| cycle.shared_operands_mut())
        .filter_map(|op| match op {
            Rep3Operand::Shared {
                arithmetic: None,
                binary,
                ..
            } => Some((*binary, op)),
            _ => None,
        })
        .unzip();

    if binary.is_empty() {
        return Ok(());
    }

    let arithmetic = upcast_many_from_binary(&binary, io_ctx)?;

    operands
        .into_iter()
        .zip(arithmetic)
        .for_each(|(operand, arith)| match operand {
            Rep3Operand::Shared {
                arithmetic: None,
                binary,
                public,
            } => {
                *operand = Rep3Operand::Shared {
                    binary: std::mem::take(binary),
                    arithmetic: Some(arith),
                    public: std::mem::take(public),
                };
            }
            _ => panic!("Expected shared operand"),
        });

    Ok(())
}

mod add;
mod addi;
mod and;
mod andi;
mod andn;
mod auipc;
mod beq;
mod bge;
mod bgeu;
mod blt;
mod bltu;
mod bne;
mod ecall;
mod fence;
mod jal;
mod jalr;
mod ld;
mod lui;
mod mul;
mod mulhu;
mod or;
mod ori;
mod rem;
mod sd;
mod slt;
mod slti;
mod sltiu;
mod sltu;
mod store;
mod sub;
mod virtual_advice;
mod virtual_assert_eq;
mod virtual_assert_halfword_alignment;
mod virtual_assert_lte;
mod virtual_assert_mulu_no_overflow;
mod virtual_assert_valid_div0;
mod virtual_assert_valid_unsigned_remainder;
mod virtual_assert_word_alignment;
mod virtual_change_divisor;
mod virtual_move;
mod virtual_movsign;
mod virtual_muli;
mod virtual_pow2;
mod virtual_pow2_w;
mod virtual_rev8w;
mod virtual_rotri;
mod virtual_shift_right_bitmask;
mod virtual_sign_extend_word;
mod virtual_sra;
mod virtual_srai;
mod virtual_srl;
mod virtual_srli;
mod virtual_xor_rot;
mod virtual_xor_rotw;
mod virtual_zero_extend_word;
mod xor;
mod xori;
