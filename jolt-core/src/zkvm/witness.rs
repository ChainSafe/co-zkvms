#![allow(static_mut_refs)]

use allocative::Allocative;
use common::constants::XLEN;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use std::sync::LazyLock;
use strum::IntoEnumIterator;

use crate::{
    utils::math::Math,
    zkvm::{instruction_lookups, lookup_table::LookupTables},
};

use super::instruction::CircuitFlags;

/// K^{1/d}
pub const DTH_ROOT_OF_K: usize = 1 << 8;

pub fn compute_d_parameter_from_log_K(log_K: usize) -> usize {
    log_K.div_ceil(DTH_ROOT_OF_K.log_2())
}

pub fn compute_d_parameter(K: usize) -> usize {
    // Calculate D dynamically such that 2^8 = K^(1/D)
    let log_K = K.log_2();
    log_K.div_ceil(DTH_ROOT_OF_K.log_2())
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /* R1CS aux variables */
    /// The "left" input to the current instruction. Typically either the
    /// rs1 value or the current program counter.
    LeftInstructionInput,
    /// The "right" input to the current instruction. Typically either the
    /// rs2 value or the immediate value.
    RightInstructionInput,
    /// Whether the current instruction should write the lookup output to
    /// the destination register
    WriteLookupOutputToRD,
    /// Whether the current instruction should write the program counter to
    /// the destination register
    WritePCtoRD,
    /// Whether the current instruction triggers a branch
    ShouldBranch,
    /// Whether the current instruction triggers a jump
    ShouldJump,
    /*  Twist/Shout witnesses */
    /// Inc polynomial for the registers instance of Twist
    RdInc,
    /// Inc polynomial for the RAM instance of Twist
    RamInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are d=8 of these polynomials, `InstructionRa(0) .. InstructionRa(7)`
    InstructionRa(usize),
    /// One-hot ra polynomial for the bytecode instance of Shout
    BytecodeRa(usize),
    /// One-hot ra/wa polynomial for the RAM instance of Twist
    /// Note that for RAM, ra and wa are the same polynomial because
    /// there is at most one load or store per cycle.
    RamRa(usize),
}

pub static mut ALL_COMMITTED_POLYNOMIALS: OnceCell<Vec<CommittedPolynomial>> = OnceCell::new();

pub struct AllCommittedPolynomials();
impl AllCommittedPolynomials {
    pub fn initialize(ram_d: usize, bytecode_d: usize) -> Self {
        unsafe {
            if let Some(existing) = ALL_COMMITTED_POLYNOMIALS.get() {
                // Check if existing polynomials match requested dimensions
                let existing_ram_d = existing
                    .iter()
                    .filter(|p| matches!(p, CommittedPolynomial::RamRa(_)))
                    .count();
                let existing_bytecode_d = existing
                    .iter()
                    .filter(|p| matches!(p, CommittedPolynomial::BytecodeRa(_)))
                    .count();

                if existing_ram_d == ram_d && existing_bytecode_d == bytecode_d {
                    // Parameters match, reuse existing polynomials
                    return AllCommittedPolynomials();
                } else {
                    // Parameters differ, need to reinitialize
                    ALL_COMMITTED_POLYNOMIALS.take();
                }
            }
        };
        let mut polynomials = vec![
            CommittedPolynomial::LeftInstructionInput,
            CommittedPolynomial::RightInstructionInput,
            CommittedPolynomial::WriteLookupOutputToRD,
            CommittedPolynomial::WritePCtoRD,
            CommittedPolynomial::ShouldBranch,
            CommittedPolynomial::ShouldJump,
            CommittedPolynomial::RdInc,
            CommittedPolynomial::RamInc,
            CommittedPolynomial::InstructionRa(0),
            CommittedPolynomial::InstructionRa(1),
            CommittedPolynomial::InstructionRa(2),
            CommittedPolynomial::InstructionRa(3),
            CommittedPolynomial::InstructionRa(4),
            CommittedPolynomial::InstructionRa(5),
            CommittedPolynomial::InstructionRa(6),
            CommittedPolynomial::InstructionRa(7),
            CommittedPolynomial::InstructionRa(8),
            CommittedPolynomial::InstructionRa(9),
            CommittedPolynomial::InstructionRa(10),
            CommittedPolynomial::InstructionRa(11),
            CommittedPolynomial::InstructionRa(12),
            CommittedPolynomial::InstructionRa(13),
            CommittedPolynomial::InstructionRa(14),
            CommittedPolynomial::InstructionRa(15),
        ];
        for i in 0..ram_d {
            polynomials.push(CommittedPolynomial::RamRa(i));
        }
        for i in 0..bytecode_d {
            polynomials.push(CommittedPolynomial::BytecodeRa(i));
        }

        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .set(polynomials)
                .expect("ALL_COMMITTED_POLYNOMIALS is already initialized");
        }

        AllCommittedPolynomials()
    }

    pub fn iter() -> impl Iterator<Item = &'static CommittedPolynomial> {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .iter()
        }
    }

    pub fn par_iter() -> impl ParallelIterator<Item = &'static CommittedPolynomial> {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .par_iter()
        }
    }
}

impl CommittedPolynomial {
    pub fn len() -> usize {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .len()
        }
    }

    // TODO(moodlezoup): return Result<Self>
    pub fn from_index(index: usize) -> Self {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")[index]
        }
    }

    // TODO(moodlezoup): return Result<usize>
    pub fn to_index(&self) -> usize {
        unsafe {
            ALL_COMMITTED_POLYNOMIALS
                .get()
                .expect("ALL_COMMITTED_POLYNOMIALS is uninitialized")
                .iter()
                .find_position(|poly| *poly == self)
                .unwrap()
                .0
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    SpartanAz,
    SpartanBz,
    SpartanCz,
    PC,
    UnexpandedPC,
    NextPC,
    NextUnexpandedPC,
    NextIsNoop,
    LeftLookupOperand,
    RightLookupOperand,
    Product,
    Rd,
    Imm,
    Rs1Value,
    Rs2Value,
    RdWriteValue,
    Rs1Ra,
    Rs2Ra,
    RdWa,
    LookupOutput,
    InstructionRaf,
    InstructionRafFlag,
    InstructionRa,
    RegistersVal,
    RamAddress,
    RamRa,
    RamReadValue,
    RamWriteValue,
    RamVal,
    RamValInit,
    RamValFinal,
    RamHammingWeight,
    OpFlags(CircuitFlags),
    LookupTableFlag(usize),
}

pub static ALL_VIRTUAL_POLYNOMIALS: LazyLock<Vec<VirtualPolynomial>> = LazyLock::new(|| {
    let mut polynomials = vec![
        VirtualPolynomial::SpartanAz,
        VirtualPolynomial::SpartanBz,
        VirtualPolynomial::SpartanCz,
        VirtualPolynomial::PC,
        VirtualPolynomial::UnexpandedPC,
        VirtualPolynomial::NextPC,
        VirtualPolynomial::NextUnexpandedPC,
        VirtualPolynomial::NextIsNoop,
        VirtualPolynomial::LeftLookupOperand,
        VirtualPolynomial::RightLookupOperand,
        VirtualPolynomial::Product,
        VirtualPolynomial::Rd,
        VirtualPolynomial::Imm,
        VirtualPolynomial::Rs1Value,
        VirtualPolynomial::Rs2Value,
        VirtualPolynomial::RdWriteValue,
        VirtualPolynomial::Rs1Ra,
        VirtualPolynomial::Rs2Ra,
        VirtualPolynomial::RdWa,
        VirtualPolynomial::LookupOutput,
        VirtualPolynomial::InstructionRaf,
        VirtualPolynomial::InstructionRafFlag,
        VirtualPolynomial::InstructionRa,
        VirtualPolynomial::RegistersVal,
        VirtualPolynomial::RamAddress,
        VirtualPolynomial::RamRa,
        VirtualPolynomial::RamReadValue,
        VirtualPolynomial::RamWriteValue,
        VirtualPolynomial::RamVal,
        VirtualPolynomial::RamValInit,
        VirtualPolynomial::RamValFinal,
        VirtualPolynomial::RamHammingWeight,
    ];
    for flag in CircuitFlags::iter() {
        polynomials.push(VirtualPolynomial::OpFlags(flag));
    }
    for table in LookupTables::iter() {
        polynomials.push(VirtualPolynomial::LookupTableFlag(
            LookupTables::<XLEN>::enum_index(&table),
        ));
    }

    polynomials
});

impl VirtualPolynomial {
    pub fn from_index(index: usize) -> Self {
        ALL_VIRTUAL_POLYNOMIALS[index]
    }

    pub fn to_index(&self) -> usize {
        ALL_VIRTUAL_POLYNOMIALS
            .iter()
            .find_position(|poly| *poly == self)
            .unwrap()
            .0
    }
}
