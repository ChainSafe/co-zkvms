use crate::poly::field::JoltField;
use enum_dispatch::enum_dispatch;
use rand::prelude::StdRng;
use std::any::TypeId;
use strum_macros::{EnumCount, EnumIter};

// use super::{Jolt, JoltProof};
use crate::jolt::instruction::{
    add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
    bgeu::BGEUInstruction, bne::BNEInstruction, lb::LBInstruction, lh::LHInstruction,
    or::ORInstruction, sb::SBInstruction, sh::SHInstruction, sll::SLLInstruction,
    slt::SLTInstruction, sltu::SLTUInstruction, sra::SRAInstruction, srl::SRLInstruction,
    sub::SUBInstruction, sw::SWInstruction, xor::XORInstruction, JoltInstruction,
    JoltInstructionSet, SubtableIndices,
};
use crate::jolt::subtable::{
    and::AndSubtable, eq::EqSubtable, eq_abs::EqAbsSubtable, eq_msb::EqMSBSubtable,
    gt_msb::GtMSBSubtable, identity::IdentitySubtable, lt_abs::LtAbsSubtable, ltu::LtuSubtable,
    or::OrSubtable, sign_extend::SignExtendSubtable, sll::SllSubtable, sra_sign::SraSignSubtable,
    srl::SrlSubtable, truncate_overflow::TruncateOverflowSubtable, xor::XorSubtable,
    JoltSubtableSet, LassoSubtable, SubtableId,
};
use paste::paste;

macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        paste! {
            #[allow(non_camel_case_types)]
            #[repr(u8)]
            #[derive(Clone, Debug, PartialEq, EnumIter, EnumCount)]
            #[enum_dispatch(JoltInstruction<F>)]
            pub enum $enum_name<F: JoltField> {
                $([<$alias>]($struct)),+
            }
        }
        impl<F: JoltField> JoltInstructionSet<F> for $enum_name<F> {}

        // Need a default so that we can derive EnumIter on `JoltR1CSInputs`
        // impl<F: JoltField> Default for $enum_name<F> {
        //     fn default() -> Self {
        //         $enum_name::iter().collect::<Vec<_>>()[0]
        //     }
        // }
    };
}

macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        paste! {
            #[allow(non_camel_case_types)]
            #[repr(u8)]
            #[enum_dispatch(LassoSubtable<F>)]
            #[derive(Debug, EnumCount, EnumIter)]
            pub enum $enum_name<F: JoltField> { $([<$alias>]($struct)),+ }
        }
        impl<F: JoltField> From<SubtableId> for $enum_name<F> {
          fn from(subtable_id: SubtableId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id {:?}", subtable_id) }
          }
        }

        impl<F: JoltField> From<$enum_name<F>> for usize {
            fn from(subtable: $enum_name<F>) -> usize {
                // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
                let byte = unsafe { *(&subtable as *const $enum_name<F> as *const u8) };
                byte as usize
            }
        }
        impl<F: JoltField> JoltSubtableSet<F> for $enum_name<F> {}
    };
}

const WORD_SIZE: usize = 32;

instruction_set!(
  RV32I,
  ADD: ADDInstruction<WORD_SIZE, F>,
  SUB: SUBInstruction<WORD_SIZE, F>,
  AND: ANDInstruction<F>,
  OR: ORInstruction<F>,
  XOR: XORInstruction<F>,
  LB: LBInstruction<F>,
  LH: LHInstruction<F>,
  SB: SBInstruction<F>,
  SH: SHInstruction<F>,
  SW: SWInstruction<F>,
  BEQ: BEQInstruction<F>,
  BGE: BGEInstruction<F>,
  BGEU: BGEUInstruction<F>,
  BNE: BNEInstruction<F>,
  SLT: SLTInstruction<F>,
  SLTU: SLTUInstruction<F>,
  SLL: SLLInstruction<WORD_SIZE, F>,
  SRA: SRAInstruction<WORD_SIZE, F>,
  SRL: SRLInstruction<WORD_SIZE, F>
);

subtable_enum!(
  RV32ISubtables,
  AND: AndSubtable<F>,
  EQ_ABS: EqAbsSubtable<F>,
  EQ_MSB: EqMSBSubtable<F>,
  EQ: EqSubtable<F>,
  GT_MSB: GtMSBSubtable<F>,
  IDENTITY: IdentitySubtable<F>,
  LT_ABS: LtAbsSubtable<F>,
  LTU: LtuSubtable<F>,
  OR: OrSubtable<F>,
  SIGN_EXTEND_8: SignExtendSubtable<F, 8>,
  SIGN_EXTEND_16: SignExtendSubtable<F, 16>,
  SLL0: SllSubtable<F, 0, WORD_SIZE>,
  SLL1: SllSubtable<F, 1, WORD_SIZE>,
  SLL2: SllSubtable<F, 2, WORD_SIZE>,
  SLL3: SllSubtable<F, 3, WORD_SIZE>,
  SRA_SIGN: SraSignSubtable<F, WORD_SIZE>,
  SRL0: SrlSubtable<F, 0, WORD_SIZE>,
  SRL1: SrlSubtable<F, 1, WORD_SIZE>,
  SRL2: SrlSubtable<F, 2, WORD_SIZE>,
  SRL3: SrlSubtable<F, 3, WORD_SIZE>,
  TRUNCATE: TruncateOverflowSubtable<F, WORD_SIZE>,
  TRUNCATE_BYTE: TruncateOverflowSubtable<F, 8>,
  XOR: XorSubtable<F>
);

// ==================== JOLT ====================

// pub enum RV32IJoltVM {}

// pub const C: usize = 4;
// pub const M: usize = 1 << 16;

// impl<F, CS> Jolt<F, CS, C, M> for RV32IJoltVM
// where
//     F: JoltField,
//     CS: CommitmentScheme<Field = F>,
// {
//     type InstructionSet = RV32I;
//     type Subtables = RV32ISubtables<F>;
// }

// pub type RV32IJoltProof<F, CS> = JoltProof<C, M, F, CS, RV32I, RV32ISubtables<F>>;

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::{Fr, G1Projective};

    use std::collections::HashSet;

    use crate::host;
    use crate::jolt::instruction::JoltInstruction;
    use crate::jolt::vm::rv32i_vm::{Jolt, RV32IJoltVM, C, M};
    use crate::poly::commitment::hyrax::HyraxScheme;
    use std::sync::Mutex;
    use strum::{EnumCount, IntoEnumIterator};

    // If multiple tests try to read the same trace artifacts simultaneously, they will fail
    lazy_static::lazy_static! {
        static ref FIB_FILE_LOCK: Mutex<()> = Mutex::new(());
        static ref SHA3_FILE_LOCK: Mutex<()> = Mutex::new(());
    }

    #[test]
    fn instruction_set_subtables() {
        let mut subtable_set: HashSet<_> = HashSet::new();
        for instruction in
            <RV32IJoltVM as Jolt<_, HyraxScheme<G1Projective>, C, M>>::InstructionSet::iter()
        {
            for (subtable, _) in instruction.subtables::<Fr>(C, M) {
                // panics if subtable cannot be cast to enum variant
                let _ = <RV32IJoltVM as Jolt<_, HyraxScheme<G1Projective>, C, M>>::Subtables::from(
                    subtable.subtable_id(),
                );
                subtable_set.insert(subtable.subtable_id());
            }
        }
        assert_eq!(
            subtable_set.len(),
            <RV32IJoltVM as Jolt<_, HyraxScheme<G1Projective>, C, M>>::Subtables::COUNT,
            "Unused enum variants in Subtables"
        );
    }

    #[test]
    fn fib_e2e() {
        let _guard = FIB_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("fibonacci-guest");
        program.set_input(&9u32);
        let (bytecode, memory_init) = program.decode();
        let (io_device, bytecode_trace, instruction_trace, memory_trace, circuit_flags) =
            program.trace();

        let preprocessing =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 20);
        let (proof, commitments) =
            <RV32IJoltVM as Jolt<Fr, HyraxScheme<G1Projective>, C, M>>::prove(
                io_device,
                bytecode_trace,
                memory_trace,
                instruction_trace,
                circuit_flags,
                preprocessing.clone(),
            );
        let verification_result = RV32IJoltVM::verify(preprocessing, proof, commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn sha3_e2e() {
        let _guard = SHA3_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("sha3-guest");
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, bytecode_trace, instruction_trace, memory_trace, circuit_flags) =
            program.trace();

        let preprocessing =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 20);
        let (jolt_proof, jolt_commitments) =
            <RV32IJoltVM as Jolt<_, HyraxScheme<G1Projective>, C, M>>::prove(
                io_device,
                bytecode_trace,
                memory_trace,
                instruction_trace,
                circuit_flags,
                preprocessing.clone(),
            );

        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
