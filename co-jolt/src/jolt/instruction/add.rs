use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;

use super::{JoltInstruction, Rep3Operand, SubtableIndices};
use crate::jolt::subtable::{
    identity::IdentitySubtable, truncate_overflow::TruncateOverflowSubtable, LassoSubtable,
};
use crate::poly::field::JoltField;
use crate::utils::instruction_utils::{
    add_and_chunk_operands, assert_valid_parameters, concatenate_lookups,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ADDInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for ADDInstruction<WORD_SIZE, F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("ADDInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(vals.len() == C);
        // The output is the TruncateOverflow(most significant chunk) || identity of other chunks
        concatenate_lookups(vals, C, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;
        vec![
            (
                Box::new(TruncateOverflowSubtable::<F, WORD_SIZE>::new()),
                SubtableIndices::from(0..msb_chunk_index + 1),
            ),
            (
                Box::new(IdentitySubtable::new()),
                SubtableIndices::from(msb_chunk_index + 1..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                add_and_chunk_operands(*x as u128, *y as u128, C, log_M)
            }
            _ => panic!("ADDInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                if WORD_SIZE == 32 {
                    (*x as u32).overflowing_add(*y as u32).0.into()
                } else if WORD_SIZE == 64 {
                    (*x as u64).overflowing_add(*y as u64).0.into()
                } else {
                    panic!("only implemented for u32 / u64")
                }
            }
            _ => panic!("ADDInstruction::lookup_entry called with non-public operands"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(
            (rng.next_u32() as u64).into(),
            (rng.next_u32() as u64).into(),
        )
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::ADDInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn add_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = ADDInstruction::<32>(x, y);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ADDInstruction::<32>(100, 0),
            ADDInstruction::<32>(0, 100),
            ADDInstruction::<32>(1, 0),
            ADDInstruction::<32>(0, u32_max),
            ADDInstruction::<32>(u32_max, 0),
            ADDInstruction::<32>(u32_max, u32_max),
            ADDInstruction::<32>(u32_max, 1 << 8),
            ADDInstruction::<32>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn add_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = ADDInstruction::<64>(x, y);
            jolt_instruction_test!(instruction);
        }
    }
}
