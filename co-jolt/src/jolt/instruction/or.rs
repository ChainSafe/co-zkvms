use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::instruction::Rep3Operand;
use crate::jolt::subtable::{or::OrSubtable, LassoSubtable};
use crate::poly::field::JoltField;
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ORInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for ORInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                (*x, *y)
            }
            _ => panic!("ORInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(OrSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_operands(*x, *y, C, log_M)
            }
            _ => panic!("ORInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                F::from(*x | *y)
            }
            _ => panic!("ORInstruction::lookup_entry called with non-public operands"),
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

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::ORInstruction;

    #[test]
    fn or_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let y = rng.next_u32() as u64;
            let instruction = ORInstruction(x, y);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ORInstruction(100, 0),
            ORInstruction(0, 100),
            ORInstruction(1, 0),
            ORInstruction(0, u32_max),
            ORInstruction(u32_max, 0),
            ORInstruction(u32_max, u32_max),
            ORInstruction(u32_max, 1 << 8),
            ORInstruction(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn or_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            let instruction = ORInstruction(x, y);
            jolt_instruction_test!(instruction);
        }
    }
}
