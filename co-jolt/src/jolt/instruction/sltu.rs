use crate::poly::field::JoltField;
use rand::prelude::StdRng;
use rand::RngCore;

use super::{JoltInstruction, Rep3Operand};
use crate::{
    jolt::{
        instruction::SubtableIndices,
        subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    },
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SLTUInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for SLTUInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("SLTUInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);
        let ltu = vals_by_subtable[0];
        let eq = vals_by_subtable[1];

        let mut sum = F::zero();
        let mut eq_prod = F::one();

        for i in 0..C {
            sum += ltu[i] * eq_prod;
            eq_prod *= eq[i];
        }
        sum
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_operands(*x, *y, C, log_M)
            }
            _ => panic!("SLTUInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                (*x < *y).into()
            }
            _ => panic!("SLTUInstruction::lookup_entry called with non-public operands"),
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

    use super::SLTUInstruction;

    #[test]
    fn sltu_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = SLTUInstruction(x, y);
            jolt_instruction_test!(instruction);
        }
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            jolt_instruction_test!(SLTUInstruction(x, x));
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SLTUInstruction(100, 0),
            SLTUInstruction(0, 100),
            SLTUInstruction(1, 0),
            SLTUInstruction(0, u32_max),
            SLTUInstruction(u32_max, 0),
            SLTUInstruction(u32_max, u32_max),
            SLTUInstruction(u32_max, 1 << 8),
            SLTUInstruction(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
