use rand::prelude::StdRng;
use rand::RngCore;

use super::{slt::SLTInstruction, JoltInstruction, Rep3Operand, SubtableIndices};
use crate::{
    jolt::subtable::{
        eq::EqSubtable, eq_abs::EqAbsSubtable, eq_msb::EqMSBSubtable, gt_msb::GtMSBSubtable,
        lt_abs::LtAbsSubtable, ltu::LtuSubtable, LassoSubtable,
    },
    poly::field::JoltField,
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BGEInstruction<F: JoltField>(pub Rep3Operand<F>, pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for BGEInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("BGEInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        // 1 - LTS(x, y) =
        F::one() - SLTInstruction(self.0.clone(), self.1.clone()).combine_lookups(vals, C, M)
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C + 1
    }

    fn subtables(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(GtMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(EqMSBSubtable::new()), SubtableIndices::from(0)),
            (Box::new(LtuSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(1..C)),
            (Box::new(LtAbsSubtable::new()), SubtableIndices::from(0)),
            (Box::new(EqAbsSubtable::new()), SubtableIndices::from(0)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                chunk_and_concatenate_operands(*x, *y, C, log_M)
            }
            _ => panic!("BGEInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                ((*x as i32) >= (*y as i32)).into()
            }
            _ => panic!("BGEInstruction::lookup_entry called with non-public operands"),
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

    use super::BGEInstruction;

    #[test]
    fn bge_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32();
            let y = rng.next_u32();

            let instruction = BGEInstruction(x as u64, y as u64);

            jolt_instruction_test!(instruction);
        }

        // Ones
        for _ in 0..256 {
            let x = rng.next_u32();
            jolt_instruction_test!(BGEInstruction(x as u64, x as u64));
        }

        // Edge-cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            BGEInstruction(100, 0),
            BGEInstruction(0, 100),
            BGEInstruction(1, 0),
            BGEInstruction(0, u32_max),
            BGEInstruction(u32_max, 0),
            BGEInstruction(u32_max, u32_max),
            BGEInstruction(u32_max, 1 << 8),
            BGEInstruction(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
