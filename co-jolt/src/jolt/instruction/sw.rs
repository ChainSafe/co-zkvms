use crate::poly::field::JoltField;
use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;

use super::{JoltInstruction, Rep3Operand, SubtableIndices};
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_operand_usize, concatenate_lookups};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SWInstruction<F: JoltField>(pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for SWInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match &self.0 {
            Rep3Operand::Public(x) => (0, *x),
            _ => panic!("SWInstruction::operands called with non-public operand"),
        }
    }

    fn combine_lookups(&self, vals: &[F], _: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        assert!(vals.len() == 2);
        concatenate_lookups(vals, 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need two IdentitySubtables
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![(
            Box::new(IdentitySubtable::<F>::new()),
            SubtableIndices::from(C - 2..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match &self.0 {
            Rep3Operand::Public(x) => chunk_operand_usize(*x, C, log_M),
            _ => panic!("SWInstruction::to_indices called with non-public operand"),
        }
    }

    fn lookup_entry(&self) -> F {
        match &self.0 {
            Rep3Operand::Public(x) => (*x & 0xffffffff).into(),
            _ => panic!("SWInstruction::lookup_entry called with non-public operand"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self((rng.next_u32() as u64).into())
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::SWInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn sw_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = SWInstruction(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SWInstruction(0),
            SWInstruction(1),
            SWInstruction(100),
            SWInstruction(u32_max),
            SWInstruction(1 << 8),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
