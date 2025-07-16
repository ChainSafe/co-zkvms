use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3PrimeFieldShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::jolt::subtable::{
    identity::IdentitySubtable, sign_extend::SignExtendSubtable, LassoSubtable,
};
use crate::poly::field::JoltField;
use crate::utils::instruction_utils::chunk_operand_usize;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LHInstruction<F: JoltField>(pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for LHInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match &self.0 {
            Rep3Operand::Public(x) => (0, *x),
            _ => panic!("LHInstruction::operands called with non-public operand"),
        }
    }

    fn combine_lookups(&self, vals: &[F], _C: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        assert!(vals.len() == 2);

        let half = vals[0];
        let sign_extension = vals[1];

        half + F::from_u64(1 << 16).unwrap() * sign_extension
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need one TruncateOverflowSubtable
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![
            (
                Box::new(IdentitySubtable::<F>::new()),
                SubtableIndices::from(C - 1),
            ),
            (
                // Sign extend the lowest 16 bits of the loaded value,
                // Which will be in the second-to-last chunk.
                Box::new(SignExtendSubtable::<F, 16>::new()),
                SubtableIndices::from(C - 1),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match &self.0 {
            Rep3Operand::Public(x) => chunk_operand_usize(*x, C, log_M),
            _ => panic!("LHInstruction::to_indices called with non-public operand"),
        }
    }

    fn lookup_entry(&self) -> F {
        match &self.0 {
            Rep3Operand::Public(x) => {
                // Sign-extend lower 16 bits of the loaded value
                ((*x & 0xffff) as i16 as i32 as u32 as u64).into()
            }
            _ => panic!("LHInstruction::lookup_entry called with non-public operand"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self((rng.next_u32() as u64).into())
    }
}

impl<F: JoltField> Rep3JoltInstruction<F> for LHInstruction<F> {
    fn operands(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), Rep3Operand::Public(0))
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, None)
    }

    fn combine_lookups<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        assert!(M == 1 << 16);
        assert!(vals.len() == 2);

        let half = vals[0];
        let sign_extension = vals[1];

        Ok(half + rep3::arithmetic::mul_public(sign_extension, F::from_u64(1 << 16).unwrap()))
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn to_indices(
        &self,
        C: usize,
        log_M: usize,
    ) -> Vec<mpc_core::protocols::rep3::Rep3BigUintShare<F>> {
        unimplemented!()
    }

    fn output<N: Rep3Network>(
        &self,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::LHInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn lh_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = LHInstruction(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            LHInstruction(0),
            LHInstruction(1),
            LHInstruction(100),
            LHInstruction(u32_max),
            LHInstruction(1 << 8),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
