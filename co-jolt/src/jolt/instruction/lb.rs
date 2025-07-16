use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::rep3::{
    network::{IoContext, Rep3Network},
    Rep3PrimeFieldShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::jolt::subtable::{
    sign_extend::SignExtendSubtable, truncate_overflow::TruncateOverflowSubtable, LassoSubtable,
};
use crate::poly::field::JoltField;
use crate::utils::instruction_utils::chunk_operand_usize;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LBInstruction<F: JoltField>(pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for LBInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match &self.0 {
            Rep3Operand::Public(x) => (0, *x),
            _ => panic!("LBInstruction::operands called with non-public operand"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(M >= 1 << 8);
        assert!(vals.len() == 2);

        let byte = vals[0];
        let sign_extension = vals[1];

        let mut result = byte;
        for i in 1..C {
            result += F::from_u64(1 << (8 * i)).unwrap() * sign_extension;
        }
        result
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need one TruncateOverflowSubtable
        assert!(M >= 1 << 8);
        vec![
            (
                // Truncate all but the lowest eight bits of the last chunk,
                // which contains the lower 8 bits of the loaded value.
                Box::new(TruncateOverflowSubtable::<F, 8>::new()),
                SubtableIndices::from(C - 1),
            ),
            (
                // Sign extend the lowest eight bits of the last chunk,
                // which contains the lower 8 bits of the loaded value.
                Box::new(SignExtendSubtable::<F, 8>::new()),
                SubtableIndices::from(C - 1),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match &self.0 {
            Rep3Operand::Public(x) => chunk_operand_usize(*x, C, log_M),
            _ => panic!("LBInstruction::to_indices called with non-public operand"),
        }
    }

    fn lookup_entry(&self) -> F {
        match &self.0 {
            Rep3Operand::Public(x) => {
                // Sign-extend lower 8 bits of the loaded value
                ((*x & 0xff) as i8 as i32 as u32 as u64).into()
            }
            _ => panic!("LBInstruction::lookup_entry called with non-public operand"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self((rng.next_u32() as u64).into())
    }
}

impl<F: JoltField> Rep3JoltInstruction<F> for LBInstruction<F> {
    fn operands(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), Rep3Operand::Public(0))
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, None)
    }

    fn combine_lookups(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
    ) -> Rep3PrimeFieldShare<F> {
        unimplemented!()
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

    fn output<N: Rep3Network>(&self, io_ctx: &mut IoContext<N>) -> Rep3PrimeFieldShare<F> {
        unimplemented!()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::LBInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn lb_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = LBInstruction(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            LBInstruction(0),
            LBInstruction(1),
            LBInstruction(100),
            LBInstruction(u32_max),
            LBInstruction(1 << 8),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
