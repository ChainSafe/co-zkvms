use eyre::Context;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::rep3::{
    self, network::{IoContext, Rep3Network}, Rep3PrimeFieldShare
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use jolt_core::field::JoltField;
use crate::utils::instruction_utils::chunk_operand_usize;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SHInstruction<F: JoltField>(pub Rep3Operand<F>);

impl<F: JoltField> JoltInstruction<F> for SHInstruction<F> {
    fn operands(&self) -> (u64, u64) {
        match &self.0 {
            Rep3Operand::Public(x) => (0, *x),
            _ => panic!("SHInstruction::operands called with non-public operand"),
        }
    }

    fn combine_lookups(&self, vals: &[F], _: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        assert!(vals.len() == 1);
        vals[0]
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need two TruncateOverflowSubtables
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![(
            Box::new(IdentitySubtable::<F>::new()),
            SubtableIndices::from(C - 1),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        match &self.0 {
            Rep3Operand::Public(x) => chunk_operand_usize(*x, C, log_M),
            _ => panic!("SHInstruction::to_indices called with non-public operand"),
        }
    }

    fn lookup_entry(&self) -> F {
        match &self.0 {
            Rep3Operand::Public(x) => {
                // Lower 16 bits of the rs2 value
                (*x & 0xffff).into()
            }
            _ => panic!("SHInstruction::lookup_entry called with non-public operand"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self((rng.next_u32() as u64).into())
    }
}

impl<F: JoltField> Rep3JoltInstruction<F> for SHInstruction<F> {
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), Rep3Operand::Public(0))
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, None)
    }

    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        assert!(M == 1 << 16);
        assert!(vals.len() == 1);
        Ok(vals[0])
    }

    fn to_indices_rep3(
        &self,
        C: usize,
        log_M: usize,
    ) -> Vec<mpc_core::protocols::rep3::Rep3BigUintShare<F>> {
        unimplemented!()
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        match &self.0 {
            Rep3Operand::Binary(x) => {
                rep3::conversion::b2a_selector(
                    &rep3::binary::and_with_public(x, &0xffff_u64.into()),
                    io_ctx,
                ).context("while computing SHInstruction output")
            }
            _ => panic!("SHInstruction::output called with non-binary operands"),
        }
    }
}


#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::SHInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn sh_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = SHInstruction(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SHInstruction(0),
            SHInstruction(1),
            SHInstruction(100),
            SHInstruction(u32_max),
            SHInstruction(1 << 8),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
