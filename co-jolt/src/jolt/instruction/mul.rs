use ark_std::log2;
use jolt_core::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare, Rep3PrimeFieldShare,
};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{
    jolt::instruction::Rep3JoltInstruction, utils::instruction_utils::concatenate_lookups_rep3,
};

use super::{JoltInstruction, Rep3Operand, SubtableIndices};
use jolt_core::{
    field::JoltField,
    utils::instruction_utils::{
        assert_valid_parameters, concatenate_lookups, multiply_and_chunk_operands,
    },
};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MULInstruction<const WORD_SIZE: usize, F: JoltField>(
    pub Rep3Operand<F>,
    pub Rep3Operand<F>,
);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for MULInstruction<WORD_SIZE, F> {
    // fn to_lookup_index(&self) -> u64 {
    //     self.0 * self.1
    // }

    fn operands(&self) -> (u64, u64) {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => (*x, *y),
            _ => panic!("MULInstruction::operands called with non-public operands"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(vals.len() == C / 2);
        concatenate_lookups(vals, C / 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(msb_chunk_index + 1..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                multiply_and_chunk_operands(*x as u128, *y as u128, C, log_M)
            }
            _ => panic!("MULInstruction::to_indices called with non-public operands"),
        }
    }

    fn lookup_entry(&self) -> F {
        match (&self.0, &self.1) {
            (Rep3Operand::Public(x), Rep3Operand::Public(y)) => {
                if WORD_SIZE == 32 {
                    let x = *x as i32;
                    let y = *y as i32;
                    (x.wrapping_mul(y) as u32 as u64).into()
                } else if WORD_SIZE == 64 {
                    let x = *x as i64;
                    let y = *y as i64;
                    (x.wrapping_mul(y) as u64).into()
                } else {
                    panic!("MUL is only implemented for 32-bit or 64-bit word sizes")
                }
            }
            _ => panic!("MULInstruction::lookup_entry called with non-public operands"),
        }
    }

    // fn materialize_entry(&self, index: u64) -> u64 {
    //     index % (1 << WORD_SIZE)
    // }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self(
                Rep3Operand::Public(rng.next_u32() as u64),
                Rep3Operand::Public(rng.next_u32() as u64),
            ),
            64 => Self(
                Rep3Operand::Public(rng.next_u64()),
                Rep3Operand::Public(rng.next_u64()),
            ),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    // fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
    //     debug_assert_eq!(r.len(), 2 * WORD_SIZE);
    //     let mut result = F::zero();
    //     for i in 0..WORD_SIZE {
    //         result += F::from_u64_unchecked(1 << (WORD_SIZE - 1 - i)) * r[WORD_SIZE + i];
    //     }
    //     result
    // }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F> for MULInstruction<WORD_SIZE, F> {
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, Some(&mut self.1))
    }

    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        assert!(vals.len() == C / 2);
        Ok(concatenate_lookups_rep3(vals, C / 2, log2(M) as usize))
    }

    fn to_indices_rep3(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>> {
        todo!()
    }

    fn output<N: mpc_core::protocols::rep3::network::Rep3Network>(
        &self,
        io_ctx: &mut mpc_core::protocols::rep3::network::IoContext<N>,
    ) -> eyre::Result<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>> {
        match (&self.0, &self.1) {
            (Rep3Operand::Arithmetic(x), Rep3Operand::Arithmetic(y)) => {
                rep3::arithmetic::mul(*x, *y, io_ctx).map_err(|e| eyre::eyre!(e))
            }
            _ => Err(eyre::eyre!(
                "MULInstruction::output called with shared operands"
            )),
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::MULInstruction;
    use crate::{
        jolt::instruction::{
            test::{
                instruction_mle_full_hypercube_test, instruction_mle_random_test,
                materialize_entry_test, prefix_suffix_test,
            },
            JoltInstruction,
        },
        jolt_instruction_test,
    };

    #[test]
    fn mul_materialize_entry() {
        materialize_entry_test::<Fr, MULInstruction<32>>();
    }

    #[test]
    fn mul_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, MULInstruction<8>>();
    }

    #[test]
    fn mul_mle_random() {
        instruction_mle_random_test::<Fr, MULInstruction<32>>();
    }

    #[test]
    fn mul_prefix_suffix() {
        prefix_suffix_test::<Fr, MULInstruction<32>>();
    }

    #[test]
    fn mul_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = MULInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            MULInstruction::<WORD_SIZE>(100, 0),
            MULInstruction::<WORD_SIZE>(0, 100),
            MULInstruction::<WORD_SIZE>(1, 0),
            MULInstruction::<WORD_SIZE>(0, u32_max),
            MULInstruction::<WORD_SIZE>(u32_max, 0),
            MULInstruction::<WORD_SIZE>(2, u32_max),
            MULInstruction::<WORD_SIZE>(u32_max, u32_max),
            MULInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            MULInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn mul_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = MULInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            MULInstruction::<WORD_SIZE>(100, 0),
            MULInstruction::<WORD_SIZE>(0, 100),
            MULInstruction::<WORD_SIZE>(1, 0),
            MULInstruction::<WORD_SIZE>(0, u64_max),
            MULInstruction::<WORD_SIZE>(u64_max, 0),
            MULInstruction::<WORD_SIZE>(u64_max, u64_max),
            MULInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            MULInstruction::<WORD_SIZE>(1 << 32, u64_max),
            MULInstruction::<WORD_SIZE>(1 << 63, 1),
            MULInstruction::<WORD_SIZE>(1, 1 << 63),
            MULInstruction::<WORD_SIZE>(u64_max - 1, 1),
            MULInstruction::<WORD_SIZE>(1, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
