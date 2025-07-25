use ark_std::log2;
use eyre::Context;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3PrimeFieldShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::instruction_utils::chunk_operand_usize;
use crate::{
    jolt::subtable::{identity::IdentitySubtable, LassoSubtable},
    utils::instruction_utils::concatenate_lookups_rep3,
};
use jolt_core::{field::JoltField, utils::instruction_utils::concatenate_lookups};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ADVICEInstruction<const WORD_SIZE: usize, F: JoltField>(pub Rep3Operand<F>);

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for ADVICEInstruction<WORD_SIZE, F> {
    fn operands(&self) -> (u64, u64) {
        match &self.0 {
            Rep3Operand::Public(x) => (*x, 0),
            _ => panic!("ADVICEInstruction::operands called with non-public operand"),
        }
    }

    fn combine_lookups(&self, vals: &[F], C: usize, M: usize) -> F {
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
        match &self.0 {
            Rep3Operand::Public(x) => chunk_operand_usize(*x, C, log_M),
            _ => panic!("ADVICEInstruction::to_indices called with non-public operand"),
        }
    }

    fn lookup_entry(&self) -> F {
        match &self.0 {
            Rep3Operand::Public(x) => (*x).into(),
            _ => panic!("ADVICEInstruction::lookup_entry called with non-public operand"),
        }
    }

    // fn materialize_entry(&self, index: u64) -> u64 {
    //     index % (1 << WORD_SIZE)
    // }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self((rng.next_u32() as u64).into()),
            64 => Self(rng.next_u64().into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F>
    for ADVICEInstruction<WORD_SIZE, F>
{
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
        Ok(concatenate_lookups_rep3(vals, C / 2, log2(M) as usize))
    }

    fn to_indices_rep3(&self, C: usize, log_M: usize) -> Vec<rep3::Rep3BigUintShare<F>> {
        todo!()
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        match &self.0 {
            Rep3Operand::Arithmetic(x) => Ok(*x),
            _ => Err(eyre::eyre!(
                "ADVICEInstruction::output called with non-arithmetic operand"
            )),
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::ADVICEInstruction;
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
    fn advice_materialize_entry() {
        materialize_entry_test::<Fr, ADVICEInstruction<32>>();
    }

    #[test]
    fn advice_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, ADVICEInstruction<8>>();
    }

    #[test]
    fn advice_mle_random() {
        instruction_mle_random_test::<Fr, ADVICEInstruction<32>>();
    }

    #[test]
    fn advice_prefix_suffix() {
        prefix_suffix_test::<Fr, ADVICEInstruction<32>>();
    }

    #[test]
    fn advice_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = ADVICEInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ADVICEInstruction::<WORD_SIZE>(0),
            ADVICEInstruction::<WORD_SIZE>(1),
            ADVICEInstruction::<WORD_SIZE>(100),
            ADVICEInstruction::<WORD_SIZE>(1 << 8),
            ADVICEInstruction::<WORD_SIZE>(1 << 16),
            ADVICEInstruction::<WORD_SIZE>(u32_max),
            ADVICEInstruction::<WORD_SIZE>(u32_max - 101),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn advice_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = ADVICEInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            ADVICEInstruction::<WORD_SIZE>(0),
            ADVICEInstruction::<WORD_SIZE>(1),
            ADVICEInstruction::<WORD_SIZE>(100),
            ADVICEInstruction::<WORD_SIZE>(1 << 8),
            ADVICEInstruction::<WORD_SIZE>(1 << 16),
            ADVICEInstruction::<WORD_SIZE>(1 << 32),
            ADVICEInstruction::<WORD_SIZE>(1 << (48 + 2)),
            ADVICEInstruction::<WORD_SIZE>(u64_max),
            ADVICEInstruction::<WORD_SIZE>(u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
