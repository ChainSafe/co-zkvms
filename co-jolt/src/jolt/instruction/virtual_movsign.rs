use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use jolt_core::{
    field::JoltField,
    jolt::{
        instruction::SubtableIndices,
        subtable::{identity::IdentitySubtable, sign_extend::SignExtendSubtable, LassoSubtable},
    },
    utils::instruction_utils::{chunk_operand_usize, concatenate_lookups},
};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_core::protocols::rep3::{Rep3BigUintShare, Rep3PrimeFieldShare};

use crate::utils::instruction_utils::{concatenate_lookups_rep3, concatenate_lookups_rep3_batched};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MOVSIGNInstruction<const WORD_SIZE: usize, F: JoltField>(pub Rep3Operand<F>);

// Constants for 32-bit and 64-bit word sizes
const ALL_ONES_32: u64 = 0xFFFF_FFFF;
const ALL_ONES_64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SIGN_BIT_32: u64 = 0x8000_0000;
const SIGN_BIT_64: u64 = 0x8000_0000_0000_0000;

impl<const WORD_SIZE: usize, F: JoltField> JoltInstruction<F> for MOVSIGNInstruction<WORD_SIZE, F> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), 0)
    }

    fn combine_lookups(&self, vals: &[F], _: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        let val = vals[0];
        let repeat = WORD_SIZE / 16;
        concatenate_lookups(&vec![val; repeat], repeat, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        assert!(M == 1 << 16);
        let msb_chunk_index = C - (WORD_SIZE / 16);
        vec![
            (
                Box::new(SignExtendSubtable::<F, 16>::new()),
                SubtableIndices::from(msb_chunk_index),
            ),
            (
                // Not used for lookup, but this implicitly range-checks
                // the remaining query chunks
                Box::new(IdentitySubtable::<F>::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0.as_public(), C, log_M)
    }

    fn lookup_entry(&self) -> F {
        match WORD_SIZE {
            32 => {
                if self.0.as_public() & SIGN_BIT_32 != 0 {
                    ALL_ONES_32.into()
                } else {
                    F::zero()
                }
            }
            64 => {
                if self.0.as_public() & SIGN_BIT_64 != 0 {
                    ALL_ONES_64.into()
                } else {
                    F::zero()
                }
            }
            _ => panic!("only implemented for u32 / u64"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self((rng.next_u32() as u64).into()),
            64 => Self(rng.next_u64().into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize, F: JoltField> Rep3JoltInstruction<F>
    for MOVSIGNInstruction<WORD_SIZE, F>
{
    fn operands_rep3(&self) -> (Rep3Operand<F>, Rep3Operand<F>) {
        (self.0.clone(), Rep3Operand::default())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand<F>, Option<&mut Rep3Operand<F>>) {
        (&mut self.0, None)
    }

    fn combine_lookups_rep3<N: Rep3Network>(
        &self,
        vals: &[Rep3PrimeFieldShare<F>],
        _: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        assert!(M == 1 << 16);
        let val = vals[0];
        let repeat = WORD_SIZE / 16;
        Ok(concatenate_lookups_rep3(
            &vec![val; repeat],
            repeat,
            log2(M) as usize,
        ))
    }

    fn combine_lookups_rep3_batched<N: Rep3Network>(
        &self,
        mut vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        let repeat = WORD_SIZE / 16;
        Ok(concatenate_lookups_rep3_batched(
            vals.remove(0).into_iter().map(|val| vec![val; repeat]),
            C,
            log2(M) as usize,
        ))
    }

    fn to_indices_rep3(&self, C: usize, log_M: usize) -> Vec<Rep3BigUintShare<F>> {
        todo!()
    }

    fn output<N: Rep3Network>(
        &self,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Rep3PrimeFieldShare<F>> {
        todo!()
    }
}
