use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use eyre::Context;
use num_bigint::BigUint;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::utils::future::FutureRep3;
use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};

use jolt_core::{
    jolt::{
        instruction::SubtableIndices,
        subtable::{identity::IdentitySubtable, sign_extend::SignExtendSubtable, LassoSubtable},
    },
    utils::instruction_utils::{chunk_operand_usize, concatenate_lookups},
};
use mpc_core::protocols::{
    rep3::network::{IoContext, Rep3Network},
    rep3_ring::{ring::ring_impl::RingElement, Rep3RingShare},
};
use mpc_core::protocols::{
    rep3::{self, Rep3BigUintShare, Rep3PrimeFieldShare},
    rep3_ring,
};

use crate::utils::instruction_utils::{
    concatenate_lookups_rep3, concatenate_lookups_rep3_batched, rep3_chunk_operand,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand};

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MOVSIGNInstruction<const WORD_SIZE: usize>(pub Rep3Operand);

// Constants for 32-bit and 64-bit word sizes
const ALL_ONES_32: u64 = 0xFFFF_FFFF;
const ALL_ONES_64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SIGN_BIT_32: u64 = 0x8000_0000;
const SIGN_BIT_64: u64 = 0x8000_0000_0000_0000;

impl<const WORD_SIZE: usize> JoltInstruction for MOVSIGNInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        let val = vals[0];
        let repeat = WORD_SIZE / 16;
        concatenate_lookups(&vec![val; repeat], repeat, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
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

    fn lookup_entry<F: JoltField>(&self) -> F {
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

impl<const WORD_SIZE: usize> Rep3JoltInstruction for MOVSIGNInstruction<WORD_SIZE> {
    fn operands_rep3(&self) -> (Rep3Operand, Rep3Operand) {
        (self.0.clone(), Rep3Operand::default())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand, Option<&mut Rep3Operand>) {
        (&mut self.0, None)
    }

    fn lhs(&self) -> &Rep3Operand {
        &self.0
    }

    fn rhs(&self) -> Option<&Rep3Operand> {
        None
    }

    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
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

    fn to_indices_rep3(
        &self,
        _: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        rep3_chunk_operand(self.0.as_binary(), C, log_M)
    }

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        let t: Vec<_> = steps
            .into_iter()
            .map(|step| step.lhs().as_binary() & RingElement(SIGN_BIT_32 as u32))
            .collect();

        let zeros = vec![Rep3RingShare::zero_share(); t.len()];
        let all_ones = vec![
            rep3_ring::binary::promote_to_trivial_share(
                io_ctx.id,
                &RingElement(ALL_ONES_32 as u32)
            );
            t.len()
        ];

        let cond = rep3_ring::conversion::bit_inject_from_bits_many::<u32, _>(
            &rep3_ring::binary::is_zero_many(&t, io_ctx)?,
            io_ctx,
        )?;
        rep3_ring::binary::cmux_many(&cond, &zeros, &all_ones, io_ctx)
            .context("Failed to cmux")?
            .into_iter()
            .zip(out)
            .for_each(|(ready, out)| {
                *out = FutureRep3Ring::cast_to_field_b2a(ready);
            });
        Ok(())
    }
}
