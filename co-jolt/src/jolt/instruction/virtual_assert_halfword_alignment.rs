use itertools::izip;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{
    field::JoltField,
    utils::{future_ring::FutureRep3Ring, instruction_utils::rep3_add_and_chunk_operands},
};
use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        PartyID, Rep3PrimeFieldShare,
    },
    rep3_ring::{self, Rep3RingShare},
};

use jolt_core::jolt::subtable::{low_bit::LowBitSubtable, LassoSubtable};
use jolt_core::utils::instruction_utils::{add_and_chunk_operands, assert_valid_parameters};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};

/// (address, offset)
#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssertHalfwordAlignmentInstruction<const WORD_SIZE: usize>(
    pub Rep3Operand,
    pub Rep3Operand,
);

impl<const WORD_SIZE: usize> JoltInstruction for AssertHalfwordAlignmentInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0.as_public(), self.1.as_public())
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, _: usize) -> F {
        assert_eq!(vals.len(), 1);
        let lowest_bit = vals[0];
        F::one() - lowest_bit
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(
            Box::new(LowBitSubtable::<F>::new()),
            SubtableIndices::from(C - 1),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        add_and_chunk_operands(
            self.0.as_public() as u128,
            self.1.as_public() as u128,
            C,
            log_M,
        )
    }

    fn lookup_entry<F: JoltField>(&self) -> F {
        match WORD_SIZE {
            32 => (((self.0.as_public() as u32 as i32 + self.1.as_public() as u32 as i32) % 2 == 0)
                as u64)
                .into(),
            64 => {
                (((self.0.as_public() as i64 + self.1.as_public() as i64) % 2 == 0) as u64).into()
            }
            _ => panic!("Only 32-bit and 64-bit word sizes are supported"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            32 => Self(
                (rng.next_u32() as u64).into(),
                ((rng.next_u32() % (1 << 12)) as u64).into(),
            ),
            64 => Self(rng.next_u64().into(), (rng.next_u64() % (1 << 12)).into()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }
}

impl<const WORD_SIZE: usize> Rep3JoltInstruction for AssertHalfwordAlignmentInstruction<WORD_SIZE> {
    fn operands_rep3(&self) -> (Rep3Operand, Rep3Operand) {
        (self.0.clone(), self.1.clone())
    }

    fn operands_mut(&mut self) -> (&mut Rep3Operand, Option<&mut Rep3Operand>) {
        (&mut self.0, Some(&mut self.1))
    }

    fn lhs(&self) -> &Rep3Operand {
        &self.0
    }

    fn rhs(&self) -> Option<&Rep3Operand> {
        Some(&self.1)
    }

    fn combine_lookups_rep3_batched<F: JoltField, N: Rep3Network>(
        &self,
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        io_ctx: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        assert_eq!(vals.len(), 1);
        Ok(vals[0]
            .iter()
            .map(|lowest_bit| {
                rep3::arithmetic::sub_public_by_shared(F::one(), *lowest_bit, io_ctx.id)
            })
            .collect::<Vec<_>>())
    }

    fn to_indices_intermediate<F: JoltField>(
        &self,
        _: PartyID,
    ) -> FutureRep3Ring<u128, Option<Rep3RingShare<u128>>> {
        FutureRep3Ring::a2b(self.0.as_arithmetic() + self.1.as_arithmetic())
    }

    fn to_indices_rep3(
        &self,
        z: Option<Rep3RingShare<u128>>,
        C: usize,
        log_M: usize,
    ) -> Vec<Rep3RingShare<u32>> {
        rep3_add_and_chunk_operands(&z.unwrap(), C, log_M)
    }

    fn output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3JoltInstruction],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // (((self.0.as_public() as u32 as i32 + self.1.as_public() as u32 as i32) % 2 == 0)
        //         as u64)
        //         .into()

        let (x, y): (Vec<_>, Vec<_>) = steps
            .into_iter()
            .map(|st| (st.lhs().as_binary(), st.rhs().unwrap().as_binary()))
            .unzip();
        let z = rep3_ring::binary::add_many(&x, &y, io_ctx)?;
        let z_is_even = z.iter().map(|z| z.is_even()).collect::<Vec<_>>();
        izip!(z_is_even, out.into_iter()).for_each(|(r, out)| {
            *out = FutureRep3Ring::bit_inject_to_field(r);
        });
        Ok(())
    }
}
