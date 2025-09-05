use ark_std::log2;
use itertools::izip;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::{field::JoltField, utils::future_ring::FutureRep3Ring};
use jolt_core::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use mpc_core::protocols::{
    rep3::{
        network::{IoContext, Rep3Network},
        Rep3PrimeFieldShare,
    },
    rep3_ring::Rep3RingShare,
};

use super::{JoltInstruction, Rep3JoltInstruction, Rep3Operand, SubtableIndices};
use crate::utils::instruction_utils::{
    chunk_operand_usize, concatenate_lookups_rep3_batched, rep3_chunk_operand,
};
use jolt_core::utils::instruction_utils::concatenate_lookups;

#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ADVICEInstruction<const WORD_SIZE: usize>(pub Rep3Operand);

impl<const WORD_SIZE: usize> JoltInstruction for ADVICEInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        match &self.0 {
            Rep3Operand::Public(x) => (*x, 0),
            _ => panic!("ADVICEInstruction::operands called with non-public operand"),
        }
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C / 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
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

    fn lookup_entry<F: JoltField>(&self) -> F {
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

impl<const WORD_SIZE: usize> Rep3JoltInstruction for ADVICEInstruction<WORD_SIZE> {
    fn operands_rep3(&self) -> (Rep3Operand, Rep3Operand) {
        (self.0.clone(), Rep3Operand::Public(0))
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
        vals: Vec<Vec<Rep3PrimeFieldShare<F>>>,
        C: usize,
        M: usize,
        _: &mut IoContext<N>,
    ) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
        Ok(concatenate_lookups_rep3_batched(
            vals,
            C / 2,
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
        _: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        izip!(steps, out).for_each(|(st, out)| {
            *out = FutureRep3Ring::cast_to_field(st.lhs().as_arithmetic_u32());
        });
        Ok(())
    }
}
