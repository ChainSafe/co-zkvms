use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualAdvice> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        // Match vanilla: VirtualAdvice does not contribute to instruction inputs.
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_index(&self, party_id: PartyID) -> FutureRep3Ring<u128, Rep3RingShare<u128>> {
        let advice = match XLEN {
            #[cfg(test)]
            8 => self.instruction.advice as u8 as u128,
            32 => self.instruction.advice as u32 as u128,
            64 => self.instruction.advice as u128,
            _ => panic!("{XLEN}-bit word size is unsupported"),
        };
        FutureRep3Ring::Ready(rep3_ring::arithmetic::promote_to_trivial_share(
            party_id,
            RingElement(advice),
        ))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // RangeCheckTable is the identity function: output = input = advice value.
        itertools::izip!(steps, out).for_each(|(step, out)| {
            // `VirtualAdvice::to_lookup_index` is `Ready(trivial_share(advice))`, so we can
            // recover the public advice value locally without any MPC communication.
            let idx_fut = Rep3LookupQuery::<XLEN>::to_lookup_index(*step, io_ctx.id);
            let advice_u128 = match idx_fut {
                FutureRep3Ring::Ready(s) => s.a.0,
                _ => unreachable!("VirtualAdvice lookup index must be Ready(trivial_share)"),
            };
            *out = FutureRep3Ring::Ready(mpc_core::protocols::rep3::arithmetic::promote_to_trivial_share(
                io_ctx.id,
                F::from_u64(advice_u128 as u64),
            ));
        });
        Ok(())
    }
}
