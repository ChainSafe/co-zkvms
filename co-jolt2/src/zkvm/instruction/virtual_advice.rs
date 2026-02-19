use mpc_core::protocols::rep3;

use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualAdvice> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        // Pass advice value as a public operand so to_lookup_output_batched can access it
        (
            Rep3Operand::Public(self.instruction.advice),
            Rep3Operand::Public(0),
        )
    }

    fn to_lookup_operands(&self, party_id: PartyID) -> (Rep3RingShare<u64>, Rep3RingShare<u128>) {
        // Vanilla: (0, advice_value truncated to XLEN bits)
        // Advice is a public value stored in the instruction.
        let advice = self.instruction.advice as u128;
        (
            Rep3RingShare::default(),
            rep3_ring::arithmetic::promote_to_trivial_share(party_id, RingElement(advice)),
        )
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // RangeCheckTable is the identity function: output = input = advice value.
        // Advice is public (stored in instruction, not shared).
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, _) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let advice = l.as_public();
            let advice_u32 = advice as u32;
            *out = FutureRep3Ring::Ready(
                rep3::arithmetic::promote_to_trivial_share(io_ctx.id, F::from(advice_u32 as u64))
                    .into(),
            );
        });
        Ok(())
    }
}
