use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualRev8W> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (self.register_state.rs1_operand(), Rep3Operand::Public(0))
    }

    fn to_lookup_index(&self, party_id: PartyID) -> FutureRep3Ring<u128, Rep3RingShare<u128>> {
        let (left, right) = <Self as Rep3LookupQuery<XLEN>>::to_instruction_inputs(self);
        let l = left.as_arithmetic_or_trivial_u128(party_id);
        let r = right.as_arithmetic_or_trivial_u128(party_id);
        FutureRep3Ring::a2b(l + r)
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // Byte reversal is a bit permutation — apply component-wise
        // W-variant: operates on lower 32 bits
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, _) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let x: Rep3RingShare<u32> = downcast(l.as_binary_or_trivial(io_ctx.id));
            let reversed = Rep3RingShare::new(x.a.0.swap_bytes(), x.b.0.swap_bytes());
            let reversed_u64 =
                Rep3RingShare::new_ring(RingElement(reversed.a.0 as u64), RingElement(reversed.b.0 as u64));
            *out = FutureRep3Ring::cast_to_field_b2a(reversed_u64);
        });
        Ok(())
    }
}
