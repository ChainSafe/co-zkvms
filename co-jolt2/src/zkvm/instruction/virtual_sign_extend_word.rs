use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualSignExtendWord> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (self.register_state.rs1_operand(), Rep3Operand::Public(0))
    }

    fn to_lookup_operands(&self, party_id: PartyID) -> (Rep3RingShare<u64>, Rep3RingShare<u128>) {
        let (left, right) = <Self as Rep3LookupQuery<XLEN>>::to_instruction_inputs(self);
        (
            Rep3RingShare::default(),
            left.as_arithmetic_or_trivial::<u128>(party_id)
                + right.as_arithmetic_or_trivial::<u128>(party_id),
        )
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // Extract sign bit of lower word, conditionally extend upper bits
        let half = XLEN / 2;
        let sign_bit_mask = RingElement(1u64 << (half - 1));
        // Use wrapping_shl to avoid overflow when half == 32 (XLEN == 64)
        let lower_mask = RingElement(1u64.wrapping_shl(half as u32).wrapping_sub(1));
        let upper_mask = RingElement(!lower_mask.0);

        let sign_bits: Vec<_> = steps
            .iter()
            .map(|st| {
                let (l, _) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                l.as_binary_or_trivial(io_ctx.id) & sign_bit_mask
            })
            .collect();
        let is_positive = rep3_ring::binary::is_zero_many(&sign_bits, io_ctx)?;
        let is_positive_u64: Vec<Rep3RingShare<u64>> = is_positive
            .iter()
            .map(|b| {
                Rep3RingShare::new_ring(
                    RingElement(u8::from(b.a.0) as u64),
                    RingElement(u8::from(b.b.0) as u64),
                )
            })
            .collect();
        let zeros: Vec<_> = (0..steps.len()).map(|_| Rep3RingShare::default()).collect();
        let upper_ones: Vec<_> = (0..steps.len())
            .map(|_| rep3_ring::binary::promote_to_trivial_share(io_ctx.id, &upper_mask))
            .collect();
        // if positive: upper = 0, else: upper = upper_mask
        let uppers = rep3_ring::binary::cmux_many(&is_positive_u64, &zeros, &upper_ones, io_ctx)?;
        // Combine: lower bits from input XOR upper bits from extension
        // (non-overlapping bit positions, so XOR = OR)
        itertools::izip!(steps, uppers, out).for_each(|(step, upper, out)| {
            let (l, _) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let result = (l.as_binary_or_trivial(io_ctx.id) & lower_mask) ^ upper;
            *out = FutureRep3Ring::cast_to_field_b2a(downcast(result));
        });
        Ok(())
    }
}
