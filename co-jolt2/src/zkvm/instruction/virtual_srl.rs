use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualSRL> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (
            self.register_state.rs1_operand(),
            self.register_state.rs2_operand(),
        )
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        use crate::utils::instruction_utils::operand_to_binary_u128;

        // Same algorithm as VirtualSRLI: the bitmask (rs2) is public because
        // VirtualShiftRightBitmask produces a trivial share from a public shift amount.
        //
        // Lookup table formula (MSB-first iteration):
        //   entry = 0
        //   for i in 0..XLEN:
        //     entry = entry * (1 + y_i) + x_i * y_i
        //
        // When y_i is public, this simplifies:
        //   y_i=1 → entry = entry * 2 + x_i  (i.e. coeff doubles, x_i contributes)
        //   y_i=0 → entry = entry * 1         (coeff unchanged, no contribution)
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let bitmask = r.as_public();
            let x_bits = operand_to_binary_u128(&l, io_ctx.id);

            let mut y_bits = Vec::with_capacity(XLEN);
            for i in 0..XLEN {
                y_bits.push(((bitmask >> (XLEN - 1 - i)) & 1) != 0);
            }

            let mut coeff = RingElement(1u64);
            let mut result_shares: Vec<Rep3RingShare<u64>> = Vec::with_capacity(XLEN);

            for i in 0..XLEN {
                let x_i = (x_bits >> (XLEN - 1 - i)) & RingElement(1u128);
                let x_i_u64: Rep3RingShare<u64> = downcast(x_i);

                if y_bits[i] {
                    result_shares.push(&x_i_u64 * coeff);
                    coeff = coeff * RingElement(2u64);
                }
            }

            let result = result_shares
                .into_iter()
                .fold(Rep3RingShare::default(), |acc, x| acc + x);

            *out = FutureRep3Ring::cast_to_field_b2a(result);
        });
        Ok(())
    }
}
