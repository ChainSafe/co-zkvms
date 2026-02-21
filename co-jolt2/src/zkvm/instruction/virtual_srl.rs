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
        // Vanilla lookup table formula (MSB-first iteration):
        //   entry = 0
        //   for i in 0..XLEN:
        //     entry = entry * (1 + y_i) + x_i * y_i
        //
        // This reads the x bits at y-masked positions as a binary number.
        // First y=1 position gets the highest weight (2^(k-1) where k = popcount(y)).
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let bitmask = r.as_public();
            let x_bits = operand_to_binary_u128(&l, io_ctx.id);

            let num_ones = (bitmask as u64).count_ones();
            let mut ones_seen = 0u32;

            let mut result = Rep3RingShare::default();
            for i in 0..XLEN {
                let y_i = ((bitmask >> (XLEN - 1 - i)) & 1) != 0;
                if y_i {
                    let weight = RingElement(1u64 << (num_ones - 1 - ones_seen));
                    let x_i = (x_bits >> (XLEN - 1 - i)) & RingElement(1u128);
                    let x_i_u64: Rep3RingShare<u64> = downcast(x_i);
                    result = result + &x_i_u64 * weight;
                    ones_seen += 1;
                }
            }

            *out = FutureRep3Ring::cast_to_field_b2a(result);
        });
        Ok(())
    }
}
