use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualSRLI> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (
            self.register_state.rs1_operand(),
            Rep3Operand::Public(self.instruction.operands.imm),
        )
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u64, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        use crate::utils::instruction_utils::operand_to_binary_u128;

        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            // l = rs1 (shared), r = imm (public shift amount)
            let shift_amount = r.as_public(); // immediate is always public
            let x_bits = operand_to_binary_u128(&l, io_ctx.id);

            // Compute VirtualSRL output: entry = sum_i (x_i * y_i * prod_{j<i}(1 + y_j))
            // Since y (shift amount) is public, we can compute this locally.
            // The y bits come from the shift amount immediate value.
            let mut y_bits = Vec::with_capacity(XLEN);
            for i in 0..XLEN {
                y_bits.push(((shift_amount >> (XLEN - 1 - i)) & 1) != 0);
            }

            // Accumulate: entry = sum over i of (x_i * y_i * coeff_i)
            // where coeff_i = product_{j < i} (1 + y_j)
            let mut coeff = RingElement(1u64);
            let mut result_shares: Vec<Rep3RingShare<u64>> = Vec::with_capacity(XLEN);

            for i in 0..XLEN {
                let x_i = (x_bits >> (XLEN - 1 - i)) & RingElement(1u128);
                let x_i_u64: Rep3RingShare<u64> = downcast(x_i);

                if y_bits[i] {
                    // Contribute x_i * coeff to result
                    result_shares.push(&x_i_u64 * coeff);
                    // Update coeff: multiply by (1 + 1) = 2
                    coeff = coeff * RingElement(2u64);
                } else {
                    // y_i = 0, contribute 0 (skip)
                    // Update coeff: multiply by (1 + 0) = 1 (no change)
                }
            }

            // Sum all contributions (binary addition, no communication)
            let result = result_shares
                .into_iter()
                .fold(Rep3RingShare::default(), |acc, x| acc + x);

            // Cast to field share
            *out = FutureRep3Ring::cast_to_field_b2a(result);
        });
        Ok(())
    }
}
