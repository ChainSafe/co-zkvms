use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualSRA> {
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

        // Arithmetic right shift = logical right shift + sign extension.
        //
        // Lookup table formula (MSB-first):
        //   srl_entry = 0; sign_extension = 0
        //   for i in 0..XLEN:
        //     srl_entry = srl_entry * (1 + y_i) + x_i * y_i
        //     if i != 0: sign_extension += (1 << i) * (1 - y_i)
        //   result = srl_entry + sign_bit * sign_extension
        //
        // sign_bit = MSB of x (shared), y_bits are public (bitmask from VirtualShiftRightBitmask).
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let bitmask = r.as_public();
            let x_bits = operand_to_binary_u128(&l, io_ctx.id);

            let mut y_bits = Vec::with_capacity(XLEN);
            for i in 0..XLEN {
                y_bits.push(((bitmask >> (XLEN - 1 - i)) & 1) != 0);
            }

            // Compute SRL entry (same as VirtualSRL)
            let num_ones = (bitmask as u64).count_ones();
            let mut ones_seen = 0u32;

            let mut srl_result = Rep3RingShare::default();
            for i in 0..XLEN {
                let x_i = (x_bits >> (XLEN - 1 - i)) & RingElement(1u128);
                let x_i_u64: Rep3RingShare<u64> = downcast(x_i);

                if y_bits[i] {
                    let weight = RingElement(1u64 << (num_ones - 1 - ones_seen));
                    srl_result = srl_result + &x_i_u64 * weight;
                    ones_seen += 1;
                }
            }

            // Compute sign extension: sum_{i=1..XLEN} (1 << i) * (1 - y_i)
            // Since y_bits are public, this is a constant.
            let mut sign_extension = 0u64;
            for i in 1..XLEN {
                if !y_bits[i] {
                    sign_extension += 1u64 << i;
                }
            }

            // sign_bit is the MSB of x (shared binary bit)
            let sign_bit = (x_bits >> (XLEN - 1)) & RingElement(1u128);
            let sign_bit_u64: Rep3RingShare<u64> = downcast(sign_bit);

            // result = srl_result + sign_bit * sign_extension
            let result = srl_result + &sign_bit_u64 * RingElement(sign_extension);

            *out = FutureRep3Ring::cast_to_field_b2a(result);
        });
        Ok(())
    }
}
