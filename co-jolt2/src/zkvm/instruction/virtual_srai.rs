use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualSRAI> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (self.register_state.rs1_operand(), Rep3Operand::Public(self.instruction.operands.imm.into()))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        steps: &[&impl Rep3LookupQuery<XLEN>],
        io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        use crate::utils::instruction_utils::operand_to_binary_wide;

        // Same as VirtualSRA but bitmask comes from immediate instead of rs2.
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let bitmask = r.as_public();
            let x_bits = operand_to_binary_wide(&l, io_ctx.id);

            let mut y_bits = Vec::with_capacity(XLEN);
            for i in 0..XLEN {
                y_bits.push(((bitmask >> (XLEN - 1 - i)) & 1) != 0);
            }

            // Compute SRL entry
            let num_ones = (bitmask as XlenInt).count_ones();
            let mut ones_seen = 0u32;

            let mut srl_result = Rep3RingShare::default();
            for i in 0..XLEN {
                let x_i = (x_bits >> (XLEN - 1 - i)) & RingElement(1 as LookupIndexInt);
                let x_i_xlen: Rep3RingShare<XlenInt> = downcast(x_i);

                if y_bits[i] {
                    let weight = RingElement((1 as XlenInt) << (num_ones - 1 - ones_seen));
                    srl_result = srl_result + &x_i_xlen * weight;
                    ones_seen += 1;
                }
            }

            // Sign extension: sum_{i=1..XLEN} (1 << i) * (1 - y_i)
            let mut sign_extension = 0 as XlenInt;
            for i in 1..XLEN {
                if !y_bits[i] {
                    sign_extension += (1 as XlenInt) << i;
                }
            }

            // sign_bit is the MSB of x (shared binary bit)
            let sign_bit = (x_bits >> (XLEN - 1)) & RingElement(1 as LookupIndexInt);
            let sign_bit_xlen: Rep3RingShare<XlenInt> = downcast(sign_bit);

            // result = srl_result + sign_bit * sign_extension
            let result = srl_result + &sign_bit_xlen * RingElement(sign_extension);

            *out = FutureRep3Ring::cast_to_field_b2a(result);
        });
        Ok(())
    }
}
