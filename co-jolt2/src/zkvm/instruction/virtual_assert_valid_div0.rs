use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualAssertValidDiv0> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        // divisor (left), quotient (right)
        // if divisor == 0: check quotient == MAX, else: return 1
        let (divisors, quotients): (Vec<_>, Vec<_>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (l.as_binary(), r.as_arithmetic_u32())
            })
            .unzip();
        let divisor_is_zero = rep3_ring::binary::is_zero_many(&divisors, io_ctx)?;
        let max_vals: Vec<_> = (0..steps.len())
            .map(|_| {
                rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, RingElement(u32::MAX))
            })
            .collect();
        let quotient_eq_max = rep3_ring::arithmetic::eq_many(&quotients, &max_vals, io_ctx)?;
        let ones: Vec<_> = (0..steps.len())
            .map(|_| rep3_ring::binary::promote_to_trivial_share(io_ctx.id, &RingElement(1u32)))
            .collect();
        let quotient_eq_max_u32: Vec<_> =
            quotient_eq_max.iter().map(|b| bit_to_ring32(*b)).collect();
        let divisor_is_zero_u32: Vec<_> =
            divisor_is_zero.iter().map(|b| bit_to_ring32(*b)).collect();
        // if divisor == 0: result = (quotient == MAX), else: result = 1
        rep3_ring::binary::cmux_many(&divisor_is_zero_u32, &quotient_eq_max_u32, &ones, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(z, out)| {
                *out = FutureRep3Ring::cast_to_field_b2a(z);
            });
        Ok(())
    }
}
