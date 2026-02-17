use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN>
    for Rep3RISCVCycle<VirtualAssertValidUnsignedRemainder>
{
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
        // remainder (left), divisor (right)
        // valid if remainder == 0 OR remainder < divisor
        let (remainders, divisors): (Vec<_>, Vec<_>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (l.as_binary(), r.as_binary())
            })
            .unzip();
        let rem_is_zero = rep3_ring::binary::is_zero_many(&remainders, io_ctx)?;
        let rem_lt_div = rep3_ring::arithmetic::lt_many(&remainders, &divisors, io_ctx)?;
        let rem_is_zero_u32: Vec<_> = rem_is_zero.iter().map(|b| bit_to_ring32(*b)).collect();
        let rem_lt_div_u32: Vec<_> = rem_lt_div.iter().map(|b| bit_to_ring32(*b)).collect();
        rep3_ring::binary::or_many(&rem_is_zero_u32, &rem_lt_div_u32, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(z, out)| {
                *out = FutureRep3Ring::cast_to_field_b2a(z);
            });
        Ok(())
    }
}
