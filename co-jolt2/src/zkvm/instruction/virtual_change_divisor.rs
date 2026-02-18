use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualChangeDivisor> {
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
        // if divisor == -1 (0xFFFFFFFF) && dividend == INT_MIN: return 1, else: return divisor
        let (dividends, divisors): (Vec<_>, Vec<_>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (
                    l.as_arithmetic_or_trivial::<u32>(io_ctx.id),
                    r.as_arithmetic_or_trivial::<u32>(io_ctx.id),
                )
            })
            .unzip();
        let neg_ones: Vec<_> = (0..steps.len())
            .map(|_| {
                rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, RingElement(u32::MAX))
            })
            .collect();
        let int_mins: Vec<_> = (0..steps.len())
            .map(|_| {
                rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, RingElement(1u32 << 31))
            })
            .collect();
        let div_eq_neg1 = rep3_ring::arithmetic::eq_many(&divisors, &neg_ones, io_ctx)?;
        let dividend_eq_int_min = rep3_ring::arithmetic::eq_many(&dividends, &int_mins, io_ctx)?;
        // Both conditions must hold
        let both = rep3_ring::binary::and_many(&div_eq_neg1, &dividend_eq_int_min, io_ctx)?;
        let both_u32: Vec<_> = both.iter().map(|b| bit_to_ring32(*b)).collect();
        let ones: Vec<_> = (0..steps.len())
            .map(|_| rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, RingElement(1u32)))
            .collect();
        // if both: output 1, else: output divisor
        rep3_ring::binary::cmux_many(&both_u32, &ones, &divisors, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(z, out)| {
                *out = FutureRep3Ring::cast_to_field_b2a(z);
            });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualChangeDivisorW> {
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
        // if divisor == -1 (0xFFFFFFFF) && dividend == INT_MIN: return 1, else: return divisor
        let (dividends, divisors): (Vec<_>, Vec<_>) = steps
            .iter()
            .map(|st| {
                let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*st);
                (
                    l.as_arithmetic_or_trivial::<u32>(io_ctx.id),
                    r.as_arithmetic_or_trivial::<u32>(io_ctx.id),
                )
            })
            .unzip();
        let neg_ones: Vec<_> = (0..steps.len())
            .map(|_| {
                rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, RingElement(u32::MAX))
            })
            .collect();
        let int_mins: Vec<_> = (0..steps.len())
            .map(|_| {
                rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, RingElement(1u32 << 31))
            })
            .collect();
        let div_eq_neg1 = rep3_ring::arithmetic::eq_many(&divisors, &neg_ones, io_ctx)?;
        let dividend_eq_int_min = rep3_ring::arithmetic::eq_many(&dividends, &int_mins, io_ctx)?;
        // Both conditions must hold
        let both = rep3_ring::binary::and_many(&div_eq_neg1, &dividend_eq_int_min, io_ctx)?;
        let both_u32: Vec<_> = both.iter().map(|b| bit_to_ring32(*b)).collect();
        let ones: Vec<_> = (0..steps.len())
            .map(|_| rep3_ring::arithmetic::promote_to_trivial_share(io_ctx.id, RingElement(1u32)))
            .collect();
        // if both: output 1, else: output divisor
        rep3_ring::binary::cmux_many(&both_u32, &ones, &divisors, io_ctx)?
            .into_iter()
            .zip(out)
            .for_each(|(z, out)| {
                *out = FutureRep3Ring::cast_to_field_b2a(z);
            });
        Ok(())
    }
}
