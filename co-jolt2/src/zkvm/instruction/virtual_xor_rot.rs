use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualXORROT32> {
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
        let n_rot: u32 = 32 % XLEN as u32;
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let xored = l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id);
            // Right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(xored.a.0.rotate_right(n_rot)),
                RingElement(xored.b.0.rotate_right(n_rot)),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(downcast(rotated));
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualXORROT24> {
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
        let n_rot: u32 = 24 % XLEN as u32;
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let xored = l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id);
            // Right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(xored.a.0.rotate_right(n_rot)),
                RingElement(xored.b.0.rotate_right(n_rot)),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(downcast(rotated));
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualXORROT16> {
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
        let n_rot: u32 = 16 % XLEN as u32;
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let xored = l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id);
            // Right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(xored.a.0.rotate_right(n_rot)),
                RingElement(xored.b.0.rotate_right(n_rot)),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(downcast(rotated));
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualXORROT63> {
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
        let n_rot: u32 = 63 % XLEN as u32;
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let xored = l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id);
            // Right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(xored.a.0.rotate_right(n_rot)),
                RingElement(xored.b.0.rotate_right(n_rot)),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(downcast(rotated));
        });
        Ok(())
    }
}
