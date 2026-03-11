use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualXORROTW16> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        const N_ROT: u32 = 16 % 32;
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let xored: Rep3RingShare<u32> =
                downcast(l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id));
            // 32-bit right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(xored.a.0.rotate_right(N_ROT)),
                RingElement(xored.b.0.rotate_right(N_ROT)),
            );
            let rotated_xlen = Rep3RingShare::new_ring(
                RingElement(rotated.a.0 as XlenInt),
                RingElement(rotated.b.0 as XlenInt),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(rotated_xlen);
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualXORROTW12> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        const N_ROT: u32 = 12 % 32;
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let xored: Rep3RingShare<u32> =
                downcast(l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id));
            // 32-bit right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(xored.a.0.rotate_right(N_ROT)),
                RingElement(xored.b.0.rotate_right(N_ROT)),
            );
            let rotated_xlen = Rep3RingShare::new_ring(
                RingElement(rotated.a.0 as XlenInt),
                RingElement(rotated.b.0 as XlenInt),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(rotated_xlen);
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualXORROTW8> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        const N_ROT: u32 = 8 % 32;
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let xored: Rep3RingShare<u32> =
                downcast(l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id));
            // 32-bit right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(xored.a.0.rotate_right(N_ROT)),
                RingElement(xored.b.0.rotate_right(N_ROT)),
            );
            let rotated_xlen = Rep3RingShare::new_ring(
                RingElement(rotated.a.0 as XlenInt),
                RingElement(rotated.b.0 as XlenInt),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(rotated_xlen);
        });
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualXORROTW7> {
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
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<XlenInt, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        const N_ROT: u32 = 7 % 32;
        itertools::izip!(steps, out).for_each(|(step, out)| {
            let (l, r) = Rep3LookupQuery::<XLEN>::to_instruction_inputs(*step);
            let xored: Rep3RingShare<u32> =
                downcast(l.as_binary_or_trivial(io_ctx.id) ^ r.as_binary_or_trivial(io_ctx.id));
            // 32-bit right rotation applied component-wise (bit permutation)
            let rotated = Rep3RingShare::new_ring(
                RingElement(xored.a.0.rotate_right(N_ROT)),
                RingElement(xored.b.0.rotate_right(N_ROT)),
            );
            let rotated_xlen = Rep3RingShare::new_ring(
                RingElement(rotated.a.0 as XlenInt),
                RingElement(rotated.b.0 as XlenInt),
            );
            *out = FutureRep3Ring::cast_to_field_b2a(rotated_xlen);
        });
        Ok(())
    }
}
