use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<LW> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        _steps: &[&impl Rep3LookupQuery<XLEN>],
        _io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        for o in out {
            *o = FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share());
        }
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<LH> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        _steps: &[&impl Rep3LookupQuery<XLEN>],
        _io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        for o in out {
            *o = FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share());
        }
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<LB> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        _steps: &[&impl Rep3LookupQuery<XLEN>],
        _io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        for o in out {
            *o = FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share());
        }
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<LBU> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        _steps: &[&impl Rep3LookupQuery<XLEN>],
        _io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        for o in out {
            *o = FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share());
        }
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<LHU> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        _steps: &[&impl Rep3LookupQuery<XLEN>],
        _io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        for o in out {
            *o = FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share());
        }
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<LWU> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        _steps: &[&impl Rep3LookupQuery<XLEN>],
        _io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        for o in out {
            *o = FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share());
        }
        Ok(())
    }
}

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualLW> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        _steps: &[&impl Rep3LookupQuery<XLEN>],
        _io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        for o in out {
            *o = FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share());
        }
        Ok(())
    }
}
