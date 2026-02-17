use super::*;

impl<const XLEN: usize> Rep3LookupQuery<XLEN> for Rep3RISCVCycle<VirtualAdvice> {
    fn to_instruction_inputs(&self) -> (Rep3Operand, Rep3Operand) {
        (Rep3Operand::Public(0), Rep3Operand::Public(0))
    }

    fn to_lookup_operands(&self) -> (Rep3Operand, Rep3Operand) {
        // Vanilla: (0, advice_value)
        todo!("to_lookup_operands: VirtualAdvice — requires advice value access")
    }

    fn to_lookup_output_batched<'a, F: JoltField, N: Rep3Network>(
        &self,
        _steps: &[&impl Rep3LookupQuery<XLEN>],
        _io_ctx: &mut IoContext<N>,
        out: impl IntoIterator<Item = &'a mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>,
    ) -> eyre::Result<()> {
        todo!("VirtualAdvice output requires advice value access")
    }
}
