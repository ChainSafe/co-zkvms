use jolt_core::poly::eq_poly::EqPlusOnePolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction::CircuitFlags;
use jolt_core::zkvm::spartan::pc::PCSumcheck;
use jolt_core::zkvm::witness::VirtualPolynomial;

use crate::poly::opening_proof::Rep3OpeningAccumulator;
use crate::subprotocols::sumcheck::PublicSumcheckInstance;
use jolt_core::field::JoltField;

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for PCSumcheck<F> {
    fn degree(&self) -> usize {
        self.degree()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim()
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        // Get r_cycle from the SpartanOuter sumcheck opening point.
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let num_cycles_bits = self.num_rounds();
        let (r_cycle, _) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Shift openings from accumulator.
        let (_, unexpanded_pc_eval_at_shift_r) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, pc_eval_at_shift_r) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let (_, is_noop_eval_at_shift_r) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let batched_eval_at_shift_r = unexpanded_pc_eval_at_shift_r
            + self.gamma() * pc_eval_at_shift_r
            + self.gamma_squared() * is_noop_eval_at_shift_r;

        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::<F>::new(r_cycle.to_vec()).evaluate(r);
        batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        self.normalize_opening_point(opening_point)
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        // claims order: [UnexpandedPC, PC, IsNoopFlag]
        let [unexpanded_pc_eval, pc_eval, is_noop_eval]: [F; 3] = claims
            .try_into()
            .expect("PCSumcheck expects 3 opening claims");

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
        );
    }
}
