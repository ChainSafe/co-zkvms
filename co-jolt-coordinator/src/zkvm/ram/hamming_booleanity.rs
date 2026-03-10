use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheck;
use jolt_core::zkvm::witness::VirtualPolynomial;

use jolt_core::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulator;
use crate::subprotocols::sumcheck::PublicSumcheckInstance;

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for HammingBooleanitySumcheck<F> {
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
        let (_, h_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        );

        let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let r_cycle_rev: Vec<F::Challenge> = r_cycle.r.iter().cloned().rev().collect();
        let eq = EqPolynomial::<F>::mle(r, &r_cycle_rev);

        (h_claim.square() - h_claim) * eq
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
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            opening_point,
            claims[0],
        );
    }
}
