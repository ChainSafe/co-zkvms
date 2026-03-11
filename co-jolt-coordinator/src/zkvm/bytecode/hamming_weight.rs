use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::bytecode::hamming_weight::HammingWeightSumcheck;
use jolt_core::zkvm::witness::CommittedPolynomial;

use crate::poly::opening_proof::Rep3OpeningAccumulator;
use crate::subprotocols::sumcheck::PublicSumcheckInstance;
use jolt_core::field::JoltField;

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for HammingWeightSumcheck<F> {
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
        _r: &[F::Challenge],
    ) -> F {
        self.gamma_powers()
            .iter()
            .enumerate()
            .map(|(i, gamma)| {
                let ra = accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeRa(i),
                        SumcheckId::BytecodeHammingWeight,
                    )
                    .1;
                ra * gamma
            })
            .sum::<F>()
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
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                jolt_core::zkvm::witness::VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();

        accumulator.append_sparse(
            transcript,
            (0..self.d()).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeHammingWeight,
            &opening_point.r,
            &r_cycle,
            claims,
        );
    }
}
