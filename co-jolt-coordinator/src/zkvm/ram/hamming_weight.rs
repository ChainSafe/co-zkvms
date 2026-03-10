use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::ram::hamming_weight::HammingWeightSumcheck;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use crate::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulator;
use crate::subprotocols::sumcheck::PublicSumcheckInstance;

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
                let (_, ra) = accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::RamHammingWeight,
                );
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
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .0
            .r
            .clone();

        accumulator.append_sparse(
            transcript,
            (0..self.d()).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamHammingWeight,
            &opening_point.r,
            &r_cycle,
            claims,
        );
    }
}
