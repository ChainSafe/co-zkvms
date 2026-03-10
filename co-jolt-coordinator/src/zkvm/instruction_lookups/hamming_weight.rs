use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use crate::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulator;

use crate::zkvm::dag::stage::Rep3SumcheckInstance;

const DEGREE: usize = 1;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3HammingWeightSumcheck<F: JoltField> {
    gamma: [F; D],
}

impl<F: JoltField> Rep3HammingWeightSumcheck<F> {
    pub fn new<T: Transcript>(transcript: &mut T) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        Self {
            gamma: gamma_powers,
        }
    }

    /// Return gamma powers so the worker can use them.
    pub fn gamma(&self) -> [F; D] {
        self.gamma
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK
    }

    fn input_claim_public(&self) -> F {
        self.gamma.iter().copied().sum()
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        _r: &[F::Challenge],
    ) -> F {
        let ra_claims = (0..D).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionHammingWeight,
                )
                .1
        });

        self.gamma
            .iter()
            .zip(ra_claims)
            .map(|(gamma, ra)| ra * gamma)
            .sum()
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        // Get r_cycle from the accumulator (stored during Spartan outer sumcheck).
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();

        accumulator.append_sparse(
            transcript,
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionHammingWeight,
            &opening_point.r,
            &r_cycle,
            claims,
        );
    }
}
