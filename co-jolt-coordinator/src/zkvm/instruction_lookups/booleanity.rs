use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use crate::poly::opening_proof::Rep3OpeningAccumulator;
use jolt_core::field::JoltField;

use crate::zkvm::dag::stage::Rep3SumcheckInstance;

const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3BooleanitySumcheck<F: JoltField> {
    gamma: [F; D],
    r_address: Vec<F::Challenge>,
    log_T: usize,
}

impl<F: JoltField> Rep3BooleanitySumcheck<F> {
    pub fn new<T: Transcript>(transcript: &mut T, log_T: usize) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let mut gamma_powers = [F::one(); D];
        for i in 1..D {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
        let r_address: Vec<F::Challenge> = transcript.challenge_vector_optimized::<F>(LOG_K_CHUNK);

        Self { gamma: gamma_powers, r_address, log_T }
    }

    /// Return gamma powers so the worker can use them.
    pub fn gamma(&self) -> [F; D] {
        self.gamma
    }

    /// Return r_address so the worker can use them.
    pub fn r_address(&self) -> &[F::Challenge] {
        &self.r_address
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K_CHUNK + self.log_T
    }

    fn input_claim_public(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(&self, accumulator: &Rep3OpeningAccumulator<F>, r_prime: &[F::Challenge]) -> F {
        let ra_claims = (0..D).map(|i| {
            accumulator
                .get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionBooleanity,
                )
                .1
        });
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter)
            .0
            .r
            .clone();

        EqPolynomial::<F>::mle(
            r_prime,
            &self.r_address.iter().cloned().rev().chain(r_cycle.iter().cloned().rev()).collect::<Vec<F::Challenge>>(),
        ) * self.gamma.iter().zip(ra_claims).fold(F::zero(), |acc, (gamma, ra)| (ra.square() - ra) * gamma + acc)
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address, r_cycle) = opening_point.split_at(LOG_K_CHUNK);
        let mut r_big_endian: Vec<F::Challenge> = r_address.iter().rev().copied().collect();
        r_big_endian.extend(r_cycle.iter().copied().rev());
        OpeningPoint::new(r_big_endian)
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        accumulator.append_sparse(
            transcript,
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionBooleanity,
            &r_sumcheck.r[..LOG_K_CHUNK],
            &r_sumcheck.r[LOG_K_CHUNK..],
            claims,
        );
    }
}
