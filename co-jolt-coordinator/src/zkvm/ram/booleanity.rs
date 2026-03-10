use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::ram::booleanity::BooleanitySumcheck;
use jolt_core::zkvm::witness::{CommittedPolynomial, DTH_ROOT_OF_K};

use jolt_core::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulator;
use crate::subprotocols::sumcheck::PublicSumcheckInstance;

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for BooleanitySumcheck<F> {
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
        let d = self.d();
        let log_K_chunk = DTH_ROOT_OF_K.log_2();

        let ra_claims: Vec<F> = (0..d)
            .map(|i| {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::RamRa(i),
                        SumcheckId::RamBooleanity,
                    )
                    .1
            })
            .collect();

        let (r_address_prime, r_cycle_prime) = r.split_at(log_K_chunk);

        // normalize_opening_point reverses each part separately (LowToHigh → BigEndian)
        let r_address_prime_rev: Vec<F::Challenge> =
            r_address_prime.iter().copied().rev().collect();
        let r_cycle_prime_rev: Vec<F::Challenge> = r_cycle_prime.iter().copied().rev().collect();

        let eq_address = EqPolynomial::<F>::mle(self.r_address(), &r_address_prime_rev);
        let eq_cycle = EqPolynomial::<F>::mle(self.r_cycle(), &r_cycle_prime_rev);

        let booleanity_sum: F = ra_claims
            .iter()
            .zip(self.gamma_powers().iter())
            .map(|(ra, gamma)| *gamma * (ra.square() - *ra))
            .sum();

        eq_address * eq_cycle * booleanity_sum
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
        let (r_address, r_cycle) = opening_point.split_at(DTH_ROOT_OF_K.log_2());
        accumulator.append_sparse(
            transcript,
            (0..self.d()).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamBooleanity,
            &r_address.r,
            &r_cycle.r,
            claims,
        );
    }
}
