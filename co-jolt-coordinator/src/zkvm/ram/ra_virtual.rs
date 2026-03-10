use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::ram::ra_virtual::RaSumcheck;
use jolt_core::zkvm::witness::CommittedPolynomial;

use crate::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulator;
use crate::subprotocols::sumcheck::PublicSumcheckInstance;

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for RaSumcheck<F> {
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
        // RaSumcheck binds LowToHigh; normalize_opening_point reverses r.
        // expected_output_claim in vanilla uses r.iter().rev(), so we match that.
        let r_rev: Vec<F::Challenge> = r.iter().cloned().rev().collect();
        let gamma = self.gamma();
        let r_cycle = self.r_cycle();

        let eq_eval = gamma[0] * EqPolynomial::<F>::mle(&r_cycle[0], &r_rev)
            + gamma[1] * EqPolynomial::<F>::mle(&r_cycle[1], &r_rev)
            + gamma[2] * EqPolynomial::<F>::mle(&r_cycle[2], &r_rev);

        let product: F = (0..self.d())
            .map(|i| {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::RamRa(i),
                        SumcheckId::RamRaVirtualization,
                    )
                    .1
            })
            .product();

        eq_eval * product
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
        for i in 0..self.d() {
            let r_address_chunk = self.r_address_chunks()[i].clone();
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                &r_address_chunk,
                &opening_point.r,
                vec![claims[i]],
            );
        }
    }
}
