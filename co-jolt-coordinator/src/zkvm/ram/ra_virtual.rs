use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::ram::ra_virtual::RaSumcheck;
use jolt_core::zkvm::witness::CommittedPolynomial;
use mpc_core::protocols::rep3::PartyID;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for RaSumcheck<F> {
    fn degree(&self) -> usize {
        <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
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
        <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
            self,
            opening_point,
        )
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
