use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::ram::ra_virtual::RaSumcheck;
use jolt_core::zkvm::witness::CommittedPolynomial;
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID};

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for RaSumcheck<F> {
    fn degree(&self) -> usize {
        <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F> {
        let degree = <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self);
        let base =
            <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::compute_prover_message(
                self,
                round,
                previous_claim,
            );

        debug_assert!(degree >= 1);
        debug_assert!(base.len() >= degree);
        debug_assert!(max_degree >= degree);

        if max_degree == degree {
            return base[..degree].to_vec();
        }

        // base = [y0, y2, ..., y_degree]. Recover y1 = previous_claim - y0.
        let y0 = base[0];
        let y1 = previous_claim - y0;

        let mut full_evals = Vec::with_capacity(degree + 1);
        full_evals.push(y0);
        full_evals.push(y1);
        full_evals.extend_from_slice(&base[1..]); // y2..y_degree

        let poly = UniPoly::<F>::from_evals(&full_evals);
        let coeffs = poly.as_vec();

        let mut msg = vec![F::zero(); max_degree];
        msg[0] = y0;
        if degree >= 2 {
            msg[1] = full_evals[2]; // y2
        }
        for k in 3..=max_degree {
            let x = F::from_u64(k as u64);
            let eval = coeffs.iter().rev().fold(F::zero(), |acc, c| acc * x + *c);
            msg[k - 1] = eval;
        }
        msg
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        <RaSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::bind(self, r_j, round)
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

    fn cache_openings_public(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        party_id: PartyID,
    ) -> Vec<F> {
        let d = self.d();
        let claims: Vec<F> = if party_id == PartyID::ID0 {
            self.ra_i_final_claims()
        } else {
            vec![F::zero(); d]
        };

        for i in 0..d {
            let share = rep3_arith::promote_to_trivial_share(party_id, claims[i]);
            accumulator.append_sparse(
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                &self.r_address_chunks()[i],
                &opening_point.r,
                vec![share],
            );
        }

        claims
    }
}

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
