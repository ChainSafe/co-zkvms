use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::utils::math::Math;
use jolt_core::zkvm::ram::booleanity::BooleanitySumcheck;
use jolt_core::zkvm::witness::{CommittedPolynomial, DTH_ROOT_OF_K};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID};

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F> {
        let degree =
            <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self);
        let base =
            <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::compute_prover_message(
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

        // degree == 3: base = [y0, y2, y3]. Recover y1 = previous_claim - y0.
        let y0 = base[0];
        let y1 = previous_claim - y0;
        let full_evals = vec![y0, y1, base[1], base[2]];
        let poly = UniPoly::<F>::from_evals(&full_evals);
        let coeffs = poly.as_vec();

        let mut msg = vec![F::zero(); max_degree];
        msg[0] = y0;
        msg[1] = base[1]; // y2
        msg[2] = base[2]; // y3
        for k in 4..=max_degree {
            let x = F::from_u64(k as u64);
            let eval = coeffs.iter().rev().fold(F::zero(), |acc, c| acc * x + *c);
            msg[k - 1] = eval;
        }
        msg
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::bind(self, r_j, round)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
            self.h_final_claims()
        } else {
            vec![F::zero(); d]
        };

        let shares: Vec<_> = claims
            .iter()
            .map(|&claim| rep3_arith::promote_to_trivial_share(party_id, claim))
            .collect();

        let (r_address, r_cycle) = opening_point.split_at(DTH_ROOT_OF_K.log_2());
        accumulator.append_sparse(
            (0..d).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamBooleanity,
            &r_address.r,
            &r_cycle.r,
            shares,
        );

        claims
    }
}

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
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

        let eq_address =
            EqPolynomial::<F>::mle(self.r_address(), &r_address_prime_rev);
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
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
