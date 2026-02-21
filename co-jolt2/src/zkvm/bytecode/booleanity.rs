use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::bytecode::booleanity::BooleanitySumcheck;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
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

        // degree == 3: base = [y0, y2, y3].  Recover y1 = previous_claim - y0.
        let y0 = base[0];
        let y1 = previous_claim - y0;
        let full_evals = vec![y0, y1, base[1], base[2]]; // evals at {0, 1, 2, 3}
        let poly = UniPoly::<F>::from_evals(&full_evals);

        let mut msg = vec![F::zero(); max_degree];
        msg[0] = y0;
        if degree >= 2 {
            msg[1] = base[1]; // y2
        }
        msg[2] = base[2]; // y3
        for k in 4..=max_degree {
            let x: F::Challenge = (k as u128).into();
            msg[k - 1] = poly.evaluate(&x);
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
    ) -> Vec<F> {
        let d = self.d();
        let log_K_chunk = self.log_K_chunk();
        let claims = self.h_final_claims();

        let shares: Vec<_> = claims
            .iter()
            .map(|&claim| rep3_arith::promote_to_trivial_share(PartyID::ID0, claim))
            .collect();

        accumulator.append_sparse(
            (0..d).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeBooleanity,
            &opening_point.r[..log_K_chunk],
            &opening_point.r[log_K_chunk..],
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
        let log_K_chunk = self.log_K_chunk();

        let ra_claims: Vec<F> = (0..d)
            .map(|i| {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeRa(i),
                        SumcheckId::BytecodeBooleanity,
                    )
                    .1
            })
            .collect();

        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;

        // Mirrors vanilla BooleanitySumcheck::expected_output_claim.
        // normalize_opening_point reverses address and cycle parts separately,
        // so r_prime is already [rev(r_address) || rev(r_cycle)] in BIG_ENDIAN.
        // We reconstruct the LowToHigh point used during binding:
        //   r_address was bound LowToHigh → reversed = r_address in big-endian
        //   r_cycle was bound LowToHigh → reversed = r_cycle in big-endian
        let expected_r: Vec<F::Challenge> = self
            .r_address()
            .iter()
            .cloned()
            .rev()
            .chain(r_cycle.iter().cloned().rev())
            .collect();

        EqPolynomial::<F>::mle(r, &expected_r)
            * self
                .gamma_powers()
                .iter()
                .zip(ra_claims.iter())
                .map(|(gamma, &ra)| (ra.square() - ra) * gamma)
                .sum::<F>()
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
        let log_K_chunk = self.log_K_chunk();

        accumulator.append_sparse(
            transcript,
            (0..self.d()).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeBooleanity,
            &opening_point.r[..log_K_chunk],
            &opening_point.r[log_K_chunk..],
            claims,
        );
    }
}
