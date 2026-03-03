use jolt_core::poly::identity_poly::UnmapRamAddressPolynomial;
use jolt_core::poly::multilinear_polynomial::PolynomialEvaluation;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::ram::raf_evaluation::RafEvaluationSumcheck;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::PartyID;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for RafEvaluationSumcheck<F> {
    fn degree(&self) -> usize {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F> {
        let degree =
            <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self);
        let base =
            <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::compute_prover_message(
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

        // degree == 2: base = [y0, y2]. Recover y1 = previous_claim - y0.
        let y0 = base[0];
        let y1 = previous_claim - y0;
        let full_evals = vec![y0, y1, base[1]];
        let poly = UniPoly::<F>::from_evals(&full_evals);
        let coeffs = poly.as_vec();

        let mut msg = vec![F::zero(); max_degree];
        msg[0] = y0;
        msg[1] = base[1]; // y2
        for k in 3..=max_degree {
            let x = F::from_u64(k as u64);
            let eval = coeffs.iter().rev().fold(F::zero(), |acc, c| acc * x + *c);
            msg[k - 1] = eval;
        }
        msg
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::bind(self, r_j, round)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
            self,
            opening_point,
        )
    }

    fn cache_openings_public(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
        party_id: PartyID,
    ) -> Vec<F> {
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .0;
        let ra_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());

        let ra_claim = if party_id == PartyID::ID0 {
            self.ra_final_claim()
        } else {
            F::zero()
        };

        accumulator.append_virtual_public(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
            ra_claim,
            party_id,
        );

        vec![ra_claim]
    }
}

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for RafEvaluationSumcheck<F> {
    fn degree(&self) -> usize {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let unmap_eval =
            UnmapRamAddressPolynomial::<F>::new(self.log_K(), self.start_address()).evaluate(r);

        let (_, ra_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);

        unmap_eval * ra_claim
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <RafEvaluationSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
            self,
            opening_point,
        )
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .0;
        let ra_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
            claims[0],
        );
    }
}
