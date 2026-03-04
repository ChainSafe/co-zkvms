use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::ram::hamming_booleanity::HammingBooleanitySumcheck;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::PartyID;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for HammingBooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F> {
        let degree =
            <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self);
        let base =
            <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::compute_prover_message(
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
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::bind(
            self, r_j, round,
        )
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
        let claim = if party_id == PartyID::ID0 {
            self.h_final_claim()
        } else {
            F::zero()
        };

        accumulator.append_virtual_public(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            opening_point,
            claim,
            party_id,
        );

        vec![claim]
    }
}

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for HammingBooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let (_, h_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        );

        let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let r_cycle_rev: Vec<F::Challenge> = r_cycle.r.iter().cloned().rev().collect();
        let eq = EqPolynomial::<F>::mle(r, &r_cycle_rev);

        (h_claim.square() - h_claim) * eq
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <HammingBooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            opening_point,
            claims[0],
        );
    }
}
