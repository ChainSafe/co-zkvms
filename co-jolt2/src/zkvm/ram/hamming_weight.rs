use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::ram::hamming_weight::HammingWeightSumcheck;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::rep3::PartyID;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F> {
        let degree =
            <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self);
        let base =
            <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::compute_prover_message(
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

        // degree == 1: linear polynomial g(x) = y0 + (y1 - y0)*x
        let y0 = base[0];
        let y1 = previous_claim - y0;
        let full_evals = vec![y0, y1];
        let poly = UniPoly::<F>::from_evals(&full_evals);
        let coeffs = poly.as_vec();

        let mut msg = vec![F::zero(); max_degree];
        msg[0] = y0;
        for k in 2..=max_degree {
            let x = F::from_u64(k as u64);
            let eval = coeffs.iter().rev().fold(F::zero(), |acc, c| acc * x + *c);
            msg[k - 1] = eval;
        }
        msg
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::bind(self, r_j, round)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
            self.ra_final_claims()
        } else {
            vec![F::zero(); d]
        };

        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .0
            .r;

        accumulator.append_sparse_public(
            (0..d).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamHammingWeight,
            &opening_point.r,
            &r_cycle,
            claims.clone(),
        );

        claims
    }
}

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        _r: &[F::Challenge],
    ) -> F {
        self.gamma_powers()
            .iter()
            .enumerate()
            .map(|(i, gamma)| {
                let (_, ra) = accumulator.get_committed_polynomial_opening(
                    CommittedPolynomial::RamRa(i),
                    SumcheckId::RamHammingWeight,
                );
                ra * gamma
            })
            .sum::<F>()
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <HammingWeightSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .0
            .r
            .clone();

        accumulator.append_sparse(
            transcript,
            (0..self.d()).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamHammingWeight,
            &opening_point.r,
            &r_cycle,
            claims,
        );
    }
}
