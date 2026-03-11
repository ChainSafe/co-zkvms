use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::zkvm::bytecode::hamming_weight::HammingWeightSumcheck;
use jolt_core::zkvm::witness::CommittedPolynomial;
use mpc_core::protocols::rep3::PartyID;

use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::subprotocols::sumcheck::PublicSumcheckInstanceWorker;
use jolt_core::field::JoltField;

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        self.degree()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim()
    }

    fn compute_prover_message_public(&mut self, round: usize, previous_claim: F, max_degree: usize) -> Vec<F> {
        let degree = self.degree();
        let base = self.compute_prover_message(round, previous_claim);

        debug_assert!(degree >= 1);
        debug_assert!(base.len() >= degree);
        debug_assert!(max_degree >= degree);

        if max_degree == degree {
            return base[..degree].to_vec();
        }

        // degree == 1: polynomial is linear, g(x) = base[0] + (previous_claim - base[0]) * x
        let y0 = base[0];
        let y1 = previous_claim - y0;

        // Build full evals at {0, 1} then extrapolate via UniPoly.
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
        self.bind(r_j, round)
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        self.normalize_opening_point(opening_point)
    }

    fn cache_openings_public(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        party_id: PartyID,
    ) -> Vec<F> {
        let d = self.d();
        let ra_claims: Vec<F> = if party_id == PartyID::ID0 { self.ra_final_claims() } else { vec![F::zero(); d] };

        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                jolt_core::zkvm::witness::VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;

        accumulator.append_sparse_public(
            (0..d).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeHammingWeight,
            &opening_point.r,
            &r_cycle,
            ra_claims.clone(),
        );

        ra_claims
    }
}
