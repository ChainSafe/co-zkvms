use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::zkvm::ram::ra_virtual::RaSumcheck;
use jolt_core::zkvm::witness::CommittedPolynomial;
use mpc_core::protocols::rep3::PartyID;

use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::subprotocols::sumcheck::PublicSumcheckInstanceWorker;
use jolt_core::field::JoltField;

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for RaSumcheck<F> {
    fn degree(&self) -> usize {
        self.degree()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim()
    }

    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F> {
        let degree = self.degree();
        let base = self.compute_prover_message(round, previous_claim);

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
        self.bind(r_j, round)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        self.normalize_opening_point(opening_point)
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
            accumulator.append_sparse_public(
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                &self.r_address_chunks()[i],
                &opening_point.r,
                vec![claims[i]],
            );
        }

        claims
    }
}
