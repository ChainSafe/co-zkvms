use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::ram::booleanity::BooleanitySumcheck;
use jolt_core::zkvm::witness::{CommittedPolynomial, DTH_ROOT_OF_K};
use mpc_core::protocols::rep3::PartyID;

use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::subprotocols::sumcheck::PublicSumcheckInstanceWorker;
use jolt_core::field::JoltField;

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for BooleanitySumcheck<F> {
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
        let claims: Vec<F> = if party_id == PartyID::ID0 { self.h_final_claims() } else { vec![F::zero(); d] };

        let (r_address, r_cycle) = opening_point.split_at(DTH_ROOT_OF_K.log_2());
        accumulator.append_sparse_public(
            (0..d).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamBooleanity,
            &r_address.r,
            &r_cycle.r,
            claims.clone(),
        );

        claims
    }
}
