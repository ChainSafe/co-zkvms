use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::zkvm::instruction::CircuitFlags;
use jolt_core::zkvm::spartan::pc::PCSumcheck;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::PartyID;

use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::subprotocols::sumcheck::PublicSumcheckInstanceWorker;
use jolt_core::field::JoltField;

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for PCSumcheck<F> {
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

        // Build full evals on nodes x = 0..=degree.
        let mut full_evals: Vec<F> = Vec::with_capacity(degree + 1);
        full_evals.push(base[0]); // y0
        full_evals.push(previous_claim - base[0]); // y1
        full_evals.extend((2..=degree).map(|k| base[k - 1])); // y2..yd

        let poly = UniPoly::<F>::from_evals(&full_evals);

        let mut msg = vec![F::zero(); max_degree];
        msg[0] = full_evals[0];
        if degree >= 2 {
            msg[1] = full_evals[2]; // y2
        }
        for k in 3..=max_degree {
            let x = F::from_u64(k as u64);
            msg[k - 1] = poly.evaluate::<F>(&x);
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
        let (unexpanded_pc_eval, pc_eval, is_noop_eval) =
            if party_id == PartyID::ID0 { self.final_shift_evals() } else { (F::zero(), F::zero(), F::zero()) };

        accumulator.append_virtual_public(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
            party_id,
        );
        accumulator.append_virtual_public(
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
            party_id,
        );
        accumulator.append_virtual_public(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
            party_id,
        );

        vec![unexpanded_pc_eval, pc_eval, is_noop_eval]
    }
}
