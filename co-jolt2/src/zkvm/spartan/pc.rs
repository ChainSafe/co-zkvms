use jolt_core::poly::eq_poly::EqPlusOnePolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::instruction::CircuitFlags;
use jolt_core::zkvm::spartan::pc::PCSumcheck;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::PartyID;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for PCSumcheck<F> {
    fn degree(&self) -> usize {
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F> {
        let degree = <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self);
        let base = <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::compute_prover_message(
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
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::bind(self, r_j, round)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
        let (unexpanded_pc_eval, pc_eval, is_noop_eval) = if party_id == PartyID::ID0 {
            self.final_shift_evals()
        } else {
            (F::zero(), F::zero(), F::zero())
        };

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

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for PCSumcheck<F> {
    fn degree(&self) -> usize {
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        // Get r_cycle from the SpartanOuter sumcheck opening point.
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let num_cycles_bits =
            <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self);
        let (r_cycle, _) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Shift openings from accumulator.
        let (_, unexpanded_pc_eval_at_shift_r) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
        );
        let (_, pc_eval_at_shift_r) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let (_, is_noop_eval_at_shift_r) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
        );

        let batched_eval_at_shift_r = unexpanded_pc_eval_at_shift_r
            + self.gamma() * pc_eval_at_shift_r
            + self.gamma_squared() * is_noop_eval_at_shift_r;

        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::<F>::new(r_cycle.to_vec()).evaluate(r);
        batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <PCSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
        // claims order: [UnexpandedPC, PC, IsNoopFlag]
        let [unexpanded_pc_eval, pc_eval, is_noop_eval]: [F; 3] = claims
            .try_into()
            .expect("PCSumcheck expects 3 opening claims");

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            unexpanded_pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
            pc_eval,
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
            is_noop_eval,
        );
    }
}
