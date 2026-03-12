use std::cell::RefCell;
use std::rc::Rc;

use allocative::Allocative;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPlusOnePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::InputClaimConstraint;
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::zkvm::instruction::CircuitFlags;
use crate::zkvm::witness::VirtualPolynomial;
use rayon::prelude::*;

#[derive(Allocative)]
struct PCSumcheckProverState<F: JoltField> {
    unexpanded_pc_poly: MultilinearPolynomial<F>,
    pc_poly: MultilinearPolynomial<F>,
    is_noop_poly: MultilinearPolynomial<F>,
    eq_plus_one_poly: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct PCSumcheck<F: JoltField> {
    input_claim: F,
    gamma: F,
    gamma_squared: F,
    log_T: usize,
    prover_state: Option<PCSumcheckProverState<F>>,
}

impl<F: JoltField> PCSumcheck<F> {
    /// Construct the verifier-side PC sumcheck from already-computed public values.
    ///
    /// This is intended for external drivers (e.g. MPC coordinators) that do not
    /// have access to a vanilla `StateManager` but want to reuse the vanilla
    /// `PCSumcheck` logic and transcript ordering.
    pub fn new_verifier_from_openings(input_claim: F, gamma: F, log_T: usize) -> Self {
        let gamma_squared = gamma.square();
        Self {
            input_claim,
            gamma,
            gamma_squared,
            log_T,
            prover_state: None,
        }
    }

    /// Construct the prover-side PC sumcheck from public witness polynomials.
    ///
    /// The polynomials must correspond to the same `r_cycle` used to build `eq_plus_one_poly`.
    /// This is intended for public-only execution models (e.g. one designated worker).
    pub fn new_prover_from_polys(
        input_claim: F,
        gamma: F,
        log_T: usize,
        unexpanded_pc_poly: MultilinearPolynomial<F>,
        pc_poly: MultilinearPolynomial<F>,
        is_noop_poly: MultilinearPolynomial<F>,
        eq_plus_one_poly: MultilinearPolynomial<F>,
    ) -> Self {
        let gamma_squared = gamma.square();
        Self {
            input_claim,
            gamma,
            gamma_squared,
            log_T,
            prover_state: Some(PCSumcheckProverState {
                unexpanded_pc_poly,
                pc_poly,
                is_noop_poly,
                eq_plus_one_poly,
            }),
        }
    }

    /// Return the final evaluations of (UnexpandedPC, PC, IsNoop) after all rounds
    /// have been bound (i.e. at the shift opening point).
    pub fn final_shift_evals(&self) -> (F, F, F) {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        (
            ps.unexpanded_pc_poly.final_sumcheck_claim(),
            ps.pc_poly.final_sumcheck_claim(),
            ps.is_noop_poly.final_sumcheck_claim(),
        )
    }

    /// Return the batching challenge `gamma` used by this instance.
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// Return `gamma^2` used by this instance.
    pub fn gamma_squared(&self) -> F {
        self.gamma_squared
    }

    pub fn degree(&self) -> usize {
        2
    }

    pub fn num_rounds(&self) -> usize {
        self.log_T
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "PCSumcheck::compute_prover_message")]
    pub fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.unexpanded_pc_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let unexpanded_pc_evals = prover_state
                    .unexpanded_pc_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let pc_evals = prover_state
                    .pc_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let eq_evals = prover_state
                    .eq_plus_one_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let is_noop_evals = prover_state
                    .is_noop_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    (unexpanded_pc_evals[0]
                        + self.gamma * pc_evals[0]
                        + self.gamma_squared * is_noop_evals[0])
                        * eq_evals[0], // eval at 0
                    (unexpanded_pc_evals[1]
                        + self.gamma * pc_evals[1]
                        + self.gamma_squared * is_noop_evals[1])
                        * eq_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "PCSumcheck::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::scope(|s| {
            s.spawn(|_| {
                prover_state
                    .unexpanded_pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .pc_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .is_noop_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
            s.spawn(|_| {
                prover_state
                    .eq_plus_one_poly
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            });
        });
    }

    pub fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for PCSumcheck<F> {
    fn degree(&self) -> usize {
        self.degree()
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds()
    }

    fn input_claim(&self) -> F {
        self.input_claim()
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap().borrow();

        // Get r_cycle from the SpartanOuter sumcheck opening point
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let num_cycles_bits = self.log_T;
        let (r_cycle, _) = outer_sumcheck_r.split_at(num_cycles_bits);

        // Get the shift evaluations from the accumulator
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
            + self.gamma * pc_eval_at_shift_r
            + self.gamma_squared * is_noop_eval_at_shift_r;

        let eq_plus_one_shift_sumcheck =
            EqPlusOnePolynomial::<F>::new(r_cycle.to_vec()).evaluate(r);

        batched_eval_at_shift_r * eq_plus_one_shift_sumcheck
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        self.normalize_opening_point(opening_point)
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::PC,
            SumcheckId::SpartanShift,
            opening_point.clone(),
        );
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::OpFlags(CircuitFlags::IsNoop),
            SumcheckId::SpartanShift,
            opening_point,
        );
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::weighted_openings(&[
            OpeningId::Virtual(VirtualPolynomial::NextUnexpandedPC, SumcheckId::SpartanOuter),
            OpeningId::Virtual(VirtualPolynomial::NextPC, SumcheckId::SpartanOuter),
            OpeningId::Virtual(
                VirtualPolynomial::NextIsNoop,
                SumcheckId::SpartanOuter,
            ),
        ])
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
    ) -> Vec<F> {
        vec![self.gamma, self.gamma_squared]
    }
}
