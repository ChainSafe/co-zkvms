use num_traits::Zero;
use std::cell::RefCell;
use std::rc::Rc;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN};
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::witness::VirtualPolynomial;
use allocative::Allocative;
use rayon::prelude::*;

const DEGREE: usize = 3;

#[derive(Allocative)]
struct HammingBooleanityProverState<F: JoltField> {
    eq_r_cycle: MultilinearPolynomial<F>,
    H: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct HammingBooleanitySumcheck<F: JoltField> {
    prover_state: Option<HammingBooleanityProverState<F>>,
    log_T: usize,
}

impl<F: JoltField> HammingBooleanitySumcheck<F> {
    /// Construct a prover instance from pre-extracted data.
    ///
    /// `ram_addrs` contains remapped RAM addresses (0 for NoOp / padding cycles).
    /// `r_cycle` is the opening point from `(LookupOutput, SpartanOuter)`.
    pub fn new_prover_from_parts(ram_addrs: &[u64], r_cycle: &[F::Challenge]) -> Self {
        let T = ram_addrs.len();
        let log_T = T.log_2();

        let H: Vec<u8> = ram_addrs.par_iter().map(|&addr| if addr == 0 { 0 } else { 1 }).collect();
        let H = MultilinearPolynomial::from(H);

        let eq_r_cycle = MultilinearPolynomial::from(EqPolynomial::<F>::evals(r_cycle));

        Self { prover_state: Some(HammingBooleanityProverState { eq_r_cycle, H }), log_T }
    }

    /// Construct a verifier instance from log_T only (no `StateManager` needed).
    pub fn new_verifier_from_parts(log_T: usize) -> Self {
        Self { prover_state: None, log_T }
    }
}

impl<F: JoltField> HammingBooleanitySumcheck<F> {
    pub fn log_T(&self) -> usize {
        self.log_T
    }

    pub fn h_final_claim(&self) -> F {
        self.prover_state.as_ref().expect("prover state missing").H.final_sumcheck_claim()
    }

    pub fn degree(&self) -> usize {
        DEGREE
    }

    pub fn num_rounds(&self) -> usize {
        self.log_T
    }

    pub fn input_claim(&self) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheck::compute_prover_message")]
    pub fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();

        (0..p.eq_r_cycle.len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = p.eq_r_cycle.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let H_evals = p.H.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let evals = [
                    H_evals[0].square() - H_evals[0],
                    H_evals[1].square() - H_evals[1],
                    H_evals[2].square() - H_evals[2],
                ];

                [
                    eq_evals[0].mul_unreduced::<9>(evals[0]),
                    eq_evals[1].mul_unreduced::<9>(evals[1]),
                    eq_evals[2].mul_unreduced::<9>(evals[2]),
                ]
            })
            .reduce(
                || [F::Unreduced::zero(); DEGREE],
                |running, new| [running[0] + new[0], running[1] + new[1], running[2] + new[2]],
            )
            .into_iter()
            .map(F::from_montgomery_reduce)
            .collect()
    }

    #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheck::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let ps = self.prover_state.as_mut().unwrap();
        rayon::join(
            || ps.eq_r_cycle.bind_parallel(r_j, BindingOrder::LowToHigh),
            || ps.H.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    pub fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point.reverse();
        opening_point.into()
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for HammingBooleanitySumcheck<F> {
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
        let accumulator = accumulator.as_ref().unwrap();
        let H_claim = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamHammingWeight, SumcheckId::RamHammingBooleanity)
            .1;

        let (r_cycle, _) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter);

        let eq = EqPolynomial::<F>::mle(r, &r_cycle.r.iter().cloned().rev().collect::<Vec<F::Challenge>>());

        (H_claim.square() - H_claim) * eq
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
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
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            opening_point,
        );
    }
}
