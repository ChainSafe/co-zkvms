use std::{cell::RefCell, rc::Rc};

#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, ProductTerm, ValueSource};
use num_traits::Zero;

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN},
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    zkvm::witness::{CommittedPolynomial, VirtualPolynomial},
};
use allocative::Allocative;
use rayon::prelude::*;

#[derive(Allocative)]
pub struct HammingWeightProverState<F: JoltField> {
    ra: Vec<MultilinearPolynomial<F>>,
}

#[derive(Allocative)]
pub struct HammingWeightSumcheck<F: JoltField> {
    gamma: Vec<F>,
    log_K_chunk: usize,
    d: usize,
    prover_state: Option<HammingWeightProverState<F>>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    /// Construct a prover instance from pre-extracted parts.
    pub fn new_prover_from_parts(gamma_powers: Vec<F>, log_K_chunk: usize, F_arrays: Vec<Vec<F>>) -> Self {
        let d = gamma_powers.len();
        let ra = F_arrays.into_iter().map(MultilinearPolynomial::from).collect::<Vec<_>>();
        Self { gamma: gamma_powers, log_K_chunk, d, prover_state: Some(HammingWeightProverState { ra }) }
    }

    /// Construct a verifier-like instance from pre-extracted parts.
    pub fn new_verifier_from_parts(gamma_powers: Vec<F>, log_K_chunk: usize) -> Self {
        let d = gamma_powers.len();
        Self { gamma: gamma_powers, log_K_chunk, d, prover_state: None }
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn gamma_powers(&self) -> &[F] {
        &self.gamma
    }

    /// Returns the final sumcheck claims for each `ra` polynomial (prover only).
    pub fn ra_final_claims(&self) -> Vec<F> {
        self.prover_state
            .as_ref()
            .expect("ra_final_claims called on verifier instance")
            .ra
            .iter()
            .map(|ra| ra.final_sumcheck_claim())
            .collect()
    }

    pub fn degree(&self) -> usize {
        1
    }

    pub fn num_rounds(&self) -> usize {
        self.log_K_chunk
    }

    pub fn input_claim(&self) -> F {
        self.gamma.iter().sum()
    }

    #[tracing::instrument(skip_all, name = "BytecodeHammingWeight::compute_prover_message")]
    pub fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self.prover_state.as_ref().unwrap();

        let prover_msg = ps
            .ra
            .par_iter()
            .zip(self.gamma.par_iter())
            .map(|(ra, gamma)| {
                let ra_sum = (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|i| ra.get_bound_coeff(2 * i))
                    .fold_with(F::Unreduced::<5>::zero(), |running, new| running + new.as_unreduced_ref())
                    .reduce(F::Unreduced::zero, |running, new| running + new);
                ra_sum.mul_trunc::<4, 9>(gamma.as_unreduced_ref())
            })
            .reduce(F::Unreduced::zero, |running, new| running + new);

        vec![F::from_montgomery_reduce(prover_msg)]
    }

    #[tracing::instrument(skip_all, name = "BytecodeHammingWeight::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.prover_state
            .as_mut()
            .unwrap()
            .ra
            .par_iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh))
    }

    pub fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for HammingWeightSumcheck<F> {
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
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F::Challenge],
    ) -> F {
        let opening_accumulator = opening_accumulator.as_ref().unwrap();
        self.gamma
            .iter()
            .enumerate()
            .map(|(i, gamma)| {
                let ra = opening_accumulator
                    .borrow()
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeRa(i),
                        SumcheckId::BytecodeHammingWeight,
                    )
                    .1;
                ra * gamma
            })
            .sum()
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
        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter)
            .0
            .r
            .clone();
        let r = opening_point.r.iter().cloned().chain(r_cycle.iter().cloned()).collect::<Vec<_>>();
        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeHammingWeight,
            r,
        );
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::sum_of_products(
            self.gamma.iter().enumerate().map(|(idx, _)| ProductTerm::single(ValueSource::challenge(idx))).collect(),
        )
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
    ) -> Vec<F> {
        self.gamma.clone()
    }
}
