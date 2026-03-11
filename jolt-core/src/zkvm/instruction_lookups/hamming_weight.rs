use std::{cell::RefCell, rc::Rc};

use allocative::Allocative;
use num_traits::Zero;
use rayon::prelude::*;

use super::{D, LOG_K_CHUNK};

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

const DEGREE: usize = 1;

#[derive(Allocative)]
struct HammingProverState<F: JoltField> {
    /// ra_i polynomials
    ra: [MultilinearPolynomial<F>; D],
}

#[derive(Allocative)]
pub struct HammingWeightSumcheck<F: JoltField> {
    gamma: [F; D],
    prover_state: Option<HammingProverState<F>>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    /// Construct a prover instance from pre-extracted parts.
    pub fn new_prover_from_parts(gamma_powers: [F; D], G: [Vec<F>; D]) -> Self {
        let ra = G
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Self {
            gamma: gamma_powers,
            prover_state: Some(HammingProverState { ra }),
        }
    }

    /// Construct a verifier instance from pre-extracted parts.
    pub fn new_verifier_from_parts(gamma_powers: [F; D]) -> Self {
        Self {
            gamma: gamma_powers,
            prover_state: None,
        }
    }

    pub fn degree(&self) -> usize {
        DEGREE
    }

    pub fn num_rounds(&self) -> usize {
        LOG_K_CHUNK
    }

    pub fn input_claim(&self) -> F {
        self.gamma.iter().sum()
    }

    #[tracing::instrument(skip_all, name = "InstructionHammingWeight::compute_prover_message")]
    pub fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();

        let result = prover_state
            .ra
            .iter()
            .zip(self.gamma.iter())
            .map(|(ra, gamma)| {
                let ra_sum = (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|i| ra.get_bound_coeff(2 * i))
                    .fold_with(F::Unreduced::<5>::zero(), |running, new| {
                        running + new.as_unreduced_ref()
                    })
                    .reduce(F::Unreduced::zero, |running, new| running + new);
                ra_sum.mul_trunc::<4, 9>(gamma.as_unreduced_ref())
            })
            .fold(F::Unreduced::<9>::zero(), |running, new| running + new);
        vec![F::from_montgomery_reduce(result)]
    }

    #[tracing::instrument(skip_all, name = "InstructionHammingWeight::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.prover_state
            .as_mut()
            .unwrap()
            .ra
            .par_iter_mut()
            .for_each(|ra| ra.bind_parallel(r_j, BindingOrder::LowToHigh))
    }

    pub fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
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
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F::Challenge],
    ) -> F {
        let ra_claims: Vec<F> = (0..D)
            .map(|i| {
                let accumulator = accumulator.as_ref().unwrap();
                let accumulator = accumulator.borrow();
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::InstructionRa(i),
                        SumcheckId::InstructionHammingWeight,
                    )
                    .1
            })
            .collect();

        self.gamma
            .iter()
            .zip(ra_claims.iter())
            .map(|(gamma, ra)| *ra * gamma)
            .sum()
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
        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        let r = opening_point
            .r
            .iter()
            .cloned()
            .chain(r_cycle.iter().cloned())
            .collect::<Vec<_>>();
        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..D).map(CommittedPolynomial::InstructionRa).collect(),
            SumcheckId::InstructionHammingWeight,
            r,
        );
    }
}
