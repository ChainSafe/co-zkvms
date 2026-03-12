use std::{cell::RefCell, rc::Rc};

use allocative::Allocative;
use num_traits::Zero;
use rayon::prelude::*;

use crate::{
    field::{JoltField, MulTrunc},
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN},
    },
    subprotocols::blindfold::{InputClaimConstraint, ValueSource},
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::math::Math,
    zkvm::witness::{compute_d_parameter, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K},
};

#[derive(Allocative)]
pub struct HammingWeightProverState<F: JoltField> {
    ra: Vec<MultilinearPolynomial<F>>,
}

#[derive(Allocative)]
pub struct HammingWeightSumcheck<F: JoltField> {
    input_claim: F,
    d: usize,
    gamma_powers: Vec<F>,
    prover_state: Option<HammingWeightProverState<F>>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    /// Construct a prover instance from pre-extracted parts.
    ///
    /// `gamma_powers`: `[1, γ, γ², ..., γ^(d-1)]`
    /// `input_claim`: the batched input claim
    /// `F_arrays`: d arrays of size `DTH_ROOT_OF_K`, each the eq-weighted histogram
    ///             of address chunk `i` over the trace.
    pub fn new_prover_from_parts(gamma_powers: Vec<F>, input_claim: F, F_arrays: Vec<Vec<F>>) -> Self {
        let d = gamma_powers.len();
        let ra: Vec<MultilinearPolynomial<F>> = F_arrays.into_iter().map(MultilinearPolynomial::from).collect();
        Self { input_claim, d, gamma_powers, prover_state: Some(HammingWeightProverState { ra }) }
    }

    /// Construct a verifier-like instance from pre-extracted parts.
    pub fn new_verifier_from_parts(gamma_powers: Vec<F>, input_claim: F) -> Self {
        let d = gamma_powers.len();
        Self { input_claim, d, gamma_powers, prover_state: None }
    }
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    pub fn d(&self) -> usize {
        self.d
    }

    pub fn gamma_powers(&self) -> &[F] {
        &self.gamma_powers
    }

    pub fn ra_final_claims(&self) -> Vec<F> {
        self.prover_state.as_ref().expect("prover state missing").ra.iter().map(|p| p.final_sumcheck_claim()).collect()
    }

    pub fn degree(&self) -> usize {
        1
    }

    pub fn num_rounds(&self) -> usize {
        DTH_ROOT_OF_K.log_2()
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "RamHammingWeightSumcheck::compute_prover_message")]
    pub fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self.prover_state.as_ref().expect("Prover state not initialized");

        let prover_msg = ps
            .ra
            .par_iter()
            .zip(self.gamma_powers.par_iter())
            .map(|(ra_poly, gamma_power)| {
                let ra_sum = (0..ra_poly.len() / 2)
                    .into_par_iter()
                    .map(|i| ra_poly.get_bound_coeff(2 * i))
                    .fold_with(F::Unreduced::<5>::zero(), |running, new| running + new.as_unreduced_ref())
                    .reduce(F::Unreduced::zero, |running, new| running + new);
                ra_sum.mul_trunc::<4, 9>(gamma_power.as_unreduced_ref())
            })
            .reduce(F::Unreduced::zero, |running, new| running + new);

        vec![F::from_montgomery_reduce(prover_msg)]
    }

    #[tracing::instrument(skip_all, name = "RamHammingWeightSumcheck::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            prover_state.ra.par_iter_mut().for_each(|ra_poly| ra_poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    pub fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
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
        let ra_claims: Vec<_> = (0..self.d)
            .map(|i| {
                accumulator
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .get_committed_polynomial_opening(CommittedPolynomial::RamRa(i), SumcheckId::RamHammingWeight)
                    .1
            })
            .collect();

        // Compute batched claim: sum_{i=0}^{d-1} gamma^i * ra_i
        ra_claims.iter().zip(self.gamma_powers.iter()).map(|(ra_claim, gamma_power)| *ra_claim * gamma_power).sum()
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        self.normalize_opening_point(opening_point)
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let (r_cycle, _) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamHammingWeight, SumcheckId::RamHammingBooleanity);
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());

        accumulator.borrow_mut().append_sparse(
            transcript,
            (0..self.d).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamHammingWeight,
            opening_point.r,
        );
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::linear(vec![(
            ValueSource::challenge(0),
            ValueSource::opening(OpeningId::Virtual(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )),
        )])
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
    ) -> Vec<F> {
        vec![self.gamma_powers.iter().copied().sum()]
    }
}
