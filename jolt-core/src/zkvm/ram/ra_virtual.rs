use num_traits::Zero;
use std::cell::RefCell;
use std::iter::zip;
use std::rc::Rc;
use std::sync::Arc;

use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::{
    OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::ra_poly::RaPolynomial;
use crate::zkvm::ram::remap_address;
use crate::zkvm::witness::{
    compute_d_parameter, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K,
};
use crate::{
    field::JoltField,
    poly::{
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::math::Math,
};
use allocative::Allocative;
use rayon::prelude::*;

#[derive(Allocative)]
pub struct RaProverState<F: JoltField> {
    /// `ra` polys to be constructed based addresses
    ra_i_polys: Vec<RaPolynomial<u8, F>>,
    /// eq poly
    eq_poly: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct RaSumcheck<F: JoltField> {
    gamma: [F; 3],
    /// Random challenge r_cycle
    r_cycle: [Vec<F::Challenge>; 3],
    r_address_chunks: Vec<Vec<F::Challenge>>,
    /// [ra(r_address, r_cycle_val), ra(r_address, r_cycle_rw), ra(r_address, r_cycle_raf)]
    ra_claim: F,
    /// Number of decomposition parts
    d: usize,
    /// Length of the trace
    T: usize,
    prover_state: Option<RaProverState<F>>,
}

impl<F: JoltField> RaSumcheck<F> {
    /// Construct a prover instance from pre-extracted parts.
    ///
    /// * `gamma` — `[1, γ, γ²]`
    /// * `ra_claim` — combined claim `γ⁰·val + γ¹·rw + γ²·raf`
    /// * `d` — number of decomposition parts
    /// * `T` — trace length
    /// * `r_cycle` — three random challenge vectors (val, rw, raf)
    /// * `r_address_chunks` — d chunks of r_address
    /// * `ra_i_polys` — d `RaPolynomial` instances built from trace addresses
    /// * `eq_poly` — gamma-weighted linear combination of eq polynomials
    pub fn new_prover_from_parts(
        gamma: [F; 3],
        ra_claim: F,
        d: usize,
        T: usize,
        r_cycle: [Vec<F::Challenge>; 3],
        r_address_chunks: Vec<Vec<F::Challenge>>,
        ra_i_polys: Vec<RaPolynomial<u8, F>>,
        eq_poly: MultilinearPolynomial<F>,
    ) -> Self {
        Self {
            gamma,
            ra_claim,
            d,
            T,
            r_cycle,
            r_address_chunks,
            prover_state: Some(RaProverState {
                ra_i_polys,
                eq_poly,
            }),
        }
    }

    /// Construct a verifier-like instance from pre-extracted parts.
    pub fn new_verifier_from_parts(
        gamma: [F; 3],
        ra_claim: F,
        d: usize,
        T: usize,
        r_cycle: [Vec<F::Challenge>; 3],
        r_address_chunks: Vec<Vec<F::Challenge>>,
    ) -> Self {
        Self {
            gamma,
            ra_claim,
            d,
            T,
            r_cycle,
            r_address_chunks,
            prover_state: None,
        }
    }
}

impl<F: JoltField> RaSumcheck<F> {
    pub fn d(&self) -> usize {
        self.d
    }

    pub fn gamma(&self) -> [F; 3] {
        self.gamma
    }

    pub fn r_cycle(&self) -> &[Vec<F::Challenge>; 3] {
        &self.r_cycle
    }

    pub fn r_address_chunks(&self) -> &[Vec<F::Challenge>] {
        &self.r_address_chunks
    }

    pub fn ra_i_final_claims(&self) -> Vec<F> {
        self.prover_state
            .as_ref()
            .expect("prover state missing")
            .ra_i_polys
            .iter()
            .map(|p| p.final_sumcheck_claim())
            .collect()
    }

    pub fn degree(&self) -> usize {
        self.d + 1
    }

    pub fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    pub fn input_claim(&self) -> F {
        self.ra_claim
    }

    #[tracing::instrument(skip_all, name = "RamRaVirtualization::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, _: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        for ra_i in prover_state.ra_i_polys.iter_mut() {
            ra_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        prover_state
            .eq_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    #[tracing::instrument(skip_all, name = "RamRaVirtualization::compute_prover_message")]
    pub fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = self.degree();
        let ra_i_polys = &ps.ra_i_polys;
        let eq_poly = &ps.eq_poly;

        (0..ra_i_polys[0].len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                let mut evals = vec![];

                let all_ra_i_evals: Vec<Vec<F>> = ra_i_polys
                    .iter()
                    .map(|ra_i_poly| ra_i_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh))
                    .collect();

                for eval_point in 0..degree {
                    let mut result = eq_evals[eval_point];
                    for ra_i_evals in all_ra_i_evals.iter() {
                        result *= ra_i_evals[eval_point];
                    }
                    let unreduced = *result.as_unreduced_ref();
                    evals.push(unreduced);
                }

                evals
            })
            .fold_with(vec![F::Unreduced::<5>::zero(); degree], |running, new| {
                zip(running, new).map(|(a, b)| a + b).collect()
            })
            .reduce(
                || vec![F::Unreduced::zero(); degree],
                |running, new| zip(running, new).map(|(a, b)| a + b).collect(),
            )
            .into_iter()
            .map(F::from_barrett_reduce)
            .collect()
    }

    pub fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for RaSumcheck<F> {
    fn degree(&self) -> usize { self.degree() }
    fn num_rounds(&self) -> usize { self.num_rounds() }
    fn input_claim(&self) -> F { self.input_claim() }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        // we need opposite endian-ness here
        let r_rev: Vec<_> = r.iter().cloned().rev().collect();
        let eq_eval = self.gamma[0] * EqPolynomial::<F>::mle(&self.r_cycle[0], &r_rev)
            + self.gamma[1] * EqPolynomial::<F>::mle(&self.r_cycle[1], &r_rev)
            + self.gamma[2] * EqPolynomial::<F>::mle(&self.r_cycle[2], &r_rev);

        // Compute the product of all ra_i evaluations
        let mut product = F::one();
        for i in 0..self.d {
            let accumulator = accumulator.as_ref().unwrap();
            let accumulator = accumulator.borrow();
            let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::RamRaVirtualization,
            );
            product *= ra_i_claim;
        }
        eq_eval * product
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
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        for i in 0..self.d {
            let opening_point =
                [self.r_address_chunks[i].as_slice(), r_cycle.r.as_slice()].concat();
            accumulator.borrow_mut().append_sparse(
                transcript,
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                opening_point,
            );
        }
    }

}
