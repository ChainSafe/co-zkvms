use num_traits::Zero;
use std::{cell::RefCell, rc::Rc};

use allocative::Allocative;
use rayon::prelude::*;

#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint, ValueSource};
use crate::{
    field::JoltField,
    poly::{
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation},
        opening_proof::{OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN},
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::math::Math,
    zkvm::witness::VirtualPolynomial,
};

#[derive(Allocative)]
pub struct RafEvaluationProverState<F: JoltField> {
    /// The ra polynomial
    ra: MultilinearPolynomial<F>,
    /// The unmap polynomial
    unmap: UnmapRamAddressPolynomial<F>,
}

#[derive(Allocative)]
pub struct RafEvaluationSumcheck<F: JoltField> {
    /// The initial claim (raf_claim)
    input_claim: F,
    /// log K (number of rounds)
    log_K: usize,
    /// Start address for unmap polynomial
    start_address: u64,
    prover_state: Option<RafEvaluationProverState<F>>,
    /// Cached ra_claim after sumcheck completion
    cached_claim: Option<F>,
}

impl<F: JoltField> RafEvaluationSumcheck<F> {
    /// Construct a prover instance from pre-extracted parts.
    pub fn new_prover_from_parts(
        input_claim: F,
        log_K: usize,
        start_address: u64,
        ra: MultilinearPolynomial<F>,
        unmap: UnmapRamAddressPolynomial<F>,
    ) -> Self {
        Self {
            input_claim,
            log_K,
            start_address,
            prover_state: Some(RafEvaluationProverState { ra, unmap }),
            cached_claim: None,
        }
    }

    /// Construct a verifier instance from pre-extracted parts.
    pub fn new_verifier_from_parts(input_claim: F, log_K: usize, start_address: u64, ra_claim: F) -> Self {
        Self { input_claim, log_K, start_address, prover_state: None, cached_claim: Some(ra_claim) }
    }
}

impl<F: JoltField> RafEvaluationSumcheck<F> {
    pub fn log_K(&self) -> usize {
        self.log_K
    }

    pub fn start_address(&self) -> u64 {
        self.start_address
    }

    pub fn ra_final_claim(&self) -> F {
        self.prover_state.as_ref().expect("prover state missing").ra.final_sumcheck_claim()
    }

    pub fn degree(&self) -> usize {
        2
    }

    pub fn num_rounds(&self) -> usize {
        self.log_K
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheck::compute_prover_message")]
    pub fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let ps = self.prover_state.as_ref().expect("Prover state not initialized");
        const DEGREE: usize = 2;

        (0..ps.ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ps.ra.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let unmap_evals = ps.unmap.sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);

                [ra_evals[0].mul_unreduced::<9>(unmap_evals[0]), ra_evals[1].mul_unreduced::<9>(unmap_evals[1])]
            })
            .reduce(|| [F::Unreduced::zero(); DEGREE], |running, new| [running[0] + new[0], running[1] + new[1]])
            .into_iter()
            .map(F::from_montgomery_reduce)
            .collect()
    }

    #[tracing::instrument(skip_all, name = "RamRafEvaluationSumcheck::bind")]
    pub fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            rayon::join(
                || prover_state.ra.bind_parallel(r_j, BindingOrder::HighToLow),
                || prover_state.unmap.bind_parallel(r_j, BindingOrder::HighToLow),
            );
        }
    }

    pub fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for RafEvaluationSumcheck<F> {
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
        _accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        // Compute unmap evaluation at r
        let unmap_eval = UnmapRamAddressPolynomial::<F>::new(self.log_K, self.start_address).evaluate(r);

        // Return unmap(r) * ra(r)
        let ra_claim = self.cached_claim.expect("ra_claim not cached");
        unmap_eval * ra_claim
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
        let r_cycle = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .0;
        let ra_opening_point = OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
        );
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::direct(OpeningId::Virtual(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter))
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        Some(OutputClaimConstraint::product(vec![
            ValueSource::challenge(0),
            ValueSource::opening(OpeningId::Virtual(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation)),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        vec![UnmapRamAddressPolynomial::<F>::new(self.log_K, self.start_address).evaluate(sumcheck_challenges)]
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::transcripts::Blake2bTranscript;
//     use ark_bn254::Fr;

//     #[test]
//     fn test_raf_evaluation_no_ops() {
//         const K: usize = 1 << 16;
//         const T: usize = 1 << 8;

//         let memory_layout = MemoryLayout {
//             max_input_size: 256,
//             max_output_size: 256,
//             input_start: 0x80000000,
//             input_end: 0x80000100,
//             output_start: 0x80001000,
//             output_end: 0x80001100,
//             stack_size: 1024,
//             stack_end: 0x7FFFFF00,
//             memory_size: 0x10000,
//             memory_end: 0x80010000,
//             panic: 0x80002000,
//             termination: 0x80002001,
//             io_end: 0x80002002,
//         };

//         // Create trace with only no-ops (address = 0)
//         let mut trace = Vec::new();
//         for i in 0..T {
//             trace.push(Cycle::NoOp(i));
//         }

//         let mut prover_transcript = Blake2bTranscript::new(b"test_no_ops");
//         let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

//         // Prove
//         let proof =
//             RafEvaluationProof::prove(&trace, &memory_layout, r_cycle, K, &mut prover_transcript);

//         // Verify
//         let mut verifier_transcript = Blake2bTranscript::new(b"test_no_ops");
//         let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

//         let r_address_result = proof.verify(K, &mut verifier_transcript, &memory_layout);

//         assert!(
//             r_address_result.is_ok(),
//             "No-op RAF evaluation verification failed"
//         );
//     }
// }
