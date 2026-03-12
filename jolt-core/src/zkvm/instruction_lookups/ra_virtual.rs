use std::cell::RefCell;
use std::rc::Rc;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{
    OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::InputClaimConstraint;
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::zkvm::instruction_lookups::D;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

pub struct InstructionRaSumcheck<F: JoltField> {
    input_claim: F,
    r_cycle: Vec<F::Challenge>,
    r_address_chunks: Vec<Vec<F::Challenge>>,
}

impl<F: JoltField> InstructionRaSumcheck<F> {
    pub fn new(
        input_claim: F,
        r_cycle: Vec<F::Challenge>,
        r_address_chunks: Vec<Vec<F::Challenge>>,
    ) -> Self {
        Self {
            input_claim,
            r_cycle,
            r_address_chunks,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for InstructionRaSumcheck<F> {
    fn degree(&self) -> usize {
        D + 1
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.unwrap();
        let acc = accumulator.borrow();

        let eq_eval = EqPolynomial::<F>::mle(&self.r_cycle, r);
        let ra_claim_prod: F = (0..D)
            .map(|i| {
                acc.get_committed_polynomial_opening(
                    CommittedPolynomial::InstructionRa(i),
                    SumcheckId::InstructionRaVirtualization,
                )
                .1
            })
            .product();
        eq_eval * ra_claim_prod
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut acc = accumulator.borrow_mut();
        for (i, r_address_chunk) in self.r_address_chunks.iter().enumerate() {
            acc.append_sparse(
                transcript,
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                r_address_chunk
                    .iter()
                    .chain(opening_point.r.iter())
                    .copied()
                    .collect(),
            );
        }
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::direct(OpeningId::Virtual(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        ))
    }
}
