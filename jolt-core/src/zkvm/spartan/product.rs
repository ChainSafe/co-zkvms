use std::cell::RefCell;
use std::rc::Rc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::opening_proof::{
    OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

pub struct ProductVirtualizationSumcheck<F: JoltField> {
    input_claim: F,
    log_T: usize,
}

impl<F: JoltField> ProductVirtualizationSumcheck<F> {
    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let accumulator = sm.get_verifier_accumulator();
        let acc = accumulator.borrow();
        let (r_point, input_claim) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter);
        Self {
            input_claim,
            log_T: r_point.r.len(),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ProductVirtualizationSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_T
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

        let (outer_sumcheck_opening, _) =
            acc.get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let (r_cycle, _) = outer_sumcheck_r.split_at(self.log_T);

        let (_, left_input_eval) = acc.get_committed_polynomial_opening(
            CommittedPolynomial::LeftInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let (_, right_input_eval) = acc.get_committed_polynomial_opening(
            CommittedPolynomial::RightInstructionInput,
            SumcheckId::ProductVirtualization,
        );

        let eq_eval = EqPolynomial::mle(&r.iter().rev().copied().collect::<Vec<_>>(), r_cycle);
        eq_eval * left_input_eval * right_input_eval
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_dense(
            transcript,
            vec![
                CommittedPolynomial::LeftInstructionInput,
                CommittedPolynomial::RightInstructionInput,
            ],
            SumcheckId::ProductVirtualization,
            opening_point.r,
        );
    }
}
