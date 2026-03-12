use std::cell::RefCell;
use std::rc::Rc;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{
    OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::InputClaimConstraint;
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::inputs::ALL_R1CS_INPUTS;
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;

pub struct InnerSumcheck<F: JoltField> {
    gamma: F,
    input_claim: F,
    key: UniformSpartanKey<F>,
    outer_sumcheck_r: Vec<F::Challenge>,
    claimed_witness_evals: Vec<F>,
}

impl<F: JoltField> InnerSumcheck<F> {
    pub fn new_verifier<
        C: JoltCurve,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        sm: &mut StateManager<'_, F, C, ProofTranscript, PCS>,
    ) -> Self {
        let gamma: F = sm.transcript.borrow_mut().challenge_scalar();

        let accumulator = sm.get_verifier_accumulator();
        let acc = accumulator.borrow();

        let (outer_sumcheck_r, claim_az) = acc
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_bz) = acc
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_cz) = acc
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);

        let input_claim = claim_az + gamma * claim_bz + gamma.square() * claim_cz;

        let claimed_witness_evals: Vec<F> = ALL_R1CS_INPUTS
            .iter()
            .map(|r1cs_input| {
                let key =
                    OpeningId::try_from(r1cs_input).expect("Failed to map R1CS input to OpeningId");
                acc.get_opening(key)
            })
            .collect();

        let padded_trace_length = sm.trace_length.next_power_of_two();
        let key = UniformSpartanKey::new(padded_trace_length);

        Self {
            gamma,
            input_claim,
            key,
            outer_sumcheck_r: outer_sumcheck_r.r,
            claimed_witness_evals,
        }
    }

    pub fn gamma(&self) -> F {
        self.gamma
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for InnerSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.key.num_vars_uniform_padded().log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        _accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let num_cycles_bits = self.key.num_steps.ilog2() as usize;
        let (_r_cycle, rx_var) = self.outer_sumcheck_r.split_at(num_cycles_bits);

        let eval_a = self.key.evaluate_uniform_a_at_point(rx_var, r);
        let eval_b = self.key.evaluate_uniform_b_at_point(rx_var, r);
        let eval_c = self.key.evaluate_uniform_c_at_point(rx_var, r);

        let left = eval_a + self.gamma * eval_b + self.gamma.square() * eval_c;
        let eval_z =
            self.key
                .evaluate_z_mle_with_segment_evals(&self.claimed_witness_evals, r, true);

        left * eval_z
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // No polynomial openings to cache for InnerSumcheck
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::weighted_openings(&[
            OpeningId::Virtual(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter),
            OpeningId::Virtual(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter),
            OpeningId::Virtual(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter),
        ])
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
    ) -> Vec<F> {
        vec![self.gamma, self.gamma.square()]
    }
}
