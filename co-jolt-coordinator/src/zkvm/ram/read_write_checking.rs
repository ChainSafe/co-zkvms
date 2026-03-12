use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
#[cfg(feature = "zk")]
use jolt_core::poly::opening_proof::OpeningId;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN};
#[cfg(feature = "zk")]
use jolt_core::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use crate::poly::opening_proof::Rep3OpeningAccumulator;
use jolt_core::field::JoltField;

use crate::zkvm::dag::stage::Rep3SumcheckInstance;
use crate::zkvm::dag::state_manager::StateManager;

const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3RamReadWriteChecking<F: JoltField> {
    K: usize,
    T: usize,
    gamma: F,
    sumcheck_switch_index: usize,
    input_claim: F,
    #[cfg(feature = "zk")]
    r_cycle: Vec<F::Challenge>,
}

impl<F: JoltField> Rep3RamReadWriteChecking<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let gamma = sm.transcript.challenge_scalar();
        let K = sm.ram_K;

        let (_, rv_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );
        let (_, wv_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamWriteValue,
            SumcheckId::SpartanOuter,
        );
        let input_claim = rv_claim + gamma * wv_claim;

        // Infer T from opening point dimension
        let (r_point, _) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );
        let T = 1 << r_point.r.len();

        Self {
            K,
            T,
            gamma,
            sumcheck_switch_index: sm.twist_sumcheck_switch_index,
            input_claim,
            #[cfg(feature = "zk")]
            r_cycle: r_point.r,
        }
    }

    /// Gamma challenge for broadcasting to workers.
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// Public input claim for broadcasting to workers.
    pub fn input_claim(&self) -> F {
        self.input_claim
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3RamReadWriteChecking<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let (r_prime, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamReadValue,
            SumcheckId::SpartanOuter,
        );

        let mut r_cycle = r[..self.sumcheck_switch_index].to_vec();
        r_cycle.extend(r[self.sumcheck_switch_index..self.T.log_2()].iter().rev());
        let r_cycle = OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle);
        let eq_eval = EqPolynomial::mle_endian(&r_prime, &r_cycle);

        let (_, ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamReadWriteChecking,
        );

        eq_eval * ra_claim * (val_claim + self.gamma * (val_claim + inc_claim))
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let log_T = self.T.log_2();
        let mut r_cycle = opening_point[self.sumcheck_switch_index..log_T].to_vec();
        r_cycle.extend(opening_point[..self.sumcheck_switch_index].iter().rev());
        let r_address = opening_point[log_T..].to_vec();
        [r_address, r_cycle].concat().into()
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        // claims: [val, ra, inc]
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            claims[0],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
            opening_point.clone(),
            claims[1],
        );

        let (_, r_cycle) = opening_point.split_at(self.K.log_2());
        accumulator.append_dense(
            transcript,
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamReadWriteChecking,
            r_cycle.r,
            vec![claims[2]],
        );
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::weighted_openings(&[
            OpeningId::Virtual(VirtualPolynomial::RamReadValue, SumcheckId::SpartanOuter),
            OpeningId::Virtual(VirtualPolynomial::RamWriteValue, SumcheckId::SpartanOuter),
        ])
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &Rep3OpeningAccumulator<F>,
    ) -> Vec<F> {
        vec![self.gamma]
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::product(vec![
                ValueSource::challenge(0),
                ValueSource::opening(OpeningId::Virtual(
                    VirtualPolynomial::RamRa,
                    SumcheckId::RamReadWriteChecking,
                )),
                ValueSource::opening(OpeningId::Virtual(
                    VirtualPolynomial::RamVal,
                    SumcheckId::RamReadWriteChecking,
                )),
            ]),
            ProductTerm::product(vec![
                ValueSource::challenge(0),
                ValueSource::challenge(1),
                ValueSource::opening(OpeningId::Virtual(
                    VirtualPolynomial::RamRa,
                    SumcheckId::RamReadWriteChecking,
                )),
                ValueSource::opening(OpeningId::Virtual(
                    VirtualPolynomial::RamVal,
                    SumcheckId::RamReadWriteChecking,
                )),
            ]),
            ProductTerm::product(vec![
                ValueSource::challenge(0),
                ValueSource::challenge(1),
                ValueSource::opening(OpeningId::Virtual(
                    VirtualPolynomial::RamRa,
                    SumcheckId::RamReadWriteChecking,
                )),
                ValueSource::opening(OpeningId::Committed(
                    CommittedPolynomial::RamInc,
                    SumcheckId::RamReadWriteChecking,
                )),
            ]),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let eq_eval = ram_eq_eval(
            &self.r_cycle,
            sumcheck_challenges,
            self.sumcheck_switch_index,
            self.T.log_2(),
        );
        vec![eq_eval, self.gamma]
    }
}

#[cfg(feature = "zk")]
fn ram_eq_eval<F: JoltField>(
    r_cycle: &[F::Challenge],
    sumcheck_challenges: &[F::Challenge],
    switch_index: usize,
    log_t: usize,
) -> F {
    let mut reordered = sumcheck_challenges[..switch_index].to_vec();
    reordered.extend(sumcheck_challenges[switch_index..log_t].iter().rev());
    EqPolynomial::mle_endian(
        &OpeningPoint::<BIG_ENDIAN, F>::new(r_cycle.to_vec()),
        &OpeningPoint::<LITTLE_ENDIAN, F>::new(reordered),
    )
}
