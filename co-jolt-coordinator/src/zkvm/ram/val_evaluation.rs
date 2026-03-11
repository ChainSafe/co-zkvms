use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
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

pub struct Rep3RamValEvaluation<F: JoltField> {
    input_claim: F,
    num_rounds: usize,
    K: usize,
}

impl<F: JoltField> Rep3RamValEvaluation<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
        init_eval: F,
    ) -> Self {
        let (opening_point, claimed_evaluation) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );

        let K = sm.ram_K;
        let r_address_len = K.log_2();
        let num_rounds = opening_point.r.len() - r_address_len;

        Self {
            input_claim: claimed_evaluation - init_eval,
            num_rounds,
            K,
        }
    }

    /// Input claim for broadcasting to workers.
    pub fn input_claim(&self) -> F {
        self.input_claim
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3RamValEvaluation<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let (opening_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, r_cycle) = opening_point.split_at(self.K.log_2());

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r.iter().zip(r_cycle.r.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
        );
        let (_, wa_claim) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValEvaluation);

        inc_claim * wa_claim * lt_eval
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        let (opening_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, _) = opening_point.split_at(self.K.log_2());

        // Must match vanilla order: append_virtual (wa) first, then append_dense (inc)
        let r = [r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            OpeningPoint::new(r),
            claims[1], // wa_claim
        );

        accumulator.append_dense(
            transcript,
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamValEvaluation,
            r_cycle.r.clone(),
            vec![claims[0]], // inc_claim
        );
    }
}
