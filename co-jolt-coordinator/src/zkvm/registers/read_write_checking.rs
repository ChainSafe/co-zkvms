use jolt_common::constants::REGISTER_COUNT;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};

use crate::poly::opening_proof::Rep3OpeningAccumulator;
use jolt_core::field::JoltField;

use crate::zkvm::dag::stage::Rep3SumcheckInstance;
use crate::zkvm::dag::state_manager::StateManager;

const K: usize = REGISTER_COUNT as usize;
const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3RegistersReadWriteChecking<F: JoltField> {
    T: usize,
    gamma: F,
    gamma_sqr: F,
    sumcheck_switch_index: usize,
    input_claim: F,
}

impl<F: JoltField> Rep3RegistersReadWriteChecking<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (r_point, rs1_rv_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);
        let (_, rs2_rv_claim) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs2Value, SumcheckId::SpartanOuter);
        let (_, rd_wv_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWriteValue,
            SumcheckId::SpartanOuter,
        );

        let gamma: F = sm.transcript.challenge_scalar();
        let input_claim = rd_wv_claim + gamma * rs1_rv_claim + gamma.square() * rs2_rv_claim;

        let T = 1 << r_point.r.len();

        Self {
            T,
            gamma,
            gamma_sqr: gamma.square(),
            sumcheck_switch_index: sm.twist_sumcheck_switch_index,
            input_claim,
        }
    }

    /// Gamma challenge for broadcasting to workers.
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// Input claim for broadcasting to workers.
    pub fn input_claim(&self) -> F {
        self.input_claim
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3RegistersReadWriteChecking<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        K.log_2() + self.T.log_2()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let (r_prime, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Rs1Value, SumcheckId::SpartanOuter);

        let mut r_cycle = r[..self.sumcheck_switch_index].to_vec();
        r_cycle.extend(r[self.sumcheck_switch_index..self.T.log_2()].iter().rev());
        let r_cycle = OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle);

        let eq_eval_cycle = EqPolynomial::mle_endian(&r_prime, &r_cycle);

        let (_, val_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs1_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rs2_ra_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, rd_wa_claim) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, inc_claim) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::RegistersReadWriteChecking,
        );

        eq_eval_cycle
            * (rd_wa_claim * (inc_claim + val_claim)
                + self.gamma * rs1_ra_claim * val_claim
                + self.gamma_sqr * rs2_ra_claim * val_claim)
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
        // claims: [val, rs1_ra, rs2_ra, rd_wa, inc]
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RegistersVal,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            claims[0],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            claims[1],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::Rs2Ra,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            claims[2],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersReadWriteChecking,
            opening_point.clone(),
            claims[3],
        );

        let (_, r_cycle) = opening_point.split_at(K.log_2());
        accumulator.append_dense(
            transcript,
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersReadWriteChecking,
            r_cycle.r,
            vec![claims[4]],
        );
    }
}
