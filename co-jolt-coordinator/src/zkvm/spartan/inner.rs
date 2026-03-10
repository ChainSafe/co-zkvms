use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::{OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::r1cs::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS};
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use jolt_core::zkvm::witness::VirtualPolynomial;

use jolt_core::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulator;

use crate::subprotocols::sumcheck::Rep3SumcheckInstance;
use crate::zkvm::dag::state_manager::StateManager;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3InnerSumcheck<F: JoltField> {
    gamma: F,
    input_claim: F,
    key: UniformSpartanKey<F>,
    outer_sumcheck_r: Vec<F::Challenge>,
    claimed_witness_evals: Vec<F>,
}

impl<F: JoltField> Rep3InnerSumcheck<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Derive gamma from transcript (matches vanilla ordering)
        let gamma: F = sm.transcript.challenge_scalar();

        // Read Az/Bz/Cz claims from accumulator
        let (outer_sumcheck_r, claim_az) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_bz) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_cz) = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);

        let input_claim = claim_az + gamma * claim_bz + gamma.square() * claim_cz;

        // Read claimed_witness_evals from accumulator for ALL_R1CS_INPUTS
        let claimed_witness_evals: Vec<F> = ALL_R1CS_INPUTS
            .iter()
            .map(|r1cs_input| {
                let key =
                    OpeningId::try_from(r1cs_input).expect("Failed to map R1CS input to OpeningId");
                sm.accumulator.get_opening(key)
            })
            .collect();

        // Derive key from trace_length
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

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3InnerSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.key.num_vars_uniform_padded().log_2()
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        _accumulator: &Rep3OpeningAccumulator<F>,
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

    fn cache_openings(
        &self,
        _accumulator: &mut Rep3OpeningAccumulator<F>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
        _claims: Vec<F>,
    ) {
        // No polynomial openings to cache (matches vanilla InnerSumcheck)
    }
}
