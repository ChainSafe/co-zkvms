use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::{OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::r1cs::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS};
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};

use crate::field::JoltField;
use crate::poly::mixed_polynomial::MixedPolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::poly::Polynomial;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::subprotocols::sumcheck::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use crate::utils::types::Rep3Value;
use crate::zkvm::dag::state_manager::StateManagerCoordinator;

use jolt_core::poly::multilinear_polynomial::BindingOrder;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3InnerSumcheckWorker<F: JoltField> {
    poly_abc_small: MixedPolynomial<F>,
    poly_z: MixedPolynomial<F>,
    num_rounds: usize,
    input_claim: F,
    party_id: PartyID,
}

impl<F: JoltField> Rep3InnerSumcheckWorker<F> {
    pub fn new(
        gamma: F,
        input_claim: F,
        outer_sumcheck_r: &[F::Challenge],
        claimed_witness_evals: Vec<Rep3PrimeFieldShare<F>>,
        padded_trace_length: usize,
        party_id: PartyID,
    ) -> Self {
        let key = UniformSpartanKey::<F>::new(padded_trace_length);
        let num_cycles_bits = key.num_steps.ilog2() as usize;
        let (_r_cycle, rx_var) = outer_sumcheck_r.split_at(num_cycles_bits);

        let num_vars_uniform = key.num_vars_uniform_padded();

        // poly_abc_small: PUBLIC — combined A/B/C matrix evaluation with RLC
        let poly_abc_small = MixedPolynomial::from_public_evals(
            key.evaluate_small_matrix_rlc(rx_var, gamma),
            party_id,
        );

        // poly_z: MIXED — shared witness evals + public constant column
        let mut bind_z = vec![Rep3Value::zero_public(); num_vars_uniform];
        for r1cs_input in ALL_R1CS_INPUTS.iter() {
            bind_z[r1cs_input.to_index()] =
                Rep3Value::Shared(claimed_witness_evals[r1cs_input.to_index()]);
        }
        // Set constant column
        let const_col = JoltR1CSInputs::num_inputs();
        if const_col < num_vars_uniform {
            bind_z[const_col] = Rep3Value::Public(F::one());
        }

        let poly_z = MixedPolynomial::new(bind_z, party_id);
        assert_eq!(
            poly_abc_small.len(),
            poly_z.len(),
            "poly_abc_small and poly_z length mismatch"
        );

        let num_rounds = num_vars_uniform.log_2();

        Self {
            poly_abc_small,
            poly_z,
            num_rounds,
            input_claim,
            party_id,
        }
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3InnerSumcheckWorker<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self) -> Rep3Value<F> {
        Rep3Value::Public(self.input_claim)
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        _previous_claim: AdditiveShare<F>,
        max_degree: usize,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        let half_len = self.poly_abc_small.len() / 2;
        let party_id = self.party_id;

        // The product abc*z is degree 2, but when batched with degree-3 instances
        // we need evaluations at {0, 2, 3} (max_degree points). We achieve this by
        // evaluating each linear factor at max_degree points and multiplying.
        let eval_degree = max_degree.max(2);
        let mut evals = vec![AdditiveShare::<F>::zero(); max_degree];

        for i in 0..half_len {
            let abc_evals = self.poly_abc_small.sumcheck_evals(
                i,
                eval_degree,
                BindingOrder::HighToLow,
                party_id,
            );
            let z_evals =
                self.poly_z
                    .sumcheck_evals(i, eval_degree, BindingOrder::HighToLow, party_id);

            // evals at {0, 2, 3, ..., eval_degree}
            for j in 0..max_degree {
                evals[j] += abc_evals[j].mul(&z_evals[j]).into_additive(party_id);
            }
        }

        evals
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize, _io_ctx: &mut IoContextPool<N>) {
        let r: F = r_j.into();
        self.poly_abc_small.bind(r, BindingOrder::HighToLow);
        self.poly_z.bind(r, BindingOrder::HighToLow);
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_worker(
        &mut self,
        _accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        // No polynomial openings to cache (matches vanilla InnerSumcheck)
        vec![]
    }
}

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
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
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
