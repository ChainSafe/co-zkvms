use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::opening_proof::{OpeningId, OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::r1cs::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS};
use jolt_core::zkvm::r1cs::key::UniformSpartanKey;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;

use crate::poly::mixed_polynomial::MixedPolynomial;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::poly::Polynomial;
use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::subprotocols::sumcheck::Rep3SumcheckInstanceWorker;
use crate::utils::types::Rep3Value;

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
        let poly_abc_small = MixedPolynomial::from_public_evals(key.evaluate_small_matrix_rlc(rx_var, gamma), party_id);

        // poly_z: MIXED — shared witness evals + public constant column
        let mut bind_z = vec![Rep3Value::zero_public(); num_vars_uniform];
        for r1cs_input in ALL_R1CS_INPUTS.iter() {
            bind_z[r1cs_input.to_index()] = Rep3Value::Shared(claimed_witness_evals[r1cs_input.to_index()]);
        }
        // Set constant column
        let const_col = JoltR1CSInputs::num_inputs();
        if const_col < num_vars_uniform {
            bind_z[const_col] = Rep3Value::Public(F::one());
        }

        let poly_z = MixedPolynomial::new(bind_z, party_id);
        assert_eq!(poly_abc_small.len(), poly_z.len(), "poly_abc_small and poly_z length mismatch");

        let num_rounds = num_vars_uniform.log_2();

        Self { poly_abc_small, poly_z, num_rounds, input_claim, party_id }
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N> for Rep3InnerSumcheckWorker<F> {
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

        if max_degree <= 3 {
            let mut evals = vec![AdditiveShare::<F>::zero(); max_degree];
            for i in 0..half_len {
                let abc_evals = self.poly_abc_small.sumcheck_evals_deg_3_high_to_low(i);
                let z_evals = self.poly_z.sumcheck_evals_deg_3_high_to_low(i);
                for j in 0..max_degree {
                    evals[j] += abc_evals[j].mul(&z_evals[j]).into_additive(party_id);
                }
            }
            return evals;
        }

        // Fallback for unusual batching degrees.
        // The product abc*z is degree 2, but when batched with higher-degree instances
        // we evaluate each linear factor at the required points and multiply.
        let eval_degree = max_degree.max(2);
        let mut evals = vec![AdditiveShare::<F>::zero(); max_degree];
        for i in 0..half_len {
            let abc_evals = self.poly_abc_small.sumcheck_evals(i, eval_degree, BindingOrder::HighToLow, party_id);
            let z_evals = self.poly_z.sumcheck_evals(i, eval_degree, BindingOrder::HighToLow, party_id);
            for j in 0..max_degree {
                evals[j] += abc_evals[j].mul(&z_evals[j]).into_additive(party_id);
            }
        }
        evals
    }

    fn bind(
        &mut self,
        r_j: F::Challenge,
        _round: usize,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) {
        let r: F = r_j.into();
        self.poly_abc_small.bind(r, BindingOrder::HighToLow);
        self.poly_z.bind(r, BindingOrder::HighToLow);
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
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
