use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::identity_poly::UnmapRamAddressPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use num_traits::Zero;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::utils::types::Rep3Value;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

const DEGREE: usize = 2;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Worker-side RAF evaluation sumcheck. All polynomials are PUBLIC (derived from
/// public addresses and eq evaluations), so round evaluations are promoted to
/// trivial additive shares — no MPC communication.
pub struct Rep3RafEvaluationWorker<F: JoltField> {
    party_id: PartyID,
    input_claim: F,
    log_K: usize,
    _start_address: u64,
    ra: MultilinearPolynomial<F>,
    unmap: UnmapRamAddressPolynomial<F>,
}

impl<F: JoltField> Rep3RafEvaluationWorker<F> {
    pub fn new<PCS: CommitmentScheme<Field = F>>(sm: &mut StateManagerWorker<'_, F, PCS>) -> Self {
        let party_id = sm.party_id;
        let memory_layout = &sm.program_io.memory_layout;
        let K = sm.ram_K;
        let cycle_witness = &sm.prover_state.cycle_witness;

        let (r_cycle_point, _raf_claim_share) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamAddress,
            SumcheckId::SpartanOuter,
        );

        let eq_r_cycle: Vec<F> = EqPolynomial::evals(&r_cycle_point.r);

        // Build ra histogram from public addresses
        let T = cycle_witness.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let ra_evals: Vec<F> = cycle_witness
            .meta()
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, meta_chunk)| {
                let mut result = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for m in meta_chunk {
                    if let Some(k) = remap_address(m.ram_addr, memory_layout) {
                        result[k as usize] += eq_r_cycle[j];
                    }
                    j += 1;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );

        // raf_claim is PUBLIC and can be recomputed locally:
        //   raf_claim = Σ_k ra(k) * unmap(k)
        // where unmap(k) = 8*k + (start_address - 8).
        let base = F::from_u64(memory_layout.trusted_advice_start - 8);
        let input_claim: F = ra_evals
            .iter()
            .enumerate()
            .map(|(k, ra_k)| *ra_k * (F::from_u64((8 * k) as u64) + base))
            .sum();

        let ra = MultilinearPolynomial::from(ra_evals);
        let unmap = UnmapRamAddressPolynomial::new(K.log_2(), memory_layout.trusted_advice_start);

        Self {
            party_id,
            input_claim,
            log_K: K.log_2(),
            _start_address: memory_layout.trusted_advice_start,
            ra,
            unmap,
        }
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3RafEvaluationWorker<F>
{
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.log_K
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
        // All PUBLIC — compute plain evaluations.
        // Evaluate each linear factor at max_degree points so the degree-2
        // product is correctly represented when batched with higher-degree instances.
        let eval_degree = max_degree.max(DEGREE);

        let evals: Vec<F> = (0..self.ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals: Vec<F> =
                    self.ra
                        .sumcheck_evals(i, eval_degree, BindingOrder::HighToLow);
                let unmap_evals: Vec<F> =
                    self.unmap
                        .sumcheck_evals(i, eval_degree, BindingOrder::HighToLow);
                let mut result = vec![F::zero(); max_degree];
                for d in 0..max_degree {
                    result[d] = ra_evals[d] * unmap_evals[d];
                }
                result
            })
            .reduce(
                || vec![F::zero(); max_degree],
                |mut running, new| {
                    for d in 0..max_degree {
                        running[d] = running[d] + new[d];
                    }
                    running
                },
            );

        // Promote to trivial additive shares so that sum across 3 parties = eval.
        evals
            .into_iter()
            .map(|e| additive::promote_to_trivial_share(e, self.party_id))
            .collect()
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize, _io_ctx: &mut IoContextPool<N>) {
        rayon::join(
            || self.ra.bind_parallel(r_j, BindingOrder::HighToLow),
            || self.unmap.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_worker(
        &mut self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .0;
        let ra_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());

        let ra_claim = self.ra.final_sumcheck_claim();

        accumulator.append_virtual_public(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
            ra_claim,
            self.party_id,
        );

        vec![rep3_arith::promote_to_trivial_share(
            self.party_id,
            ra_claim,
        )]
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3RafEvaluation<F: JoltField> {
    input_claim: F,
    log_K: usize,
    start_address: u64,
}

impl<F: JoltField> Rep3RafEvaluation<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let K = sm.ram_K;
        let raf_claim = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .1;

        Self {
            input_claim: raf_claim,
            log_K: K.log_2(),
            start_address: sm.program_io.memory_layout.trusted_advice_start,
        }
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3RafEvaluation<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.log_K
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let unmap_eval =
            UnmapRamAddressPolynomial::<F>::new(self.log_K, self.start_address).evaluate(r);

        let ra_claim = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation)
            .1;

        unmap_eval * ra_claim
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
        r_address: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        let r_cycle = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter)
            .0;
        let ra_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
            claims[0],
        );
    }
}
