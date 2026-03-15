use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::identity_poly::UnmapRamAddressPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::utils::math::Math;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use rayon::prelude::*;

use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::utils::types::Rep3Value;
use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::Rep3SumcheckInstanceWorker;
use crate::zkvm::dag::state_manager::StateManagerWorker;

const DEGREE: usize = 2;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Worker-side RAF evaluation sumcheck. All polynomials are PUBLIC (derived from
/// public addresses and eq evaluations), so round evaluations are promoted to
/// trivial additive shares — no MPC communication.
pub struct Rep3RafEvaluationWorker<F: JoltField> {
    party_id: PartyID,
    input_claim: Rep3PrimeFieldShare<F>,
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

        let (r_cycle_point, raf_claim_share) =
            sm.accumulator.get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter);

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
                    running.par_iter_mut().zip(new.into_par_iter()).for_each(|(x, y)| *x += y);
                    running
                },
            );

        let ra = MultilinearPolynomial::from(ra_evals);
        let unmap = UnmapRamAddressPolynomial::new(K.log_2(), memory_layout.trusted_advice_start);

        Self {
            party_id,
            input_claim: raf_claim_share,
            log_K: K.log_2(),
            _start_address: memory_layout.trusted_advice_start,
            ra,
            unmap,
        }
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N> for Rep3RafEvaluationWorker<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.log_K
    }

    fn input_claim(&self) -> Rep3Value<F> {
        Rep3Value::Shared(self.input_claim)
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        // All PUBLIC — compute plain evaluations.
        let base: [F; DEGREE] = (0..self.ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = self.ra.sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let unmap_evals = self.unmap.sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);
                [ra_evals[0] * unmap_evals[0], ra_evals[1] * unmap_evals[1]]
            })
            .reduce(|| [F::zero(); DEGREE], |running, new| [running[0] + new[0], running[1] + new[1]]);

        let y0 = additive::promote_to_trivial_share(base[0], self.party_id);
        let y2 = additive::promote_to_trivial_share(base[1], self.party_id);
        if max_degree == DEGREE {
            return vec![y0, y2];
        }

        let y1 = previous_claim - y0;
        let mut evals = Vec::with_capacity(max_degree);
        evals.push(y0);
        evals.push(y2);

        // Interpolate from y(0), y(1), y(2) in additive-share space so the
        // padded degree-3 batch points preserve the shared claim exactly.
        for k in 3..=max_degree {
            let x = F::from_u64(k as u64);
            let l0 = (x - F::one()) * (x - F::from_u64(2)) * F::TWO_INV;
            let l1 = -x * (x - F::from_u64(2));
            let l2 = x * (x - F::one()) * F::TWO_INV;
            evals.push(y0 * l0 + y1 * l1 + y2 * l2);
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
        rayon::join(
            || self.ra.bind_parallel(r_j, BindingOrder::HighToLow),
            || self.unmap.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }

    fn normalize_opening_point(&self, opening_point: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_worker(
        &mut self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let r_cycle =
            accumulator.get_virtual_polynomial_opening(VirtualPolynomial::RamAddress, SumcheckId::SpartanOuter).0;
        let ra_opening_point = OpeningPoint::new([r_address.r.as_slice(), r_cycle.r.as_slice()].concat());

        let ra_claim = self.ra.final_sumcheck_claim();

        accumulator.append_virtual_public(
            VirtualPolynomial::RamRa,
            SumcheckId::RamRafEvaluation,
            ra_opening_point,
            ra_claim,
            self.party_id,
        );

        vec![rep3_arith::promote_to_trivial_share(self.party_id, ra_claim)]
    }
}

// ---------------------------------------------------------------------------
