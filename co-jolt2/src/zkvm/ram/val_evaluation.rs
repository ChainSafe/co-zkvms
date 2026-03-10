use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding,
};
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use rayon::prelude::*;

use jolt_core::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::utils::types::Rep3Value;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::Rep3SumcheckInstanceWorker;
use crate::zkvm::dag::state_manager::StateManagerWorker;

const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3RamValEvaluationWorker<F: JoltField> {
    party_id: PartyID,
    input_claim: F,
    num_rounds: usize,
    K: usize,
    /// Committed RamInc polynomial (SHARED)
    inc: Rep3DensePolynomial<F>,
    /// Write-address polynomial: wa(k, j) = eq(r_address, remap(ram_addr[j])) (PUBLIC)
    wa: MultilinearPolynomial<F>,
    /// LT polynomial (PUBLIC)
    lt: MultilinearPolynomial<F>,
}

impl<F: JoltField> Rep3RamValEvaluationWorker<F> {
    pub fn new<PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerWorker<'_, F, PCS>,
        input_claim: F,
        inc: Rep3DensePolynomial<F>,
    ) -> Self {
        let party_id = sm.party_id;
        let T = sm.prover_state.cycle_witness.len();
        let K = sm.ram_K;

        let (opening_point, _) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );

        let r_address_len = K.log_2();
        let (r_address_slice, r_cycle_slice) = opening_point.split_at(r_address_len);
        let r_address: Vec<F::Challenge> = r_address_slice.into();
        let r_cycle: Vec<F::Challenge> = r_cycle_slice.into();

        // wa: PUBLIC polynomial from eq(r_address) and public ram_addr
        let eq_r_address = EqPolynomial::evals(&r_address);
        let memory_layout = &sm.program_io.memory_layout;
        let wa_evals: Vec<F> = sm
            .prover_state
            .cycle_witness
            .meta()
            .par_iter()
            .map(|m| {
                let addr = m.ram_addr;
                remap_address(addr, memory_layout)
                    .map(|k| eq_r_address[k as usize])
                    .unwrap_or(F::zero())
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa_evals);

        // LT polynomial (PUBLIC) — same construction as vanilla
        let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
        for (i, r) in r_cycle.iter().rev().enumerate() {
            let (evals_left, evals_right) = lt.split_at_mut(1 << i);
            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r;
                    *x += *r - *y;
                });
        }
        let lt = MultilinearPolynomial::from(lt);

        let num_rounds = r_cycle.len().pow2().log_2();

        Self {
            party_id,
            input_claim,
            num_rounds,
            K,
            inc,
            wa,
            lt,
        }
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3RamValEvaluationWorker<F>
{
    fn degree(&self) -> usize {
        DEGREE
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
        // inc(SHARED) * wa(PUBLIC) * lt(PUBLIC) = SHARED → AdditiveShare
        let evals: Vec<AdditiveShare<F>> = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = self.inc.sumcheck_evals(i, DEGREE, BindingOrder::HighToLow);
                let wa_evals: [F; DEGREE] = self
                    .wa
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let lt_evals: [F; DEGREE] = self
                    .lt
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                // inc(SHARED) * wa(PUB) * lt(PUB) → SHARED (mul_public twice) → AdditiveShare
                let mut result = [AdditiveShare::<F>::zero(); DEGREE];
                for d in 0..DEGREE {
                    let prod = rep3_arith::mul_public(inc_evals[d], wa_evals[d] * lt_evals[d]);
                    result[d] = prod.into_additive();
                }
                result
            })
            .reduce(
                || [AdditiveShare::<F>::zero(); DEGREE],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            )
            .to_vec();

        // Pad if needed
        let mut result = vec![AdditiveShare::<F>::zero(); max_degree];
        for (i, &e) in evals.iter().enumerate() {
            if i < max_degree {
                result[i] = e;
            }
        }
        result
    }

    fn bind(
        &mut self,
        r_j: F::Challenge,
        _round: usize,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) {
        self.wa.bind_parallel(r_j, BindingOrder::HighToLow);
        self.lt.bind_parallel(r_j, BindingOrder::HighToLow);
        self.inc.bind(r_j.into(), BindingOrder::HighToLow);
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
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let (opening_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, _) = opening_point.split_at(self.K.log_2());

        let inc_claim = self.inc.final_sumcheck_claim();
        let wa_claim = self.wa.final_sumcheck_claim();

        // inc is SHARED, committed
        accumulator.append_dense(
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamValEvaluation,
            r_cycle.r.clone(),
            &[inc_claim],
        );

        // wa is PUBLIC (virtual)
        let r = [r_address.r.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.append_virtual_public(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            OpeningPoint::new(r),
            wa_claim,
            self.party_id,
        );

        vec![
            inc_claim,
            rep3_arith::promote_to_trivial_share(self.party_id, wa_claim),
        ]
    }
}

// ---------------------------------------------------------------------------
