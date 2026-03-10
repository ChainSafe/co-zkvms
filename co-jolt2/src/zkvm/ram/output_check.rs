use jolt_common::constants::{RAM_START_ADDRESS, RAM_WORD_SIZE};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::program_io_polynomial::ProgramIOPolynomial;
use jolt_core::poly::range_mask_polynomial::RangeMaskPolynomial;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use rayon::prelude::*;
use tracer::JoltDevice;

use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::mixed_polynomial::MixedPolynomial;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::poly::Polynomial;
use crate::utils::types::Rep3Value;
use jolt_core::field::JoltField;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::Rep3SumcheckInstanceWorker;
use crate::zkvm::dag::state_manager::StateManagerWorker;
use crate::zkvm::instruction_lookups::booleanity::extend_degree_3_evals;

const DEGREE_OUTPUT: usize = 3;
const DEGREE_VAL_FINAL: usize = 2;

// ===========================================================================
// OutputSumcheck
// ===========================================================================

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3OutputSumcheckWorker<F: JoltField> {
    party_id: PartyID,
    K: usize,
    /// val_init (SHARED) — initial RAM state as shared MLE
    val_init: Rep3DensePolynomial<F>,
    /// val_final (MIXED) — final RAM state, keeping known-public regions public
    val_final: MixedPolynomial<F>,
    /// val_io (PUBLIC) — I/O-masked final state MLE
    val_io: MultilinearPolynomial<F>,
    /// eq_poly (PUBLIC) — EQ(r_address, ·)
    eq_poly: MultilinearPolynomial<F>,
    /// io_mask (PUBLIC) — range mask for I/O region
    io_mask: MultilinearPolynomial<F>,
}

impl<F: JoltField> Rep3OutputSumcheckWorker<F> {
    pub fn new<PCS: CommitmentScheme<Field = F>>(
        val_init: Rep3DensePolynomial<F>,
        val_final: MixedPolynomial<F>,
        r_address: Vec<F::Challenge>,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Self {
        let party_id = sm.party_id;
        let K = val_final.len();
        let memory_layout = &sm.program_io.memory_layout;
        let ws = RAM_WORD_SIZE as usize;

        // Build val_io (PUBLIC) from program_io — for correct execution this
        // matches val_final at I/O addresses and is 0 elsewhere.
        let io_start = remap_address(memory_layout.input_start, memory_layout).unwrap() as usize;
        let io_end = remap_address(RAM_START_ADDRESS, memory_layout).unwrap() as usize;

        let mut val_io_evals = vec![0u64; K];
        let program_io = &sm.program_io;
        // Populate input words
        let mut input_index = io_start;
        for chunk in program_io.inputs.chunks(ws) {
            val_io_evals[input_index] = jolt_core::zkvm::ram::bytes_to_ram_word(chunk);
            input_index += 1;
        }
        // Populate output words
        let mut output_index =
            remap_address(memory_layout.output_start, memory_layout).unwrap() as usize;
        for chunk in program_io.outputs.chunks(ws) {
            val_io_evals[output_index] = jolt_core::zkvm::ram::bytes_to_ram_word(chunk);
            output_index += 1;
        }
        // Panic bit
        let panic_index = remap_address(memory_layout.panic, memory_layout).unwrap() as usize;
        val_io_evals[panic_index] = program_io.panic as u64;
        // Termination bit
        if !program_io.panic {
            let termination_index =
                remap_address(memory_layout.termination, memory_layout).unwrap() as usize;
            val_io_evals[termination_index] = 1;
        }
        let val_io: MultilinearPolynomial<F> = val_io_evals.into();

        // io_mask (PUBLIC): 1 for I/O addresses, 0 elsewhere
        let mut io_mask_evals = vec![0u8; K];
        for k in io_start..io_end {
            io_mask_evals[k] = 1;
        }
        let io_mask: MultilinearPolynomial<F> = io_mask_evals.into();

        // eq_poly (PUBLIC): EQ(r_address, ·)
        let eq_poly: MultilinearPolynomial<F> = EqPolynomial::<F>::evals(&r_address).into();

        Self {
            party_id,
            K,
            val_init,
            val_final,
            val_io,
            eq_poly,
            io_mask,
        }
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3OutputSumcheckWorker<F>
{
    fn degree(&self) -> usize {
        DEGREE_OUTPUT
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> Rep3Value<F> {
        Rep3Value::Public(F::zero()) // Zero-check
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        // P(k) = eq(k) * io_mask(k) * (val_final(k) - val_io(k))
        //      = eq(k) * io_mask(k) * val_final(k) - eq(k) * io_mask(k) * val_io(k)
        //        ^^^^^^^^^^^^^^^^^^^^^ SHARED ^^^^    ^^^^^^^^^^^^ PUBLIC ^^^^^^^^^^^
        let party_id = self.party_id;
        let half_len = self.eq_poly.len() / 2;

        let base: [AdditiveShare<F>; DEGREE_OUTPUT] = (0..half_len)
            .into_par_iter()
            .map(|i| {
                let eq_evals = sumcheck_evals_deg_3_high_to_low_public::<F>(&self.eq_poly, i);
                let mask_evals = sumcheck_evals_deg_3_high_to_low_public::<F>(&self.io_mask, i);
                let vf_evals = self.val_final.sumcheck_evals_deg_3_high_to_low(i);
                let vio_evals = sumcheck_evals_deg_3_high_to_low_public::<F>(&self.val_io, i);

                let mut result = [AdditiveShare::<F>::zero(); DEGREE_OUTPUT];
                for d in 0..DEGREE_OUTPUT {
                    let eq_mask = eq_evals[d] * mask_evals[d]; // PUBLIC
                    let vio = vio_evals[d]; // PUBLIC

                    // eq_mask * (val_final[d] - val_io[d])
                    match vf_evals[d] {
                        Rep3Value::Public(vf) => {
                            let term = eq_mask * (vf - vio);
                            result[d] += additive::promote_to_trivial_share(term, party_id);
                        }
                        Rep3Value::Shared(vf) => {
                            // eq_mask * val_final[d] (SHARED) - eq_mask * val_io[d] (PUBLIC)
                            let shared_term = rep3_arith::mul_public(vf, eq_mask).into_additive();
                            let public_term = eq_mask * vio;
                            result[d] +=
                                additive::sub_shared_by_public(shared_term, public_term, party_id);
                        }
                        Rep3Value::Additive(_) => unreachable!("val_final must not be additive"),
                    }
                }
                result
            })
            .reduce(
                || [AdditiveShare::<F>::zero(); DEGREE_OUTPUT],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        extend_degree_3_evals::<F>(previous_claim, &base, max_degree)
    }

    fn bind(
        &mut self,
        r_j: F::Challenge,
        _round: usize,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) {
        self.val_init.bind(r_j.into(), BindingOrder::HighToLow);
        self.val_final.bind(r_j.into(), BindingOrder::HighToLow);
        self.val_io.bind_parallel(r_j, BindingOrder::HighToLow);
        self.eq_poly.bind_parallel(r_j, BindingOrder::HighToLow);
        self.io_mask.bind_parallel(r_j, BindingOrder::HighToLow);
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
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let val_final_claim = match self.val_final.final_sumcheck_claim() {
            Rep3Value::Public(f) => rep3_arith::promote_to_trivial_share(self.party_id, f),
            Rep3Value::Shared(s) => s,
            Rep3Value::Additive(_) => unreachable!("val_final claim must not be additive"),
        };
        let val_init_claim = self.val_init.final_sumcheck_claim();

        accumulator.append_virtual(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
            val_final_claim,
        );
        accumulator.append_virtual(
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
            val_init_claim,
        );

        vec![val_final_claim, val_init_claim]
    }
}

#[inline]
fn sumcheck_evals_deg_3_high_to_low_public<F: JoltField>(
    poly: &MultilinearPolynomial<F>,
    index: usize,
) -> [F; DEGREE_OUTPUT] {
    let half = poly.len() / 2;
    let eval_0 = poly.get_bound_coeff(index);
    let eval_1 = poly.get_bound_coeff(index + half);
    let slope = eval_1 - eval_0;
    let eval_2 = eval_1 + slope;
    let eval_3 = eval_2 + slope;
    [eval_0, eval_2, eval_3]
}

// ===========================================================================
// ValFinalSumcheck
// ===========================================================================

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3ValFinalSumcheckWorker<F: JoltField> {
    party_id: PartyID,
    T: usize,
    input_claim: F,
    /// RamInc polynomial (SHARED)
    inc: Rep3DensePolynomial<F>,
    /// wa = eq(r_address, ram_addr[j]) (PUBLIC)
    wa: MultilinearPolynomial<F>,
}

impl<F: JoltField> Rep3ValFinalSumcheckWorker<F> {
    pub fn new<PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerWorker<'_, F, PCS>,
        input_claim: F,
    ) -> Self {
        let party_id = sm.party_id;
        let cycle_witness = &sm.prover_state.cycle_witness;
        let T = cycle_witness.len();
        let memory_layout = &sm.program_io.memory_layout;

        let r_address = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0
            .r;

        let eq_r_address = EqPolynomial::evals(&r_address);

        // wa (PUBLIC): eq(r_address, remap(ram_addr[j]))
        let wa: Vec<F> = cycle_witness
            .meta()
            .par_iter()
            .map(|m| {
                remap_address(m.ram_addr, memory_layout)
                    .map_or(F::zero(), |k| eq_r_address[k as usize])
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        // Take ownership — this is the last consumer of ram_inc
        let inc = sm.prover_state.cycle_witness.take_ram_inc();

        Self {
            party_id,
            T,
            input_claim,
            inc,
            wa,
        }
    }
}

impl<F: JoltField, N: Rep3NetworkWorker> Rep3SumcheckInstanceWorker<F, N>
    for Rep3ValFinalSumcheckWorker<F>
{
    fn degree(&self) -> usize {
        DEGREE_VAL_FINAL
    }
    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }
    fn input_claim(&self) -> Rep3Value<F> {
        Rep3Value::Public(self.input_claim)
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Vec<AdditiveShare<F>> {
        // inc(SHARED) * wa(PUBLIC) → SHARED → AdditiveShare
        let base: Vec<AdditiveShare<F>> = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_evals =
                    self.inc
                        .sumcheck_evals(j, DEGREE_VAL_FINAL, BindingOrder::HighToLow);
                let wa_evals: Vec<F> =
                    self.wa
                        .sumcheck_evals(j, DEGREE_VAL_FINAL, BindingOrder::HighToLow);

                let mut result = vec![AdditiveShare::<F>::zero(); DEGREE_VAL_FINAL];
                for d in 0..DEGREE_VAL_FINAL {
                    result[d] = rep3_arith::mul_public(inc_evals[d], wa_evals[d]).into_additive();
                }
                result
            })
            .reduce(
                || vec![AdditiveShare::<F>::zero(); DEGREE_VAL_FINAL],
                |mut r, n| {
                    for d in 0..DEGREE_VAL_FINAL {
                        r[d] += n[d];
                    }
                    r
                },
            );

        extend_degree_2_evals(previous_claim, &base, max_degree)
    }

    fn bind(
        &mut self,
        r_j: F::Challenge,
        _round: usize,
        _io_ctx: &mut IoContextPool<N>,
        _preproc: &mut PreprocessingPool<F>,
    ) {
        rayon::join(
            || self.inc.bind(r_j.into(), BindingOrder::HighToLow),
            || self.wa.bind_parallel(r_j, BindingOrder::HighToLow),
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
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let r_address = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        let inc_claim = self.inc.final_sumcheck_claim();
        let wa_claim = self.wa.final_sumcheck_claim();

        accumulator.append_dense(
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamValFinalEvaluation,
            r_cycle_prime.r,
            &[inc_claim],
        );
        accumulator.append_virtual_public(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
            wa_opening_point,
            wa_claim,
            self.party_id,
        );

        vec![
            inc_claim,
            rep3_arith::promote_to_trivial_share(self.party_id, wa_claim),
        ]
    }
}

fn extend_degree_2_evals<F: JoltField>(
    previous_claim: AdditiveShare<F>,
    base: &[AdditiveShare<F>],
    max_degree: usize,
) -> Vec<AdditiveShare<F>> {
    debug_assert_eq!(base.len(), DEGREE_VAL_FINAL);
    debug_assert!(max_degree >= DEGREE_VAL_FINAL);

    if max_degree == DEGREE_VAL_FINAL {
        return base.to_vec();
    }

    let y0 = base[0];
    let y1 = previous_claim - y0;
    let y2 = base[1];

    let mut evals = vec![AdditiveShare::<F>::zero(); max_degree];
    evals[0] = y0;
    evals[1] = y2;

    for x in 3..=max_degree {
        let xf = F::from(x as u64);
        let l0 = (xf - F::from(1u64)) * (xf - F::from(2u64)) * F::TWO_INV;
        let l1 = -xf * (xf - F::from(2u64));
        let l2 = xf * (xf - F::from(1u64)) * F::TWO_INV;
        evals[x - 1] = y0 * l0 + y1 * l1 + y2 * l2;
    }

    evals
}
