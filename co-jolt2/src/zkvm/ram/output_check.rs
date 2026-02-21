use jolt2_common::constants::RAM_START_ADDRESS;
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
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use rayon::prelude::*;
use tracer::JoltDevice;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

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
    /// val_init (PUBLIC) — initial RAM state as MLE
    val_init: MultilinearPolynomial<F>,
    /// val_final (SHARED) — final RAM state in field
    val_final: Rep3DensePolynomial<F>,
}

impl<F: JoltField> Rep3OutputSumcheckWorker<F> {
    pub fn new<PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: Vec<u64>,
        final_ram_field: Vec<Rep3PrimeFieldShare<F>>,
        _r_address: Vec<F::Challenge>,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Self {
        let party_id = sm.party_id;
        let K = final_ram_field.len();
        let val_final = Rep3DensePolynomial::new(final_ram_field);
        let val_init: MultilinearPolynomial<F> = initial_ram_state.into();

        Self {
            party_id,
            K,
            val_init,
            val_final,
        }
    }
}

impl<F: JoltField> Rep3SumcheckInstanceWorker<F> for Rep3OutputSumcheckWorker<F> {
    fn degree(&self) -> usize {
        DEGREE_OUTPUT
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim_public(&self) -> F {
        F::zero() // Zero-check
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        _previous_claim: AdditiveShare<F>,
        max_degree: usize,
    ) -> Vec<AdditiveShare<F>> {
        // Output sumcheck is a zero-check with input claim 0; for honest provers
        // the round message is identically zero. We still bind val_init/val_final
        // so the correct opening claims are cached at the end.
        vec![AdditiveShare::<F>::zero(); max_degree]
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.val_init.bind_parallel(r_j, BindingOrder::HighToLow);
        self.val_final.bind(r_j.into(), BindingOrder::HighToLow);
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_worker(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<Rep3PrimeFieldShare<F>> {
        let val_final_claim = self.val_final.final_sumcheck_claim();
        let val_init_claim = self.val_init.final_sumcheck_claim();

        accumulator.append_virtual(
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
            val_final_claim,
        );
        accumulator.append_virtual_public(
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
            val_init_claim,
            self.party_id,
        );

        vec![val_final_claim]
    }
}

// ---------------------------------------------------------------------------
// OutputSumcheck Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3OutputSumcheck<F: JoltField> {
    K: usize,
    r_address: Vec<F::Challenge>,
    program_io: JoltDevice,
    val_init: MultilinearPolynomial<F>,
}

impl<F: JoltField> Rep3OutputSumcheck<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let K = sm.ram_K;
        let r_address = sm.transcript.challenge_vector_optimized::<F>(K.log_2());
        let initial_memory_state = super::build_initial_memory_state(&sm.preprocessing.shared.ram, &sm.program_io, K);
        let val_init = MultilinearPolynomial::<F>::from(initial_memory_state);
        Self {
            K,
            r_address,
            program_io: sm.program_io.clone(),
            val_init,
        }
    }

    pub fn r_address(&self) -> &[F::Challenge] {
        &self.r_address
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3OutputSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE_OUTPUT
    }
    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }
    fn input_claim_public(&self) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let val_final_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;

        let r_address_prime = &r[..self.r_address.len()];

        let io_mask = RangeMaskPolynomial::<F>::new(
            remap_address(
                self.program_io.memory_layout.input_start,
                &self.program_io.memory_layout,
            )
            .unwrap() as u128,
            remap_address(RAM_START_ADDRESS, &self.program_io.memory_layout)
                .unwrap() as u128,
        );
        let val_io = ProgramIOPolynomial::new(&self.program_io);

        let eq_eval: F = EqPolynomial::<F>::mle(&self.r_address, r_address_prime);
        let io_mask_eval = io_mask.evaluate_mle(r_address_prime);
        let val_io_eval: F = val_io.evaluate(r_address_prime);

        eq_eval * io_mask_eval * (val_final_claim - val_io_eval)
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
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        // claims: [val_final]
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
            claims[0],
        );
        let val_init_claim = self.val_init.evaluate(&opening_point.r);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
            val_init_claim,
        );
    }
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
            .ram_addr
            .par_iter()
            .map(|&addr| {
                remap_address(addr, memory_layout).map_or(F::zero(), |k| eq_r_address[k as usize])
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        // Take ownership — this is the last consumer of ram_inc
        let inc = sm
            .prover_state
            .cycle_witness
            .ram_inc
            .take()
            .expect("ram_inc not populated");

        Self {
            party_id,
            T,
            input_claim,
            inc,
            wa,
        }
    }
}

impl<F: JoltField> Rep3SumcheckInstanceWorker<F> for Rep3ValFinalSumcheckWorker<F> {
    fn degree(&self) -> usize {
        DEGREE_VAL_FINAL
    }
    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }
    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        _previous_claim: AdditiveShare<F>,
        max_degree: usize,
    ) -> Vec<AdditiveShare<F>> {
        // inc(SHARED) * wa(PUBLIC) → SHARED → AdditiveShare
        let evals: [AdditiveShare<F>; DEGREE_VAL_FINAL] = (0..self.inc.len() / 2)
            .into_par_iter()
            .map(|j| {
                let inc_evals =
                    self.inc
                        .sumcheck_evals(j, DEGREE_VAL_FINAL, BindingOrder::HighToLow);
                let wa_evals: [F; DEGREE_VAL_FINAL] = self
                    .wa
                    .sumcheck_evals_array::<DEGREE_VAL_FINAL>(j, BindingOrder::HighToLow);

                [
                    rep3_arith::mul_public(inc_evals[0], wa_evals[0]).into_additive(),
                    rep3_arith::mul_public(inc_evals[1], wa_evals[1]).into_additive(),
                ]
            })
            .reduce(
                || [AdditiveShare::<F>::zero(); DEGREE_VAL_FINAL],
                |r, n| [r[0] + n[0], r[1] + n[1]],
            );

        let mut result = vec![AdditiveShare::<F>::zero(); max_degree];
        for (i, &e) in evals.iter().enumerate() {
            if i < max_degree {
                result[i] = e;
            }
        }
        result
    }

    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
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
        &self,
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

// ---------------------------------------------------------------------------
// ValFinalSumcheck Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3ValFinalSumcheck<F: JoltField> {
    T: usize,
    val_init_eval: F,
    val_final_claim: F,
}

impl<F: JoltField> Rep3ValFinalSumcheck<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let val_init_eval = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValInit,
                SumcheckId::RamOutputCheck,
            )
            .1;
        let val_final_claim = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .1;

        Self {
            T: sm.trace_length,
            val_init_eval,
            val_final_claim,
        }
    }

    pub fn input_claim(&self) -> F {
        self.val_final_claim - self.val_init_eval
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3ValFinalSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE_VAL_FINAL
    }
    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }
    fn input_claim_public(&self) -> F {
        self.val_final_claim - self.val_init_eval
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        _r: &[F::Challenge],
    ) -> F {
        let inc_claim = accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::RamInc,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;
        let wa_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValFinalEvaluation,
            )
            .1;
        inc_claim * wa_claim
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
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        let r_address = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamValFinal,
                SumcheckId::RamOutputCheck,
            )
            .0;
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.append_dense(
            transcript,
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamValFinalEvaluation,
            r_cycle_prime.r,
            vec![claims[0]],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
            wa_opening_point,
            claims[1],
        );
    }
}
