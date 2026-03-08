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
use mpc_core::protocols::additive::{self, AdditiveShare};
use mpc_core::protocols::rep3::{arithmetic as rep3_arith, PartyID, Rep3PrimeFieldShare};
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use rayon::prelude::*;
use tracer::JoltDevice;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::mixed_polynomial::MixedPolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::poly::Polynomial;
use crate::utils::types::Rep3Value;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use crate::zkvm::dag::state_manager::{StateManager, StateManagerWorker};
use crate::zkvm::instruction_lookups::booleanity::extend_degree_3_evals;

const DEGREE_OUTPUT: usize = 3;
const DEGREE_VAL_FINAL: usize = 2;

// ===========================================================================
// OutputSumcheck
// ===========================================================================

// ---------------------------------------------------------------------------
// OutputSumcheck Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3OutputSumcheck<F: JoltField> {
    K: usize,
    r_address: Vec<F::Challenge>,
    program_io: JoltDevice,
}

impl<F: JoltField> Rep3OutputSumcheck<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let K = sm.ram_K;
        let r_address = sm.transcript.challenge_vector_optimized::<F>(K.log_2());
        Self {
            K,
            r_address,
            program_io: sm.program_io.clone(),
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
            remap_address(RAM_START_ADDRESS, &self.program_io.memory_layout).unwrap() as u128,
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
        // claims: [val_final, val_init]
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValFinal,
            SumcheckId::RamOutputCheck,
            opening_point.clone(),
            claims[0],
        );
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamValInit,
            SumcheckId::RamOutputCheck,
            opening_point,
            claims[1],
        );
    }
}

// ===========================================================================
// ValFinalSumcheck
// ===========================================================================

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
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
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
