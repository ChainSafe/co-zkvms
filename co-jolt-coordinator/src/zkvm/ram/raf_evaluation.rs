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
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::utils::types::Rep3Value;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use crate::zkvm::dag::state_manager::{StateManager, StateManagerWorker};

const DEGREE: usize = 2;

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
        sm: &mut StateManager<'_, F, ProofTranscript, PCS>,
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
