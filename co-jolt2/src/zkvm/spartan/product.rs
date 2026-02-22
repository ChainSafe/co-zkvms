use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::BindingOrder;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::split_eq_poly::GruenSplitEqPolynomial;
use jolt_core::transcripts::Transcript;
use jolt_core::utils::math::Math;
use jolt_core::zkvm::witness::CommittedPolynomial;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::dense_mlpoly::Rep3DensePolynomial;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::utils::types::Rep3Value;
use crate::zkvm::dag::stage::{Rep3SumcheckInstance, Rep3SumcheckInstanceWorker};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};
use crate::zkvm::instruction_lookups::booleanity::{extend_degree_3_evals, gruen_evals_deg_3};

const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

pub struct Rep3ProductVirtualizationSumcheckWorker<F: JoltField> {
    party_id: PartyID,
    input_claim: F,
    log_T: usize,
    left_input_poly: Rep3DensePolynomial<F>,
    right_input_poly: Rep3DensePolynomial<F>,
    eq_r_cycle: GruenSplitEqPolynomial<F>,
}

impl<F: JoltField> Rep3ProductVirtualizationSumcheckWorker<F> {
    pub fn new<PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerWorker<'_, F, PCS>,
        input_claim: F,
    ) -> Self {
        let party_id = sm.party_id;
        let cycle_witness = &sm.prover_state.cycle_witness;
        let n = cycle_witness.len();

        let (r_cycle_point, _) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Product,
            SumcheckId::SpartanOuter,
        );
        let log_T = r_cycle_point.r.len();

        let mut left: Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>> =
            Vec::with_capacity(n);
        let mut right: Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>> =
            Vec::with_capacity(n);
        for t in 0..n {
            let (l, r) = cycle_witness.row(t).to_instruction_inputs(party_id);
            left.push(l);
            right.push(r);
        }

        Self {
            party_id,
            input_claim,
            log_T,
            left_input_poly: Rep3DensePolynomial::new(left),
            right_input_poly: Rep3DensePolynomial::new(right),
            eq_r_cycle: GruenSplitEqPolynomial::new(&r_cycle_point.r, BindingOrder::LowToHigh),
        }
    }
}

impl<F: JoltField> Rep3SumcheckInstanceWorker<F> for Rep3ProductVirtualizationSumcheckWorker<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "ProductVirtSumcheck::compute_prover_message_share")]
    fn compute_prover_message_share(
        &mut self,
        _round: usize,
        previous_claim: AdditiveShare<F>,
        max_degree: usize,
    ) -> Vec<AdditiveShare<F>> {
        let eq = &self.eq_r_cycle;

        let quadratic_coeffs: [AdditiveShare<F>; DEGREE - 1] = if eq.E_in_current_len() == 1 {
            (0..eq.len() / 2)
                .into_par_iter()
                .map(|j| {
                    let eq_eval = eq.E_out_current()[j];

                    let left_0 = self.left_input_poly.get_bound_coeff(2 * j);
                    let left_1 = self.left_input_poly.get_bound_coeff(2 * j + 1);
                    let right_0 = self.right_input_poly.get_bound_coeff(2 * j);
                    let right_1 = self.right_input_poly.get_bound_coeff(2 * j + 1);

                    let t0 = (left_0 * right_0) * eq_eval;
                    let t_inf = ((left_1 - left_0) * (right_1 - right_0)) * eq_eval;
                    [t0, t_inf]
                })
                .reduce(
                    || [AdditiveShare::zero(), AdditiveShare::zero()],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            let num_x_in_bits = eq.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..eq.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let E_out_eval = eq.E_out_current()[x_out];

                    let chunk_evals = chunk
                        .par_iter()
                        .map(|j| {
                            let x_in = j & x_bitmask;
                            let E_in_eval = eq.E_in_current()[x_in];

                            let left_0 = self.left_input_poly.get_bound_coeff(2 * j);
                            let left_1 = self.left_input_poly.get_bound_coeff(2 * j + 1);
                            let right_0 = self.right_input_poly.get_bound_coeff(2 * j);
                            let right_1 = self.right_input_poly.get_bound_coeff(2 * j + 1);

                            let t0 = (left_0 * right_0) * E_in_eval;
                            let t_inf = ((left_1 - left_0) * (right_1 - right_0)) * E_in_eval;
                            [t0, t_inf]
                        })
                        .reduce(
                            || [AdditiveShare::zero(), AdditiveShare::zero()],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [chunk_evals[0] * E_out_eval, chunk_evals[1] * E_out_eval]
                })
                .reduce(
                    || [AdditiveShare::zero(), AdditiveShare::zero()],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        let base = gruen_evals_deg_3(
            eq,
            Rep3Value::Additive(quadratic_coeffs[0]),
            Rep3Value::Additive(quadratic_coeffs[1]),
            previous_claim,
            self.party_id,
        );

        extend_degree_3_evals(previous_claim, &base, max_degree)
    }

    #[tracing::instrument(skip_all, name = "ProductVirtSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_cycle.bind(r_j);
        let r: F = r_j.into();
        rayon::join(
            || self.left_input_poly.bind(r, BindingOrder::LowToHigh),
            || self.right_input_poly.bind(r, BindingOrder::LowToHigh),
        );
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_worker(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Vec<mpc_core::protocols::rep3::Rep3PrimeFieldShare<F>> {
        let left_eval = self.left_input_poly.final_sumcheck_claim();
        let right_eval = self.right_input_poly.final_sumcheck_claim();

        accumulator.append_dense(
            vec![
                CommittedPolynomial::LeftInstructionInput,
                CommittedPolynomial::RightInstructionInput,
            ],
            SumcheckId::ProductVirtualization,
            opening_point.r,
            &[left_eval, right_eval],
        );

        vec![left_eval, right_eval]
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3ProductVirtualizationSumcheck<F: JoltField> {
    input_claim: F,
    log_T: usize,
}

impl<F: JoltField> Rep3ProductVirtualizationSumcheck<F> {
    pub fn new<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (r_point, input_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Product,
            SumcheckId::SpartanOuter,
        );
        Self {
            input_claim,
            log_T: r_point.r.len(),
        }
    }

    pub fn input_claim(&self) -> F {
        self.input_claim
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3ProductVirtualizationSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim_public(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let (outer_sumcheck_opening, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Product,
            SumcheckId::SpartanOuter,
        );
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let (r_cycle, _) = outer_sumcheck_r.split_at(self.log_T);

        let (_, left_input_eval) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::LeftInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let (_, right_input_eval) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RightInstructionInput,
            SumcheckId::ProductVirtualization,
        );

        let eq_eval =
            EqPolynomial::mle(&r.iter().rev().copied().collect::<Vec<_>>(), r_cycle);
        eq_eval * left_input_eval * right_input_eval
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        accumulator.append_dense(
            transcript,
            vec![
                CommittedPolynomial::LeftInstructionInput,
                CommittedPolynomial::RightInstructionInput,
            ],
            SumcheckId::ProductVirtualization,
            opening_point.r,
            claims,
        );
    }
}
