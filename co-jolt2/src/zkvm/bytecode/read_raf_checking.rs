use jolt2_common::constants::REGISTER_COUNT;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::bytecode::read_raf_checking::ReadRafSumcheck;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::rep3::PartyID;

use crate::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::subprotocols::sumcheck::PublicSumcheckInstanceWorker;

impl<F: JoltField> PublicSumcheckInstanceWorker<F> for ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn compute_prover_message_public(
        &mut self,
        round: usize,
        previous_claim: F,
        max_degree: usize,
    ) -> Vec<F> {
        let base =
            <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::compute_prover_message(
                self,
                round,
                previous_claim,
            );

        // ReadRafSumcheck has variable degree per round:
        // log_K phase returns 2 evals, log_T phase returns d+1 evals.
        let round_degree = base.len();
        debug_assert!(round_degree >= 1);
        debug_assert!(max_degree >= round_degree);

        if max_degree == round_degree {
            return base[..round_degree].to_vec();
        }

        // Build full evals at {0, 1, ..., round_degree}.
        // base = [y0, y2] (degree=2, log_K phase) or [y0, y2, ..., y_{d+1}] (cycle phase).
        let y0 = base[0];
        let y1 = previous_claim - y0;

        let mut full_evals = Vec::with_capacity(round_degree + 1);
        full_evals.push(y0);
        full_evals.push(y1);
        full_evals.extend_from_slice(&base[1..]); // y2..y_round_degree

        let coeffs = UniPoly::<F>::from_evals(&full_evals).as_vec();

        let mut msg = vec![F::zero(); max_degree];
        msg[0] = y0;
        if round_degree >= 2 {
            msg[1] = full_evals[2]; // y2
        }
        for k in 3..=max_degree {
            // Evaluate polynomial at the integer k using F arithmetic
            // (NOT F::Challenge which has different representation)
            let x = F::from_u64(k as u64);
            let eval = coeffs.iter().rev().fold(F::zero(), |acc, c| acc * x + *c);
            msg[k - 1] = eval;
        }
        msg
    }

    fn bind(&mut self, r_j: F::Challenge, round: usize) {
        <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::bind(self, r_j, round)
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
            self,
            opening_point,
        )
    }

    fn cache_openings_public(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        party_id: PartyID,
    ) -> Vec<F> {
        let d = self.d();
        let log_K = self.log_K();
        let log_K_chunk = self.log_K_chunk();

        let (r_address, r_cycle) = opening_point.split_at(log_K);

        let mut claims = Vec::with_capacity(d);
        for i in 0..d {
            let r_addr_chunk = &r_address.r[log_K_chunk * i..log_K_chunk * (i + 1)];
            let claim = if party_id == PartyID::ID0 {
                self.ra_final_claim(i)
            } else {
                F::zero()
            };
            claims.push(claim);

            accumulator.append_sparse_public(
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_addr_chunk,
                &r_cycle.r,
                vec![claim],
            );
        }

        claims
    }
}
