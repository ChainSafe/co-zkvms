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
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <ReadRafSumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        // Mirrors vanilla ReadRafSumcheck::expected_output_claim.
        let log_K = self.log_K();
        let d = self.d();

        let (r_address_prime, r_cycle_prime_raw) = r.split_at(log_K);
        // r_cycle was bound LowToHigh, so reverse for EqPolynomial::mle
        let r_cycle_prime: Vec<F::Challenge> = r_cycle_prime_raw.iter().rev().copied().collect();

        // Replicate get_r_cycle_verif from vanilla using Rep3OpeningAccumulator.
        let reg_count_bits = (REGISTER_COUNT as usize).ilog2() as usize;

        let (r_cycle_1, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Imm, SumcheckId::SpartanOuter);
        let (r_rs1ra, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::Rs1Ra,
            SumcheckId::RegistersReadWriteChecking,
        );
        let (_, r_cycle_2) = r_rs1ra.split_at(reg_count_bits);
        let (r_rdwa, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
        );
        let (_, r_cycle_3) = r_rdwa.split_at(reg_count_bits);

        let r_cycles = [r_cycle_1.r, r_cycle_2.r, r_cycle_3.r];

        // int_poly = IdentityPolynomial evaluated at r_address_prime.
        // We delegate to the vanilla type which holds it.
        let int_poly_eval = self.int_poly_evaluate(r_address_prime);

        // gamma values: [gamma^0, gamma^1, gamma^2]
        // gamma_sqr and gamma_cub stored on self.
        let gamma_int = [
            int_poly_eval * self.gamma_cub(), // RAF for Stage1
            F::zero(),                        // No RAF for Stage2
            int_poly_eval * self.gamma_sqr(), // RAF for Stage3
        ];

        let ra_claims: Vec<F> = (0..d)
            .map(|i| {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeRa(i),
                        SumcheckId::BytecodeReadRaf,
                    )
                    .1
            })
            .collect();

        let val_evals = self.val_polys_evaluate(r_address_prime);
        let gammas = self.gamma_stages();
        let val: F = val_evals
            .iter()
            .zip(r_cycles.iter())
            .zip(gammas.iter())
            .zip(gamma_int.iter())
            .map(|(((val_eval, r_cycle), gamma), int_poly)| {
                (*val_eval + *int_poly) * EqPolynomial::<F>::mle(r_cycle, &r_cycle_prime) * *gamma
            })
            .sum();

        ra_claims.iter().fold(val, |running, &ra| running * ra)
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

    fn cache_openings(
        &self,
        accumulator: &mut Rep3OpeningAccumulator<F>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        let log_K = self.log_K();
        let log_K_chunk = self.log_K_chunk();
        let d = self.d();

        let (r_address, r_cycle) = opening_point.split_at(log_K);

        for i in 0..d {
            let r_addr_chunk = &r_address.r[log_K_chunk * i..log_K_chunk * (i + 1)];
            accumulator.append_sparse(
                transcript,
                vec![CommittedPolynomial::BytecodeRa(i)],
                SumcheckId::BytecodeReadRaf,
                r_addr_chunk,
                &r_cycle.r,
                vec![claims[i]],
            );
        }
    }
}
