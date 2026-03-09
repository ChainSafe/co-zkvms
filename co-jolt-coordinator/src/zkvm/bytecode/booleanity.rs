use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstance;
use jolt_core::transcripts::{KeccakTranscript, Transcript};
use jolt_core::zkvm::bytecode::booleanity::BooleanitySumcheck;
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use mpc_core::protocols::rep3::PartyID;

use crate::field::JoltField;
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::subprotocols::sumcheck::{PublicSumcheckInstance, PublicSumcheckInstanceWorker};

impl<F: JoltField, T: Transcript> PublicSumcheckInstance<F, T> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::degree(self)
    }

    fn num_rounds(&self) -> usize {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::num_rounds(self)
    }

    fn input_claim_public(&self) -> F {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::input_claim(self)
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let d = self.d();

        let ra_claims: Vec<F> = (0..d)
            .map(|i| {
                accumulator
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::BytecodeRa(i),
                        SumcheckId::BytecodeBooleanity,
                    )
                    .1
            })
            .collect();

        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;

        // Mirrors vanilla BooleanitySumcheck::expected_output_claim.
        // normalize_opening_point reverses address and cycle parts separately,
        // so r_prime is already [rev(r_address) || rev(r_cycle)] in BIG_ENDIAN.
        // We reconstruct the LowToHigh point used during binding:
        //   r_address was bound LowToHigh → reversed = r_address in big-endian
        //   r_cycle was bound LowToHigh → reversed = r_cycle in big-endian
        let expected_r: Vec<F::Challenge> = self
            .r_address()
            .iter()
            .cloned()
            .rev()
            .chain(r_cycle.iter().cloned().rev())
            .collect();

        EqPolynomial::<F>::mle(r, &expected_r)
            * self
                .gamma_powers()
                .iter()
                .zip(ra_claims.iter())
                .map(|(gamma, &ra)| (ra.square() - ra) * gamma)
                .sum::<F>()
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        <BooleanitySumcheck<F> as SumcheckInstance<F, KeccakTranscript>>::normalize_opening_point(
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
        let log_K_chunk = self.log_K_chunk();

        accumulator.append_sparse(
            transcript,
            (0..self.d()).map(CommittedPolynomial::BytecodeRa).collect(),
            SumcheckId::BytecodeBooleanity,
            &opening_point.r[..log_K_chunk],
            &opening_point.r[log_K_chunk..],
            claims,
        );
    }
}
