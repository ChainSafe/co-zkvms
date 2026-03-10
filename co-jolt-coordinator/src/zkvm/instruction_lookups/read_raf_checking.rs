use crate::field::JoltField;
use crate::poly::opening_proof::Rep3OpeningAccumulator;
use crate::zkvm::dag::stage::Rep3SumcheckInstance;
use jolt_common::constants::XLEN;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide};
use jolt_core::poly::multilinear_polynomial::PolynomialEvaluation;
use jolt_core::poly::opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN};
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::lookup_table::LookupTables;
use jolt_core::zkvm::witness::VirtualPolynomial;
use strum::{EnumCount, IntoEnumIterator};

const LOG_K: usize = XLEN * 2; // 128
const DEGREE: usize = 3;

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3ReadRafSumcheck<F: JoltField> {
    gamma: F,
    gamma_squared: F,
    rv_claim: F,
    raf_claim: F,
    log_T: usize,
}

impl<F: JoltField> Rep3ReadRafSumcheck<F> {
    /// Construct the coordinator-side ReadRaf sumcheck instance.
    ///
    /// Draws gamma from transcript, then computes
    /// `raf_claim = left_operand_claim + gamma * right_operand_claim`.
    pub fn new<T: Transcript>(
        transcript: &mut T,
        rv_claim: F,
        left_operand_claim: F,
        right_operand_claim: F,
        log_T: usize,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar();
        let raf_claim = left_operand_claim + gamma * right_operand_claim;
        Self {
            gamma,
            gamma_squared: gamma.square(),
            rv_claim,
            raf_claim,
            log_T,
        }
    }

    pub fn gamma(&self) -> F {
        self.gamma
    }

    pub fn rv_claim(&self) -> F {
        self.rv_claim
    }

    pub fn raf_claim(&self) -> F {
        self.raf_claim
    }
}

impl<F: JoltField, T: Transcript> Rep3SumcheckInstance<F, T> for Rep3ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim_public(&self) -> F {
        self.rv_claim + self.gamma * self.raf_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: &Rep3OpeningAccumulator<F>,
        r: &[F::Challenge],
    ) -> F {
        let (r_address_prime, r_cycle_prime) = r.split_at(LOG_K);

        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(r_address_prime);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(r_address_prime);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(r_address_prime);

        let val_evals: Vec<_> = LookupTables::<XLEN>::iter()
            .map(|table| table.evaluate_mle::<F, F::Challenge>(r_address_prime))
            .collect();

        let r_cycle = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        let eq_eval_cycle = EqPolynomial::<F>::mle(&r_cycle, r_cycle_prime);

        let ra_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let table_flag_claims: Vec<F> = (0..LookupTables::<XLEN>::COUNT)
            .map(|i| {
                accumulator
                    .get_virtual_polynomial_opening(
                        VirtualPolynomial::LookupTableFlag(i),
                        SumcheckId::InstructionReadRaf,
                    )
                    .1
            })
            .collect();

        let raf_flag_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRafFlag,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let rv_val_claim: F = val_evals
            .into_iter()
            .zip(table_flag_claims)
            .map(|(val, flag)| val * flag)
            .sum();

        let val_eval = rv_val_claim
            + (F::one() - raf_flag_claim)
                * (self.gamma * left_operand_eval + self.gamma_squared * right_operand_eval)
            + raf_flag_claim * self.gamma_squared * identity_poly_eval;

        eq_eval_cycle * ra_claim * val_eval
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
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
        claims: Vec<F>,
    ) {
        let (_r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);

        let num_tables = LookupTables::<XLEN>::COUNT;
        // Claims order: table_flags..., ra, raf_flag
        assert_eq!(claims.len(), num_tables + 2);

        for i in 0..num_tables {
            accumulator.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
                claims[i],
            );
        }

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            r_sumcheck,
            claims[num_tables],
        );

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle,
            claims[num_tables + 1],
        );
    }
}
