use std::cell::RefCell;
use std::rc::Rc;

use common::constants::XLEN;

use strum::{EnumCount, IntoEnumIterator};

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide};
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::{
    OpeningId, OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::subprotocols::blindfold::InputClaimConstraint;
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
use crate::zkvm::lookup_table::LookupTables;
use crate::zkvm::witness::VirtualPolynomial;

const LOG_K: usize = D * LOG_K_CHUNK;
const DEGREE: usize = 3;

pub struct ReadRafSumcheck<F: JoltField> {
    gamma: F,
    gamma_squared: F,
    rv_claim: F,
    raf_claim: F,
    log_T: usize,
}

impl<F: JoltField> ReadRafSumcheck<F> {
    pub fn new_verifier<T: Transcript>(
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

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for ReadRafSumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        LOG_K + self.log_T
    }

    fn input_claim(&self) -> F {
        self.rv_claim + self.gamma * self.raf_claim
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.unwrap();
        let acc = accumulator.borrow();

        let (r_address_prime, r_cycle_prime) = r.split_at(LOG_K);

        let left_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Left).evaluate(r_address_prime);
        let right_operand_eval =
            OperandPolynomial::<F>::new(LOG_K, OperandSide::Right).evaluate(r_address_prime);
        let identity_poly_eval = IdentityPolynomial::<F>::new(LOG_K).evaluate(r_address_prime);

        let val_evals: Vec<_> = LookupTables::<XLEN>::iter()
            .map(|table| table.evaluate_mle::<F, F::Challenge>(r_address_prime))
            .collect();

        let r_cycle = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r;
        let eq_eval_cycle = EqPolynomial::<F>::mle(&r_cycle, r_cycle_prime);

        let ra_claim = acc
            .get_virtual_polynomial_opening(
                VirtualPolynomial::InstructionRa,
                SumcheckId::InstructionReadRaf,
            )
            .1;

        let table_flag_claims: Vec<F> = (0..LookupTables::<XLEN>::COUNT)
            .map(|i| {
                acc.get_virtual_polynomial_opening(
                    VirtualPolynomial::LookupTableFlag(i),
                    SumcheckId::InstructionReadRaf,
                )
                .1
            })
            .collect();

        let raf_flag_claim = acc
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

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        r_sumcheck: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let mut acc = accumulator.borrow_mut();
        let (_r_address, r_cycle) = r_sumcheck.clone().split_at(LOG_K);

        let num_tables = LookupTables::<XLEN>::COUNT;

        for i in 0..num_tables {
            acc.append_virtual(
                transcript,
                VirtualPolynomial::LookupTableFlag(i),
                SumcheckId::InstructionReadRaf,
                r_cycle.clone(),
            );
        }

        acc.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
            r_sumcheck,
        );

        acc.append_virtual(
            transcript,
            VirtualPolynomial::InstructionRafFlag,
            SumcheckId::InstructionReadRaf,
            r_cycle,
        );
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::weighted_openings(&[
            OpeningId::Virtual(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter),
            OpeningId::Virtual(VirtualPolynomial::LeftLookupOperand, SumcheckId::SpartanOuter),
            OpeningId::Virtual(VirtualPolynomial::RightLookupOperand, SumcheckId::SpartanOuter),
        ])
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
    ) -> Vec<F> {
        vec![self.gamma, self.gamma_squared]
    }
}
