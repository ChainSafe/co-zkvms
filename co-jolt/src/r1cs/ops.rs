use jolt_core::field::JoltField;
use jolt_core::r1cs::builder::{Constraint, OffsetEqConstraint};
use jolt_core::r1cs::ops::*;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};

use crate::poly::Rep3MultilinearPolynomial;
use crate::utils::element::{SharedOrPublic, SharedOrPublicIter};

pub trait LinearCombinationExt<F: JoltField> {
    fn evaluate_row_rep3_mixed(
        &self,
        flattened_polynomials: &[&Rep3MultilinearPolynomial<F>],
        row: usize,
        party_id: PartyID,
    ) -> SharedOrPublic<F>;
}

impl<F: JoltField> LinearCombinationExt<F> for LC {
    fn evaluate_row_rep3_mixed(
        &self,
        flattened_polynomials: &[&Rep3MultilinearPolynomial<F>],
        row: usize,
        party_id: PartyID,
    ) -> SharedOrPublic<F> {
        self.terms()
            .iter()
            .map(|term| match term.0 {
                Variable::Input(var_index) | Variable::Auxiliary(var_index) => {
                    flattened_polynomials[var_index]
                        .get_coeff(row)
                        .mul(&F::from_i64(term.1).into())
                }
                Variable::Constant => F::from_i64(term.1).into(),
            })
            .sum_for(party_id)
    }
}
