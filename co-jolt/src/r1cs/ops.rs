use jolt_core::field::JoltField;
use jolt_core::r1cs::builder::{Constraint, OffsetEqConstraint};
use jolt_core::r1cs::ops::*;
use mpc_core::protocols::rep3::{PartyID, Rep3PrimeFieldShare};

use crate::poly::Rep3MultilinearPolynomial;
use crate::utils::element::{SharedOrPublic, SharedOrPublicIter};

pub use jolt_core::r1cs::ops::*;

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

/// Conversions and arithmetic for concrete ConstraintInput
#[macro_export]
macro_rules! impl_r1cs_input_lc_conversions {
    ($ConcreteInput:ty, $C:expr) => {
        impl<F: JoltField> Into<$crate::r1cs::ops::Variable> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::Variable {
                $crate::r1cs::ops::Variable::Input(self.to_index::<$C>())
            }
        }

        impl<F: JoltField> Into<$crate::r1cs::ops::Term> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::Term {
                $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(self.to_index::<$C>()),
                    1,
                )
            }
        }

        impl<F: JoltField> Into<$crate::r1cs::ops::LC> for $ConcreteInput {
            fn into(self) -> $crate::r1cs::ops::LC {
                $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(self.to_index::<$C>()),
                    1,
                )
                .into()
            }
        }

        impl<F: JoltField> $ConcreteInput {
            fn lc_from_vec(inputs: Vec<$ConcreteInput>) -> $crate::r1cs::ops::LC {
                let terms: Vec<$crate::r1cs::ops::Term> =
                    inputs.into_iter().map(Into::into).collect();
                $crate::r1cs::ops::LC::new(terms)
            }
        }

        impl<F: JoltField, T: Into<$crate::r1cs::ops::LC>> std::ops::Add<T> for $ConcreteInput {
            type Output = $crate::r1cs::ops::LC;

            fn add(self, rhs: T) -> Self::Output {
                let lhs_lc: $crate::r1cs::ops::LC = self.into();
                let rhs_lc: $crate::r1cs::ops::LC = rhs.into();
                lhs_lc + rhs_lc
            }
        }

        impl<F: JoltField, T: Into<$crate::r1cs::ops::LC>> std::ops::Sub<T> for $ConcreteInput {
            type Output = $crate::r1cs::ops::LC;

            fn sub(self, rhs: T) -> Self::Output {
                let lhs_lc: $crate::r1cs::ops::LC = self.into();
                let rhs_lc: $crate::r1cs::ops::LC = rhs.into();
                lhs_lc - rhs_lc
            }
        }

        impl<F: JoltField> std::ops::Mul<i64> for $ConcreteInput {
            type Output = $crate::r1cs::ops::Term;

            fn mul(self, rhs: i64) -> Self::Output {
                $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(self.to_index::<$C>()),
                    rhs,
                )
            }
        }

        impl<F: JoltField> std::ops::Mul<$ConcreteInput> for i64 {
            type Output = $crate::r1cs::ops::Term;

            fn mul(self, rhs: $ConcreteInput) -> Self::Output {
                $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(rhs.to_index::<$C>()),
                    self,
                )
            }
        }
        impl<F: JoltField> std::ops::Add<$ConcreteInput> for i64 {
            type Output = $crate::r1cs::ops::LC;

            fn add(self, rhs: $ConcreteInput) -> Self::Output {
                let term1 = $crate::r1cs::ops::Term(
                    $crate::r1cs::ops::Variable::Input(rhs.to_index::<$C>()),
                    1,
                );
                let term2 = $crate::r1cs::ops::Term($crate::r1cs::ops::Variable::Constant, self);
                $crate::r1cs::ops::LC::new(vec![term1, term2])
            }
        }
    };
}
