use jolt_core::field::JoltField;
pub use jolt_core::poly::unipoly::*;
use mpc_core::protocols::additive::AdditiveShare;

pub fn unipoly_from_additive_evals<F: JoltField>(
    evals: &[AdditiveShare<F>],
) -> UniPoly<AdditiveShare<F>> {
    UniPoly {
        coeffs: AdditiveShare::from_fe_vec(
            UniPoly::from_evals(AdditiveShare::as_fe_vec_ref(evals)).coeffs,
        ),
    }
}
