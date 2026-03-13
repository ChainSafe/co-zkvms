use jolt_core::field::JoltField;
use mpc_core::protocols::additive::AdditiveShare;

/// Lagrange interpolation through 4 points (0, y0), (1, y1), (2, y2), (3, y3) at x.
///
/// Denominators are constants: -6, 2, -2, 6. Their inverses are precomputed.
pub fn lagrange_interp_4<F: JoltField>(
    y0: AdditiveShare<F>,
    y1: AdditiveShare<F>,
    y2: AdditiveShare<F>,
    y3: AdditiveShare<F>,
    x: F,
) -> AdditiveShare<F> {
    let inv6 = F::from(6u64).inverse().unwrap();
    let inv2 = F::TWO_INV;
    let inv_neg6 = -inv6;
    let inv_neg2 = -inv2;

    let xm1 = x - F::one();
    let xm2 = x - F::from(2u64);
    let xm3 = x - F::from(3u64);

    let l0 = xm1 * xm2 * xm3 * inv_neg6;
    let l1 = x * xm2 * xm3 * inv2;
    let l2 = x * xm1 * xm3 * inv_neg2;
    let l3 = x * xm1 * xm2 * inv6;

    y0 * l0 + y1 * l1 + y2 * l2 + y3 * l3
}

pub mod tracing;
