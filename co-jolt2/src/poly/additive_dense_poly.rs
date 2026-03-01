use crate::field::JoltField;
use mpc_core::protocols::additive::AdditiveShare;

/// A dense multilinear polynomial stored as additive shares.
///
/// Used for suffix polys and operand Q polys where all downstream operations
/// are linear (bind with public challenge, read coefficients, scalar mult).
/// This avoids the `mul_vec` reshare cost — the FWHT pointwise product
/// `Rep3 * Rep3 → Additive` is computed locally with zero communication.
#[derive(Clone, Debug)]
pub(crate) struct AdditiveDensePoly<F: JoltField> {
    coeffs: Vec<AdditiveShare<F>>,
    /// After first bind, subsequent binds work on bound_coeffs.
    bound: Vec<AdditiveShare<F>>,
    /// Current logical length (halves on each bind).
    current_len: usize,
    is_bound: bool,
}

impl<F: JoltField> AdditiveDensePoly<F> {
    pub(crate) fn new(coeffs: Vec<AdditiveShare<F>>) -> Self {
        let len = coeffs.len();
        Self {
            coeffs,
            bound: Vec::new(), // deferred — allocated on first bind()
            current_len: len,
            is_bound: false,
        }
    }

    pub(crate) fn zeros(len: usize) -> Self {
        Self::new(vec![AdditiveShare::zero(); len])
    }

    pub(crate) fn len(&self) -> usize {
        self.current_len
    }

    pub(crate) fn get_coeff(&self, index: usize) -> AdditiveShare<F> {
        if self.is_bound {
            self.bound[index]
        } else {
            self.coeffs[index]
        }
    }

    /// Bind the high variable with a public challenge (HighToLow order).
    /// Halves the polynomial: new[i] = left[i] + r * (right[i] - left[i]).
    pub(crate) fn bind(&mut self, r: F) {
        let n = self.current_len / 2;
        if self.is_bound {
            for i in 0..n {
                let left = self.bound[i];
                let right = self.bound[i + n];
                self.bound[i] = left + (right - left) * r;
            }
        } else {
            // Allocate bound buffer on first bind (deferred from new())
            if self.bound.len() < n {
                self.bound.resize(n, AdditiveShare::zero());
            }
            for i in 0..n {
                let left = self.coeffs[i];
                let right = self.coeffs[i + n];
                self.bound[i] = left + (right - left) * r;
            }
            self.is_bound = true;
        }
        self.current_len = n;
    }
}
