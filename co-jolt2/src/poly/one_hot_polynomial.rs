use std::sync::Arc;

use allocative::Allocative;
use mpc_core::protocols::{rep3::Rep3PrimeFieldShare, rep3_ring::Rep3RingShare};

use crate::field::JoltField;

/// Represents a one-hot multilinear polynomial (ra/wa) used
/// in Twist/Shout. Perhaps somewhat unintuitively, the implementation
/// in this file is currently only used to compute the Dory
/// commitment and in the opening proof reduction sumcheck.
#[derive(Clone, Debug)] // Allocative
pub struct Rep3OneHotPolynomial<F: JoltField> {
    /// The size of the "address" space for this polynomial.
    pub K: usize,
    /// The indices of the nonzero coefficients for each j \in {0, 1}^T.
    /// In other words, the raf/waf corresponding to this
    /// ra/wa polynomial.
    /// If empty, this polynomial is 0 for all j.
    pub nonzero_indices: Arc<Vec<Option<Rep3RingShare<u8>>>>,
    /// The number of variables that have been bound over the
    /// course of sumcheck so far.
    num_variables_bound: usize,
    /// The array described in Section 6.3 of the Twist/Shout paper.
    G: Vec<F>,
    // /// The array described in Section 6.3 of the Twist/Shout paper.
    // H: Arc<RwLock<RaPolynomial<u8, F>>>,
}

impl<F: JoltField> Default for Rep3OneHotPolynomial<F> {
    fn default() -> Self {
        Self {
            K: 1,
            nonzero_indices: Arc::new(vec![]),
            num_variables_bound: 0,
            G: vec![],
            // H: Arc::new(RwLock::new(RaPolynomial::None)),
        }
    }
}

impl<F: JoltField> Rep3OneHotPolynomial<F> {
    /// The number of rows in the coefficient matrix used to
    /// commit to this polynomial using Dory
    // pub fn num_rows(&self) -> usize {
    //     let T = self.nonzero_indices.len() as u128;
    //     let row_length = DoryGlobals::get_num_columns() as u128;
    //     (T * self.K as u128 / row_length) as usize
    // }

    pub fn get_num_vars(&self) -> usize {
        self.K.log_2() + self.nonzero_indices.len().log_2()
    }

    // #[cfg(test)]
    // fn to_dense_poly(&self) -> DensePolynomial<F> {
    //     let T = DoryGlobals::get_T();
    //     let mut dense_coeffs: Vec<F> = vec![F::zero(); self.K * T];
    //     for (t, k) in self.nonzero_indices.iter().enumerate() {
    //         if let Some(k) = k {
    //             dense_coeffs[*k as usize * T + t] = F::one();
    //         }
    //     }
    //     DensePolynomial::new(dense_coeffs)
    // }

    // pub fn evaluate<C>(&self, r: &[C]) -> F
    // where
    //     C: Copy + Send + Sync + Into<F>,
    //     F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    // {
    //     assert_eq!(r.len(), self.get_num_vars());
    //     let (r_left, r_right) = r.split_at(self.num_rows().log_2());
    //     let eq_left = EqPolynomial::<F>::evals(r_left);
    //     let eq_right = EqPolynomial::<F>::evals(r_right);
    //     let mut left_product = unsafe_allocate_zero_vec(eq_right.len());
    //     self.vector_matrix_product(&eq_left, F::one(), &mut left_product);
    //     left_product
    //         .into_par_iter()
    //         .zip_eq(eq_right.par_iter())
    //         .map(|(l, r)| l * r)
    //         .sum()
    // }

    pub fn from_indices(nonzero_indices: Vec<Option<Rep3RingShare<u8>>>, K: usize) -> Self {
        // debug_assert_eq!(DoryGlobals::get_T(), nonzero_indices.len());
        assert!(K <= 1 << 8, "K must be <= 256 for index to fit into u8");

        Self {
            K,
            nonzero_indices: Arc::new(nonzero_indices),
            ..Default::default()
        }
    }
}
