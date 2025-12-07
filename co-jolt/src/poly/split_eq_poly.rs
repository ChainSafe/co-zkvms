#[cfg(test)]
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::split_eq_poly::SplitEqPolynomial;

use crate::field::JoltField;

/// A SplitEqPolynomial chunk assigned to a single worker, with:
/// - Dao–Thaler factorization Eq(i_A, i_C, i_B) = E2(i_A, i_B) * E1(i_C),
/// - a *contiguous* 1D slice of the global Eq table [global_start, global_end),
/// - plus metadata to map between:
///     - global EQ indices,
///     - (row, col) indices in the factored table,
///     - local slice indices used to align with DenseInterleavedPolynomial.
pub struct DistributedSplitEqPolynomial<F> {
    /// Number of currently unbound variables *seen by this worker* (A|C),
    /// i.e. total Eq variables minus already-bound ones and minus chunk bits B.
    pub num_vars: usize,

    // -------- factored Dao–Thaler structure over A|C|B --------
    //
    // Semantics match SplitEqPolynomial:
    //
    //   Eq_rect(i_A, i_C, i_B) = E2(i_A, i_B) * E1(i_C)
    //
    // where:
    //   - i_C indexes the "inner" variables C (columns, bound first),
    //   - (i_A, i_B) indexes the "outer" variables A and chunk bits B (rows).
    pub E1: Vec<F>,
    /// Current number of columns in the Dao–Thaler factorization: 2^{|C|} or 1.
    pub E1_len: usize,

    pub E2: Vec<F>,
    /// Current number of rows in the Dao–Thaler factorization: 2^{|A|+|B|_active}.
    pub E2_len: usize,

    /// Length of this worker’s Eq slice in *points*:
    ///   len = global_end - global_start
    ///
    /// This is the number of Eq points this worker logically owns, even if after binding
    /// the attached polynomial P only covers a prefix of them.
    pub len: usize,

    /// Global row index of E2[0]. I.e. E2[row_offset] corresponds to the global row
    /// with index row_start + row_offset in the full Eq table.
    pub row_start: usize,

    /// First global Eq index assigned to this worker (flattened row-major index into the
    /// *full* Eq table before Dao–Thaler factorization and chunking).
    pub global_start: usize,

    /// One-past last global Eq index assigned to this worker.
    pub global_end: usize,
}

impl<F: JoltField> DistributedSplitEqPolynomial<F> {
    #[tracing::instrument(skip_all, name = "DistributedSplitEqPolynomial::new", level = "trace")]
    pub fn new(w: &[F], log_workers: usize, worker: usize, eq_pairs: usize) -> Self {
        let base = SplitEqPolynomial::new(w);

        let num_workers = 1usize << log_workers;
        assert!(worker < num_workers);

        // E1_len = number of columns = 2^{|C|}.
        // E2_len = number of rows    = 2^{|A|+|B|}.
        let E1_len_global = base.E1_len;
        let E2_len_global = base.E2_len;
        let total_points = E1_len_global * E2_len_global;

        // The coordinator assigns each worker a *contiguous* 1D slice of the flattened Eq
        // table using [global_start, global_end), measured in Eq points (not rows).
        let global_start = worker * eq_pairs;
        assert!(
            global_start < total_points,
            "worker {} starts past end of eq table (start={}, total={})",
            worker,
            global_start,
            total_points
        );

        // All non-last workers get exactly `eq_pairs` points, last worker runs to the end.
        let global_end = if worker + 1 == num_workers {
            total_points
        } else {
            core::cmp::min(global_start + eq_pairs, total_points)
        };
        assert!(global_end > global_start);

        // Compute the minimal *row interval* [row_start, row_end) in the global Eq
        // table that covers this 1D slice [global_start, global_end). Rows are
        // indexed in row-major order with row width E1_len_global.
        //
        //   row_start = floor(global_start / E1_len)
        //   row_end   = ceil(global_end  / E1_len)
        //
        let row_start = global_start / E1_len_global;
        let mut row_end = (global_end + E1_len_global - 1) / E1_len_global; // ceil
        if row_end > E2_len_global {
            row_end = E2_len_global;
        }
        assert!(row_start < row_end);
        assert!(row_end <= E2_len_global);

        // Restrict E2 to the rows actually needed for this worker.
        let e2_start = row_start;
        let e2_end = row_end;
        let e2_len = e2_end - e2_start;
        let E2 = base.E2[e2_start..e2_end].to_vec();

        // Worker’s logical Eq slice length in points.
        let len = global_end - global_start;

        Self {
            num_vars: w.len() - log_workers,
            E1: base.E1,
            E1_len: base.E1_len,
            E2,
            E2_len: e2_len,
            len,
            row_start,
            global_start,
            global_end,
        }
    }

    /// Current number of unbound variables (A|C) in this worker’s view.
    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    pub fn len(&self) -> usize {
        // Number of Eq points in this worker's contiguous slice
        // [global_start, global_end), after any bindings.
        self.len
    }

    /// Bind one sumcheck variable (same order/convention as `SplitEqPolynomial::bind`).
    ///
    /// Semantics:
    /// - If C is non-empty (E1_len > 1), we bind a C-variable:
    ///     - E1_len halves, E1 entries are linearly combined by r,
    ///     - if E1_len becomes 1, we collapse into linear-time mode by scaling E2.
    /// - If C is empty (E1_len == 1), we bind an A|B-variable:
    ///     - E2_len halves, each new row is a combination of two old rows,
    ///     - row_start halves because rows are merged pairwise.
    ///
    /// In all cases, the *global* Eq table halves in length and each new global index
    /// corresponds to floor(old_index / 2), so we also remap [global_start, global_end)
    /// and `worker_len` accordingly.
    pub fn bind(&mut self, r: F) {
        // ---------------- bind coefficients (as in SplitEqPolynomial) ----------------
        if self.E1_len == 1 {
            // E1 is fully bound, so we are binding a variable that affects the outer
            // dimension (A|B). This corresponds to merging pairs of rows in E2.
            let n = self.E2_len / 2;
            for i in 0..n {
                let a = self.E2[2 * i];
                let b = self.E2[2 * i + 1];
                self.E2[i] = a + r * (b - a);
            }
            self.E2_len = n;

            // After merging rows pairwise, the global row index of E2[0] halves as well.
            self.row_start /= 2;
        } else {
            // E1 still has >1 columns, so we bind an inner C-variable (Dao–Thaler column
            // dimension). This halves E1_len and linearly folds pairs of column entries.
            let n = self.E1_len / 2;
            for i in 0..n {
                let a = self.E1[2 * i];
                let b = self.E1[2 * i + 1];
                self.E1[i] = a + r * (b - a);
            }
            self.E1_len = n;

            // Once E1 collapses to a single column, Dao–Thaler reduces to the usual
            // linear-time sumcheck, and E2 simply stores the full Eq evaluations for
            // the remaining outer variables. We fold E1 into E2 in-place.
            if self.E1_len == 1 {
                let scale = self.E1[0];
                self.E2[..self.E2_len]
                    .iter_mut()
                    .for_each(|eval| *eval *= scale);
            }
        }

        // One Eq variable is now bound from this worker’s perspective.
        self.num_vars = self.num_vars.saturating_sub(1);

        // ---------------- remap global index interval ----------------
        //
        // Global Eq indices are treated as a flat array in row-major order.
        // Binding any variable halves the total number of Eq points; each new global
        // index corresponds to floor(old_index / 2). Therefore the worker’s slice
        // [global_start, global_end) maps to:
        //
        //   global_start' = floor(global_start / 2)
        //   global_end'   = floor((global_end - 1) / 2) + 1
        //
        // The (x + 1) >> 1 idiom implements exactly this for unsigned integers.
        self.global_start = self.global_start >> 1;
        self.global_end = (self.global_end + 1) >> 1;

        // Length of this worker’s Eq slice in points also halves, rounded up.
        self.len = (self.len + 1) >> 1;
    }

    #[cfg(test)]
    pub fn merge(&self) -> DensePolynomial<F> {
        let cols = self.E1_len;
        let mut merged = Vec::new();

        // For each row in this worker's rectangle
        for (row_offset, &e2) in self.E2[..self.E2_len].iter().enumerate() {
            let r = self.row_start + row_offset; // global row index
            let row_first = r * cols;
            let row_last = row_first + cols;

            // Intersection of this row with [global_start, global_end)
            let from = self.global_start.max(row_first) - row_first; // col_from
            let to = self.global_end.min(row_last) - row_first; // col_to

            if from >= to {
                continue; // this row contributes no points for this worker
            }

            // Push E2[r] * E1[j] for j in [from, to)
            if cols == 1 {
                // degenerate (no Dao–Thaler) case
                for _j in from..to {
                    merged.push(e2);
                }
            } else {
                for j in from..to {
                    merged.push(e2 * self.E1[j]);
                }
            }
        }

        DensePolynomial::new_padded(merged)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::field::JoltField;
    use itertools::Itertools;
    use jolt_core::poly::{eq_poly::EqPolynomial, split_eq_poly::SplitEqPolynomial};
    use snarks_core::math::Math;
    use std::env;

    #[test]
    fn test_merge2() {
        type F = ark_bn254::Fr;
        let W: usize = env::var("NUM_WORKERS")
            .unwrap_or_else(|_| "2".to_string())
            .parse()
            .unwrap();
        let W_log2 = W.log_2();
        let R: usize = env::var("R")
            .unwrap_or_else(|_| "4".to_string())
            .parse()
            .unwrap();

        let EQ_PAIRS: usize = env::var("EQ_PAIRS")
            .unwrap_or_else(|_| "4".to_string())
            .parse()
            .unwrap();
        let r = (1..R + 1).map(|i| F::from(i as u64 * 11)).collect_vec();
        let base = SplitEqPolynomial::new(&r);

        println!("base: E2: {:?}", base.E2);
        println!("----------------");
        for w in 0..W {
            let eq_chunk = DistributedSplitEqPolynomial::new(&r, W_log2, w, EQ_PAIRS);
            println!("eq_chunks[{}]: E2: {:?}", w, eq_chunk.E2);
            println!("--------");
            println!("eq_chunks[{}] merged: {:?}", w, eq_chunk.merge().Z);
            println!("--------");
            let hack = split_eq_chunk_custom_hack(&r, W_log2, w, EQ_PAIRS);
            println!("hack merged: {:?}", hack.merge().Z);
            println!("----------------");
            assert_eq!(hack.merge().Z, eq_chunk.merge().Z)
        }

        // let chunk0 = chunk2.merge();
        // let hack = hack[1].merge();

        // assert_eq!(chunk0.Z, hack.Z);
    }

    pub fn split_eq_chunk_custom_hack<F: JoltField>(
        w: &[F],
        log_chunks: usize,
        k: usize,
        eq_pairs: usize,
    ) -> SplitEqPolynomial<F> {
        let num_vars = w.len() - log_chunks;
        let rows = 1 << w.len();
        let offset = eq_pairs * k;
        let cutoff = if k < (1 << log_chunks) - 1 {
            eq_pairs * (k + 1)
        } else {
            rows
        };
        // Hack put entire chunk in E2
        let E2 = EqPolynomial::evals(w)[offset..cutoff].to_vec();

        SplitEqPolynomial {
            num_vars,
            E1: vec![F::ZERO],
            E1_len: 1,
            E2_len: E2.len(),
            E2,
        }
    }
}
