//! Pseudorandom Correlation Function (PCF) for subfield VOLE from Boyle et al. 2022.
//!
//! Produces VOLE correlations `y_0 = y_1 + r * Delta` lazily from compact keys,
//! using `t` independent RDCF instances + an EA-code sparse matrix.
//!
//! Reference: Boyle et al. "Pseudorandom Correlation Functions from Variable-Density
//! LPN, Revisited" (2022), Figure 12.

use super::rdcf::{self, Prg, RdcfKey0, RdcfKey1, Seed};
use mpc_types::field::PrimeField;
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};

// ── Parameters ──────────────────────────────────────────────────────────────

/// PCF parameters defining the EA-LPN code and RDCF tree depth.
#[derive(Debug, Clone)]
pub struct PcfParams {
    /// Number of VOLE outputs (rows of the code matrix).
    pub n: usize,
    /// Noise vector length = t * 2^m (columns of the code matrix).
    pub big_n: usize,
    /// Number of RDCF blocks (noise weight).
    pub t: usize,
    /// Number of non-zero entries per EA-code row.
    pub l: usize,
    /// RDCF tree depth: each block has domain [0, 2^m).
    pub m: u32,
    /// Public seed for the implicit EA-code matrix.
    pub matrix_seed: [u8; 32],
}

impl PcfParams {
    /// Block size = 2^m (number of points per RDCF domain).
    #[inline]
    pub fn block_size(&self) -> usize {
        1usize << self.m
    }

    /// Conservative parameters: t=664, l=7, m=20 (~1M block).
    pub fn conservative(n: usize, matrix_seed: [u8; 32]) -> Self {
        let t = 664;
        let m = 20u32;
        Self {
            n,
            big_n: t * (1 << m),
            t,
            l: 7,
            m,
            matrix_seed,
        }
    }

    /// Aggressive parameters: t=68, l=62, m=30 (~1B block).
    pub fn aggressive(n: usize, matrix_seed: [u8; 32]) -> Self {
        let t = 68;
        let m = 30u32;
        Self {
            n,
            big_n: t * (1 << m),
            t,
            l: 62,
            m,
            matrix_seed,
        }
    }
}

// ── PCF Keys ────────────────────────────────────────────────────────────────

/// PCF key for party 0 (the VOLE sender).
///
/// Holds per-block RDCF keys, noise positions, and payloads.
#[derive(Debug, Clone)]
pub struct PcfKey0<F: PrimeField> {
    /// RDCF keys for each of the `t` blocks.
    pub rdcf_keys: Vec<RdcfKey0<F>>,
    /// Noise positions: `alpha_i in [0, 2^m)` for block i.
    pub alphas: Vec<u32>,
    /// Noise payloads: `beta_i in Fp \ {0}` for block i.
    pub betas: Vec<F>,
}

/// PCF key for party 1 (the VOLE receiver).
///
/// Holds the global correlation `Delta` and per-block RDCF root seeds.
#[derive(Debug, Clone)]
pub struct PcfKey1<F: PrimeField> {
    /// Global VOLE correlation.
    pub delta: F,
    /// RDCF keys for each of the `t` blocks.
    pub rdcf_keys: Vec<RdcfKey1>,
}

// ── VOLE Output ─────────────────────────────────────────────────────────────

/// A single VOLE correlation share.
///
/// Correctness: `y_0 = y_1 + r * Delta`.
#[derive(Debug, Clone)]
pub struct VoleShare0<F: PrimeField> {
    /// The "message" component `r`.
    pub r: F,
    /// The masked output `y_0`.
    pub y: F,
}

// ── PCF Gen ─────────────────────────────────────────────────────────────────

/// Generate PCF keys for VOLE (run by P0 as trusted dealer).
///
/// Samples a global correlation `Delta`, per-block noise `(alpha_i, beta_i)`,
/// and creates RDCF keys for each block. The RDCF payload is `beta_i * Delta`.
pub fn pcf_gen<F: PrimeField>(
    params: &PcfParams,
    prg: &Prg,
    rng: &mut impl RngCore,
) -> (PcfKey0<F>, PcfKey1<F>) {
    let delta: F = F::from(rng.next_u64());
    let block_size = params.block_size() as u32;

    let mut rdcf_keys_0 = Vec::with_capacity(params.t);
    let mut rdcf_keys_1 = Vec::with_capacity(params.t);
    let mut alphas = Vec::with_capacity(params.t);
    let mut betas = Vec::with_capacity(params.t);

    for _ in 0..params.t {
        let alpha = rng.next_u32() % block_size;
        // Sample beta in Fp \ {0}
        let beta = loop {
            let b = F::from(rng.next_u64());
            if !b.is_zero() {
                break b;
            }
        };

        let (k0, k1) = rdcf::rdcf_setup(prg, params.m, alpha, beta * delta, rng);

        rdcf_keys_0.push(k0);
        rdcf_keys_1.push(k1);
        alphas.push(alpha);
        betas.push(beta);
    }

    (
        PcfKey0 { rdcf_keys: rdcf_keys_0, alphas, betas },
        PcfKey1 { delta, rdcf_keys: rdcf_keys_1 },
    )
}

// ── EA-Code Sparse Row ──────────────────────────────────────────────────────

/// An entry in a sparse EA-code row: (column_position, coefficient).
pub type SparseEntry<F> = (usize, F);

/// Compute the sparse row `B_x` of the implicit EA-code matrix.
///
/// Returns `l` entries `(position, coefficient)` where positions are in `[0, big_N)`.
/// Deterministic: both parties compute the same row for the same `x`.
pub fn compute_ea_row<F: PrimeField>(x: usize, params: &PcfParams) -> Vec<SparseEntry<F>> {
    let mut hasher = Sha3_256::new();
    hasher.update(params.matrix_seed);
    hasher.update(x.to_le_bytes());
    let hash: [u8; 32] = hasher.finalize().into();

    let mut rng = rand_chacha::ChaCha20Rng::from_seed(hash);
    let mut entries = Vec::with_capacity(params.l);
    for _ in 0..params.l {
        let pos = rng.next_u64() as usize % params.big_n;
        let coeff = F::from(rng.next_u64());
        entries.push((pos, coeff));
    }
    entries
}

// ── PCF Eval ────────────────────────────────────────────────────────────────

/// Evaluate PCF for party 0 at a single point `x`, producing `(r, y_0)`.
pub fn pcf_eval_0<F: PrimeField>(
    prg: &Prg,
    key: &PcfKey0<F>,
    params: &PcfParams,
    x: usize,
) -> VoleShare0<F> {
    let row = compute_ea_row::<F>(x, params);
    let block_size = params.block_size();

    let mut y = F::zero();
    let mut r = F::zero();

    for &(pos, coeff) in &row {
        let block_idx = pos / block_size;
        let offset = (pos % block_size) as u32;

        // RDCF eval_0 at the offset within this block
        y += coeff * rdcf::rdcf_eval_0(prg, &key.rdcf_keys[block_idx], offset);

        // r correction: coeff * beta_i * [offset < alpha_i]
        if offset < key.alphas[block_idx] {
            r += coeff * key.betas[block_idx];
        }
    }

    VoleShare0 { r, y }
}

/// Evaluate PCF for party 1 at a single point `x`, producing `y_1`.
pub fn pcf_eval_1<F: PrimeField>(
    prg: &Prg,
    key: &PcfKey1<F>,
    params: &PcfParams,
    x: usize,
) -> F {
    let row = compute_ea_row::<F>(x, params);
    let block_size = params.block_size();

    let mut y = F::zero();
    for &(pos, coeff) in &row {
        let block_idx = pos / block_size;
        let offset = (pos % block_size) as u32;
        y += coeff * rdcf::rdcf_eval_1::<F>(prg, &key.rdcf_keys[block_idx], offset);
    }
    y
}

// ── Batch Eval (parallel) ───────────────────────────────────────────────────

/// Evaluate PCF for party 0 at multiple points in parallel.
pub fn pcf_eval_0_batch<F: PrimeField>(
    prg: &Prg,
    key: &PcfKey0<F>,
    params: &PcfParams,
    xs: &[usize],
) -> Vec<VoleShare0<F>> {
    xs.par_iter()
        .with_min_len(32)
        .map(|&x| pcf_eval_0(prg, key, params, x))
        .collect()
}

/// Evaluate PCF for party 1 at multiple points in parallel.
pub fn pcf_eval_1_batch<F: PrimeField>(
    prg: &Prg,
    key: &PcfKey1<F>,
    params: &PcfParams,
    xs: &[usize],
) -> Vec<F> {
    xs.par_iter()
        .with_min_len(32)
        .map(|&x| pcf_eval_1(prg, key, params, x))
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use rand::SeedableRng;

    fn test_params(n: usize) -> PcfParams {
        // Small params for testing: t=4, m=8 (block size 256), l=3
        PcfParams {
            n,
            big_n: 4 * 256,
            t: 4,
            l: 3,
            m: 8,
            matrix_seed: [0xAB; 32],
        }
    }

    #[test]
    fn pcf_vole_correctness() {
        let params = test_params(100);
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

        let (k0, k1) = pcf_gen::<Fr>(&params, &prg, &mut rng);

        let xs: Vec<usize> = (0..params.n).collect();
        let (shares_0, ys_1) = rayon::join(
            || pcf_eval_0_batch(&prg, &k0, &params, &xs),
            || pcf_eval_1_batch(&prg, &k1, &params, &xs),
        );

        for i in 0..params.n {
            // Correctness: y_0 = y_1 + r * Delta
            let lhs = shares_0[i].y;
            let rhs = ys_1[i] + shares_0[i].r * k1.delta;
            assert_eq!(lhs, rhs, "VOLE failed at x={i}");
        }
    }

    #[test]
    fn pcf_vole_nonzero_r() {
        let params = test_params(500);
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(99);

        let (k0, _k1) = pcf_gen::<Fr>(&params, &prg, &mut rng);

        let xs: Vec<usize> = (0..params.n).collect();
        let shares = pcf_eval_0_batch(&prg, &k0, &params, &xs);

        let nonzero_r = shares.iter().filter(|s| !s.r.is_zero()).count();
        // With l=3 entries per row hitting t=4 blocks of size 256,
        // probability of r=0 is low for most rows.
        assert!(
            nonzero_r > params.n / 4,
            "too few nonzero r values: {nonzero_r}/{}",
            params.n
        );
    }

    #[test]
    fn ea_row_deterministic() {
        let params = test_params(10);
        let row_a = compute_ea_row::<Fr>(42, &params);
        let row_b = compute_ea_row::<Fr>(42, &params);
        assert_eq!(row_a.len(), row_b.len());
        for i in 0..row_a.len() {
            assert_eq!(row_a[i].0, row_b[i].0);
            assert_eq!(row_a[i].1, row_b[i].1);
        }
    }

    #[test]
    fn ea_row_different_inputs() {
        let params = test_params(10);
        let row_a = compute_ea_row::<Fr>(0, &params);
        let row_b = compute_ea_row::<Fr>(1, &params);
        // With overwhelming probability, different inputs give different rows
        let same = row_a
            .iter()
            .zip(row_b.iter())
            .all(|(a, b)| a.0 == b.0 && a.1 == b.1);
        assert!(!same, "different inputs should yield different rows");
    }

    #[test]
    fn pcf_batch_matches_single() {
        let params = test_params(50);
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(777);

        let (k0, k1) = pcf_gen::<Fr>(&params, &prg, &mut rng);

        let xs: Vec<usize> = (0..params.n).collect();
        let batch_0 = pcf_eval_0_batch(&prg, &k0, &params, &xs);
        let batch_1 = pcf_eval_1_batch(&prg, &k1, &params, &xs);

        for (i, &x) in xs.iter().enumerate() {
            let single_0 = pcf_eval_0(&prg, &k0, &params, x);
            let single_1 = pcf_eval_1(&prg, &k1, &params, x);
            assert_eq!(batch_0[i].r, single_0.r, "r mismatch at x={x}");
            assert_eq!(batch_0[i].y, single_0.y, "y0 mismatch at x={x}");
            assert_eq!(batch_1[i], single_1, "y1 mismatch at x={x}");
        }
    }
}
