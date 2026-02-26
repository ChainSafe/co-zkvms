//! Relaxed Distributed Comparison Function (RDCF) from Boyle et al. 2022.
//!
//! The RDCF secret-shares a step function `f(x) = beta * [x < alpha]` between
//! two parties. Given keys `(K0, K1)` produced by [`rdcf_setup`]:
//!
//!   `eval_0(K0, x) - eval_1(K1, x) = beta  if x < alpha`
//!   `eval_0(K0, x) - eval_1(K1, x) = 0     if x >= alpha`
//!
//! The construction uses a GGM-like binary tree traversal with a fixed-key AES PRG.
//!
//! Reference: Boyle et al. "Pseudorandom Correlation Functions from Variable-Density
//! LPN, Revisited" (2022), Figure 11.

use mpc_types::field::PrimeField;
use rand::RngCore;
use rayon::prelude::*;

// ── PRG from fixed-key AES (Matyas-Meyer-Oseas) ──────────────────────────────

/// A 128-bit seed used in the GGM tree.
pub type Seed = [u8; 16];

/// The PRG `G: {0,1}^128 -> (Seed_convert, Seed_left, Seed_right)`.
///
/// Uses three fixed AES-128 keys in Matyas-Meyer-Oseas mode:
///   output = `AES_k(x) XOR x`
///
/// The three outputs correspond to:
/// - **convert**: used to derive a field element (group element)
/// - **left**: left-child seed in the GGM tree
/// - **right**: right-child seed in the GGM tree
pub struct Prg {
    key_convert: aes::Aes128,
    key_left: aes::Aes128,
    key_right: aes::Aes128,
}

// Well-known fixed keys (publicly known constants derived from SHA-256 IV).
// Security comes from the seed randomness, not these keys.
const PRG_KEY_CONVERT: [u8; 16] = [
    0x6a, 0x09, 0xe6, 0x67, 0xbb, 0x67, 0xae, 0x85,
    0x3c, 0x6e, 0xf3, 0x72, 0xa5, 0x4f, 0xf5, 0x3a,
];
const PRG_KEY_LEFT: [u8; 16] = [
    0x51, 0x0e, 0x52, 0x7f, 0x9b, 0x05, 0x68, 0x8c,
    0x1f, 0x83, 0xd9, 0xab, 0x5b, 0xe0, 0xcd, 0x19,
];
const PRG_KEY_RIGHT: [u8; 16] = [
    0xd8, 0x07, 0xaa, 0x98, 0xa3, 0x03, 0x02, 0x42,
    0x13, 0x19, 0x8a, 0x2e, 0x03, 0x70, 0x73, 0x44,
];

impl Prg {
    /// Create a new PRG instance with the well-known fixed keys.
    pub fn new() -> Self {
        use aes::cipher::KeyInit;
        Self {
            key_convert: aes::Aes128::new(&PRG_KEY_CONVERT.into()),
            key_left: aes::Aes128::new(&PRG_KEY_LEFT.into()),
            key_right: aes::Aes128::new(&PRG_KEY_RIGHT.into()),
        }
    }

    /// Evaluate the PRG: `G(seed) -> (convert_bytes, left_seed, right_seed)`.
    #[inline]
    pub fn expand(&self, seed: &Seed) -> (Seed, Seed, Seed) {
        use aes::cipher::BlockEncrypt;
        let block = aes::Block::from(*seed);

        let mut c = block;
        self.key_convert.encrypt_block(&mut c);
        let convert: Seed = xor_blocks(&c.into(), seed);

        let mut l = block;
        self.key_left.encrypt_block(&mut l);
        let left: Seed = xor_blocks(&l.into(), seed);

        let mut r = block;
        self.key_right.encrypt_block(&mut r);
        let right: Seed = xor_blocks(&r.into(), seed);

        (convert, left, right)
    }

    /// Apply `ConvertG` to a seed: `AES_k_convert(seed) XOR seed`, then map to field.
    ///
    /// This matches the paper's `Convert` function which applies the PRG's
    /// convert key before mapping to a field element.
    #[inline]
    pub fn convert_to_field<F: PrimeField>(&self, seed: &Seed) -> F {
        use aes::cipher::BlockEncrypt;
        let mut block = aes::Block::from(*seed);
        self.key_convert.encrypt_block(&mut block);
        let converted: Seed = xor_blocks(&block.into(), seed);
        let mut buf = [0u8; 32];
        buf[16..].copy_from_slice(&converted);
        F::from_be_bytes_mod_order(&buf)
    }
}

impl Default for Prg {
    fn default() -> Self {
        Self::new()
    }
}

// Prg is safe to share across threads (fixed AES keys are immutable).
unsafe impl Sync for Prg {}
unsafe impl Send for Prg {}

/// Map raw bytes to a field element via `from_be_bytes_mod_order`.
///
/// Used for bytes that have *already* been through the PRG convert key
/// (i.e. the `gamma_bytes` output of [`Prg::expand`]).
#[inline]
fn bytes_to_field<F: PrimeField>(bytes: &Seed) -> F {
    let mut buf = [0u8; 32];
    buf[16..].copy_from_slice(bytes);
    F::from_be_bytes_mod_order(&buf)
}

#[inline]
fn xor_blocks(a: &Seed, b: &Seed) -> Seed {
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = a[i] ^ b[i];
    }
    out
}

// ── RDCF Keys ────────────────────────────────────────────────────────────────

/// RDCF key for party 0 (knows the comparison point `alpha`).
///
/// Size: `m * (16 + sizeof(F)) + sizeof(F) + 8` bytes.
/// For m=20, F=Bn254 Fr (32 bytes): ~1.0 kB.
#[derive(Debug, Clone)]
pub struct RdcfKey0<F: PrimeField> {
    /// Comparison point `alpha in [0, 2^m)`.
    pub alpha: u32,
    /// Tree depth.
    pub m: u32,
    /// Off-path seeds at each level (length = m).
    pub k_bar: Vec<Seed>,
    /// Correction values at each level (length = m).
    pub b_bar: Vec<F>,
    /// Output for the `x == alpha` case.
    pub y: F,
}

/// RDCF key for party 1 (just a single root seed).
///
/// Size: 20 bytes.
#[derive(Debug, Clone)]
pub struct RdcfKey1 {
    /// Root seed of the GGM tree.
    pub k: Seed,
    /// Tree depth.
    pub m: u32,
}

// ── RDCF Setup ───────────────────────────────────────────────────────────────

/// Generate RDCF keys for the step function `f(x) = beta * [x < alpha]`.
///
/// Both parties can independently evaluate their key at any point `x in [0, 2^m)`.
/// The difference of their outputs equals `beta` for `x < alpha` and `0` otherwise.
pub fn rdcf_setup<F: PrimeField>(
    prg: &Prg,
    m: u32,
    alpha: u32,
    beta: F,
    rng: &mut impl RngCore,
) -> (RdcfKey0<F>, RdcfKey1) {
    debug_assert!(
        m <= 30 && alpha < (1u64 << m) as u32,
        "alpha={alpha} must be < 2^m where m={m}"
    );

    // Random root seed for party 1
    let mut root = [0u8; 16];
    rng.fill_bytes(&mut root);

    let mut k_bar = Vec::with_capacity(m as usize);
    let mut b_bar = Vec::with_capacity(m as usize);

    // Walk the tree from root to leaf `alpha`, recording off-path data.
    //
    // Notation (from the plan / Boyle22 Figure 11):
    //   c_j     = gamma_j if alpha_j == 0, else 0    (on-path correction)
    //   c_bar_j = gamma_j if alpha_j == 1, else 0    (off-path correction)
    //   S_bar[j] = c_bar_j + sum(c_i for i < j)
    //   B_bar[j] = S_bar[j] + alpha_j * beta
    //
    // `p1_corr` tracks sum(c_i for all i) = party 1's correction at x=alpha.
    let mut current = root;
    let mut sum_c = F::zero(); // sum of c_i for i = 1..j-1
    let mut p1_corr = F::zero(); // sum of c_i for all levels (party 1's correction at alpha)

    for j in 1..=m {
        let (gamma_bytes, left, right) = prg.expand(&current);
        let gamma = bytes_to_field::<F>(&gamma_bytes);
        let alpha_j = (alpha >> (m - j)) & 1;

        let (c_j, c_bar_j);
        if alpha_j == 0 {
            // On-path is left; off-path is right
            current = left;
            k_bar.push(right);
            c_j = gamma;
            c_bar_j = F::zero();
        } else {
            // On-path is right; off-path is left
            current = right;
            k_bar.push(left);
            c_j = F::zero();
            c_bar_j = gamma;
        }

        // S_bar[j] = c_bar_j + sum(c_i for i < j)
        let s_bar_j = c_bar_j + sum_c;
        b_bar.push(s_bar_j + F::from(alpha_j as u64) * beta);

        // Update sum_c for next iteration: add c_j
        sum_c += c_j;
        // p1_corr accumulates all c_j (= gamma when alpha_j == 0)
        p1_corr += c_j;
    }

    // `current` is now the leaf seed at position `alpha`.
    // Party 1 evaluating at alpha would output:
    //   convert(current) + p1_corr
    // We set y = that value so eval_0(alpha) - eval_1(alpha) = 0.
    let y = prg.convert_to_field::<F>(&current) + p1_corr;

    (
        RdcfKey0 { alpha, m, k_bar, b_bar, y },
        RdcfKey1 { k: root, m },
    )
}

// ── RDCF Eval (single point) ─────────────────────────────────────────────────

/// Evaluate the RDCF for party 0 at a single point `x`.
pub fn rdcf_eval_0<F: PrimeField>(prg: &Prg, key: &RdcfKey0<F>, x: u32) -> F {
    let m = key.m;

    if x == key.alpha {
        return key.y;
    }

    // Find first bit where x and alpha differ (1-indexed, MSB-first).
    let xor = x ^ key.alpha;
    // `leading_zeros()` counts from bit 31 down. We want the position
    // relative to our m-bit domain (bits m-1 down to 0).
    // xor has its highest set bit at position `31 - leading_zeros` (from LSB, 0-indexed).
    // In our m-bit MSB-first 1-indexed scheme, that corresponds to level:
    //   first_diff_level = m - (31 - leading_zeros)
    let first_diff_level = m - (31 - xor.leading_zeros()); // 1-indexed

    // Start from the off-path seed at level `first_diff_level`
    let mut current = key.k_bar[(first_diff_level - 1) as usize];

    // Walk remaining levels
    let mut correction_sum = F::zero();
    for i in (first_diff_level + 1)..=m {
        let (gamma_bytes, left, right) = prg.expand(&current);
        let gamma = bytes_to_field::<F>(&gamma_bytes);
        let x_i = (x >> (m - i)) & 1;
        if x_i == 0 {
            correction_sum += gamma;
            current = left;
        } else {
            current = right;
        }
    }

    prg.convert_to_field::<F>(&current) + key.b_bar[(first_diff_level - 1) as usize] + correction_sum
}

/// Evaluate the RDCF for party 1 at a single point `x`.
pub fn rdcf_eval_1<F: PrimeField>(prg: &Prg, key: &RdcfKey1, x: u32) -> F {
    let m = key.m;
    let mut current = key.k;
    let mut correction_sum = F::zero();

    for j in 1..=m {
        let (gamma_bytes, left, right) = prg.expand(&current);
        let gamma = bytes_to_field::<F>(&gamma_bytes);
        let x_j = (x >> (m - j)) & 1;
        if x_j == 0 {
            correction_sum += gamma;
            current = left;
        } else {
            current = right;
        }
    }

    prg.convert_to_field::<F>(&current) + correction_sum
}

// ── RDCF Batch Eval (parallel) ───────────────────────────────────────────────

/// Evaluate the RDCF for party 0 at multiple points in parallel.
pub fn rdcf_eval_0_batch<F: PrimeField>(
    prg: &Prg,
    key: &RdcfKey0<F>,
    xs: &[u32],
) -> Vec<F> {
    xs.par_iter()
        .with_min_len(64)
        .map(|&x| rdcf_eval_0(prg, key, x))
        .collect()
}

/// Evaluate the RDCF for party 1 at multiple points in parallel.
pub fn rdcf_eval_1_batch<F: PrimeField>(
    prg: &Prg,
    key: &RdcfKey1,
    xs: &[u32],
) -> Vec<F> {
    xs.par_iter()
        .with_min_len(64)
        .map(|&x| rdcf_eval_1(prg, key, x))
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use rand::SeedableRng;

    fn check_rdcf(prg: &Prg, k0: &RdcfKey0<Fr>, k1: &RdcfKey1, alpha: u32, beta: Fr) {
        let m = k0.m;
        let domain = 1u32 << m;
        let xs: Vec<u32> = (0..domain).collect();

        let (y0s, y1s) = rayon::join(
            || rdcf_eval_0_batch::<Fr>(prg, k0, &xs),
            || rdcf_eval_1_batch::<Fr>(prg, k1, &xs),
        );

        for x in 0..domain {
            let diff = y0s[x as usize] - y1s[x as usize];
            let expected = if x < alpha { beta } else { Fr::zero() };
            assert_eq!(diff, expected, "m={m}, x={x}, alpha={alpha}");
        }
    }

    #[test]
    fn rdcf_correctness_small() {
        let m = 4u32;
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        let beta = Fr::from(7u64);

        for alpha in 0..(1u32 << m) {
            let (k0, k1) = rdcf_setup(&prg, m, alpha, beta, &mut rng);
            check_rdcf(&prg, &k0, &k1, alpha, beta);
        }
    }

    #[test]
    fn rdcf_correctness_medium() {
        let m = 10u32;
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(123);

        for _ in 0..10 {
            let alpha = rng.next_u32() % (1 << m);
            let beta = Fr::from(rng.next_u64());
            let (k0, k1) = rdcf_setup(&prg, m, alpha, beta, &mut rng);
            check_rdcf(&prg, &k0, &k1, alpha, beta);
        }
    }

    #[test]
    fn rdcf_pseudorandom_outputs() {
        let m = 8u32;
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(999);

        let alpha = 100;
        let beta = Fr::from(42u64);
        let (k0, k1) = rdcf_setup(&prg, m, alpha, beta, &mut rng);

        let xs: Vec<u32> = (0..(1u32 << m)).collect();
        let (y0s, y1s) = rayon::join(
            || rdcf_eval_0_batch::<Fr>(&prg, &k0, &xs),
            || rdcf_eval_1_batch::<Fr>(&prg, &k1, &xs),
        );

        let nonzero_0 = y0s.iter().filter(|y| !y.is_zero()).count();
        let nonzero_1 = y1s.iter().filter(|y| !y.is_zero()).count();
        assert!(nonzero_0 > 200, "party 0 outputs too many zeros: {nonzero_0}");
        assert!(nonzero_1 > 200, "party 1 outputs too many zeros: {nonzero_1}");
    }

    #[test]
    fn rdcf_alpha_zero() {
        let m = 6u32;
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(77);

        let beta = Fr::from(99u64);
        let (k0, k1) = rdcf_setup(&prg, m, 0, beta, &mut rng);
        check_rdcf(&prg, &k0, &k1, 0, beta);
    }

    #[test]
    fn rdcf_alpha_max() {
        let m = 6u32;
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(88);

        let alpha = (1u32 << m) - 1;
        let beta = Fr::from(55u64);
        let (k0, k1) = rdcf_setup(&prg, m, alpha, beta, &mut rng);
        check_rdcf(&prg, &k0, &k1, alpha, beta);
    }

    #[test]
    fn rdcf_batch_matches_single() {
        let m = 8u32;
        let prg = Prg::new();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(555);

        let alpha = 42;
        let beta = Fr::from(17u64);
        let (k0, k1) = rdcf_setup(&prg, m, alpha, beta, &mut rng);

        let xs: Vec<u32> = (0..(1u32 << m)).collect();
        let batch_0 = rdcf_eval_0_batch::<Fr>(&prg, &k0, &xs);
        let batch_1 = rdcf_eval_1_batch::<Fr>(&prg, &k1, &xs);

        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(batch_0[i], rdcf_eval_0::<Fr>(&prg, &k0, x));
            assert_eq!(batch_1[i], rdcf_eval_1::<Fr>(&prg, &k1, x));
        }
    }
}
