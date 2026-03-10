use allocative::Allocative;
use mpc_core::protocols::rep3_ring::ring::int_ring::IntRing2k;
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use jolt_core::utils::math::Math;

/// Compact multilinear polynomial over ring coefficients represented as full Rep3 shares.
///
/// For each coefficient:
/// - `shares[i]` is this party's full **arithmetic** Rep3 ring share (both `a` and `b` limbs).
/// - `shares_bin[i]` is this party's full **binary** (XOR) Rep3 ring share.
#[derive(Default, Debug, PartialEq, Clone)]
pub struct Rep3CompactPolynomial<T: IntRing2k = u64> {
    num_vars: usize,
    len: usize,
    pub shares: Vec<Rep3RingShare<T>>,
    pub shares_bin: Vec<Rep3RingShare<T>>,
}

// Rep3RingShare<T> doesn't implement Allocative, so we implement it manually.
impl<T: IntRing2k> Allocative for Rep3CompactPolynomial<T> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(
            allocative::Key::new("shares"),
            &(self.shares.len() * std::mem::size_of::<Rep3RingShare<T>>()),
        );
        visitor.visit_field(
            allocative::Key::new("shares_bin"),
            &(self.shares_bin.len() * std::mem::size_of::<Rep3RingShare<T>>()),
        );
        visitor.exit();
    }
}

impl<T: IntRing2k> Rep3CompactPolynomial<T> {
    pub fn from_shares(shares: Vec<Rep3RingShare<T>>, shares_bin: Vec<Rep3RingShare<T>>) -> Self {
        assert_eq!(
            shares.len(),
            shares_bin.len(),
            "Rep3CompactPolynomial: shares length mismatch"
        );
        assert!(
            shares.len().is_power_of_two(),
            "Rep3CompactPolynomial: length must be power-of-two (got {})",
            shares.len()
        );
        let len = shares.len();
        Self {
            num_vars: len.log_2(),
            len,
            shares,
            shares_bin,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }
}
