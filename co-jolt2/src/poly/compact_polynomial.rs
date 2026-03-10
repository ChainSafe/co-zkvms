use allocative::Allocative;
use crate::zkvm::instruction::types::rep3_operand::Rep3Operand;
use jolt2_common::constants::{ArithmeticWideInt, XlenInt};
use mpc_core::protocols::rep3_ring::Rep3RingShare;
use snarks_core::math::Math;

/// Compact multilinear polynomial over u64 coefficients stored as `Rep3Operand`.
///
/// Each coefficient is either `Rep3Operand::Public(value)` (no MPC correction needed)
/// or `Rep3Operand::Shared { binary, arithmetic, .. }` (requires ring B2A + wrap correction).
/// Padding (NoOp) rows are `Rep3Operand::Public(0)`.
#[derive(Default, Debug, PartialEq, Clone)]
pub struct Rep3CompactPolynomial {
    num_vars: usize,
    len: usize,
    pub coeffs: Vec<Rep3Operand>,
}

impl Allocative for Rep3CompactPolynomial {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(
            allocative::Key::new("coeffs"),
            &(self.coeffs.len() * std::mem::size_of::<Rep3Operand>()),
        );
        visitor.exit();
    }
}

impl Rep3CompactPolynomial {
    /// Construct from a vec of operands, padding to next power of two with `Public(0)`.
    pub fn from_operands(mut coeffs: Vec<Rep3Operand>) -> Self {
        assert!(!coeffs.is_empty(), "Rep3CompactPolynomial: empty coefficients");
        let len = coeffs.len().next_power_of_two();
        coeffs.resize(len, Rep3Operand::Public(0));
        Self {
            num_vars: len.log_2(),
            len,
            coeffs,
        }
    }

    /// Compatibility constructor: zips arithmetic + binary ring shares into `Shared` operands.
    ///
    /// Both vectors must have equal, power-of-two length. Each element becomes
    /// `Rep3Operand::Shared { binary, arithmetic: Some(upcast(arith)), public: None }`.
    pub fn from_shares(
        shares: Vec<Rep3RingShare<ArithmeticWideInt>>,
        shares_bin: Vec<Rep3RingShare<XlenInt>>,
    ) -> Self {
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
        let coeffs: Vec<Rep3Operand> = shares
            .into_iter()
            .zip(shares_bin)
            .map(|(arith, bin)| {
                Rep3Operand::Shared {
                    binary: bin,
                    arithmetic: Some(arith),
                    public: None,
                }
            })
            .collect();
        let len = coeffs.len();
        Self {
            num_vars: len.log_2(),
            len,
            coeffs,
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

    /// Count of shared (non-public) coefficients.
    pub fn shared_count(&self) -> usize {
        self.coeffs.iter().filter(|c| matches!(c, Rep3Operand::Shared { .. })).count()
    }
}
