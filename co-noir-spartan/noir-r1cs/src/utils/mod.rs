mod print_abi;
pub mod serde_ark;
// pub mod serde_hex;
pub mod serde_jsonify;
// pub mod sumcheck;

use std::fmt::{Display, Formatter, Result as FmtResult};

use acir::AcirField;
use num_bigint::BigUint;

pub use self::print_abi::PrintAbi;
use crate::{FieldElement, NoirElement};

/// Convert a Noir field element to a native FieldElement
#[inline(always)]
pub fn noir_to_native(n: NoirElement) -> FieldElement {
    let number = BigUint::from_bytes_be(&n.to_be_bytes());
    FieldElement::from(number)
}

/// Pretty print a float using SI-prefixes.
pub fn human(value: f64) -> impl Display {
    struct Human(f64);
    impl Display for Human {
        fn fmt(&self, f: &mut Formatter) -> FmtResult {
            let log10 = if self.0.is_normal() {
                self.0.abs().log10()
            } else {
                0.0
            };
            let si_power = ((log10 / 3.0).floor() as isize).clamp(-10, 10);
            let value = self.0 * 10_f64.powi((-si_power * 3) as i32);
            let digits = f.precision().unwrap_or(3) - 1 - (log10 - 3.0 * si_power as f64) as usize;
            let separator = if f.alternate() { "" } else { "\u{202F}" };
            if f.width() == Some(6) && digits == 0 {
                write!(f, " ")?;
            }
            write!(f, "{value:.digits$}{separator}")?;
            let suffix = "qryzafpnμm kMGTPEZYRQ"
                .chars()
                .nth((si_power + 10) as usize)
                .unwrap();
            if suffix != ' ' || f.width() == Some(6) {
                write!(f, "{suffix}")?;
            }
            Ok(())
        }
    }
    Human(value)
}
