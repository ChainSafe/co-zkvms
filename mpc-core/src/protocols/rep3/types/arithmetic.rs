use crate::field::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

use crate::protocols::additive::AdditivePrimeFieldShare;
use crate::protocols::rep3::PartyID;
use crate::serde_compat::{ark_de, ark_se};

/// This type represents a replicated shared value. Since a replicated share of a field element contains additive shares of two parties, this type contains two field elements.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, CanonicalSerialize, CanonicalDeserialize, Serialize, Deserialize)]
pub struct Rep3PrimeFieldShare<F: PrimeField> {
    /// Share of this party
    #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
    pub a: F,
    /// Share of the prev party
    #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
    pub b: F,
}

impl<F: PrimeField> Default for Rep3PrimeFieldShare<F> {
    fn default() -> Self {
        Self::zero_share()
    }
}

impl<F: PrimeField> Rep3PrimeFieldShare<F> {
    /// Constructs the type from two additive shares.
    pub fn new(a: F, b: F) -> Self {
        Self { a, b }
    }

    /// Constructs a zero share.
    pub fn zero_share() -> Self {
        Self { a: F::zero(), b: F::zero() }
    }

    /// Unwraps the type into two additive shares.
    pub fn ab(self) -> (F, F) {
        (self.a, self.b)
    }

    /// Double the share in place
    // pub fn double_in_place(&mut self) {
    //     self.a.double_in_place();
    //     self.b.double_in_place();
    // }

    // /// Double the share in place
    // pub fn double(&self) -> Self {
    //     Self {
    //         a: self.a.double(),
    //         b: self.b.double(),
    //     }
    // }

    /// Double the share in place
    pub fn square(&self) -> AdditivePrimeFieldShare<F> {
        self * self
    }

    /// Converts the share into an additive share.
    pub fn into_additive(self) -> AdditivePrimeFieldShare<F>
    where
        F: jolt_core::field::FieldExt,
    {
        AdditivePrimeFieldShare((self.a + self.b) * F::TWO_INV)
    }

    // /// Generate a random share
    // pub fn rand<N: Rep3Network>(io_context: &mut IoContext<N>) -> Self {
    //     let (a, b) = io_context.rngs.rand.random_fes();
    //     Self::new(a, b)
    // }

    /// Promotes a public field element to a replicated share by setting the additive share of the party with id=0 and leaving all other shares to be 0. Thus, the replicated shares of party 0 and party 1 are set.
    pub fn promote_from_trivial(val: &F, id: PartyID) -> Self {
        match id {
            PartyID::ID0 => Self::new(*val, F::zero()),
            PartyID::ID1 => Self::new(F::zero(), *val),
            PartyID::ID2 => Self::zero_share(),
        }
    }
}

impl<F: PrimeField> std::ops::Add for Rep3PrimeFieldShare<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Rep3PrimeFieldShare::<F> { a: self.a + rhs.a, b: self.b + rhs.b }
    }
}
impl<F: PrimeField> std::ops::Add<&Rep3PrimeFieldShare<F>> for &'_ Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn add(self, rhs: &Rep3PrimeFieldShare<F>) -> Self::Output {
        Rep3PrimeFieldShare::<F> { a: self.a + rhs.a, b: self.b + rhs.b }
    }
}

impl<F: PrimeField> std::ops::AddAssign<Rep3PrimeFieldShare<F>> for Rep3PrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<F: PrimeField> std::ops::AddAssign<&Rep3PrimeFieldShare<F>> for Rep3PrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: &Rep3PrimeFieldShare<F>) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<F: PrimeField> std::ops::Sub for Rep3PrimeFieldShare<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Rep3PrimeFieldShare::<F> { a: self.a - rhs.a, b: self.b - rhs.b }
    }
}

impl<F: PrimeField> std::ops::Sub<&Rep3PrimeFieldShare<F>> for &'_ Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn sub(self, rhs: &Rep3PrimeFieldShare<F>) -> Self::Output {
        Rep3PrimeFieldShare::<F> { a: self.a - rhs.a, b: self.b - rhs.b }
    }
}

impl<F: PrimeField> std::ops::SubAssign<Rep3PrimeFieldShare<F>> for Rep3PrimeFieldShare<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl<F: PrimeField> std::ops::Mul for Rep3PrimeFieldShare<F> {
    type Output = AdditivePrimeFieldShare<F>;

    // Local part of mul only
    fn mul(self, rhs: Rep3PrimeFieldShare<F>) -> Self::Output {
        AdditivePrimeFieldShare(self.a * rhs.a + self.a * rhs.b + self.b * rhs.a)
    }
}

impl<F: PrimeField> std::ops::Mul<F> for Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Self::Output { a: self.a * rhs, b: self.b * rhs }
    }
}

impl<F: PrimeField> std::ops::Mul<F> for &Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Self::Output { a: self.a * rhs, b: self.b * rhs }
    }
}

impl<F: PrimeField> std::ops::Mul<&Rep3PrimeFieldShare<F>> for &'_ Rep3PrimeFieldShare<F> {
    type Output = AdditivePrimeFieldShare<F>;

    // Local part of mul only
    fn mul(self, rhs: &Rep3PrimeFieldShare<F>) -> Self::Output {
        AdditivePrimeFieldShare(self.a * rhs.a + self.a * rhs.b + self.b * rhs.a)
    }
}

impl<F: PrimeField> std::ops::MulAssign<F> for Rep3PrimeFieldShare<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.a *= rhs;
        self.b *= rhs;
    }
}

impl<F: PrimeField> std::ops::Neg for Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn neg(self) -> Self::Output {
        Rep3PrimeFieldShare::<F> { a: -self.a, b: -self.b }
    }
}

impl<F: PrimeField> ark_ff::Zero for Rep3PrimeFieldShare<F> {
    fn zero() -> Self {
        Self { a: F::zero(), b: F::zero() }
    }

    fn is_zero(&self) -> bool {
        panic!("is_zero is not a meaningful operation for Rep3PrimeFieldShare, use interative zero check instead");
    }
}

impl<F: PrimeField> std::iter::Sum<Rep3PrimeFieldShare<F>> for Rep3PrimeFieldShare<F> {
    fn sum<I: Iterator<Item = Rep3PrimeFieldShare<F>>>(iter: I) -> Self {
        let mut sum = Rep3PrimeFieldShare::<F>::zero_share();
        for share in iter {
            sum += share;
        }
        sum
    }
}
