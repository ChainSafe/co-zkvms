use ark_ff::{PrimeField, Zero};

use super::AdditivePrimeFieldShare;


impl<F: PrimeField> std::ops::Add for AdditivePrimeFieldShare<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl<F: PrimeField> std::ops::Add<&AdditivePrimeFieldShare<F>> for &'_ AdditivePrimeFieldShare<F> {
    type Output = AdditivePrimeFieldShare<F>;

    fn add(self, rhs: &AdditivePrimeFieldShare<F>) -> Self::Output {
        *self + *rhs
    }
}

impl<F: PrimeField> std::ops::AddAssign<AdditivePrimeFieldShare<F>> for AdditivePrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<F: PrimeField> std::ops::AddAssign<&AdditivePrimeFieldShare<F>> for AdditivePrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: &AdditivePrimeFieldShare<F>) {
        self.0 += rhs.0;
    }
}

impl<F: PrimeField> std::ops::Sub for AdditivePrimeFieldShare<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<F: PrimeField> std::ops::Sub<&AdditivePrimeFieldShare<F>> for &'_ AdditivePrimeFieldShare<F> {
    type Output = AdditivePrimeFieldShare<F>;

    fn sub(self, rhs: &AdditivePrimeFieldShare<F>) -> Self::Output {
        *self - *rhs
    }
}

impl<F: PrimeField> std::ops::SubAssign<AdditivePrimeFieldShare<F>> for AdditivePrimeFieldShare<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}


impl<F: PrimeField> std::ops::Mul<F> for AdditivePrimeFieldShare<F> {
    type Output = AdditivePrimeFieldShare<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl<F: PrimeField> std::ops::Mul<F> for &AdditivePrimeFieldShare<F> {
    type Output = AdditivePrimeFieldShare<F>;

    fn mul(self, rhs: F) -> Self::Output {
        *self * rhs
    }
}

impl<F: PrimeField> std::ops::MulAssign<F> for AdditivePrimeFieldShare<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.0 *= rhs;
    }
}

impl<F: PrimeField> std::ops::Neg for AdditivePrimeFieldShare<F> {
    type Output = AdditivePrimeFieldShare<F>;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<F: PrimeField> std::iter::Sum<AdditivePrimeFieldShare<F>> for AdditivePrimeFieldShare<F> {
    fn sum<I: Iterator<Item = AdditivePrimeFieldShare<F>>>(iter: I) -> Self {
        let mut sum = Self::zero();
        for share in iter {
            sum += share;
        }
        sum
    }
}
