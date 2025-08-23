use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

use crate::protocols::rep3::id::PartyID;
// use crate::serde_compat::{ark_de, ark_se};

/// This type represents a replicated shared value. Since a replicated share of a field element contains additive shares of two parties, this type contains two field elements.
#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    CanonicalSerialize,
    CanonicalDeserialize,
    Serialize,
    Deserialize,
)]
#[repr(transparent)]
pub struct AdditivePrimeFieldShare<F: PrimeField>(pub(crate) F);

impl<F: PrimeField> Default for AdditivePrimeFieldShare<F> {
    fn default() -> Self {
        Self(F::zero())
    }
}

impl<F: PrimeField> AdditivePrimeFieldShare<F> {
    /// Constructs a zero share.
    pub fn zero() -> Self {
        Self(F::zero())
    }

    /// Double the share in place
    pub fn double_in_place(&mut self) {
        self.0.double_in_place();
    }

    /// Double the share in place
    pub fn double(&self) -> Self {
        Self(self.0.double())
    }

    /// Promotes a public field element to a replicated share by setting the additive share of the party with id=0 and leaving all other shares to be 0. Thus, the replicated shares of party 0 and party 1 are set.
    pub fn promote_from_trivial(public_value: F, id: PartyID) -> Self {
        match id {
            PartyID::ID0 => Self(public_value),
            PartyID::ID1 => Self(F::zero()),
            PartyID::ID2 => Self(F::zero()),
        }
    }
    
    /// Casts the additive share into a field element. Use with caution.
    pub fn into_fe(self) -> F {
        self.0
    }

    /// Casts a field element into an additive share. Don't use this to `promote_from_trivial`.
    pub fn from_fe(value: F) -> Self {
        Self(value)
    }

    /// Casts a vector of additive shares into a vector of field elements.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes that the size and alignment of `AdditivePrimeFieldShare<F>` and `F` are the same.
    #[inline]
    pub fn into_fe_vec(mut v: Vec<AdditivePrimeFieldShare<F>>) -> Vec<F> {
        debug_assert_eq!(
            std::mem::size_of::<AdditivePrimeFieldShare<F>>(),
            std::mem::size_of::<F>()
        );
        debug_assert_eq!(
            std::mem::align_of::<AdditivePrimeFieldShare<F>>(),
            std::mem::align_of::<F>()
        );
        let ptr = v.as_mut_ptr() as *mut F;
        let len = v.len();
        let cap = v.capacity();
        std::mem::forget(v);
        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }

    /// Casts a vector of field elements into a vector of additive shares.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it assumes that the size and alignment of `AdditivePrimeFieldShare<F>` and `F` are the same.
    #[inline]
    pub fn from_fe_vec(mut v: Vec<F>) -> Vec<AdditivePrimeFieldShare<F>> {
        debug_assert_eq!(
            std::mem::size_of::<AdditivePrimeFieldShare<F>>(),
            std::mem::size_of::<F>()
        );
        debug_assert_eq!(
            std::mem::align_of::<AdditivePrimeFieldShare<F>>(),
            std::mem::align_of::<F>()
        );
        let ptr = v.as_mut_ptr() as *mut AdditivePrimeFieldShare<F>;
        let len = v.len();
        let cap = v.capacity();
        std::mem::forget(v);
        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }

    /// Returns a view of the underlying field elements.
    #[inline]
    pub fn as_fe_vec_mut(s: &mut [AdditivePrimeFieldShare<F>]) -> &mut [F] {
        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut F, s.len()) }
    }

    /// Returns a view of the underlying field elements.
    #[inline]
    pub fn as_fe_vec_ref(s: &[AdditivePrimeFieldShare<F>]) -> &[F] {
        unsafe { std::slice::from_raw_parts(s.as_ptr() as *const F, s.len()) }
    }

    /// Multiplies the share by a public field element.
    #[inline(always)]
    pub fn mul_public_0_optimized(self, other: F) -> Self {
        if other.is_zero() {
            Self::zero()
        } else {
            self * other
        }
    }

    /// Multiplies the share by a public field element.
    #[inline(always)]
    pub fn mul_public_1_optimized(self, other: F) -> Self {
        if other.is_one() {
            self
        } else {
            self * other
        }
    }

    /// Multiplies the share by a public field element.
    #[inline(always)]
    pub fn mul_public_01_optimized(self, other: F) -> Self {
        if other.is_zero() {
            Self::zero()
        } else {
            self.mul_public_1_optimized(other)
        }
    }
}

