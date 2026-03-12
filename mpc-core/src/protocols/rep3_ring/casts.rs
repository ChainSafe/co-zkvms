//! Casts
//!
//! Implements casts for sharings of different datatypes

use super::conversion;
use crate::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
};
use crate::field::PrimeField;
use crate::protocols::{
    rep3::{Rep3BigUintShare, Rep3PrimeFieldShare},
    rep3_ring::{
        Rep3RingShare, Rep3RingSignedShare,
        ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
    },
};
use num_bigint::BigUint;
use num_traits::AsPrimitive;
use rand::{distributions::Standard, prelude::Distribution};
use rayon::prelude::*;
use std::any::TypeId;

/// Selects the appropriate implementation for the ring cast. In case of a downcast, the excess bits are just truncated.
pub fn ring_cast_selector<T, U, N>(
    x: Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3RingShare<U>>
where
    T: IntRing2k + AsPrimitive<U>,
    U: IntRing2k,
    N: Rep3Network,
    Standard: Distribution<T> + Distribution<U>,
{
    cast_a2b(x, io_context)
}

/// Selects the appropriate implementation for the ring_to_field cast.
pub fn ring_to_field_selector<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3PrimeFieldShare<F>>
where
    Standard: Distribution<T>,
{
    ring_to_field_a2b(x, io_context)
}

/// Selects the appropriate implementation for the ring_to_field cast.
pub fn ring_to_field_many_selector<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    io_context: &mut IoContext<N>,
) -> std::io::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    ring_to_field_a2b_many(x, io_context)
}

/// Selects the appropriate implementation for the field_to_ring cast.
pub fn field_to_ring_selector<F: PrimeField, T: IntRing2k, N: Rep3Network>(
    x: Rep3PrimeFieldShare<F>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3RingShare<T>>
where
    Standard: Distribution<T>,
{
    field_to_ring_a2b(x, io_context)
}

/// A downcast of a Rep3RingShare from a larger ring to a smaller ring, truncating the excess bits.
/// Does not require network interaction
pub fn downcast<T, U>(share: Rep3RingShare<T>) -> Rep3RingShare<U>
where
    T: IntRing2k + AsPrimitive<U>,
    U: IntRing2k,
{
    assert!(T::K >= U::K);

    Rep3RingShare {
        a: RingElement(share.a.0.as_()),
        b: RingElement(share.b.0.as_()),
    }
}

/// An upcast of a Rep3RingShare from a smaller ring to a larger ring
/// Does require network interaction
pub fn upcast_a2b<T, U, N>(
    share: Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3RingShare<U>>
where
    T: IntRing2k + AsPrimitive<U>,
    U: IntRing2k,
    N: Rep3Network,
    Standard: Distribution<T> + Distribution<U>,
{
    assert!(T::K < U::K);

    // A special case for Bit
    if TypeId::of::<T>() == TypeId::of::<Bit>() {
        let share = crate::downcast(&share).expect("We already checked types");
        return conversion::bit_inject_from_bit(share, io_context);
    }

    let binary = conversion::a2b(share, io_context)?;
    let binary = Rep3RingShare {
        a: RingElement(binary.a.0.as_()),
        b: RingElement(binary.b.0.as_()),
    };
    conversion::b2a(&binary, io_context)
}

/// An upcast of a Rep3RingShare from a smaller ring to a larger ring
/// Does require network interaction
#[tracing::instrument(skip_all, level = "trace")]
pub fn upcast_many_from_binary<T, U, N>(
    binary: &[Rep3RingShare<T>],
    io_context: &mut IoContext<N>,
) -> std::io::Result<Vec<Rep3RingShare<U>>>
where
    T: IntRing2k + AsPrimitive<U>,
    U: IntRing2k,
    N: Rep3Network,
    Standard: Distribution<T> + Distribution<U>,
{
    assert!(T::K < U::K);

    // A special case for Bit
    if TypeId::of::<T>() == TypeId::of::<Bit>() {
        unimplemented!()
    }

    let binary_upcasted = binary
        .par_iter()
        .map(|s| Rep3RingShare {
            a: RingElement(s.a.0.as_()),
            b: RingElement(s.b.0.as_()),
        })
        .collect::<Vec<_>>();

    conversion::b2a_many(&binary_upcasted, io_context)
}

/// A cast of a Rep3RingShare from a ring to another ring. In case of a downcast, the excess bits are just truncated.
pub fn cast_a2b<T, U, N>(
    share: Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3RingShare<U>>
where
    T: IntRing2k + AsPrimitive<U>,
    U: IntRing2k,
    N: Rep3Network,
    Standard: Distribution<T> + Distribution<U>,
{
    if T::K >= U::K {
        Ok(downcast(share))
    } else {
        upcast_a2b(share, io_context)
    }
}

/// A cast of a Rep3PrimeFieldShare to a Rep3RingShare. Truncates the excess bits.
pub fn field_to_ring_a2b<F: PrimeField, T: IntRing2k, N: Rep3Network>(
    share: Rep3PrimeFieldShare<F>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3RingShare<T>>
where
    Standard: Distribution<T>,
{
    let binary = rep3::conversion::a2b(share, io_context)?;
    let ring_share = Rep3RingShare {
        a: RingElement(T::cast_from_biguint(&binary.a)),
        b: RingElement(T::cast_from_biguint(&binary.b)),
    };
    conversion::b2a(&ring_share, io_context)
}

/// A cast of a Rep3RingShare to a Rep3PrimeFieldShare
pub fn ring_to_field_a2b<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    share: Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3PrimeFieldShare<F>>
where
    Standard: Distribution<T>,
{
    // A special case for Bit
    if TypeId::of::<T>() == TypeId::of::<Bit>() {
        let share =
            crate::downcast::<_, Rep3RingShare<Bit>>(&share).expect("We already checked types");
        let biguint_share = Rep3BigUintShare::new(
            BigUint::from(share.a.0.convert() as u64),
            BigUint::from(share.b.0.convert() as u64),
        );

        return rep3::conversion::bit_inject(&biguint_share, io_context);
    }

    let binary = conversion::a2b(share, io_context)?;
    let biguint_share = Rep3BigUintShare::new(
        T::cast_to_biguint(&binary.a.0),
        T::cast_to_biguint(&binary.b.0),
    );
    rep3::conversion::b2a(&biguint_share, io_context)
}

/// A cast of a Rep3RingShare to a Rep3PrimeFieldShare
#[tracing::instrument(skip_all, level = "trace")]
pub fn ring_to_field_a2b_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    shares: &[Rep3RingShare<T>],
    io_context: &mut IoContext<N>,
) -> std::io::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    // A special case for Bit
    if TypeId::of::<T>() == TypeId::of::<Bit>() {
        let shares = shares.to_vec();
        let biguint_shares = shares
            .into_iter()
            .map(|share| {
                let share = crate::downcast::<_, Rep3RingShare<Bit>>(&share)
                    .expect("We already checked types");
                let biguint_share = Rep3BigUintShare::new(
                    BigUint::from(share.a.0.convert() as u64),
                    BigUint::from(share.b.0.convert() as u64),
                );
                biguint_share
            })
            .collect::<Vec<_>>();

        return rep3::conversion::bit_inject_many(&biguint_shares, io_context);
    }

    let binary = conversion::a2b_many(shares, io_context)?;
    let biguint_shares = binary
        .into_iter()
        .map(|binary| {
            Rep3BigUintShare::new(
                T::cast_to_biguint(&binary.a.0),
                T::cast_to_biguint(&binary.b.0),
            )
        })
        .collect::<Vec<_>>();

    rep3::conversion::b2a_many(&biguint_shares, io_context)
}

/// A cast of a Rep3RingShare to a Rep3PrimeFieldShare
#[tracing::instrument(skip_all, level = "trace")]
pub fn binary_ring_to_field_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    binary: &[Rep3RingShare<T>],
    io_context: &mut IoContext<N>,
) -> std::io::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    // A special case for Bit
    if TypeId::of::<T>() == TypeId::of::<Bit>() {
        let shares = binary.to_vec();
        let biguint_shares = shares
            .into_iter()
            .map(|share| {
                let share = crate::downcast::<_, Rep3RingShare<Bit>>(&share)
                    .expect("We already checked types");
                let biguint_share = Rep3BigUintShare::new(
                    BigUint::from(share.a.0.convert() as u64),
                    BigUint::from(share.b.0.convert() as u64),
                );
                biguint_share
            })
            .collect::<Vec<_>>();

        return rep3::conversion::bit_inject_many(&biguint_shares, io_context);
    }

    let biguint_shares = binary
        .into_iter()
        .map(|binary| {
            Rep3BigUintShare::new(
                T::cast_to_biguint(&binary.a.0),
                T::cast_to_biguint(&binary.b.0),
            )
        })
        .collect::<Vec<_>>();

    rep3::conversion::b2a_many(&biguint_shares, io_context)
}

/// A cast of a Rep3RingShare to a Rep3PrimeFieldShare
#[tracing::instrument(skip_all, level = "trace")]
pub fn signed_binary_ring_to_field_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    singed: Vec<Rep3RingSignedShare<T>>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    let (binary, signs): (Vec<_>, Vec<_>) = singed
        .into_iter()
        .map(|Rep3RingSignedShare { abs, sign }| (abs, sign))
        .unzip();

    let positive = binary_ring_to_field_many(&binary, io_context)?;
    let negative = positive
        .iter()
        .map(|x| rep3::arithmetic::neg(*x))
        .collect::<Vec<_>>();
    let signs = conversion::bit_inject_from_bits_to_field_many(&signs, io_context)?;

    rep3::arithmetic::cmux_many::<F, N>(&signs, &positive, &negative, io_context)
}
