//! Arithmetic
//!
//! This module contains operations with arithmetic shares

use core::panic;
use mpc_types::protocols::additive::AdditivePrimeFieldShare;
use num_traits::cast::ToPrimitive;

use ark_ff::PrimeField;
use itertools::{Itertools, izip};
use num_bigint::BigUint;
use num_traits::One;
use num_traits::Zero;

use crate::IoResult;
use crate::protocols::rep3::{PartyID, detail, network::Rep3Network};
use rayon::prelude::*;

use super::{
    Rep3BigUintShare, Rep3PrimeFieldShare, binary, conversion, network::IoContext,
    rngs::Rep3CorrelatedRng,
};

use ark_linear_sumcheck::rng::FeedableRNG;
use eyre::Context;
use rand::Rng;
use rand::RngCore;

use snarks_core::field::FieldExt;

use crate::protocols::additive::AdditiveShare;
use crate::protocols::rep3::rngs::SSRandom;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Type alias for a [`Rep3PrimeFieldShare`]
pub type FieldShare<F> = Rep3PrimeFieldShare<F>;
/// Type alias for a [`Rep3BigUintShare`]
pub type BinaryShare<F> = Rep3BigUintShare<F>;

/// Performs addition between a shared value and a public value.
pub fn add_public<F: PrimeField>(shared: FieldShare<F>, public: F, id: PartyID) -> FieldShare<F> {
    let mut res = shared;
    match id {
        PartyID::ID0 => res.a += public,
        PartyID::ID1 => res.b += public,
        PartyID::ID2 => {}
    }
    res
}

/// Performs addition between a shared value and a public value in place.
pub fn add_assign_public<F: PrimeField>(shared: &mut FieldShare<F>, public: F, id: PartyID) {
    match id {
        PartyID::ID0 => shared.a += public,
        PartyID::ID1 => shared.b += public,
        PartyID::ID2 => {}
    }
}

/// Performs subtraction between a shared value and a public value, returning shared - public.
pub fn sub_shared_by_public<F: PrimeField>(
    shared: FieldShare<F>,
    public: F,
    id: PartyID,
) -> FieldShare<F> {
    add_public(shared, -public, id)
}

/// Performs subtraction between a shared value and a public value, returning public - shared.
pub fn sub_public_by_shared<F: PrimeField>(
    public: F,
    shared: FieldShare<F>,
    id: PartyID,
) -> FieldShare<F> {
    add_public(-shared, public, id)
}

/// Performs multiplication of two shared values.
pub fn mul<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let local_a = (a * b).into_fe() + io_context.rngs.rand.masking_field_element::<F>();
    let local_b = io_context.network.reshare(local_a)?;
    Ok(FieldShare {
        a: local_a,
        b: local_b,
    })
}

/// Performs a reshare on all shares in the vector.
pub fn reshare_vec<F: PrimeField, N: Rep3Network>(
    local_a: Vec<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    let local_b = io_context.network.reshare_many(&local_a)?;
    if local_b.len() != local_a.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of reshare_vec: Invalid number of elements received",
        ));
    }
    Ok(izip!(local_a, local_b)
        .map(|(a, b)| FieldShare::new(a, b))
        .collect())
}

/// Performs a reshare on all shares in the vector.
pub async fn reshare_vec_async<F: PrimeField, N: Rep3Network>(
    local_a: Vec<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    let local_b = io_context
        .network
        .reshare_many_async(local_a.clone())
        .await?;
    if local_b.len() != local_a.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of reshare_vec: Invalid number of elements received",
        ));
    }
    Ok(izip!(local_a, local_b)
        .map(|(a, b)| FieldShare::new(a, b))
        .collect())
}

/// Performs multiplication of a shared value and a public value.
pub fn mul_public<F: PrimeField>(shared: FieldShare<F>, public: F) -> FieldShare<F> {
    shared * public
}

/// Performs element-wise multiplication of two vectors of shared values.
///
/// Use this function for small vecs. For large vecs see [`local_mul_vec`] and [`reshare_vec`]
pub fn mul_vec<F: PrimeField, N: Rep3Network>(
    lhs: &[FieldShare<F>],
    rhs: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let local_a = izip!(lhs.iter(), rhs.iter())
        .map(|(lhs, rhs)| (lhs * rhs).into_fe() + io_context.rngs.rand.masking_field_element::<F>())
        .collect_vec();
    reshare_vec(local_a, io_context)
}

/// Performs element-wise multiplication of two vectors of shared values.
pub fn mul_vec_par<F: PrimeField, N: Rep3Network>(
    lhs: &[FieldShare<F>],
    rhs: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    debug_assert_eq!(lhs.len(), rhs.len());
    let rngs = tracing::trace_span!("rngs").in_scope(|| {
        (0..lhs.len())
            .map(|_| io_context.rngs.rand.masking_field_element::<F>())
            .collect_vec()
    });
    let local_a = tracing::trace_span!("cpu mul par").in_scope(|| {
        lhs.par_iter()
            .zip(rhs.par_iter())
            .zip(rngs.par_iter())
            .map(|((lhs, rhs), rng)| (lhs * rhs).into_fe() + *rng)
            .collect()
    });
    reshare_vec(local_a, io_context)
}

/// Performs division of two shared values, returning a / b.
pub fn div<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    mul(a, inv(b, io_context)?, io_context)
}

/// Performs division of a shared value by a public value, returning shared / public.
pub fn div_shared_by_public<F: PrimeField>(
    shared: FieldShare<F>,
    public: F,
) -> eyre::Result<FieldShare<F>> {
    if public.is_zero() {
        eyre::bail!("Cannot invert zero");
    }
    let b_inv = public.inverse().unwrap();
    Ok(mul_public(shared, b_inv))
}

/// Performs division of a public value by a shared value, returning public / shared.
pub fn div_public_by_shared<F: PrimeField, N: Rep3Network>(
    public: F,
    shared: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    Ok(mul_public(inv(shared, io_context)?, public))
}

/// Negates a shared value.
pub fn neg<F: PrimeField>(a: FieldShare<F>) -> FieldShare<F> {
    -a
}

/// Computes the inverse of a shared value.
pub fn inv<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let r = rand(io_context);
    let y = mul_open(a, r, io_context)?;
    if y.is_zero() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of inverse in MPC: cannot compute inverse of zero",
        ));
    }
    let y_inv = y
        .inverse()
        .expect("we checked if y is zero. Must be possible to invert.");
    Ok(r * y_inv)
}

/// Computes the inverse of a vector of shared field elements
pub fn inv_vec<F: PrimeField, N: Rep3Network>(
    a: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    let r = (0..a.len()).map(|_| rand(io_context)).collect_vec();
    let y = mul_open_vec(a, &r, io_context)?;
    if y.iter().any(|y| y.is_zero()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of inverse in MPC: cannot compute inverse of zero",
        ));
    }

    // we can unwrap as we checked that none of the y is zero
    Ok(izip!(r, y).map(|(r, y)| r * y.inverse().unwrap()).collect())
}

/// Performs the opening of a shared value and returns the equivalent public value.
pub fn open<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<F> {
    let c = io_context.network.reshare(a.b)?;
    Ok(a.a + a.b + c)
}

/// Performs the opening of a shared value and returns the equivalent public value.
pub fn open_bit<F: PrimeField, N: Rep3Network>(
    a: Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BigUint> {
    let c = io_context.network.reshare(a.b.to_owned())?;
    Ok(a.a ^ a.b ^ c)
}

/// Performs the opening of a shared value and returns the equivalent public value.
pub fn open_vec<'a, F: PrimeField, N: Rep3Network>(
    a: impl IntoIterator<Item = &'a FieldShare<F>>,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<F>> {
    let a = a.into_iter();
    let (a, b) = a
        .map(|share| (share.a, share.b))
        .collect::<(Vec<F>, Vec<F>)>();
    let c = io_context.network.reshare_many(&b)?;
    Ok(izip!(a, b, c).map(|(a, b, c)| a + b + c).collect_vec())
}

/// Computes a CMUX: If cond is 1, returns truthy, otherwise returns falsy.
/// Implementations should not overwrite this method.
pub fn cmux<F: PrimeField, N: Rep3Network>(
    cond: FieldShare<F>,
    truthy: FieldShare<F>,
    falsy: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let b_min_a = truthy - falsy;
    let d = mul(cond, b_min_a, io_context)?;
    Ok(falsy + d)
}

/// Computes a CMUX: If cond is 1, returns truthy, otherwise returns falsy.
/// Implementations should not overwrite this method.
pub fn cmux_vec<F: PrimeField, N: Rep3Network>(
    cond: FieldShare<F>,
    truthy: &[FieldShare<F>],
    falsy: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    debug_assert_eq!(truthy.len(), falsy.len());
    let result_a = truthy
        .iter()
        .zip(falsy.iter())
        .map(|(t, f)| {
            ((*t - *f) * cond).into_fe() + f.a + io_context.rngs.rand.masking_field_element::<F>()
        })
        .collect_vec();
    reshare_vec(result_a, io_context)
}

/// Convenience method for \[a\] + \[b\] * c
pub fn add_mul_public<F: PrimeField>(a: FieldShare<F>, b: FieldShare<F>, c: F) -> FieldShare<F> {
    a + mul_public(b, c)
}

/// Convenience method for \[a\] + \[b\] * \[c\]
pub fn add_mul<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    c: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let mul = mul(c, b, io_context)?;
    Ok(a + mul)
}

/// Transforms a public value into a shared value: \[a\] = a.
pub fn promote_to_trivial_share<F: PrimeField>(id: PartyID, public_value: F) -> FieldShare<F> {
    match id {
        PartyID::ID0 => Rep3PrimeFieldShare::new(public_value, F::zero()),
        PartyID::ID1 => Rep3PrimeFieldShare::new(F::zero(), public_value),
        PartyID::ID2 => Rep3PrimeFieldShare::zero_share(),
    }
}

/// Transforms a vector of public values into a vector of shared values: \[a\] = a.
pub fn promote_to_trivial_shares<F: PrimeField>(
    public_values: Vec<F>,
    id: PartyID,
) -> Vec<FieldShare<F>> {
    public_values
        .into_iter()
        .map(|value| promote_to_trivial_share(id, value))
        .collect()
}

/// This function performs a multiplication directly followed by an opening. This safes one round of communication in some MPC protocols compared to calling `mul` and `open` separately.
pub fn mul_open<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<F> {
    let a = (a * b).into_fe() + io_context.rngs.rand.masking_field_element::<F>();
    let (b, c) = io_context.network.broadcast(a)?;
    Ok(a + b + c)
}

/// This function performs a multiplication directly followed by an opening. This safes one round of communication in some MPC protocols compared to calling `mul` and `open` separately.
pub fn mul_open_vec<F: PrimeField, N: Rep3Network>(
    a: &[FieldShare<F>],
    b: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<F>> {
    let mut a = izip!(a, b)
        .map(|(a, b)| (a * b).into_fe() + io_context.rngs.rand.masking_field_element::<F>())
        .collect_vec();
    let (b, c) = io_context.network.broadcast_many(&a)?;
    izip!(a.iter_mut(), b, c).for_each(|(a, b, c)| *a += b + c);
    Ok(a)
}

/// Generate a random [`FieldShare`].
pub fn rand<F: PrimeField, N: Rep3Network>(io_context: &mut IoContext<N>) -> FieldShare<F> {
    let (a, b) = io_context.rngs.rand.random_fes();
    FieldShare::new(a, b)
}

/// Computes the square root of a shared value.
pub fn sqrt<F: PrimeField, N: Rep3Network>(
    share: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let r_squ = rand(io_context);
    let r_inv = rand(io_context);

    let rr = mul(r_squ, r_squ, io_context)?;

    // parallel mul of rr with a and r_squ with r_inv
    let lhs = vec![rr, r_squ];
    let rhs = vec![share, r_inv];
    let mul = mul_vec(&lhs, &rhs, io_context)?;

    // Open mul
    io_context
        .network
        .send_next_many(&mul.iter().map(|s| s.b.to_owned()).collect_vec())?;
    let c = io_context.network.recv_prev_many::<F>()?;
    if c.len() != 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of square root: invalid number of elements received",
        ));
    }
    let y_sq = (mul[0].a + mul[0].b + c[0]).sqrt();
    let y_inv = mul[1].a + mul[1].b + c[1];

    // postprocess the square and inverse
    let y_sq = match y_sq {
        Some(y) => y,
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "During execution of square root in MPC: cannot compute square root",
            ));
        }
    };

    if y_inv.is_zero() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of square root in MPC: cannot compute inverse of zero",
        ));
    }
    let y_inv = y_inv.inverse().unwrap();

    let r_squ_inv = r_inv * y_inv;
    let a_sqrt = r_squ_inv * y_sq;

    Ok(a_sqrt)
}

/// Performs a pow operation using a shared value as base and a public value as exponent.
pub fn pow_public<F: PrimeField, N: Rep3Network>(
    shared: FieldShare<F>,
    public: F,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    // TODO: are negative exponents allowed in circom?
    let mut res = promote_to_trivial_share(io_context.id, F::one());
    let mut public: BigUint = public.into_bigint().into();
    let mut shared: FieldShare<F> = shared;
    while !public.is_zero() {
        if public.bit(0) {
            res = mul(res, shared, io_context)?;
        }
        shared = mul(shared, shared, io_context)?;
        public >>= 1;
    }
    mul(res, shared, io_context)
}

/// Returns 1 if lhs < rhs and 0 otherwise. Checks if one shared value is less than another shared value. The result is a shared value that has value 1 if the first shared value is less than the second shared value and 0 otherwise.
pub fn lt<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a < b is equivalent to !(a >= b)
    let tmp = ge(lhs, rhs, io_context)?;
    Ok(sub_public_by_shared(F::one(), tmp, io_context.id))
}

/// Returns 1 if lhs >= rhs and 0 otherwise. Checks if one shared value is greater than or equal to another shared value. The result is a shared value that has value 1 if the first shared value is greater than or equal to the second shared value and 0 otherwise.
pub fn lt_many<F: PrimeField, N: Rep3Network>(
    lhs: &[BinaryShare<F>],
    rhs: &[BinaryShare<F>],
    io_context: &mut IoContext<N>,xz
) -> IoResult<Vec<FieldShare<F>>> {
    // a < b is equivalent to !(a >= b)
    let res = ge_many(lhs, rhs, io_context)?;
    Ok(res
        .into_iter()
        .map(|tmp| sub_public_by_shared(F::one(), tmp, io_context.id))
        .collect())
}

/// Returns 1 if lhs < rhs and 0 otherwise. Checks if a shared value is less than the public value. The result is a shared value that has value 1 if the shared value is less than the public value and 0 otherwise.
pub fn lt_public<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a < b is equivalent to !(a >= b)
    let tmp = ge_public(lhs, rhs, io_context)?;
    Ok(sub_public_by_shared(F::one(), tmp, io_context.id))
}

/// Returns 1 if lhs <= rhs and 0 otherwise. Checks if one shared value is less than or equal to another shared value. The result is a shared value that has value 1 if the first shared value is less than or equal to the second shared value and 0 otherwise.
pub fn le<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a <= b is equivalent to b >= a
    ge(rhs, lhs, io_context)
}

/// Returns 1 if lhs <= rhs and 0 otherwise. Checks if a shared value is less than or equal to a public value. The result is a shared value that has value 1 if the shared value is less than or equal to the public value and 0 otherwise.
pub fn le_public<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let res = le_public_bit(lhs, rhs, io_context)?;
    conversion::bit_inject(&res, io_context)
}

/// Same as le_public but without using bit_inject on the result. Returns 1 if lhs <= rhs and 0 otherwise. Checks if a shared value is less than or equal to a public value. The result is a shared value that has value 1 if the shared value is less than or equal to the public value and 0 otherwise.
pub fn le_public_bit<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    detail::unsigned_ge_const_lhs(rhs, lhs, io_context)
}

/// Returns 1 if lhs > rhs and 0 otherwise. Checks if one shared value is greater than another shared value. The result is a shared value that has value 1 if the first shared value is greater than the second shared value and 0 otherwise.
pub fn gt<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a > b is equivalent to !(a <= b)
    let tmp = le(lhs, rhs, io_context)?;
    Ok(sub_public_by_shared(F::one(), tmp, io_context.id))
}

/// Returns 1 if lhs > rhs and 0 otherwise. Checks if a shared value is greater than the public value. The result is a shared value that has value 1 if the shared value is greater than the public value and 0 otherwise.
pub fn gt_public<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a > b is equivalent to !(a <= b)
    let tmp = le_public(lhs, rhs, io_context)?;
    Ok(sub_public_by_shared(F::one(), tmp, io_context.id))
}

/// Returns 1 if lhs >= rhs and 0 otherwise. Checks if one shared value is greater than or equal to another shared value. The result is a shared value that has value 1 if the first shared value is greater than or equal to the second shared value and 0 otherwise.
pub fn ge<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let res = ge_bit(lhs, rhs, io_context)?;
    conversion::bit_inject(&res, io_context)
}

/// Returns 1 if lhs >= rhs and 0 otherwise. Checks if one shared value is greater than or equal to another shared value. The result is a shared value that has value 1 if the first shared value is greater than or equal to the second shared value and 0 otherwise.
pub fn ge_many<F: PrimeField, N: Rep3Network>(
    lhs: &[BinaryShare<F>],
    rhs: &[BinaryShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    let res = ge_bit_many(lhs, rhs, io_context)?;
    conversion::bit_inject_many(&res, io_context)
}

/// Same as ge but without using bit_inject on the result. Returns 1 if lhs >= rhs and 0 otherwise. Checks if one shared value is greater than or equal to another shared value. The result is a shared value that has value 1 if the first shared value is greater than or equal to the second shared value and 0 otherwise.
pub fn ge_bit<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    detail::unsigned_ge(lhs, rhs, io_context)
}

/// Same as ge but without using bit_inject on the result. Returns 1 if lhs >= rhs and 0 otherwise. Checks if one shared value is greater than or equal to another shared value. The result is a shared value that has value 1 if the first shared value is greater than or equal to the second shared value and 0 otherwise.
pub fn ge_bit_many<F: PrimeField, N: Rep3Network>(
    lhs: &[BinaryShare<F>],
    rhs: &[BinaryShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<BinaryShare<F>>> {
    detail::unsigned_ge_many(lhs, rhs, io_context)
}

/// Returns 1 if lhs >= rhs and 0 otherwise. Checks if a shared value is greater than or equal to a public value. The result is a shared value that has value 1 if the shared value is greater than or equal to the public value and 0 otherwise.
pub fn ge_public<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let res = ge_public_bit(lhs, rhs, io_context)?;
    conversion::bit_inject(&res, io_context)
}

/// Same as ge_public but without using bit_inject on the result. Returns 1 if lhs >= rhs and 0 otherwise. Checks if a shared value is greater than or equal to a public value. The result is a shared value that has value 1 if the shared value is greater than or equal to the public value and 0 otherwise.
pub fn ge_public_bit<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    detail::unsigned_ge_const_rhs(lhs, rhs, io_context)
}

//TODO FN REMARK - I think we can skip the bit_inject.
//Circom has dedicated op codes for bool ops so we would know
//for bool_and/bool_or etc that we are a boolean value (and therefore
//bit len 1).
//
//We leave it like that and come back to that later. Maybe it doesn't matter...

/// Checks if two shared values are equal. The result is a shared value that has value 1 if the two shared values are equal and 0 otherwise.
pub fn eq<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let is_zero = eq_bit(a, b, io_context)?;
    conversion::bit_inject(&is_zero, io_context)
}

/// Checks if two slices of shared values are equal element-wise.
/// Returns a vector of shared values, where each element is 1 if the corresponding elements are equal and 0 otherwise.
pub fn eq_many<F: PrimeField, N: Rep3Network>(
    a: &[FieldShare<F>],
    b: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    if a.len() != b.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "During execution of eq_many: Invalid number of elements received. Length of a : {} and length of b: {}",
                a.len(),
                b.len()
            ),
        ));
    }
    let is_zero = eq_bit_many(a, b, io_context)?;
    conversion::bit_inject_many(&is_zero, io_context)
}

/// Checks if a shared value is equal to a public value. The result is a shared value that has value 1 if the two values are equal and 0 otherwise.
pub fn eq_public<F: PrimeField, N: Rep3Network>(
    shared: FieldShare<F>,
    public: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let is_zero = eq_bit_public(shared, public, io_context)?;
    conversion::bit_inject(&is_zero, io_context)
}

/// Checks if a slice of shared values is equal to a slice of public values element-wise.
/// Returns a vector of shared values, where each element is 1 if the corresponding elements are equal and 0 otherwise.
pub fn eq_public_many<F: PrimeField, N: Rep3Network>(
    shared: &[FieldShare<F>],
    public: &[F],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    if shared.len() != public.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "During execution of eq_public_many: Invalid number of elements received. Length of shared : {} and length of public: {}",
                shared.len(),
                public.len()
            ),
        ));
    }
    let is_zero = eq_bit_public_many(shared, public, io_context)?;
    conversion::bit_inject_many(&is_zero, io_context)
}

/// Same as eq_bit but without using bit_inject on the result. Checks if a shared value is equal to a public value. The result is a shared value that has value 1 if the two values are equal and 0 otherwise.
pub fn eq_bit_public<F: PrimeField, N: Rep3Network>(
    shared: FieldShare<F>,
    public: F,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let public = promote_to_trivial_share(io_context.id, public);
    eq_bit(shared, public, io_context)
}

/// Same as eq_bit_many but without using bit_inject on the result. Checks if a slice of shared values is equal to a slice of public values element-wise.
/// Returns a vector of shared values, where each element is 1 if the corresponding elements are equal and 0 otherwise.
pub fn eq_bit_public_many<F: PrimeField, N: Rep3Network>(
    shared: &[FieldShare<F>],
    public: &[F],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<BinaryShare<F>>> {
    if shared.len() != public.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "During execution of eq_bit_public_many: Invalid number of elements received. Length of shared : {} and length of public: {}",
                shared.len(),
                public.len()
            ),
        ));
    }
    let public = public
        .iter()
        .map(|&p| promote_to_trivial_share(io_context.id, p))
        .collect::<Vec<_>>();
    eq_bit_many(shared, &public, io_context)
}

/// Same as eq but without using bit_inject on the result. Checks whether two prime field shares are equal and return a binary share of 0 or 1. 1 means they are equal.
pub fn eq_bit<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let diff = a - b;
    let bits = conversion::a2b_selector(diff, io_context)?;
    let is_zero = binary::is_zero(&bits, io_context)?;
    Ok(is_zero)
}

/// Same as eq_many but without using bit_inject on the result. Checks whether two slice of prime field shares are equal and returns a Vec of binary shares of 0 or 1. 1 means they are equal.
pub fn eq_bit_many<F: PrimeField, N: Rep3Network>(
    a: &[FieldShare<F>],
    b: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<BinaryShare<F>>> {
    if a.len() != b.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "During execution of eq_bit_many: Invalid number of elements received. Length of a : {} and length of b: {}",
                a.len(),
                b.len()
            ),
        ));
    }
    let mut diff = Vec::with_capacity(a.len());
    for (a_, b_) in izip!(a.iter(), b.iter()) {
        diff.push(*a_ - *b_);
    }
    let bits = conversion::a2b_many(&diff, io_context)?;
    let is_zero = binary::is_zero_many(bits, io_context)?;
    Ok(is_zero)
}

/// Checks if two shared values are not equal. The result is a shared value that has value 1 if the two values are not equal and 0 otherwise.
pub fn neq<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let eq = eq(a, b, io_context)?;
    Ok(sub_public_by_shared(F::one(), eq, io_context.id))
}

/// Checks if a shared value is not equal to a public value. The result is a shared value that has value 1 if the two values are not equal and 0 otherwise.
pub fn neq_public<F: PrimeField, N: Rep3Network>(
    shared: FieldShare<F>,
    public: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let public = promote_to_trivial_share(io_context.id, public);
    neq(shared, public, io_context)
}

/// Outputs whether a shared value is zero (true) or not (false).
pub fn is_zero<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<bool> {
    let zero_share = FieldShare::default();
    let res = eq_bit(zero_share, a, io_context)?;
    let x = open_bit(res, io_context)?;
    Ok(x.is_one())
}

/// Computes `shared*2^public`. This is the same as `shared << public`.
///
/// #Panics
/// If public is larger than the bit size of the modulus of the underlying `PrimeField`.
pub fn pow_2_public<F: PrimeField>(shared: FieldShare<F>, public: F) -> FieldShare<F> {
    if public.is_zero() {
        shared
    } else {
        let shift: BigUint = public.into();
        let shift = shift.to_u32().expect("can cast shift operand to u32");
        if shift >= F::MODULUS_BIT_SIZE {
            panic!(
                "Expected left shift to be maximal {}, but was {}",
                F::MODULUS_BIT_SIZE,
                shift
            );
        } else {
            mul_public(shared, F::from(2u64).pow(public.into_bigint()))
        }
    }
}

/// computes XOR using arithmetic operations, only valid when x and y are known to be 0 or 1.
pub(crate) fn arithmetic_xor<F: PrimeField, N: Rep3Network>(
    x: Rep3PrimeFieldShare<F>,
    y: Rep3PrimeFieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    let mut d = (x * y).into_fe() + io_context.rngs.rand.masking_field_element::<F>();
    d.double_in_place();
    let e = x.a + y.a;
    let res_a = e - d;

    let res_b = io_context.network.reshare(res_a)?;
    Ok(FieldShare { a: res_a, b: res_b })
}

pub(crate) fn arithmetic_xor_many<F: PrimeField, N: Rep3Network>(
    x: &[Rep3PrimeFieldShare<F>],
    y: &[Rep3PrimeFieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3PrimeFieldShare<F>>> {
    debug_assert_eq!(x.len(), y.len());

    let mut a = Vec::with_capacity(x.len());
    for (x, y) in x.iter().zip(y.iter()) {
        let mut d = (x * y).into_fe() + io_context.rngs.rand.masking_field_element::<F>();
        d.double_in_place();
        let e = x.a + y.a;
        let res_a = e - d;
        a.push(res_a);
    }

    let b = io_context.network.reshare_many(&a)?;
    let res = a
        .into_iter()
        .zip(b)
        .map(|(a, b)| FieldShare { a, b })
        .collect();
    Ok(res)
}

pub fn generate_shares_rep3<F: PrimeField, R: Rng>(
    val: F,
    rng: &mut R,
) -> Vec<Rep3PrimeFieldShare<F>> {
    let t0 = F::rand(rng);
    let t1 = F::rand(rng);
    let t2 = val - t0 - t1;

    let p_share_0 = Rep3PrimeFieldShare::new(t0, t2); // Party 0 gets (t_0, t_2)
    let p_share_1 = Rep3PrimeFieldShare::new(t1, t0); // Party 1 gets (t_1, t_0)
    let p_share_2 = Rep3PrimeFieldShare::new(t2, t1); // Party 2 gets (t_2, t_1)
    vec![p_share_0, p_share_1, p_share_2]
}

pub fn get_mask_scalar_rep3<F: PrimeField, R: RngCore + FeedableRNG>(
    rng: &mut SSRandom<R>,
) -> (F, F) {
    let mask_share = (
        F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0),
        F::rand(&mut rng.rng_1) - F::rand(&mut rng.rng_0),
    );
    rng.update();
    mask_share
}

#[tracing::instrument(skip_all, name = "product", level = "trace")]
pub fn product<F: PrimeField, N: Rep3Network>(
    shares: &[Rep3PrimeFieldShare<F>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Rep3PrimeFieldShare<F>> {
    shares
        .iter()
        .skip(1)
        .try_fold(shares[0], |acc, x| mul(acc, *x, io_ctx))
        .context("while computing product")
}

pub fn product_into_additive<F: PrimeField + FieldExt, N: Rep3Network>(
    shares: &[Rep3PrimeFieldShare<F>],
    io_ctx: &mut IoContext<N>,
    public_extra: Option<F>,
) -> eyre::Result<AdditiveShare<F>> {
    let num_multiplications = shares.len();

    if num_multiplications == 1 {
        return Ok((shares[0] * public_extra.unwrap_or(F::one())).into_additive());
    } else if num_multiplications == 2 {
        return Ok(shares[0] * (shares[1] * public_extra.unwrap_or(F::one())));
    }

    let product_except_last = shares
        .iter()
        .skip(1)
        .take(num_multiplications - 2)
        .try_fold(shares[0] * public_extra.unwrap_or(F::one()), |acc, x| {
            mul(acc, *x, io_ctx)
        })
        .context("while computing product")?;

    Ok(product_except_last * *shares.last().unwrap())
}

pub fn product_many<F: PrimeField, N: Rep3Network, S>(
    shares: impl IntoIterator<Item = S>,
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    S: AsRef<[Rep3PrimeFieldShare<F>]>,
{
    let mut shares = shares.into_iter();
    let first_share = shares.next().unwrap();
    let first_share_ref = first_share.as_ref();
    shares
        .try_fold(
            (0..first_share_ref.len())
                .map(|i| first_share_ref[i])
                .collect_vec(),
            |acc, x| mul_vec(&acc, x.as_ref(), io_ctx),
        )
        .context("while computing product")
}

pub fn product_many_into_additive<F: PrimeField + FieldExt, N: Rep3Network>(
    shares: &[&[Rep3PrimeFieldShare<F>]],
    io_ctx: &mut IoContext<N>,
    public_extra: Option<F>,
) -> eyre::Result<Vec<AdditiveShare<F>>> {
    let num_multiplications = shares[0].len();
    if num_multiplications == 1 {
        return Ok(shares
            .iter()
            .map(|x| (x[0] * public_extra.unwrap_or(F::one())).into_additive())
            .collect());
    } else if num_multiplications == 2 {
        return Ok(shares
            .iter()
            .map(|x| x[0] * (x[1] * public_extra.unwrap_or(F::one())))
            .collect());
    }

    let products_except_last = shares
        .iter()
        .skip(1)
        .take(num_multiplications - 2)
        .try_fold(
            shares
                .iter()
                .map(|x| x[0] * public_extra.unwrap_or(F::one()))
                .collect_vec(),
            |acc, x| mul_vec(&acc, *x, io_ctx),
        )
        .context("while computing product")?;
    Ok(products_except_last
        .into_iter()
        .zip(shares.into_iter().map(|x| x.last().unwrap()))
        .map(|(a, &b)| a * b)
        .collect())
}

pub fn reshare_additive<F: PrimeField, N: Rep3Network>(
    additive: AdditiveShare<F>,
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Rep3PrimeFieldShare<F>> {
    let additive_share: F = additive.into_fe();
    let prev_share: F = io_ctx.network.reshare(additive_share)?;
    Ok(Rep3PrimeFieldShare::new(additive_share, prev_share))
}

pub fn reshare_additive_many<F: PrimeField, N: Rep3Network>(
    additive_shares: &[AdditiveShare<F>],
    io_ctx: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let additive_shares = AdditiveShare::as_fe_vec_ref(&additive_shares);
    let b_shares: Vec<F> = io_ctx.network.reshare_many(additive_shares)?;
    Ok(additive_shares
        .into_par_iter()
        .zip(b_shares.into_par_iter())
        .map(|(a, b)| Rep3PrimeFieldShare::new(*a, b))
        .collect())
}

/// Performs multiplication of a shared value and a public value.
#[inline]
pub fn mul_public_0_1_optimized<F: PrimeField>(shared: FieldShare<F>, public: F) -> FieldShare<F> {
    if public.is_zero() {
        Rep3PrimeFieldShare::zero_share()
    } else if public.is_one() {
        shared
    } else {
        Rep3PrimeFieldShare::new(
            field_mul_0_1_optimized(&shared.a, &public),
            field_mul_0_1_optimized(&shared.b, &public),
        )
    }
}

pub fn sum_batched<F: PrimeField>(
    vals: &[Vec<Rep3PrimeFieldShare<F>>],
) -> Vec<Rep3PrimeFieldShare<F>> {
    let bathes_len = vals[0].len();
    (0..bathes_len)
        .map(|i| {
            vals.iter()
                .map(|val| val[i])
                .sum::<Rep3PrimeFieldShare<F>>()
        })
        .collect()
}

/// Reconstructs a vector of field elements from its arithmetic replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_field_elements_vec<F: PrimeField>(
    shares: Vec<Vec<Rep3PrimeFieldShare<F>>>,
) -> Vec<F> {
    let [s0, s1, s2]: [Vec<_>; 3] = shares.try_into().unwrap();
    crate::protocols::rep3::combine_field_elements(&s0, &s1, &s2)
}

#[inline(always)]
pub fn field_mul_0_1_optimized<F: PrimeField>(a: &F, b: &F) -> F {
    if a.is_zero() || b.is_zero() {
        F::zero()
    } else if a.is_one() {
        *b
    } else if b.is_one() {
        *a
    } else {
        *a * *b
    }
}

/// Checks if two shared values are not equal. The result is a shared value that has value 1 if the two values are not equal and 0 otherwise.
pub fn neq_many<F: PrimeField, N: Rep3Network>(
    a: &[FieldShare<F>],
    b: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> eyre::Result<Vec<FieldShare<F>>> {
    Ok(eq_many(a, b, io_context)?
        .into_iter()
        .map(|x| sub_public_by_shared(F::one(), x, io_context.id))
        .collect())
}
