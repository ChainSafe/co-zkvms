//! Binary
//!
//! This module contains operations with binary shares

use ark_ff::{One, PrimeField};
use itertools::{Itertools as _, izip};
use num_bigint::BigUint;
use rand::{rngs::StdRng, Rng, SeedableRng};
use ark_ff::BigInteger;
use itertools;

use crate::{
    IoResult,
    protocols::rep3::{
        PartyID, Rep3BigUintShare, Rep3PrimeFieldShare,
        arithmetic::{self},
        conversion,
        network::Rep3Network,
    },
};

use super::network::IoContext;
use num_traits::cast::ToPrimitive;

type ArithmeticShare<F> = Rep3PrimeFieldShare<F>;
type BinaryShare<F> = Rep3BigUintShare<F>;

/// Performs a bitwise XOR operation on two shared values.
pub fn xor<F: PrimeField>(a: &BinaryShare<F>, b: &BinaryShare<F>) -> BinaryShare<F> {
    a ^ b
}

/// Performs a bitwise XOR operation on a shared value and a public value.
pub fn xor_public<F: PrimeField>(
    shared: &BinaryShare<F>,
    public: &BigUint,
    id: PartyID,
) -> BinaryShare<F> {
    let mut res = shared.to_owned();
    match id {
        PartyID::ID0 => res.a ^= public,
        PartyID::ID1 => res.b ^= public,
        PartyID::ID2 => {}
    }
    res
}

/// Performs element-wise bitwise XOR operation on the provided public and shared values.
pub fn xor_public_vec<F: PrimeField>(
    shared: &[BinaryShare<F>],
    public: &[BigUint],
    id: PartyID,
) -> Vec<BinaryShare<F>> {
    shared
        .iter()
        .zip(public)
        .map(|(shared, public)| {
            let mut res = shared.to_owned();
            match id {
                PartyID::ID0 => res.a ^= public,
                PartyID::ID1 => res.b ^= public,
                PartyID::ID2 => {}
            }
            res
        })
        .collect()
}

/// Performs a bitwise OR operation on two shared values.
pub fn or<F: PrimeField, N: Rep3Network>(
    a: &BinaryShare<F>,
    b: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let xor = a ^ b;
    let and = and(a, b, io_context)?;
    Ok(xor ^ and)
}

/// Performs a bitwise OR operation on two shared values.
pub fn or_vec<F: PrimeField, N: Rep3Network>(
    a: &[BinaryShare<F>],
    b: &[BinaryShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3BigUintShare<F>>> {
    let xor = izip!(a, b).map(|(a, b)| a ^ b);
    let and = and_vec(a, b, io_context)?;
    Ok(izip!(xor, and).map(|(xor, and)| xor ^ and).collect())
}

/// Performs a bitwise OR operation on a shared value and a public value.
pub fn or_public<F: PrimeField>(
    shared: &BinaryShare<F>,
    public: &BigUint,
    id: PartyID,
) -> BinaryShare<F> {
    let tmp = shared & public;
    let xor = xor_public(shared, public, id);
    xor ^ tmp
}

/// Performs a bitwise AND operation on two shared values.
pub fn and<F: PrimeField, N: Rep3Network>(
    a: &BinaryShare<F>,
    b: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    debug_assert!(a.a.bits() <= u64::from(F::MODULUS_BIT_SIZE));
    debug_assert!(b.a.bits() <= u64::from(F::MODULUS_BIT_SIZE));
    let (mut mask, mask_b) = io_context
        .rngs
        .rand
        .random_biguint(usize::try_from(F::MODULUS_BIT_SIZE).expect("u32 fits into usize"));
    mask ^= mask_b;
    let local_a = (a & b) ^ mask;
    let local_b = io_context.network.reshare(local_a.clone())?;
    Ok(BinaryShare::new(local_a, local_b))
}

/// Performs element-wise bitwise AND operation on the provided shared values.
pub fn and_vec<F: PrimeField, N: Rep3Network>(
    a: &[BinaryShare<F>],
    b: &[BinaryShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<BinaryShare<F>>> {
    let local_a = izip!(a, b)
        .map(|(a, b)| {
            let (mut mask, mask_b) = io_context
                .rngs
                .rand
                .random_biguint(usize::try_from(F::MODULUS_BIT_SIZE).expect("u32 fits into usize"));

            mask ^= mask_b;
            (a & b) ^ mask
        })
        .collect_vec();
    let local_b = io_context.network.reshare(local_a.clone())?;
    Ok(izip!(local_a, local_b)
        .map(|(a, b)| BinaryShare::new(a, b))
        .collect_vec())
}

/// Performs a bitwise AND operation on a shared value and a public value.
pub fn and_with_public<F: PrimeField>(shared: &BinaryShare<F>, public: &BigUint) -> BinaryShare<F> {
    shared & public
}

/// Shifts a share by a public value `F` to the right.
///
/// # Panics
/// This method panics if `public` is larger than the of bits of
/// the underlying `PrimeField`'s modulus'.
pub fn shift_r_public<F: PrimeField>(shared: &BinaryShare<F>, public: F) -> BinaryShare<F> {
    // some special casing
    if public.is_zero() {
        return shared.to_owned();
    }
    let shift: BigUint = public.into();
    let shift = shift.to_usize().expect("can cast shift operand to usize");
    shared >> shift
}

/// Shifts a share by a public value `F` to the left.
///
/// # Panics
/// This method panics if `public` is larger than the of bits of
/// the underlying `PrimeField`'s modulus'.
pub fn shift_l_public<F: PrimeField>(shared: &BinaryShare<F>, public: F) -> BinaryShare<F> {
    // some special casing
    if public.is_zero() {
        return shared.to_owned();
    }
    let shift: BigUint = public.into();
    let shift = shift.to_usize().expect("can cast shift operand to usize");
    shared << shift
}

/// Shifts a public value `F` by a share to the left.
pub fn shift_l_public_by_shared<F: PrimeField, N: Rep3Network>(
    public: F,
    shared: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<ArithmeticShare<F>> {
    // This case is equivalent to a*2^b
    // Strategy: limit size of b to k bits
    // bit-decompose b into bits b_i

    // TODO: this sucks... we need something better here...
    let io_0 = io_context.fork()?;
    let io_1 = io_context.fork()?;
    let io_2 = io_context.fork()?;
    let io_3 = io_context.fork()?;
    let io_4 = io_context.fork()?;
    let io_5 = io_context.fork()?;
    let io_6 = io_context.fork()?;
    let io_7 = io_context.fork()?;
    let mut contexts = [io_0, io_1, io_2, io_3, io_4, io_5, io_6, io_7];
    let party_id = io_context.id;
    let mut individual_bit_shares = Vec::with_capacity(8);
    for (i, context) in izip!(0..8, contexts.iter_mut()) {
        let bit = Rep3BigUintShare::new(
            (shared.a.clone() >> i) & BigUint::one(),
            (shared.b.clone() >> i) & BigUint::one(),
        );
        individual_bit_shares.push(conversion::b2a_selector(&bit, context)?);
    }
    // v_i = 2^2^i * <b_i> + 1 - <b_i>
    let mut vs: Vec<_> = individual_bit_shares
        .into_iter()
        .enumerate()
        .map(|(i, b_i)| {
            let two = F::from(2u64);
            // i is 8 at most there `as u32` is ok
            let two_to_two_to_i = two.pow([2u64.pow(i as u32)]);
            let v = arithmetic::mul_public(b_i, two_to_two_to_i);
            let v = arithmetic::add_public(v, F::one(), party_id);
            v - b_i
        })
        .collect();

    // v = \prod v_i
    // TODO: This should be done in a multiplication tree
    let mut v = vs.pop().unwrap();
    for v_i in vs {
        v = arithmetic::mul(v, v_i, io_context)?;
    }
    // TODO could use try_fold from futures::stream
    // let last = vs.pop().unwrap();
    // let v = self.runtime.block_on(
    //     futures::stream::iter(vs.into_iter().map(|v| Ok(v)))
    //         .try_fold(last, |a, b|  move {
    //             arithmetic::mul(a, b, &mut self.io_context)
    //         }),
    // )?;
    Ok(arithmetic::mul_public(v, public))
}

pub fn add_many<F: PrimeField, N: Rep3Network>(
    a: &[BinaryShare<F>],
    b: &[BinaryShare<F>],
    bitlen: usize,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<BinaryShare<F>>> {
    super::detail::low_depth_binary_add_many(a, b, io_context, bitlen)
}

pub fn add_many_mod_p<F: PrimeField, N: Rep3Network>(
    a: &[BinaryShare<F>],
    b: &[BinaryShare<F>],
    bitlen: usize,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<BinaryShare<F>>> {
    super::detail::low_depth_binary_add_mod_p_many(a, b, io_context, bitlen)
}

/// Performs the opening of a shared value and returns the equivalent public value.
pub fn open<F: PrimeField, N: Rep3Network>(
    a: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BigUint> {
    let c = io_context.network.reshare(a.b.clone())?;
    Ok(&a.a ^ &a.b ^ c)
}

/// Transforms a public value into a shared value: \[a\] = a.
pub fn promote_to_trivial_share<F: PrimeField>(
    id: PartyID,
    public_value: &BigUint,
) -> BinaryShare<F> {
    match id {
        PartyID::ID0 => BinaryShare::new(public_value.to_owned(), BigUint::ZERO),
        PartyID::ID1 => BinaryShare::new(BigUint::ZERO, public_value.to_owned()),
        PartyID::ID2 => BinaryShare::zero_share(),
    }
}

/// Computes a CMUX: If `c` is `1`, returns `x_t`, otherwise returns `x_f`.
pub fn cmux<F: PrimeField, N: Rep3Network>(
    c: &BinaryShare<F>,
    x_t: &BinaryShare<F>,
    x_f: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let xor = x_f ^ x_t;
    let mut and = and(c, &xor, io_context)?;
    and ^= x_f;
    Ok(and)
}

/// Computes an element-wise CMUX: If `$c_i$` is `1`, returns `$x^t_i$`, otherwise returns `$x^f_i$`.
pub fn cmux_many<F: PrimeField, N: Rep3Network>(
    c: &[BinaryShare<F>],
    x_t: &[BinaryShare<F>],
    x_f: &[BinaryShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<BinaryShare<F>>> {
    assert_eq!(c.len(), x_t.len());
    assert_eq!(c.len(), x_f.len());
    let xor = izip!(x_f, x_t).map(|(x_f, x_t)| x_f ^ x_t).collect_vec();
    let mut and = and_vec(c, &xor, io_context)?;
    for (and, x_f) in izip!(and.iter_mut(), x_f) {
        *and ^= x_f;
    }
    Ok(and)
}

//TODO most likely the inputs here are only one bit therefore we
//do not have to perform an or over the whole length of prime field
//but only one bit.
//Do we want that to be configurable? Semms like a waste?
/// Compute a OR tree of the input vec
pub fn or_tree<F: PrimeField, N: Rep3Network>(
    mut inputs: Vec<BinaryShare<F>>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let mut num = inputs.len();

    tracing::debug!("starting or tree over {} elements", inputs.len());
    while num > 1 {
        tracing::trace!("binary tree still has {} elements", num);
        let mod_ = num & 1;
        num >>= 1;

        let (a_vec, tmp) = inputs.split_at(num);
        let (b_vec, leftover) = tmp.split_at(num);

        let mut res = Vec::with_capacity(num);
        // TODO WE WANT THIS BATCHED!!!
        // THIS IS SUPER BAD
        for (a, b) in izip!(a_vec.iter(), b_vec.iter()) {
            res.push(or(a, b, io_context)?);
        }

        res.extend_from_slice(leftover);
        inputs = res;

        num += mod_;
    }
    let result = inputs[0].clone();
    tracing::debug!("we did it!");
    Ok(result)
}

/// Computes a binary circuit to check whether the replicated binary-shared input x is zero or not. The output is a binary sharing of one bit.
pub fn is_zero<F: PrimeField, N: Rep3Network>(
    x: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let bit_len = F::MODULUS_BIT_SIZE as usize;
    let mask = (BigUint::from(1u64) << bit_len) - BigUint::one();

    // negate
    let mut x = x ^ &mask;

    // do ands in a tree
    // TODO: Make and tree more communication efficient, ATM we send the full element for each level, even though they halve in size
    let mut len = bit_len;
    while len > 1 {
        if len % 2 == 1 {
            len += 1;
            // pad with a 1 (= 1 xor 1 xor 1) in MSB position
            // since this is publicly known we just set the bit in each party's share and its replication
            x.a.set_bit(len as u64 - 1, true);
            x.b.set_bit(len as u64 - 1, true);
        }
        len /= 2;
        let mask = (BigUint::from(1u64) << len) - BigUint::one();
        let y = &x >> len;
        x = and(&(&x & &mask), &(&y & &mask), io_context)?;
    }
    // extract LSB
    Ok(x & BigUint::one())
}

/// Computes a binary circuit to check whether each of the replicated binary-shared inputs in the vector x is zero or not. The output is a vector of binary sharings of one bit.
pub fn is_zero_many<F: PrimeField, N: Rep3Network>(
    mut x: Vec<Rep3BigUintShare<F>>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Vec<Rep3BigUintShare<F>>> {
    let bit_len = F::MODULUS_BIT_SIZE as usize;
    let mask = (BigUint::from(1u64) << bit_len) - BigUint::one();

    // mask negate
    for x_ in x.iter_mut() {
        *x_ ^= &mask; // Negate bits
        *x_ &= &mask; // remove additional bits
    }
    let mut y = x.clone();

    // do ands in a tree
    // TODO: Make and tree more communication efficient, ATM we send the full element for each level, even though they halve in size
    let mut len = bit_len;
    while len > 1 {
        if len % 2 == 1 {
            len += 1;
            // pad with a 1 (= 1 xor 1 xor 1) in MSB position
            // since this is publicly known we just set the bit in each party's share and its replication
            for x in x.iter_mut() {
                x.a.set_bit(len as u64 - 1, true);
                x.b.set_bit(len as u64 - 1, true);
            }
        }
        len /= 2;
        let mask = (BigUint::from(1u64) << len) - BigUint::one();
        for (x_, y_) in izip!(x.iter_mut(), y.iter_mut()) {
            y_.a = (&x_.a >> len) & &mask;
            y_.b = (&x_.b >> len) & &mask;
            x_.a &= &mask;
            x_.b &= &mask;
        }
        x = and_vec(&x, &y, io_context)?;
    }
    // extract LSB
    for x_ in x.iter_mut() {
        *x_ &= BigUint::one();
    }
    Ok(x)
}

/// Converts a vector of bits in little-endian order to a share.
pub fn from_le_bits<F: PrimeField>(bits: &[BinaryShare<F>]) -> BinaryShare<F> {
    bits.iter()
        .rev()
        .fold(BinaryShare::zero_share(), |int, bit| {
            shift_l_public(&int, F::one()) ^ bit.clone()
        })
}

/// Shares a binary secret using the Rep3 protocol.
pub fn share_rep3_binary<F: PrimeField, R: Rng>(
    secret: BigUint,
    rng: &mut R,
) -> [Rep3BigUintShare<F>; 3] {
    let a1 = BigUint::from(rng.r#gen::<u64>());
    let a2 = BigUint::from(rng.r#gen::<u64>());

    let a3 = (secret ^ a1.clone()) ^ a2.clone();

    let s1 = Rep3BigUintShare::new(a1.clone(), a2.clone());
    let s2 = Rep3BigUintShare::new(a2, a3.clone());
    let s3 = Rep3BigUintShare::new(a3, a1);

    [s1, s2, s3]
}

pub fn generate_shares_rep3<F: PrimeField, R: Rng>(
    val: F,
    rng: &mut R,
) -> Vec<Rep3BigUintShare<F>> {
    let val = BigUint::from_bytes_le(&val.into_bigint().to_bytes_le());
    let t0 = BigUint::from(rng.r#gen::<u64>());
    let t1 = BigUint::from(rng.r#gen::<u64>());
    let t2 = (val ^ t0.clone()) ^ t1.clone();

    let p_share_0 = Rep3BigUintShare::new(t0.clone(), t2.clone());
    let p_share_1 = Rep3BigUintShare::new(t1.clone(), t0);
    let p_share_2 = Rep3BigUintShare::new(t2, t1);
    vec![p_share_0, p_share_1, p_share_2]
}

/// Reconstructs a vector of field elements from its binary replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_binary_elements<F: PrimeField>(
    share1: &[Rep3BigUintShare<F>],
    share2: &[Rep3BigUintShare<F>],
    share3: &[Rep3BigUintShare<F>],
) -> Vec<F> {
    assert_eq!(share1.len(), share2.len());
    assert_eq!(share2.len(), share3.len());

    itertools::multizip((share1, share2, share3))
        .map(|(x1, x2, x3)| (x1.a.clone() ^ x2.a.clone() ^ x3.a.clone()).into())
        .collect::<Vec<_>>()
}

/// Reconstructs a value (represented as [BigUint]) from its binary replicated shares. Since binary operations can lead to results >= p, the result is not guaranteed to be a valid field element.
pub fn combine_binary_element<F: PrimeField>(
    share1: Rep3BigUintShare<F>,
    share2: Rep3BigUintShare<F>,
    share3: Rep3BigUintShare<F>,
) -> F {
    (share1.a ^ share2.a ^ share3.a).into()
}

#[cfg(test)]
mod tests {
    use crate::protocols::rep3;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    use super::*;

    type F = ark_bn254::Fr;
    #[test]
    fn test_share_rep3_binary() {
        let mut rng = test_rng();
        let secret = F::rand(&mut rng);
        let shares = generate_shares_rep3(secret.clone(), &mut rng);

        let combined = combine_binary_element(shares[0], shares[1], shares[2]);
        assert_eq!(combined, secret);
    }

    #[test]
    fn test_share_rep3_binary() {
        let secret1 = BigUint::from(123u64);
        let secret2 = BigUint::from(456u64);
        let rng = &mut StdRng::from_seed(Default::default());
        let shares1 = share_rep3_binary::<ark_bn254::Fr, _>(secret1.clone(), rng);
        let shares2 = share_rep3_binary::<ark_bn254::Fr, _>(secret2.clone(), rng);
        let shares = [
            shares1[0].clone() ^ shares2[0].clone(),
            shares1[1].clone() ^ shares2[1].clone(),
            shares1[2].clone() ^ shares2[2].clone(),
        ];

        let combined =
            combine_binary_element(shares[0].clone(), shares[1].clone(), shares[2].clone());
        assert_eq!(combined, secret1 ^ secret2);
    }
}
