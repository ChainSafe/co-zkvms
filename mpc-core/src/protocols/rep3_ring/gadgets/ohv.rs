//! OHV
//!
//! This module contains some algorithms to create a random one-hot encoded vector for the Rep3 protocol.

use ark_ff::{One, Zero};
use mpc_types::field::PrimeField;
use itertools::{Itertools, izip};
use mpc_types::protocols::{
    rep3::{Rep3BigUintShare, Rep3PrimeFieldShare},
    rep3_ring::{
        Rep3RingShare,
        ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
    },
};
use rand::{distributions::Standard, prelude::Distribution};
use rayon::prelude::*;

use crate::{
    IoResult,
    protocols::{
        rep3::network::{IoContext, Rep3Network},
        rep3_ring::{
            self,
            binary::{self, pack_bits, pack_bits_many, unpack_bits, unpack_bits_many},
        },
    },
};

/// Generates a random one-hot-encoded vector of size k bits.
/// The output is (r, e), where r is a binary sharing of the index of the set bit, wheras e is a vector of size 2^k with all bits zero except at index r.
/// The algorithm is a rewrite of Protocol 5 from [https://eprint.iacr.org/2024/1317.pdf](https://eprint.iacr.org/2024/1317.pdf) for rep3.
pub fn rand_ohv<T: IntRing2k, N: Rep3Network>(
    k: usize,
    io_context: &mut IoContext<N>,
) -> IoResult<(Rep3RingShare<T>, Vec<Rep3RingShare<Bit>>)>
where
    Standard: Distribution<T>,
{
    debug_assert!(k >= 1);
    debug_assert!(k <= T::K); // Make sure datatype is large enough for bitsize
    let (mut a, mut b) = io_context.random_elements::<T>();
    if k != T::K {
        let mask = (T::one() << k) - T::one();
        a &= mask;
        b &= mask
    }

    let bits = Rep3RingShare::new(a, b);
    let e = ohv(k, bits, io_context)?;

    Ok((bits, e))
}

/// Generates a random one-hot-encoded vector of size k bits.
/// The output is (r, e), where r is a binary sharing of the index of the set bit, wheras e is a vector of size 2^k with all bits zero except at index r.
/// The algorithm is a rewrite of Protocol 5 from [https://eprint.iacr.org/2024/1317.pdf](https://eprint.iacr.org/2024/1317.pdf) for rep3.
pub fn rand_ohv_to_field<F: PrimeField, N: Rep3Network>(
    len: usize,
    io_context: &mut IoContext<N>,
) -> IoResult<(Rep3BigUintShare<F>, Vec<Rep3PrimeFieldShare<F>>)> {
    let k = len.next_power_of_two().ilog2() as usize;

    let (r, e) = if k == 1 {
        rand_ohv_to_field_inner::<Bit, _, _>(k, io_context)?
    } else if k <= 8 {
        rand_ohv_to_field_inner::<u8, _, _>(k, io_context)?
    } else if k <= 16 {
        rand_ohv_to_field_inner::<u16, _, _>(k, io_context)?
    } else if k <= 32 {
        rand_ohv_to_field_inner::<u32, _, _>(k, io_context)?
    } else {
        panic!("Table is too large")
    };

    Ok((r, e))
}

fn rand_ohv_to_field_inner<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    k: usize,
    io_context: &mut IoContext<N>,
) -> IoResult<(Rep3BigUintShare<F>, Vec<Rep3PrimeFieldShare<F>>)> {
    let (r, e) = rand_ohv::<Bit, _>(k, io_context)?;
    let r = Rep3BigUintShare::new(
        r.a.convert().cast_to_biguint(),
        r.b.convert().cast_to_biguint(),
    );
    let e = rep3_ring::conversion::bit_inject_from_bits_to_field_many(&e, io_context)?;
    Ok((r, e))
}

/// Generates a one-hot-encoded vector of size k bits from a given secret shared index which is already decomposed into shared bits.
/// The output is (r, e), where r is a binary sharing of the index of the set bit, wheras e is a vector of size 2^k with all bits zero except at index r.
/// The algorithm is a rewrite of Protocol 5 from [https://eprint.iacr.org/2024/1317.pdf](https://eprint.iacr.org/2024/1317.pdf) for rep3.
pub fn ohv<T: IntRing2k, N: Rep3Network>(
    k: usize,
    mut bits: Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3RingShare<Bit>>> {
    debug_assert!(k > 0);
    debug_assert!(k <= T::K); // Make sure datatype is large enough for bitsize

    let new_k = k - 1;
    let vk = bits.get_bit(new_k);

    if new_k == 0 {
        return Ok(vec![!vk, vk]);
    }

    let mask = (RingElement::one() << new_k) - RingElement::one();
    bits &= mask; // Remove the vk

    let mut f = ohv(new_k, bits, io_context)?; // ohv is recursively called k - 1 times
    let mut e = pack_and(&f[..f.len() - 1], &vk, io_context)?; // This has communication (2^new_k - 1 bits)
    e.push(e.iter().fold(vk, |a, b| &a ^ b));

    for (e, f) in e.iter().zip(f.iter_mut()) {
        *f ^= e;
    }
    f.extend(e);
    Ok(f)
}

/// Generates a one-hot-encoded vector of size k bits from a given secret shared index which is already decomposed into shared bits.
/// The output is (r, e), where r is a binary sharing of the index of the set bit, wheras e is a vector of size 2^k with all bits zero except at index r.
/// The algorithm is a rewrite of Protocol 5 from [https://eprint.iacr.org/2024/1317.pdf](https://eprint.iacr.org/2024/1317.pdf) for rep3.
#[tracing::instrument(skip_all, level = "trace")]
pub fn ohv_many<T: IntRing2k, N: Rep3Network>(
    k: usize,
    mut bits: Vec<Rep3RingShare<T>>,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Vec<Rep3RingShare<Bit>>>> {
    debug_assert!(k > 0);
    debug_assert!(k <= T::K); // Make sure datatype is large enough for bitsize

    let new_k = k - 1;
    let vks = bits
        .iter()
        .map(|bit| bit.get_bit(new_k))
        .collect::<Vec<_>>();

    if new_k == 0 {
        return Ok(vks.into_iter().map(|vk| vec![!vk, vk]).collect());
    }

    let mask = (RingElement::one() << new_k) - RingElement::one();
    bits.par_iter_mut().for_each(|b| *b &= mask);

    let mut f = ohv_many(new_k, bits, io_context)?; // ohv is recursively called k - 1 times
    let len = f[0].len();
    debug_assert!(!f.iter().any(|x| x.len() != len));
    let e = pack_and_many(
        f.par_iter().map(|f| &f[..len - 1]),
        &vks,
        len - 1,
        io_context,
    )?;
    e.into_par_iter()
        .zip_eq(vks)
        .map(|(mut e, vk)| {
            e.push(e.iter().fold(vk, |a, b| &a ^ b));
            e
        })
        .zip_eq(f.par_iter_mut())
        .for_each(|(e, f)| {
            izip!(e.iter(), f.iter_mut()).for_each(|(e, f)| *f ^= e);
            f.extend(e);
        });

    Ok(f)
}

fn and_pre_bit<T: IntRing2k, N: Rep3Network>(
    a: &Rep3RingShare<T>,
    b: &Rep3RingShare<Bit>,
    io_context: &mut IoContext<N>,
) -> RingElement<T>
where
    Standard: Distribution<T>,
{
    let (mut res, mask_b) = io_context.random_elements::<RingElement<T>>();
    res ^= mask_b;
    if b.a.0.convert() {
        res ^= &a.a;
        res ^= &a.b;
    }
    if b.b.0.convert() {
        res ^= &a.a;
    }
    res
}

fn pack_and<N: Rep3Network>(
    input: &[Rep3RingShare<Bit>],
    rhs: &Rep3RingShare<Bit>,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3RingShare<Bit>>> {
    let len = input.len();
    debug_assert!(len >= 1);

    if len <= 128 {
        let padded_len = len.next_power_of_two();
        let result = match padded_len {
            1 => {
                vec![binary::and(&input[0], rhs, io_context)?]
            }
            2 | 4 | 8 => {
                let packed = pack_bits::<u8>(input);
                let local_a = and_pre_bit(&packed, rhs, io_context);
                let local_b = io_context.network.reshare(local_a)?;
                unpack_bits(Rep3RingShare::new_ring(local_a, local_b), len)
            }
            16 => {
                let packed = pack_bits::<u16>(input);
                let local_a = and_pre_bit(&packed, rhs, io_context);
                let local_b = io_context.network.reshare(local_a)?;
                unpack_bits(Rep3RingShare::new_ring(local_a, local_b), len)
            }
            32 => {
                let packed = pack_bits::<u32>(input);
                let local_a = and_pre_bit(&packed, rhs, io_context);
                let local_b = io_context.network.reshare(local_a)?;
                unpack_bits(Rep3RingShare::new_ring(local_a, local_b), len)
            }
            64 => {
                let packed = pack_bits::<u64>(input);
                let local_a = and_pre_bit(&packed, rhs, io_context);
                let local_b = io_context.network.reshare(local_a)?;
                unpack_bits(Rep3RingShare::new_ring(local_a, local_b), len)
            }
            128 => {
                let packed = pack_bits::<u128>(input);
                let local_a = and_pre_bit(&packed, rhs, io_context);
                let local_b = io_context.network.reshare(local_a)?;
                unpack_bits(Rep3RingShare::new_ring(local_a, local_b), len)
            }
            _ => {
                unreachable!()
            }
        };
        Ok(result)
    } else {
        type Packtype = u64;
        const BITLEN: usize = std::mem::size_of::<Packtype>() * 8;

        let mut result = Vec::with_capacity(len);
        let mut to_send = Vec::with_capacity(len.div_ceil(BITLEN));
        for els in input.chunks(BITLEN) {
            let packed = pack_bits::<Packtype>(els);
            let u64_a = and_pre_bit(&packed, rhs, io_context);
            to_send.push(u64_a);
        }
        let received = io_context.network.reshare(to_send.to_owned())?;

        let mut remeining = len;
        for (a, b) in to_send.into_iter().zip(received) {
            let rcv = std::cmp::min(BITLEN, remeining);
            result.extend(unpack_bits(Rep3RingShare::new_ring(a, b), rcv));
            remeining -= rcv;
        }
        debug_assert_eq!(remeining, 0);
        Ok(result)
    }
}

#[tracing::instrument(skip_all, level = "trace")]
fn pack_and_many<'a, N: Rep3Network>(
    inputs: impl IntoParallelIterator<Item = &'a [Rep3RingShare<Bit>]>,
    rhs: &Vec<Rep3RingShare<Bit>>,
    len: usize,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Vec<Rep3RingShare<Bit>>>> {
    debug_assert!(len >= 1);
    if len <= 128 {
        let padded_len = len.next_power_of_two();
        let result = match padded_len {
            1 => binary::and_many(
                &inputs
                    .into_par_iter()
                    .map(|x| x[0].clone())
                    .collect::<Vec<_>>(),
                rhs,
                io_context,
            )?
            .into_iter()
            .map(|x| vec![x])
            .collect::<Vec<_>>(),
            2 | 4 | 8 => {
                let packed = pack_bits_many::<u8>(inputs);
                let local_a = and_pre_bit_many(&packed, rhs, io_context);
                let local_b = io_context.network.reshare_many(&local_a)?;
                unpack_bits_many(local_a, local_b, len)
            }
            16 => {
                // let packed = pack::<u16>(inputs);
                // let local_a = and_pre_bit(&packed, rhs, io_context);
                // let local_b = io_context.network.reshare(local_a)?;
                // unpack(Rep3RingShare::new_ring(local_a, local_b), len)
                let packed = pack_bits_many::<u16>(inputs);
                let local_a = and_pre_bit_many(&packed, rhs, io_context);
                let local_b = io_context.network.reshare_many(&local_a)?;
                unpack_bits_many(local_a, local_b, len)
            }
            32 => {
                let packed = pack_bits_many::<u32>(inputs);
                let local_a = and_pre_bit_many(&packed, rhs, io_context);
                let local_b = io_context.network.reshare_many(&local_a)?;
                unpack_bits_many(local_a, local_b, len)
            }
            64 => {
                let packed = pack_bits_many::<u64>(inputs);
                let local_a = and_pre_bit_many(&packed, rhs, io_context);
                let local_b = io_context.network.reshare_many(&local_a)?;
                unpack_bits_many(local_a, local_b, len)
            }
            128 => {
                let packed = pack_bits_many::<u128>(inputs);
                let local_a = and_pre_bit_many(&packed, rhs, io_context);
                let local_b = io_context.network.reshare_many(&local_a)?;
                unpack_bits_many(local_a, local_b, len)
            }
            _ => unreachable!(),
        };
        Ok(result)
    } else {
        type Packtype = u64;
        const BITLEN: usize = std::mem::size_of::<Packtype>() * 8;

        let mut to_send = Vec::with_capacity(len.div_ceil(BITLEN));

        let input_chunked = inputs
            .into_par_iter()
            .map(|input| input.chunks(BITLEN).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut results = vec![Vec::with_capacity(len); input_chunked.len()];

        let input_chunks = transpose(input_chunked);

        for els in input_chunks {
            let packed = pack_bits_many::<Packtype>(els);
            let u64_a = and_pre_bit_many(&packed, rhs, io_context);
            to_send.push(u64_a);
        }
        let received = io_context.network.reshare_many(&to_send)?;

        let mut remaining = len;
        izip!(to_send, received).for_each(|(a, b)| {
            let rcv = std::cmp::min(BITLEN, remaining);
            results
                .par_iter_mut()
                .zip_eq(a)
                .zip_eq(b)
                .for_each(|((result, a), b)| {
                    result.extend(unpack_bits(Rep3RingShare::new_ring(a, b), rcv));
                });
            remaining -= rcv;
        });

        debug_assert_eq!(remaining, 0);
        Ok(results)
    }
}

fn transpose<I, T>(matrix: I) -> Vec<Vec<T>>
where
    I: IntoIterator<Item = Vec<T>>,
{
    let mut it = matrix.into_iter();
    let first_row = match it.next() {
        Some(r) => r,
        None => return Vec::new(),
    };
    let cols = first_row.len();
    let (low, _) = it.size_hint();
    let mut out: Vec<Vec<T>> = (0..cols).map(|_| Vec::with_capacity(low + 1)).collect();

    // push first row
    for (c, v) in first_row.into_iter().enumerate() {
        out[c].push(v);
    }
    // push remaining rows
    for row in it {
        assert_eq!(row.len(), cols, "ragged matrix");
        for (c, v) in row.into_iter().enumerate() {
            out[c].push(v);
        }
    }
    out
}

#[inline]
fn and_pre_bit_many<T: IntRing2k, N: Rep3Network>(
    a: &[Rep3RingShare<T>],
    b: &[Rep3RingShare<Bit>],
    io_context: &mut IoContext<N>,
) -> Vec<RingElement<T>>
where
    Standard: Distribution<T>,
{
    izip!(a, b)
        .map(|(a, b)| and_pre_bit(a, b, io_context))
        .collect()
}
