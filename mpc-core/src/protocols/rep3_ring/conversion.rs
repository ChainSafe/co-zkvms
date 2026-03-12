//! Conversions
//!
//! This module contains conversions between share types

use super::{detail, arithmetic};
use crate::{
    IoResult,
    protocols::{
        rep3::{
            self,
            conversion::A2BType,
            network::{IoContext, Rep3Network},
        },
        rep3_ring,
    },
};
use itertools::{Itertools, izip};
use crate::field::PrimeField;
use crate::protocols::{
    rep3::{Rep3PrimeFieldShare, id::PartyID},
    rep3_ring::{
        Rep3RingShare,
        ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
    },
};
use rand::{distributions::Standard, prelude::Distribution};
use std::ops::Neg;

use rayon::prelude::*;

/// Depending on the `A2BType` of the io_context, this function selects the appropriate implementation for the arithmetic-to-binary conversion.
pub fn a2b_selector<T: IntRing2k, N: Rep3Network>(
    x: Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3RingShare<T>>
where
    Standard: Distribution<T>,
{
    match io_context.a2b_type {
        A2BType::Direct => a2b(x, io_context),
    }
}

/// Depending on the `A2BType` of the io_context, this function selects the appropriate implementation for the binary-to-arithmetic conversion.
pub fn b2a_selector<T: IntRing2k, N: Rep3Network>(
    x: &Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> std::io::Result<Rep3RingShare<T>>
where
    Standard: Distribution<T>,
{
    match io_context.a2b_type {
        A2BType::Direct => b2a(x, io_context),
    }
}

/// Transforms the replicated shared value x from an arithmetic sharing to a binary sharing. I.e., x = x_1 + x_2 + x_3 gets transformed into x = x'_1 xor x'_2 xor x'_3.
pub fn a2b<T: IntRing2k, N: Rep3Network>(
    x: Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3RingShare<T>>
where
    Standard: Distribution<T>,
{
    let mut x01 = Rep3RingShare::zero_share();
    let mut x2 = Rep3RingShare::zero_share();

    let (mut r, r2) = io_context.rngs.rand.random_elements::<RingElement<T>>();
    r ^= r2;

    match io_context.id {
        PartyID::ID0 => {
            x01.a = r;
            x2.b = x.b;
        }
        PartyID::ID1 => {
            let val = x.a + x.b;
            x01.a = val ^ r;
        }
        PartyID::ID2 => {
            x01.a = r;
            x2.a = x.a;
        }
    }

    // reshare x01
    io_context.network.send_next(x01.a.to_owned())?;
    let local_b = io_context.network.recv_prev()?;
    x01.b = local_b;

    detail::low_depth_binary_add(&x01, &x2, io_context)
}

/// Transforms the replicated shared value x from an arithmetic sharing to a binary sharing. I.e., x = x_1 + x_2 + x_3 gets transformed into x = x'_1 xor x'_2 xor x'_3.
#[tracing::instrument(skip_all, level = "trace")]
pub fn a2b_many<T: IntRing2k, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3RingShare<T>>>
where
    Standard: Distribution<T>,
{
    let mut x2 = vec![Rep3RingShare::zero_share(); x.len()];

    let mut r_vec = Vec::with_capacity(x.len());
    for _ in 0..x.len() {
        let (mut r, r2) = io_context.rngs.rand.random_elements::<RingElement<T>>();
        r ^= r2;
        r_vec.push(r);
    }

    let x01_a = match io_context.id {
        PartyID::ID0 => {
            for (x2, x) in izip!(x2.iter_mut(), x) {
                x2.b = x.b;
            }
            r_vec
        }

        PartyID::ID1 => izip!(x, r_vec)
            .map(|(x, r)| {
                let val = x.a + x.b;
                val ^ r
            })
            .collect(),
        PartyID::ID2 => {
            for (x2, x) in izip!(x2.iter_mut(), x) {
                x2.a = x.a;
            }
            r_vec
        }
    };

    // reshare x01
    let x01_b = io_context.network.reshare_many(&x01_a)?;
    let x01 = izip!(x01_a, x01_b)
        .map(|(a, b)| Rep3RingShare { a, b })
        .collect::<Vec<_>>();

    detail::low_depth_binary_add_many(&x01, &x2, io_context)
}

/// Transforms the replicated shared value x from a binary sharing to an arithmetic sharing. I.e., x = x_1 xor x_2 xor x_3 gets transformed into x = x'_1 + x'_2 + x'_3.
pub fn b2a<T: IntRing2k, N: Rep3Network>(
    x: &Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3RingShare<T>>
where
    Standard: Distribution<T>,
{
    let mut y = Rep3RingShare::zero_share();
    let mut res = Rep3RingShare::zero_share();

    let (mut r, r2) = io_context.rngs.rand.random_elements::<RingElement<T>>();
    r ^= r2;

    match io_context.id {
        PartyID::ID0 => {
            let k3 = io_context
                .rngs
                .bitcomp2
                .random_elements_3keys::<RingElement<T>>();

            res.b = (k3.0 + k3.1 + k3.2).neg();
            y.a = r;
        }
        PartyID::ID1 => {
            let k2 = io_context
                .rngs
                .bitcomp1
                .random_elements_3keys::<RingElement<T>>();

            res.a = (k2.0 + k2.1 + k2.2).neg();
            y.a = r;
        }
        PartyID::ID2 => {
            let k2 = io_context
                .rngs
                .bitcomp1
                .random_elements_3keys::<RingElement<T>>();
            let k3 = io_context
                .rngs
                .bitcomp2
                .random_elements_3keys::<RingElement<T>>();

            let k2_comp = k2.0 + k2.1 + k2.2;
            let k3_comp = k3.0 + k3.1 + k3.2;
            let val = k2_comp + k3_comp;
            y.a = val ^ r;
            res.a = k3_comp.neg();
            res.b = k2_comp.neg();
        }
    }

    // reshare y
    io_context.network.send_next(y.a.to_owned())?;
    let local_b = io_context.network.recv_prev()?;
    y.b = local_b;

    let z = detail::low_depth_binary_add(x, &y, io_context)?;

    match io_context.id {
        PartyID::ID0 => {
            io_context.network.send_next(z.b.to_owned())?;
            let rcv: RingElement<T> = io_context.network.recv_prev()?;
            res.a = z.a ^ z.b ^ rcv;
        }
        PartyID::ID1 => {
            let rcv: RingElement<T> = io_context.network.recv_prev()?;
            res.b = z.a ^ z.b ^ rcv;
        }
        PartyID::ID2 => {
            io_context.network.send_next(z.b)?;
        }
    }
    Ok(res)
}

/// Transforms the replicated shared value x from a binary sharing to an arithmetic sharing. I.e., x = x_1 xor x_2 xor x_3 gets transformed into x = x'_1 + x'_2 + x'_3.
#[tracing::instrument(skip_all, level = "trace")]
pub fn b2a_many<T: IntRing2k, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3RingShare<T>>>
where
    Standard: Distribution<T>,
{
    let mut res = vec![Rep3RingShare::zero_share(); x.len()];

    let r_vec = (0..x.len())
        .map(|_| {
            let (r1, r2) = io_context.rngs.rand.random_elements::<RingElement<T>>();
            r1 ^ r2
        })
        .collect::<Vec<_>>();

    let y_a = match io_context.id {
        PartyID::ID0 => {
            res.iter_mut().for_each(|res| {
                let k3 = io_context
                    .rngs
                    .bitcomp2
                    .random_elements_3keys::<RingElement<T>>();
                res.b = (k3.0 + k3.1 + k3.2).neg();
            });
            r_vec
        }
        PartyID::ID1 => {
            res.iter_mut().for_each(|res| {
                let k2 = io_context
                    .rngs
                    .bitcomp1
                    .random_elements_3keys::<RingElement<T>>();

                res.a = (k2.0 + k2.1 + k2.2).neg();
            });
            r_vec
        }
        PartyID::ID2 => izip!(res.iter_mut(), r_vec)
            .map(|(res, r)| {
                let k2 = io_context
                    .rngs
                    .bitcomp1
                    .random_elements_3keys::<RingElement<T>>();
                let k3 = io_context
                    .rngs
                    .bitcomp2
                    .random_elements_3keys::<RingElement<T>>();

                let k2_comp = k2.0 + k2.1 + k2.2;
                let k3_comp = k3.0 + k3.1 + k3.2;
                let val = k2_comp + k3_comp;
                res.a = k3_comp.neg();
                res.b = k2_comp.neg();
                val ^ r
            })
            .collect(),
    };

    // reshare y
    let y_b = io_context.network.reshare_many(&y_a)?;
    let y: Vec<_> = izip!(y_a, y_b)
        .map(|(a, b)| Rep3RingShare { a, b })
        .collect();
    let z = detail::low_depth_binary_add_many(x, &y, io_context)?;

    match io_context.id {
        PartyID::ID0 => {
            let z_b = z.iter().map(|z| z.b.to_owned()).collect_vec();
            let rcv = io_context.network.reshare_many(&z_b)?;
            izip!(res.iter_mut(), rcv, z).for_each(|(res, rcv, z)| {
                res.a = z.a ^ z.b ^ rcv;
            });
        }
        PartyID::ID1 => {
            let rcv = io_context
                .network
                .recv_many::<RingElement<T>>(io_context.id.prev_id())?;
            izip!(res.iter_mut(), rcv, z).for_each(|(res, rcv, z)| {
                res.b = z.a ^ z.b ^ rcv;
            });
        }
        PartyID::ID2 => {
            let z_b = z.iter().map(|z| z.b.to_owned()).collect_vec();
            io_context
                .network
                .send_many(io_context.id.next_id(), &z_b)?;
        }
    }
    Ok(res)
}

/// Translates one shared bit into an arithmetic sharing of the same bit. I.e., the shared bit x = x_1 xor x_2 xor x_3 gets transformed into x = x'_1 + x'_2 + x'_3, with x being either 0 or 1.
pub fn bit_inject<T: IntRing2k, N: Rep3Network>(
    x: &Rep3RingShare<T>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3RingShare<T>>
where
    Standard: Distribution<T>,
{
    // standard bit inject
    assert!(x.a.bits() <= 1);

    let mut b0 = Rep3RingShare::default();
    let mut b1 = Rep3RingShare::default();
    let mut b2 = Rep3RingShare::default();

    match io_context.id {
        PartyID::ID0 => {
            b0.a = x.a.to_owned();
            b2.b = x.b.to_owned();
        }
        PartyID::ID1 => {
            b1.a = x.a.to_owned();
            b0.b = x.b.to_owned();
        }
        PartyID::ID2 => {
            b2.a = x.a.to_owned();
            b1.b = x.b.to_owned();
        }
    };

    let d = arithmetic::arithmetic_xor(b0, b1, io_context)?;
    let e = arithmetic::arithmetic_xor(d, b2, io_context)?;
    Ok(e)
}

/// Translates a vector of shared bits into a vector of arithmetic sharings of the same bits. See [bit_inject] for details.
pub fn bit_inject_many<T: IntRing2k, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3RingShare<T>>>
where
    Standard: Distribution<T>,
{
    // standard bit inject
    assert!(x.iter().all(|a| a.a.bits() <= 1));

    let mut b0 = vec![Rep3RingShare::default(); x.len()];
    let mut b1 = vec![Rep3RingShare::default(); x.len()];
    let mut b2 = vec![Rep3RingShare::default(); x.len()];

    match io_context.id {
        PartyID::ID0 => {
            for (b0, b2, x) in izip!(&mut b0, &mut b2, x.iter().cloned()) {
                b0.a = x.a;
                b2.b = x.b;
            }
        }
        PartyID::ID1 => {
            for (b1, b0, x) in izip!(&mut b1, &mut b0, x.iter().cloned()) {
                b1.a = x.a;
                b0.b = x.b;
            }
        }
        PartyID::ID2 => {
            for (b2, b1, x) in izip!(&mut b2, &mut b1, x.iter().cloned()) {
                b2.a = x.a;
                b1.b = x.b;
            }
        }
    };

    let d = arithmetic::arithmetic_xor_many(&b0, &b1, io_context)?;
    let e = arithmetic::arithmetic_xor_many(&d, &b2, io_context)?;
    Ok(e)
}

/// Translates one shared bit into an arithmetic sharing of the same bit. I.e., the shared bit x = x_1 xor x_2 xor x_3 gets transformed into x = x'_1 + x'_2 + x'_3, with x being either 0 or 1.
pub fn bit_inject_from_bit<T: IntRing2k, N: Rep3Network>(
    x: &Rep3RingShare<Bit>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3RingShare<T>>
where
    Standard: Distribution<T>,
{
    // standard bit inject

    let mut b0 = Rep3RingShare::default();
    let mut b1 = Rep3RingShare::default();
    let mut b2 = Rep3RingShare::default();

    match io_context.id {
        PartyID::ID0 => {
            b0.a = RingElement(T::from(x.a.0.convert()));
            b2.b = RingElement(T::from(x.b.0.convert()));
        }
        PartyID::ID1 => {
            b1.a = RingElement(T::from(x.a.0.convert()));
            b0.b = RingElement(T::from(x.b.0.convert()));
        }
        PartyID::ID2 => {
            b2.a = RingElement(T::from(x.a.0.convert()));
            b1.b = RingElement(T::from(x.b.0.convert()));
        }
    };

    let d = arithmetic::arithmetic_xor(b0, b1, io_context)?;
    let e = arithmetic::arithmetic_xor(d, b2, io_context)?;
    Ok(e)
}

/// Translates a vector of shared bits into a vector of arithmetic sharings of the same bits. See [bit_inject] for details.
#[tracing::instrument(skip_all, level = "trace")]
pub fn bit_inject_from_bits_many<T: IntRing2k, N: Rep3Network>(
    x: &[Rep3RingShare<Bit>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3RingShare<T>>>
where
    Standard: Distribution<T>,
{
    let mut b0 = vec![Rep3RingShare::default(); x.len()];
    let mut b1 = vec![Rep3RingShare::default(); x.len()];
    let mut b2 = vec![Rep3RingShare::default(); x.len()];

    match io_context.id {
        PartyID::ID0 => {
            for (b0, b2, x) in izip!(&mut b0, &mut b2, x.iter().cloned()) {
                b0.a = RingElement(T::from(x.a.0.convert()));
                b2.b = RingElement(T::from(x.b.0.convert()));
            }
        }
        PartyID::ID1 => {
            for (b1, b0, x) in izip!(&mut b1, &mut b0, x.iter().cloned()) {
                b1.a = RingElement(T::from(x.a.0.convert()));
                b0.b = RingElement(T::from(x.b.0.convert()));
            }
        }
        PartyID::ID2 => {
            for (b2, b1, x) in izip!(&mut b2, &mut b1, x.iter().cloned()) {
                b2.a = RingElement(T::from(x.a.0.convert()));
                b1.b = RingElement(T::from(x.b.0.convert()));
            }
        }
    };

    let d = arithmetic::arithmetic_xor_many(&b0, &b1, io_context)?;
    let r = arithmetic::arithmetic_xor_many(&d, &b2, io_context)?;
    Ok(r)
}

/// Translates a vector of shared bits into a vector of arithmetic sharings of the same bits. See [bit_inject] for details.
#[tracing::instrument(skip_all, level = "trace")]
pub fn bit_inject_from_bits_to_field_many<F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<Bit>], // TODO: impl IntoParallelIterator
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3PrimeFieldShare<F>>> {
    let mut b0 = vec![Rep3PrimeFieldShare::default(); x.len()];
    let mut b1 = vec![Rep3PrimeFieldShare::default(); x.len()];
    let mut b2 = vec![Rep3PrimeFieldShare::default(); x.len()];

    match io_context.id {
        PartyID::ID0 => {
            b0.iter_mut()
                .zip_eq(&mut b2)
                .zip_eq(x)
                .for_each(|((b0, b2), x)| {
                    b0.a = F::from(x.a.0.convert() as u64);
                    b2.b = F::from(x.b.0.convert() as u64);
                });
        }
        PartyID::ID1 => {
            b1.iter_mut()
                .zip_eq(&mut b0)
                .zip_eq(x)
                .for_each(|((b1, b0), x)| {
                    b1.a = F::from(x.a.0.convert() as u64);
                    b0.b = F::from(x.b.0.convert() as u64);
                });
        }
        PartyID::ID2 => {
            b2.iter_mut()
                .zip_eq(&mut b1)
                .zip_eq(x)
                .for_each(|((b2, b1), x)| {
                    b2.a = F::from(x.a.0.convert() as u64);
                    b1.b = F::from(x.b.0.convert() as u64);
                });
        }
    };

    let d = rep3::arithmetic::arithmetic_xor_many(&b0, &b1, io_context)?;
    let e = rep3::arithmetic::arithmetic_xor_many(&d, &b2, io_context)?;
    Ok(e)
}
