//! Conversions
//!
//! This module contains conversions between share types

use crate::{IoResult, protocols::rep3::arithmetic::BinaryShare};

use super::{
    PartyID, Rep3BigUintShare, Rep3PrimeFieldShare, arithmetic, detail,
    network::{IoContext, Rep3Network},
};
use itertools::{Itertools as _, izip};
use crate::field::PrimeField;
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};

/// Selects between online and preprocessed MPC execution modes.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord, Hash,
)]
pub enum MPCType {
    /// Online MPC execution (no preprocessing required).
    #[default]
    Online,
    /// Preprocessed MPC execution (uses correlated randomness generated offline).
    Preprocessed,
}

/// Transforms the replicated shared value x from an arithmetic sharing to a binary sharing. I.e., x = x_1 + x_2 + x_3 gets transformed into x = x'_1 xor x'_2 xor x'_3.
pub fn a2b<F: PrimeField, N: Rep3Network>(
    x: Rep3PrimeFieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3BigUintShare<F>> {
    let mut x01 = Rep3BigUintShare::zero_share();
    let mut x2 = Rep3BigUintShare::zero_share();

    let (mut r, r2) = io_context
        .rngs
        .rand
        .random_biguint(F::MODULUS_BIT_SIZE as usize);
    r ^= r2;

    match io_context.id {
        PartyID::ID0 => {
            x01.a = r;
            x2.b = x.b.into_biguint();
        }
        PartyID::ID1 => {
            let val: BigUint = (x.a + x.b).into_biguint();
            x01.a = val ^ r;
        }
        PartyID::ID2 => {
            x01.a = r;
            x2.a = x.a.into_biguint();
        }
    }

    // reshare x01
    io_context.network.send_next(x01.a.to_owned())?;
    let local_b = io_context.network.recv_prev()?;
    x01.b = local_b;

    detail::low_depth_binary_add_mod_p::<F, N>(&x01, &x2, io_context, F::MODULUS_BIT_SIZE as usize)
}

/// Transforms the provided replicated shared values from an arithmetic sharing to a binary sharing. I.e., x = x_1 + x_2 + x_3 gets transformed into x = x'_1 xor x'_2 xor x'_3. Reduces the mul-depth by batching all elements together.
#[tracing::instrument(skip_all, level = "trace")]
pub fn a2b_many<F: PrimeField, N: Rep3Network>(
    x: &[Rep3PrimeFieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3BigUintShare<F>>> {
    let mut x2 = vec![BinaryShare::<F>::zero_share(); x.len()];

    let mut r_vec = Vec::with_capacity(x.len());
    for _ in 0..x.len() {
        let (mut r, r2) = io_context
            .rngs
            .rand
            .random_biguint(F::MODULUS_BIT_SIZE as usize);
        r ^= &r2;
        r_vec.push(r);
    }

    let x01_a = match io_context.id {
        PartyID::ID0 => {
            for (x2, x) in izip!(x2.iter_mut(), x) {
                x2.b = x.b.into_biguint();
            }
            r_vec
        }

        PartyID::ID1 => izip!(x, r_vec)
            .map(|(x, r)| {
                let tmp: BigUint = (x.a + x.b).into_biguint();
                tmp ^ r
            })
            .collect(),
        PartyID::ID2 => {
            for (x2, x) in izip!(x2.iter_mut(), x) {
                x2.a = x.a.into_biguint();
            }
            r_vec
        }
    };

    // reshare x01
    let x01_b = io_context.network.reshare_many(&x01_a)?;
    let x01 = izip!(x01_a, x01_b)
        .map(|(a, b)| BinaryShare::new(a, b))
        .collect_vec();

    detail::low_depth_binary_add_mod_p_many::<F, N>(
        &x01,
        &x2,
        io_context,
        F::MODULUS_BIT_SIZE as usize,
    )
}

/// Transforms the replicated shared value x from a binary sharing to an arithmetic sharing. I.e., x = x_1 xor x_2 xor x_3 gets transformed into x = x'_1 + x'_2 + x'_3. This implementation currently works only for a binary sharing of a valid field element, i.e., x = x_1 xor x_2 xor x_3 < p.
///
/// Keep in mind: Only works if the input is actually a binary sharing of a valid field element
/// If the input has the correct number of bits, but is >= P, then either x can be reduced with self.low_depth_sub_p_cmux(x) first, or self.low_depth_binary_add_2_mod_p(x, y) is extended to subtract 2P in parallel as well. The second solution requires another multiplexer in the end.
pub fn b2a<F: PrimeField, N: Rep3Network>(
    x: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    let mut y = Rep3BigUintShare::zero_share();
    let mut res = Rep3PrimeFieldShare::zero_share();

    let (mut r, r2) = io_context
        .rngs
        .rand
        .random_biguint(F::MODULUS_BIT_SIZE as usize);
    r ^= r2;

    match io_context.id {
        PartyID::ID0 => {
            let k3 = io_context.rngs.bitcomp2.random_fes_3keys::<F>();

            res.b = (k3.0 + k3.1 + k3.2).neg();
            y.a = r;
        }
        PartyID::ID1 => {
            let k2 = io_context.rngs.bitcomp1.random_fes_3keys::<F>();

            res.a = (k2.0 + k2.1 + k2.2).neg();
            y.a = r;
        }
        PartyID::ID2 => {
            let k2 = io_context.rngs.bitcomp1.random_fes_3keys::<F>();
            let k3 = io_context.rngs.bitcomp2.random_fes_3keys::<F>();

            let k2_comp = k2.0 + k2.1 + k2.2;
            let k3_comp = k3.0 + k3.1 + k3.2;
            let val: BigUint = (k2_comp + k3_comp).into_biguint();
            y.a = val ^ r;
            res.a = k3_comp.neg();
            res.b = k2_comp.neg();
        }
    }

    // reshare y
    io_context.network.send_next(y.a.to_owned())?;
    let local_b = io_context.network.recv_prev()?;
    y.b = local_b;

    let z = detail::low_depth_binary_add_mod_p::<F, N>(
        x,
        &y,
        io_context,
        F::MODULUS_BIT_SIZE as usize,
    )?;

    match io_context.id {
        PartyID::ID0 => {
            io_context.network.send_next(z.b.to_owned())?;
            let rcv: BigUint = io_context.network.recv_prev()?;
            res.a = (z.a ^ z.b ^ rcv).into();
        }
        PartyID::ID1 => {
            let rcv: BigUint = io_context.network.recv_prev()?;
            res.b = (z.a ^ z.b ^ rcv).into();
        }
        PartyID::ID2 => {
            io_context.network.send_next(z.b)?;
        }
    }
    Ok(res)
}

/// Transforms the replicated shared value x from a binary sharing to an arithmetic sharing. I.e., x = x_1 xor x_2 xor x_3 gets transformed into x = x'_1 + x'_2 + x'_3. This implementation currently works only for a binary sharing of a valid field element, i.e., x = x_1 xor x_2 xor x_3 < p.
///
/// Keep in mind: Only works if the input is actually a binary sharing of a valid field element
/// If the input has the correct number of bits, but is >= P, then either x can be reduced with self.low_depth_sub_p_cmux(x) first, or self.low_depth_binary_add_2_mod_p(x, y) is extended to subtract 2P in parallel as well. The second solution requires another multiplexer in the end.
#[tracing::instrument(skip_all, level = "trace")]
pub fn b2a_many<'a, F: PrimeField, N: Rep3Network>(
    x: impl IntoIterator<Item = &'a Rep3BigUintShare<F>, IntoIter: ExactSizeIterator>,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3PrimeFieldShare<F>>> {
    let x = x.into_iter();
    let mut res = vec![Rep3PrimeFieldShare::zero_share(); x.len()];

    let mut r_vec = Vec::with_capacity(x.len());
    for _ in 0..x.len() {
        let (mut r, r2) = io_context
            .rngs
            .rand
            .random_biguint(F::MODULUS_BIT_SIZE as usize);
        r ^= &r2;
        r_vec.push(r);
    }

    let y_a = match io_context.id {
        PartyID::ID0 => {
            res.iter_mut().for_each(|res| {
                let k3 = io_context.rngs.bitcomp2.random_fes_3keys::<F>();

                res.b = (k3.0 + k3.1 + k3.2).neg();
            });
            r_vec
        }
        PartyID::ID1 => {
            res.iter_mut().for_each(|res| {
                let k2 = io_context.rngs.bitcomp1.random_fes_3keys::<F>();

                res.a = (k2.0 + k2.1 + k2.2).neg();
            });
            r_vec
        }
        PartyID::ID2 => izip!(res.iter_mut(), r_vec)
            .map(|(res, r)| {
                let k2 = io_context.rngs.bitcomp1.random_fes_3keys::<F>();
                let k3 = io_context.rngs.bitcomp2.random_fes_3keys::<F>();

                let k2_comp = k2.0 + k2.1 + k2.2;
                let k3_comp = k3.0 + k3.1 + k3.2;
                let val: BigUint = (k2_comp + k3_comp).into_biguint();

                res.a = k3_comp.neg();
                res.b = k2_comp.neg();
                val ^ r
            })
            .collect(),
    };

    // reshare y
    let y_b: Vec<_> = io_context.network.reshare_many(&y_a)?;
    let y: Vec<_> = izip!(y_a, y_b)
        .map(|(a, b)| BinaryShare::new(a, b))
        .collect();

    let z = detail::low_depth_binary_add_mod_p_many::<F, N>(
        x,
        &y,
        io_context,
        F::MODULUS_BIT_SIZE as usize,
    )?;

    match io_context.id {
        PartyID::ID0 => {
            let z_b = z.iter().map(|z| z.b.to_owned()).collect_vec();
            let rcv = io_context.network.reshare_many(&z_b)?;
            izip!(res.iter_mut(), rcv, z).for_each(|(res, rcv, z)| {
                res.a = (z.a ^ z.b ^ rcv).into();
            });
        }
        PartyID::ID1 => {
            let rcv = io_context
                .network
                .recv_many::<BigUint>(io_context.id.prev_id())?;
            izip!(res.iter_mut(), rcv, z).for_each(|(res, rcv, z)| {
                res.b = (z.a ^ z.b ^ rcv).into();
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

/// Translates one shared bits into an arithmetic sharing of the same bit. I.e., the shared bit x = x_1 xor x_2 xor x_3 gets transformed into x = x'_1 + x'_2 + x'_3, with x being either 0 or 1.
pub fn bit_inject<F: PrimeField, N: Rep3Network>(
    x: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    // standard bit inject
    assert!(x.a.bits() <= 1);

    let mut b0 = Rep3PrimeFieldShare::<F>::default();
    let mut b1 = Rep3PrimeFieldShare::<F>::default();
    let mut b2 = Rep3PrimeFieldShare::<F>::default();

    match io_context.id {
        PartyID::ID0 => {
            b0.a = x.a.to_owned().into();
            b2.b = x.b.to_owned().into();
        }
        PartyID::ID1 => {
            b1.a = x.a.to_owned().into();
            b0.b = x.b.to_owned().into();
        }
        PartyID::ID2 => {
            b2.a = x.a.to_owned().into();
            b1.b = x.b.to_owned().into();
        }
    };

    let d = arithmetic::arithmetic_xor(b0, b1, io_context)?;
    let e = arithmetic::arithmetic_xor(d, b2, io_context)?;
    Ok(e)
}

/// Translates a vector of shared bit into a vector of arithmetic sharings of the same bits. See [bit_inject] for details.
pub fn bit_inject_many<F: PrimeField, N: Rep3Network>(
    x: &[Rep3BigUintShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<Rep3PrimeFieldShare<F>>> {
    // standard bit inject
    assert!(x.iter().all(|a| a.a.bits() <= 1));

    let mut b0 = vec![Rep3PrimeFieldShare::<F>::default(); x.len()];
    let mut b1 = vec![Rep3PrimeFieldShare::<F>::default(); x.len()];
    let mut b2 = vec![Rep3PrimeFieldShare::<F>::default(); x.len()];

    match io_context.id {
        PartyID::ID0 => {
            for (b0, b2, x) in izip!(&mut b0, &mut b2, x.iter().cloned()) {
                b0.a = x.a.into();
                b2.b = x.b.into();
            }
        }
        PartyID::ID1 => {
            for (b1, b0, x) in izip!(&mut b1, &mut b0, x.iter().cloned()) {
                b1.a = x.a.into();
                b0.b = x.b.into();
            }
        }
        PartyID::ID2 => {
            for (b2, b1, x) in izip!(&mut b2, &mut b1, x.iter().cloned()) {
                b2.a = x.a.into();
                b1.b = x.b.into();
            }
        }
    };

    let d = arithmetic::arithmetic_xor_many(&b0, &b1, io_context)?;
    let e = arithmetic::arithmetic_xor_many(&d, &b2, io_context)?;
    Ok(e)
}
