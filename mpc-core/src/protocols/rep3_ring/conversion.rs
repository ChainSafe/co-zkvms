//! Conversions
//!
//! This module contains conversions between share types

use super::{arithmetic, detail};
use crate::field::PrimeField;
use crate::preprocessing::dabits::DaBitBatch;
use crate::preprocessing::edabits::EdaBitsRingBatch;
use crate::protocols::{
    rep3::{PartyID, Rep3PrimeFieldShare},
    rep3_ring::{
        Rep3RingShare,
        ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
    },
};
use crate::{
    IoResult,
    protocols::rep3::{
            self,
            conversion::MPCType,
            network::{IoContext, Rep3Network},
        },
};
use itertools::{Itertools, izip};
use rand::{distributions::Standard, prelude::Distribution};
use std::ops::Neg;

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
    let x01 = izip!(x01_a, x01_b).map(|(a, b)| Rep3RingShare { a, b }).collect::<Vec<_>>();

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
            let k3 = io_context.rngs.bitcomp2.random_elements_3keys::<RingElement<T>>();

            res.b = (k3.0 + k3.1 + k3.2).neg();
            y.a = r;
        }
        PartyID::ID1 => {
            let k2 = io_context.rngs.bitcomp1.random_elements_3keys::<RingElement<T>>();

            res.a = (k2.0 + k2.1 + k2.2).neg();
            y.a = r;
        }
        PartyID::ID2 => {
            let k2 = io_context.rngs.bitcomp1.random_elements_3keys::<RingElement<T>>();
            let k3 = io_context.rngs.bitcomp2.random_elements_3keys::<RingElement<T>>();

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
                let k3 = io_context.rngs.bitcomp2.random_elements_3keys::<RingElement<T>>();
                res.b = (k3.0 + k3.1 + k3.2).neg();
            });
            r_vec
        }
        PartyID::ID1 => {
            res.iter_mut().for_each(|res| {
                let k2 = io_context.rngs.bitcomp1.random_elements_3keys::<RingElement<T>>();

                res.a = (k2.0 + k2.1 + k2.2).neg();
            });
            r_vec
        }
        PartyID::ID2 => izip!(res.iter_mut(), r_vec)
            .map(|(res, r)| {
                let k2 = io_context.rngs.bitcomp1.random_elements_3keys::<RingElement<T>>();
                let k3 = io_context.rngs.bitcomp2.random_elements_3keys::<RingElement<T>>();

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
    let y: Vec<_> = izip!(y_a, y_b).map(|(a, b)| Rep3RingShare { a, b }).collect();
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
            let rcv = io_context.network.recv_many::<RingElement<T>>(io_context.id.prev_id())?;
            izip!(res.iter_mut(), rcv, z).for_each(|(res, rcv, z)| {
                res.b = z.a ^ z.b ^ rcv;
            });
        }
        PartyID::ID2 => {
            let z_b = z.iter().map(|z| z.b.to_owned()).collect_vec();
            io_context.network.send_many(io_context.id.next_id(), &z_b)?;
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
            b0.iter_mut().zip_eq(&mut b2).zip_eq(x).for_each(|((b0, b2), x)| {
                b0.a = F::from(x.a.0.convert() as u64);
                b2.b = F::from(x.b.0.convert() as u64);
            });
        }
        PartyID::ID1 => {
            b1.iter_mut().zip_eq(&mut b0).zip_eq(x).for_each(|((b1, b0), x)| {
                b1.a = F::from(x.a.0.convert() as u64);
                b0.b = F::from(x.b.0.convert() as u64);
            });
        }
        PartyID::ID2 => {
            b2.iter_mut().zip_eq(&mut b1).zip_eq(x).for_each(|((b2, b1), x)| {
                b2.a = F::from(x.a.0.convert() as u64);
                b1.b = F::from(x.b.0.convert() as u64);
            });
        }
    };

    let d = rep3::arithmetic::arithmetic_xor_many(&b0, &b1, io_context)?;
    let e = rep3::arithmetic::arithmetic_xor_many(&d, &b2, io_context)?;
    Ok(e)
}

// ---------------------------------------------------------------------------
// Preprocessed conversions (moved from edabits.rs and dabits.rs)
// ---------------------------------------------------------------------------

/// Ring-domain B2A: convert binary XOR-shares to arithmetic ring shares
/// using preprocessed EdaBits (Protocol Π₂ in ring domain).
///
/// Online communication: 2 rounds (1 broadcast + 1 reshare_many).
#[tracing::instrument(skip_all, level = "trace")]
pub fn b2a_preproc_many<T: IntRing2k, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    batch: &EdaBitsRingBatch<T>,
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3RingShare<T>>>
where
    Standard: Distribution<T>,
{
    let n = x_binary.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    debug_assert_eq!(batch.gammas.len(), n);
    debug_assert_eq!(batch.alphas_flat.len(), n * T::K);

    let k = T::K;

    // Precompute powers of 2 in the ring.
    let pow2: Vec<RingElement<T>> = {
        let mut pow2 = Vec::with_capacity(k);
        let mut cur = RingElement(T::one());
        for _ in 0..k {
            pow2.push(cur);
            cur = cur + cur;
        }
        pow2
    };

    // --- Round 1: P0 broadcasts masked values ---
    let ms: Vec<RingElement<T>> = if io.id == PartyID::ID0 {
        let ms: Vec<_> = x_binary.iter().zip(&batch.gammas).map(|(x, gamma)| x.a ^ x.b ^ *gamma).collect();
        io.network.send_many(PartyID::ID1, &ms)?;
        io.network.send_many(PartyID::ID2, &ms)?;
        ms
    } else {
        io.network.recv_many(PartyID::ID0)?
    };

    // --- Local computation + masking ---
    let maskings: Vec<RingElement<T>> = (0..n).map(|_| io.rngs.rand.masking_element::<RingElement<T>>()).collect();
    let party_id = io.id;

    let s_selfs: Vec<RingElement<T>> = ms
        .iter()
        .zip(x_binary.iter())
        .zip(maskings.iter())
        .enumerate()
        .map(|(idx, ((m, x), z))| {
            if party_id == PartyID::ID0 {
                return *z;
            }

            let beta = match party_id {
                PartyID::ID0 => unreachable!(),
                PartyID::ID1 => *m ^ x.a,
                PartyID::ID2 => *m ^ x.b,
            };

            let mut v = RingElement(T::zero());
            let alpha_base = idx * k;
            for i in 0..k {
                let beta_bit = ((beta.0 >> i) & T::one()) == T::one();
                let alpha = batch.alphas_flat[alpha_base + i];
                let signed_alpha = if beta_bit { -alpha } else { alpha };
                v = v + pow2[i] * signed_alpha;
            }

            if party_id == PartyID::ID1 {
                v = v + beta;
            }

            v + *z
        })
        .collect();

    // --- Round 2: reshare ---
    let s_prevs = io.network.reshare_many(&s_selfs)?;

    Ok(s_selfs.into_iter().zip(s_prevs).map(|(s_self, s_prev)| Rep3RingShare::new_ring(s_self, s_prev)).collect())
}

/// Convert binary Rep3 bit shares to arithmetic field shares using Π₁ daBits.
///
/// **Protocol (1 round, 3 bits):**
/// 1. P0 broadcasts m₀ = b.a ⊕ b.b ⊕ γ to P1,P2
/// 2. P1 sends m₁ = θ ⊕ b.a to P0
/// 3. All compute σ, then ⟦b⟧^A = (−1)^σ · ⟦v⟧^A + ⟦β⟧^A_{G₁}
pub fn bit_inject_field_preproc_many<F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<Bit>],
    batch: &DaBitBatch<F>,
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    debug_assert_eq!(batch.gammas.len(), n);
    debug_assert_eq!(batch.thetas.len(), n);
    debug_assert_eq!(batch.v_shares.len(), n);

    let party_id = io.id;

    // --- Round 1: P0 broadcasts m₀, P1 sends m₁ ---
    let (m0s, m1s): (Vec<u8>, Vec<u8>) = match party_id {
        PartyID::ID0 => {
            let m0s: Vec<u8> = x
                .iter()
                .zip(&batch.gammas)
                .map(|(xi, &gamma)| (xi.a.0.convert() ^ xi.b.0.convert() ^ gamma) as u8)
                .collect();
            io.network.send_many(PartyID::ID1, &m0s)?;
            io.network.send_many(PartyID::ID2, &m0s)?;
            let m1s: Vec<u8> = io.network.recv_many(PartyID::ID1)?;
            (m0s, m1s)
        }
        PartyID::ID1 => {
            let m0s: Vec<u8> = io.network.recv_many(PartyID::ID0)?;
            let m1s: Vec<u8> =
                x.iter().zip(&batch.thetas).map(|(xi, &theta)| (xi.a.0.convert() ^ theta) as u8).collect();
            io.network.send_many(PartyID::ID0, &m1s)?;
            (m0s, m1s)
        }
        PartyID::ID2 => {
            let m0s: Vec<u8> = io.network.recv_many(PartyID::ID0)?;
            (m0s, vec![])
        }
    };

    // --- Local computation ---
    let results: Vec<Rep3PrimeFieldShare<F>> = match party_id {
        PartyID::ID0 => x
            .iter()
            .zip(m0s.iter())
            .zip(m1s.iter())
            .zip(batch.gammas.iter())
            .zip(batch.v_shares.iter())
            .map(|((((xi, &_m0), &m1), &gamma), v)| {
                let sigma = (m1 != 0) ^ xi.a.0.convert() ^ xi.b.0.convert() ^ gamma;
                let neg1_sigma = if sigma { -F::one() } else { F::one() };
                Rep3PrimeFieldShare::new(v.a * neg1_sigma, v.b * neg1_sigma)
            })
            .collect(),
        PartyID::ID1 => m0s
            .iter()
            .zip(x.iter())
            .zip(batch.thetas.iter())
            .zip(batch.v_shares.iter())
            .map(|(((&m0, xi), &theta), v)| {
                let beta = (m0 != 0) ^ xi.a.0.convert();
                let sigma = beta ^ theta;
                let neg1_sigma = if sigma { -F::one() } else { F::one() };
                Rep3PrimeFieldShare::new(v.a * neg1_sigma + F::from(beta as u64), v.b * neg1_sigma)
            })
            .collect(),
        PartyID::ID2 => m0s
            .iter()
            .zip(x.iter())
            .zip(batch.thetas.iter())
            .zip(batch.v_shares.iter())
            .map(|(((&m0, xi), &theta), v)| {
                let beta = (m0 != 0) ^ xi.b.0.convert();
                let sigma = beta ^ theta;
                let neg1_sigma = if sigma { -F::one() } else { F::one() };
                Rep3PrimeFieldShare::new(v.a * neg1_sigma, v.b * neg1_sigma + F::from(beta as u64))
            })
            .collect(),
    };

    Ok(results)
}

// ---------------------------------------------------------------------------
// MPCType-based selectors
// ---------------------------------------------------------------------------

/// Dispatch B2A conversion based on `io_context.mpc_type`.
///
/// - `Online` → Kogge-Stone adder (`b2a_many`)
/// - `Preprocessed` → EdaBits Protocol Π₂ (`b2a_preproc_many`)
pub fn b2a_many_selector<T: IntRing2k, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    batch: Option<&EdaBitsRingBatch<T>>,
    io_context: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3RingShare<T>>>
where
    Standard: Distribution<T>,
{
    match io_context.mpc_type {
        MPCType::Online => Ok(b2a_many(x, io_context)?),
        MPCType::Preprocessed => {
            b2a_preproc_many(x, batch.expect("b2a_many_selector: preprocessed mode requires batch"), io_context)
        }
    }
}
