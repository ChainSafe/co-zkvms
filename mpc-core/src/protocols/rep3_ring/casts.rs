//! Casts
//!
//! Implements casts for sharings of different datatypes

use super::conversion;
use crate::field::PrimeField;
use crate::preprocessing::edabits::{DaRing, EdaBits, EdaBitsBatch, EdaBitsBatchRef};
use crate::protocols::rep3::{
    self, PartyID, Rep3PrimeFieldShare, arithmetic as rep3_arith,
    conversion::MPCType,
    network::{IoContext, Rep3Network},
};
use crate::protocols::{
    rep3::Rep3BigUintShare,
    rep3_ring::{
        Rep3RingShare, Rep3RingSignedShare, arithmetic as rep3_ring_arith,
        ring::{bit::Bit, int_ring::IntRing2k, ring_impl::RingElement},
    },
};
use num_bigint::BigUint;
use num_traits::AsPrimitive;
use rand::{distributions::Standard, prelude::Distribution};
use rayon::prelude::*;
use std::any::TypeId;

pub struct R2fB2aScratch<T: IntRing2k, F: PrimeField> {
    ms: Vec<RingElement<T>>,
    maskings: Vec<F>,
    s_selfs: Vec<F>,
    out: Vec<Rep3PrimeFieldShare<F>>,
    pow2: Vec<F>,
}

impl<T: IntRing2k, F: PrimeField> Default for R2fB2aScratch<T, F> {
    fn default() -> Self {
        Self { ms: Vec::new(), maskings: Vec::new(), s_selfs: Vec::new(), out: Vec::new(), pow2: Vec::new() }
    }
}

impl<T: IntRing2k, F: PrimeField> R2fB2aScratch<T, F> {
    fn ensure_pow2(&mut self) {
        if self.pow2.len() == T::K {
            return;
        }
        self.pow2.clear();
        self.pow2.reserve(T::K);
        let mut cur = F::one();
        for _ in 0..T::K {
            self.pow2.push(cur);
            cur = cur + cur;
        }
    }

    pub fn output(&self) -> &[Rep3PrimeFieldShare<F>] {
        &self.out
    }
}

/// A downcast of a Rep3RingShare from a larger ring to a smaller ring, truncating the excess bits.
/// Does not require network interaction
pub fn downcast<T, U>(share: Rep3RingShare<T>) -> Rep3RingShare<U>
where
    T: IntRing2k + AsPrimitive<U>,
    U: IntRing2k,
{
    assert!(T::K >= U::K);

    Rep3RingShare { a: RingElement(share.a.0.as_()), b: RingElement(share.b.0.as_()) }
}

/// An upcast of a Rep3RingShare from a smaller ring to a larger ring
/// Does require network interaction
pub fn upcast_a2b<T, U, N>(share: Rep3RingShare<T>, io_context: &mut IoContext<N>) -> std::io::Result<Rep3RingShare<U>>
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
    let binary = Rep3RingShare { a: RingElement(binary.a.0.as_()), b: RingElement(binary.b.0.as_()) };
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
        .map(|s| Rep3RingShare { a: RingElement(s.a.0.as_()), b: RingElement(s.b.0.as_()) })
        .collect::<Vec<_>>();

    conversion::b2a_many(&binary_upcasted, io_context)
}

/// A cast of a Rep3RingShare from a ring to another ring. In case of a downcast, the excess bits are just truncated.
pub fn cast_a2b<T, U, N>(share: Rep3RingShare<T>, io_context: &mut IoContext<N>) -> std::io::Result<Rep3RingShare<U>>
where
    T: IntRing2k + AsPrimitive<U>,
    U: IntRing2k,
    N: Rep3Network,
    Standard: Distribution<T> + Distribution<U>,
{
    if T::K >= U::K { Ok(downcast(share)) } else { upcast_a2b(share, io_context) }
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
        let share = crate::downcast::<_, Rep3RingShare<Bit>>(&share).expect("We already checked types");
        let biguint_share =
            Rep3BigUintShare::new(BigUint::from(share.a.0.convert() as u64), BigUint::from(share.b.0.convert() as u64));

        return rep3::conversion::bit_inject(&biguint_share, io_context);
    }

    let binary = conversion::a2b(share, io_context)?;
    let biguint_share = Rep3BigUintShare::new(T::cast_to_biguint(&binary.a.0), T::cast_to_biguint(&binary.b.0));
    rep3::conversion::b2a(&biguint_share, io_context)
}

/// A cast of a Rep3RingShare to a Rep3PrimeFieldShare
#[tracing::instrument(skip_all, level = "trace")]
pub fn r2f_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
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
                let share = crate::downcast::<_, Rep3RingShare<Bit>>(&share).expect("We already checked types");
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
        .map(|binary| Rep3BigUintShare::new(T::cast_to_biguint(&binary.a.0), T::cast_to_biguint(&binary.b.0)))
        .collect::<Vec<_>>();

    rep3::conversion::b2a_many(&biguint_shares, io_context)
}

/// A cast of a Rep3RingShare to a Rep3PrimeFieldShare
#[tracing::instrument(skip_all, level = "trace")]
pub fn r2f_b2a_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
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
                let share = crate::downcast::<_, Rep3RingShare<Bit>>(&share).expect("We already checked types");
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
        .map(|binary| Rep3BigUintShare::new(T::cast_to_biguint(&binary.a.0), T::cast_to_biguint(&binary.b.0)))
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
    let (binary, signs): (Vec<_>, Vec<_>) =
        singed.into_iter().map(|Rep3RingSignedShare { abs, sign }| (abs, sign)).unzip();

    let positive = r2f_b2a_many(&binary, io_context)?;
    let negative = positive.iter().map(|x| rep3::arithmetic::neg(*x)).collect::<Vec<_>>();
    let signs = conversion::bit_inject_from_bits_to_field_many(&signs, io_context)?;

    rep3::arithmetic::cmux_many::<F, N>(&signs, &positive, &negative, io_context)
}

// ---------------------------------------------------------------------------
// Preprocessed R2F conversions (moved from edabits.rs)
// ---------------------------------------------------------------------------

/// Convert arithmetic ring shares `[x]` over `Z_{2^K}` into arithmetic
/// field shares `[x]` over `Fp`, using masked openings with DaRing tuples.
///
/// # Assumptions
/// `x` represents a non-negative integer smaller than `2^K`, and `Fp`
/// is large enough for the application.
pub fn r2f_preproc_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    eda: &[DaRing<T, F>],
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>> {
    debug_assert_eq!(x.len(), eda.len());

    let masked = x.iter().zip(eda).map(|(x, eda)| *x - eda.r_ring).collect::<Vec<_>>();

    let opened = rep3_ring_arith::open_vec(&masked, io)?;

    Ok(opened
        .into_iter()
        .zip(eda)
        .map(|(c_open, eda)| {
            let c_f = F::from(Into::<u128>::into(c_open.0));
            let c_fp_share = rep3_arith::promote_to_trivial_share(io.id, c_f);
            eda.r_fp + c_fp_share
        })
        .collect())
}

/// Convert a single binary (XOR-shared) ring share `[x]` over `Z_{2^K}` into an
/// arithmetic field share `[x]` over `Fp`, using Protocol Π₂.
pub fn r2f_b2a_preproc<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: Rep3RingShare<T>,
    eda: EdaBits<T, F>,
    io: &mut IoContext<N>,
) -> eyre::Result<Rep3PrimeFieldShare<F>>
where
    Standard: Distribution<T>,
{
    let batch = EdaBitsBatch { gammas: vec![eda.gamma], alphas_flat: eda.alphas };
    let mut out = r2f_b2a_preproc_many::<T, F, N>(&[x_binary], &batch, io)?;
    Ok(out.remove(0))
}

/// Batched Protocol Π₂ B2A conversion: binary Rep3 ring shares → arithmetic field shares.
///
/// **Online communication:**
/// - Round 1: P0 broadcasts N packed ring elements (K bits each) to P1 and P2.
/// - Round 2: ShareConvert via `reshare_many` (one field element per conversion).
pub fn r2f_b2a_preproc_many<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    batch: &EdaBitsBatch<T, F>,
    io: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    let mut scratch = R2fB2aScratch::default();
    r2f_b2a_preproc_many_into(x_binary, batch.as_ref(), io, &mut scratch)?;
    Ok(std::mem::take(&mut scratch.out))
}

#[tracing::instrument(skip_all, level = "trace", name = "r2f_b2a_preproc_many_into")]
pub fn r2f_b2a_preproc_many_into<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    batch: EdaBitsBatchRef<'_, T, F>,
    io: &mut IoContext<N>,
    scratch: &mut R2fB2aScratch<T, F>,
) -> eyre::Result<()>
where
    Standard: Distribution<T>,
{
    let n = x_binary.len();
    scratch.out.clear();
    if n == 0 {
        return Ok(());
    }
    debug_assert_eq!(batch.gammas.len(), n);
    debug_assert_eq!(batch.alphas_flat.len(), n * T::K);

    scratch.ensure_pow2();
    let k = T::K;

    // --- Round 1: P0 broadcasts masked values ---
    if io.id == PartyID::ID0 {
        scratch.ms.clear();
        scratch.ms.extend(x_binary.iter().zip(batch.gammas.iter()).map(|(x, gamma)| x.a ^ x.b ^ *gamma));
        io.network.send_many(PartyID::ID1, &scratch.ms)?;
        io.network.send_many(PartyID::ID2, &scratch.ms)?;
    } else {
        scratch.ms = io.network.recv_many(PartyID::ID0)?;
    }

    // --- Local computation: fused v_component + masking → s_selfs ---
    scratch.maskings.clear();
    scratch.maskings.extend((0..n).map(|_| io.masking_field_element::<F>()));

    scratch.s_selfs.clear();
    scratch.s_selfs.resize(n, F::zero());

    let party_id = io.id;
    let ms = &scratch.ms;
    let maskings = &scratch.maskings;
    let pow2 = &scratch.pow2;
    let alphas_flat = batch.alphas_flat;
    scratch.s_selfs.par_iter_mut().enumerate().with_min_len(256).for_each(|(idx, s_self)| {
        let z = maskings[idx];
        if party_id == PartyID::ID0 {
            *s_self = z;
            return;
        }

        let x = x_binary[idx];
        let m = ms[idx];
        let beta = match party_id {
            PartyID::ID0 => unreachable!(),
            PartyID::ID1 => m ^ x.a,
            PartyID::ID2 => m ^ x.b,
        };

        let mut v = F::zero();
        let alpha_base = idx * k;
        for i in 0..k {
            let beta_bit = ((beta.0 >> i) & T::one()) == T::one();
            let alpha = alphas_flat[alpha_base + i];
            let signed_alpha = if beta_bit { -alpha } else { alpha };
            v += pow2[i] * signed_alpha;
        }

        if party_id == PartyID::ID1 {
            v += F::from(Into::<u128>::into(beta.0));
        }

        *s_self = v + z;
    });

    let s_prevs = io.network.reshare_many(&scratch.s_selfs)?;
    scratch.out.extend(
        scratch
            .s_selfs
            .iter()
            .copied()
            .zip(s_prevs.into_iter())
            .map(|(s_self, s_prev)| Rep3PrimeFieldShare::new(s_self, s_prev)),
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// MPCType-based selectors
// ---------------------------------------------------------------------------

/// Dispatch ring-to-field conversion based on `io_context.mpc_type`.
///
/// - `Online` → A2B-based conversion (`r2f_many`)
/// - `Preprocessed` → EdaBits masked opening (`r2f_preproc_many`)
pub fn r2f_many_selector<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x: &[Rep3RingShare<T>],
    eda: Option<&[DaRing<T, F>]>,
    io_context: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    match io_context.mpc_type {
        MPCType::Online => Ok(r2f_many(x, io_context)?),
        MPCType::Preprocessed => {
            r2f_preproc_many(x, eda.expect("r2f_many_selector: preprocessed mode requires eda"), io_context)
        }
    }
}

/// Dispatch binary-ring-to-field conversion based on `io_context.mpc_type`.
///
/// - `Online` → B2A via BigUint (`r2f_b2a_many`)
/// - `Preprocessed` → EdaBits Protocol Π₂ (`r2f_b2a_preproc_many`)
pub fn r2f_b2a_many_selector<T: IntRing2k, F: PrimeField, N: Rep3Network>(
    x_binary: &[Rep3RingShare<T>],
    batch: Option<&EdaBitsBatch<T, F>>,
    io_context: &mut IoContext<N>,
) -> eyre::Result<Vec<Rep3PrimeFieldShare<F>>>
where
    Standard: Distribution<T>,
{
    match io_context.mpc_type {
        MPCType::Online => Ok(r2f_b2a_many(x_binary, io_context)?),
        MPCType::Preprocessed => r2f_b2a_preproc_many(
            x_binary,
            batch.expect("r2f_b2a_many_selector: preprocessed mode requires batch"),
            io_context,
        ),
    }
}
