use crate::field::JoltField;
use mpc_core::protocols::{
    rep3::{
        network::{IoContextPool, Rep3NetworkWorker},
        Rep3PrimeFieldShare,
    },
    rep3_ring::{
        self,
        edabits::PreprocessingPool,
        ring::{bit::Bit, int_ring::IntRing2k},
        Rep3RingShare, Rep3RingSignedShare,
    },
};

use rand::{distributions::Standard, prelude::Distribution};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum FutureRep3Ring<R: IntRing2k, T, Args = ()> {
    Ready(T),
    Pending(FutureOp<R>, Args),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FutureOp<R: IntRing2k> {
    // Out: Rep3PrimeFieldShare<F>
    BitInject(Rep3RingShare<Bit>),
    CastToField(Rep3RingShare<R>),
    CastToFieldB2A(Rep3RingShare<R>),
    CastToFieldSigned(Rep3RingSignedShare<R>),

    // Out: Rep3RingShare<R>
    RingMulA2B(Rep3RingShare<R>, Rep3RingShare<R>), // TODO: make recursive
    RingA2B(Rep3RingShare<R>),
    // And(Rep3RingShare<R>, ),
    Fulfilled,
}

impl<R: IntRing2k, T, Args: Default> FutureRep3Ring<R, T, Args> {
    // ===== Into Field Ops =====

    pub fn cast_to_field(a: Rep3RingShare<R>) -> Self {
        FutureRep3Ring::Pending(FutureOp::CastToField(a), Default::default())
    }

    pub fn cast_to_field_b2a(a: Rep3RingShare<R>) -> Self {
        FutureRep3Ring::Pending(FutureOp::CastToFieldB2A(a), Default::default())
    }

    pub fn cast_to_field_signed_b2a(a: Rep3RingSignedShare<R>) -> Self {
        FutureRep3Ring::Pending(FutureOp::CastToFieldSigned(a), Default::default())
    }

    pub fn bit_inject_to_field(a: Rep3RingShare<Bit>) -> Self {
        FutureRep3Ring::Pending(FutureOp::BitInject(a), Default::default())
    }

    // ===== Into Ring Ops =====

    pub fn a2b(a: Rep3RingShare<R>) -> Self {
        FutureRep3Ring::Pending(FutureOp::RingA2B(a), Default::default())
    }

    pub fn mul_a2b(a: Rep3RingShare<R>, b: Rep3RingShare<R>) -> Self {
        FutureRep3Ring::Pending(FutureOp::RingMulA2B(a, b), Default::default())
    }
}

impl<R: IntRing2k, T> FutureRep3Ring<R, T, Option<usize>> {
    // Workeraround to allow reusing fulfilled values
    pub fn fulfilled(index: usize) -> Self {
        FutureRep3Ring::Pending(FutureOp::Fulfilled, Some(index))
    }
}

pub trait Rep3RingFutureExt<R: IntRing2k, U, T, Args = ()> {
    fn fulfill_batched<N: Rep3NetworkWorker, MapFn: Fn(U, Args) -> T + Send>(
        self,
        io_ctx: &mut IoContextPool<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(U, Args) -> T + Send + Sync;
}

/// Thread-local buckets for parallel partition of field futures.
struct FieldBuckets<R: IntRing2k, T, Args> {
    ready_idx: Vec<usize>,
    ready_val: Vec<T>,

    bit_idx: Vec<usize>,
    bit_x: Vec<Rep3RingShare<Bit>>,
    bit_args: Vec<Args>,

    cast_idx: Vec<usize>,
    cast_x: Vec<Rep3RingShare<R>>,
    cast_args: Vec<Args>,

    b2a_idx: Vec<usize>,
    b2a_x: Vec<Rep3RingShare<R>>,
    b2a_args: Vec<Args>,

    signed_idx: Vec<usize>,
    signed_x: Vec<Rep3RingSignedShare<R>>,
    signed_args: Vec<Args>,

    fulfilled_idx: Vec<usize>,
    fulfilled_args: Vec<Args>,
}

impl<R: IntRing2k, T, Args> FieldBuckets<R, T, Args> {
    fn new() -> Self {
        Self {
            ready_idx: Vec::new(),
            ready_val: Vec::new(),
            bit_idx: Vec::new(),
            bit_x: Vec::new(),
            bit_args: Vec::new(),
            cast_idx: Vec::new(),
            cast_x: Vec::new(),
            cast_args: Vec::new(),
            b2a_idx: Vec::new(),
            b2a_x: Vec::new(),
            b2a_args: Vec::new(),
            signed_idx: Vec::new(),
            signed_x: Vec::new(),
            signed_args: Vec::new(),
            fulfilled_idx: Vec::new(),
            fulfilled_args: Vec::new(),
        }
    }

    fn extend(&mut self, other: Self) {
        self.ready_idx.extend(other.ready_idx);
        self.ready_val.extend(other.ready_val);
        self.bit_idx.extend(other.bit_idx);
        self.bit_x.extend(other.bit_x);
        self.bit_args.extend(other.bit_args);
        self.cast_idx.extend(other.cast_idx);
        self.cast_x.extend(other.cast_x);
        self.cast_args.extend(other.cast_args);
        self.b2a_idx.extend(other.b2a_idx);
        self.b2a_x.extend(other.b2a_x);
        self.b2a_args.extend(other.b2a_args);
        self.signed_idx.extend(other.signed_idx);
        self.signed_x.extend(other.signed_x);
        self.signed_args.extend(other.signed_args);
        self.fulfilled_idx.extend(other.fulfilled_idx);
        self.fulfilled_args.extend(other.fulfilled_args);
    }
}

impl<R: IntRing2k, F: JoltField, T, Args> Rep3RingFutureExt<R, Rep3PrimeFieldShare<F>, T, Args>
    for Vec<FutureRep3Ring<R, T, Args>>
where
    T: Clone + Default + Send,
    Args: Send + Copy,
    Standard: Distribution<R>,
{
    #[tracing::instrument(
        skip_all,
        name = "FutureRep3Ring::fulfill_batched_to_field",
        level = "trace"
    )]
    fn fulfill_batched<N: Rep3NetworkWorker, MapFn>(
        self,
        io_ctx: &mut IoContextPool<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(Rep3PrimeFieldShare<F>, Args) -> T + Send + Sync,
    {
        let len = self.len();

        // Parallel fold/reduce partition into indexed buckets
        let buckets: FieldBuckets<R, T, Args> = self
            .into_par_iter()
            .enumerate()
            .fold(FieldBuckets::new, |mut acc, (i, f)| {
                match f {
                    FutureRep3Ring::Ready(x) => {
                        acc.ready_idx.push(i);
                        acc.ready_val.push(x);
                    }
                    FutureRep3Ring::Pending(FutureOp::BitInject(x), args) => {
                        acc.bit_idx.push(i);
                        acc.bit_x.push(x);
                        acc.bit_args.push(args);
                    }
                    FutureRep3Ring::Pending(FutureOp::CastToField(x), args) => {
                        acc.cast_idx.push(i);
                        acc.cast_x.push(x);
                        acc.cast_args.push(args);
                    }
                    FutureRep3Ring::Pending(FutureOp::CastToFieldB2A(x), args) => {
                        acc.b2a_idx.push(i);
                        acc.b2a_x.push(x);
                        acc.b2a_args.push(args);
                    }
                    FutureRep3Ring::Pending(FutureOp::CastToFieldSigned(x), args) => {
                        acc.signed_idx.push(i);
                        acc.signed_x.push(x);
                        acc.signed_args.push(args);
                    }
                    FutureRep3Ring::Pending(FutureOp::Fulfilled, args) => {
                        acc.fulfilled_idx.push(i);
                        acc.fulfilled_args.push(args);
                    }
                    _ => unimplemented!(),
                }
                acc
            })
            .reduce(FieldBuckets::new, |mut a, b| {
                a.extend(b);
                a
            });

        let mut out = vec![T::default(); len];

        // Ready — direct write
        for k in 0..buckets.ready_idx.len() {
            out[buckets.ready_idx[k]] = buckets.ready_val[k].clone();
        }

        // Bit Inject
        if !buckets.bit_x.is_empty() {
            let c = io_ctx.par_chunks(buckets.bit_x, None, |xs, io_ctx| {
                rep3_ring::conversion::bit_inject_from_bits_to_field_many(&xs, io_ctx)
            })?;
            for k in 0..c.len() {
                out[buckets.bit_idx[k]] = map(c[k], buckets.bit_args[k]);
            }
        }

        // Cast
        if !buckets.cast_x.is_empty() {
            let c = io_ctx.par_chunks(buckets.cast_x, None, |xs, io_ctx| {
                rep3_ring::casts::ring_to_field_many_selector(&xs, io_ctx)
            })?;
            for k in 0..c.len() {
                out[buckets.cast_idx[k]] = map(c[k], buckets.cast_args[k]);
            }
        }

        // Cast B2A
        if !buckets.b2a_x.is_empty() {
            let shares = io_ctx.par_chunks(buckets.b2a_x, None, |xs, io_ctx| {
                rep3_ring::casts::binary_ring_to_field_many(&xs, io_ctx)
            })?;
            for k in 0..shares.len() {
                out[buckets.b2a_idx[k]] = map(shares[k], buckets.b2a_args[k]);
            }
        }

        // Cast B2A Signed
        if !buckets.signed_x.is_empty() {
            let shares = io_ctx.par_chunks(buckets.signed_x, None, |chunk, io_ctx| {
                rep3_ring::casts::signed_binary_ring_to_field_many(chunk, io_ctx)
            })?;
            for k in 0..shares.len() {
                out[buckets.signed_idx[k]] = map(shares[k], buckets.signed_args[k]);
            }
        }

        // Fulfilled
        for k in 0..buckets.fulfilled_idx.len() {
            out[buckets.fulfilled_idx[k]] = map(Default::default(), buckets.fulfilled_args[k]);
        }

        Ok(out)
    }
}

/// Fulfill batched `CastToFieldB2A` futures using an [`EdaBitsPool`] for
/// PCG-based B2A conversion (1 binary open round).
///
/// Generic over ring type `R` — callers downcast suffix bits to the smallest
/// ring that fits, so edaBits use fewer alphas and less communication.
///
/// Only handles `CastToFieldB2A` and `Ready` variants; other `FutureOp`
/// variants will panic.
// #[tracing::instrument(skip_all, name = "fulfill_batched_with_pool", level = "trace")]
// pub fn fulfill_batched_with_pool<R, F, T, Args, N, MapFn>(
//     futures: Vec<FutureRep3Ring<R, T, Args>>,
//     io_ctx: &mut IoContextPool<N>,
//     preproc: &mut PreprocessingPool<F>,
//     map: MapFn,
// ) -> eyre::Result<Vec<T>>
// where
//     R: IntRing2k,
//     Standard: Distribution<R>,
//     F: JoltField,
//     T: Clone + Default + Send,
//     Args: Send + Copy,
//     N: Rep3NetworkWorker,
//     MapFn: Fn(Rep3PrimeFieldShare<F>, Args) -> T + Send + Sync,
// {
//     use mpc_core::protocols::rep3_ring::{dabits, edabits};

//     let len = futures.len();

//     // Parallel fold/reduce partition into indexed buckets
//     struct PoolBuckets<R: IntRing2k, T, Args> {
//         ready_idx: Vec<usize>,
//         ready_val: Vec<T>,
//         bit_idx: Vec<usize>,
//         bit_x: Vec<Rep3RingShare<Bit>>,
//         bit_args: Vec<Args>,
//         b2a_idx: Vec<usize>,
//         b2a_x: Vec<Rep3RingShare<R>>,
//         b2a_args: Vec<Args>,
//     }

//     impl<R: IntRing2k, T, Args> PoolBuckets<R, T, Args> {
//         fn new() -> Self {
//             Self {
//                 ready_idx: Vec::new(),
//                 ready_val: Vec::new(),
//                 bit_idx: Vec::new(),
//                 bit_x: Vec::new(),
//                 bit_args: Vec::new(),
//                 b2a_idx: Vec::new(),
//                 b2a_x: Vec::new(),
//                 b2a_args: Vec::new(),
//             }
//         }
//         fn extend(&mut self, other: Self) {
//             self.ready_idx.extend(other.ready_idx);
//             self.ready_val.extend(other.ready_val);
//             self.bit_idx.extend(other.bit_idx);
//             self.bit_x.extend(other.bit_x);
//             self.bit_args.extend(other.bit_args);
//             self.b2a_idx.extend(other.b2a_idx);
//             self.b2a_x.extend(other.b2a_x);
//             self.b2a_args.extend(other.b2a_args);
//         }
//     }

//     let buckets: PoolBuckets<R, T, Args> = futures
//         .into_par_iter()
//         .enumerate()
//         .fold(PoolBuckets::new, |mut acc, (i, f)| {
//             match f {
//                 FutureRep3Ring::Ready(x) => {
//                     acc.ready_idx.push(i);
//                     acc.ready_val.push(x);
//                 }
//                 FutureRep3Ring::Pending(FutureOp::BitInject(x), args) => {
//                     acc.bit_idx.push(i);
//                     acc.bit_x.push(x);
//                     acc.bit_args.push(args);
//                 }
//                 FutureRep3Ring::Pending(FutureOp::CastToFieldB2A(x), args) => {
//                     acc.b2a_idx.push(i);
//                     acc.b2a_x.push(x);
//                     acc.b2a_args.push(args);
//                 }
//                 other => panic!(
//                     "fulfill_batched_with_pool: unexpected variant {:?}",
//                     std::mem::discriminant(&other)
//                 ),
//             }
//             acc
//         })
//         .reduce(PoolBuckets::new, |mut a, b| {
//             a.extend(b);
//             a
//         });

//     let mut out = vec![T::default(); len];

//     // Ready — direct write
//     for k in 0..buckets.ready_idx.len() {
//         out[buckets.ready_idx[k]] = buckets.ready_val[k].clone();
//     }

//     // Bit Inject (single-bit → field via daBits) — distributed across forks
//     if !buckets.bit_x.is_empty() {
//         let batch = preproc.take_dabits(buckets.bit_x.len());
//         let c = io_ctx.par_chunks_dabits(buckets.bit_x, batch, None, |xs, batch, ctx| {
//             dabits::bit_inject_field_many(&xs, &batch, ctx)
//         })?;
//         for k in 0..c.len() {
//             out[buckets.bit_idx[k]] = map(c[k], buckets.bit_args[k]);
//         }
//     }

//     // Cast B2A (binary/XOR ring → field) via Protocol Π₂ edaBits — distributed across forks
//     if !buckets.b2a_x.is_empty() {
//         let batch = preproc.take_edabits::<R>(buckets.b2a_x.len());
//         let shares = io_ctx.par_chunks_preproc(buckets.b2a_x, batch, None, |xs, batch, ctx| {
//             edabits::ring_to_field_b2a_many::<R, F, _>(&xs, &batch, ctx)
//         })?;
//         for k in 0..shares.len() {
//             out[buckets.b2a_idx[k]] = map(shares[k], buckets.b2a_args[k]);
//         }
//     }

//     Ok(out)
// }

impl<R, T, Args> Rep3RingFutureExt<R, Rep3RingShare<R>, T, Args> for Vec<FutureRep3Ring<R, T, Args>>
where
    R: IntRing2k,
    T: Send + Default + Clone,
    Args: Send + Copy,
    Standard: Distribution<R>,
{
    #[tracing::instrument(
        skip_all,
        name = "FutureRep3Ring::fulfill_batched_to_ring",
        level = "trace"
    )]
    fn fulfill_batched<N: Rep3NetworkWorker, MapFn>(
        self,
        io_ctx: &mut IoContextPool<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(Rep3RingShare<R>, Args) -> T + Send + Sync,
    {
        let len = self.len();

        // Parallel fold/reduce partition into indexed buckets
        struct RingBuckets<R: IntRing2k, T, Args> {
            ready_idx: Vec<usize>,
            ready_val: Vec<T>,
            mul_idx: Vec<usize>,
            mul_x: Vec<Rep3RingShare<R>>,
            mul_y: Vec<Rep3RingShare<R>>,
            mul_args: Vec<Args>,
            a2b_idx: Vec<usize>,
            a2b_x: Vec<Rep3RingShare<R>>,
            a2b_args: Vec<Args>,
        }

        impl<R: IntRing2k, T, Args> RingBuckets<R, T, Args> {
            fn new() -> Self {
                Self {
                    ready_idx: Vec::new(),
                    ready_val: Vec::new(),
                    mul_idx: Vec::new(),
                    mul_x: Vec::new(),
                    mul_y: Vec::new(),
                    mul_args: Vec::new(),
                    a2b_idx: Vec::new(),
                    a2b_x: Vec::new(),
                    a2b_args: Vec::new(),
                }
            }
            fn extend(&mut self, other: Self) {
                self.ready_idx.extend(other.ready_idx);
                self.ready_val.extend(other.ready_val);
                self.mul_idx.extend(other.mul_idx);
                self.mul_x.extend(other.mul_x);
                self.mul_y.extend(other.mul_y);
                self.mul_args.extend(other.mul_args);
                self.a2b_idx.extend(other.a2b_idx);
                self.a2b_x.extend(other.a2b_x);
                self.a2b_args.extend(other.a2b_args);
            }
        }

        let buckets: RingBuckets<R, T, Args> = self
            .into_par_iter()
            .enumerate()
            .fold(RingBuckets::new, |mut acc, (i, f)| {
                match f {
                    FutureRep3Ring::Ready(x) => {
                        acc.ready_idx.push(i);
                        acc.ready_val.push(x);
                    }
                    FutureRep3Ring::Pending(FutureOp::RingMulA2B(a, b), args) => {
                        acc.mul_idx.push(i);
                        acc.mul_x.push(a);
                        acc.mul_y.push(b);
                        acc.mul_args.push(args);
                    }
                    FutureRep3Ring::Pending(FutureOp::RingA2B(x), args) => {
                        acc.a2b_idx.push(i);
                        acc.a2b_x.push(x);
                        acc.a2b_args.push(args);
                    }
                    _ => unimplemented!(),
                }
                acc
            })
            .reduce(RingBuckets::new, |mut a, b| {
                a.extend(b);
                a
            });

        let mut out = vec![T::default(); len];

        // Ready — direct write
        for k in 0..buckets.ready_idx.len() {
            out[buckets.ready_idx[k]] = buckets.ready_val[k].clone();
        }

        // Mul + A2B
        if !buckets.mul_x.is_empty() {
            let t = rep3_ring::arithmetic::mul_vec(&buckets.mul_x, &buckets.mul_y, io_ctx.main())?;
            let c = rep3_ring::conversion::a2b_many(&t, io_ctx.main())?;
            for k in 0..c.len() {
                out[buckets.mul_idx[k]] = map(c[k], buckets.mul_args[k]);
            }
        }

        // A2B
        if !buckets.a2b_x.is_empty() {
            let c = rep3_ring::conversion::a2b_many(&buckets.a2b_x, io_ctx.main())?;
            for k in 0..c.len() {
                out[buckets.a2b_idx[k]] = map(c[k], buckets.a2b_args[k]);
            }
        }

        Ok(out)
    }
}
