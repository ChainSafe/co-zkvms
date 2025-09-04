use crate::field::JoltField;
use itertools::Itertools;
use mpc_core::protocols::{
    rep3::{
        network::{IoContext, Rep3Network, Rep3NetworkWorker, WorkerIoContext},
        Rep3PrimeFieldShare,
    },
    rep3_ring::{
        self,
        ring::{bit::Bit, int_ring::IntRing2k},
        Rep3RingShare,
    },
};

use rand::{distributions::Standard, prelude::Distribution};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub enum FutureRep3Ring<R: IntRing2k, T, Args = ()> {
    Ready(T),
    Pending(FutureOp<R>, Args),
}

#[derive(Debug, Clone)]
pub enum FutureOp<R: IntRing2k> {
    // Out: Rep3PrimeFieldShare<F>
    BitInject(Rep3RingShare<Bit>),
    CastToField(Rep3RingShare<R>),
    CastToFieldB2A(Rep3RingShare<R>),

    // Out: Rep3RingShare<R>
    RingMulA2B(Rep3RingShare<R>, Rep3RingShare<R>), // TODO: make recursive
    RingA2B(Rep3RingShare<R>),

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
        io_ctx: &mut WorkerIoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(U, Args) -> T + Send + Sync;
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
        io_ctx: &mut WorkerIoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(Rep3PrimeFieldShare<F>, Args) -> T + Send + Sync,
    {
        let mut fufilled = vec![T::default(); self.len()];
        let (mut bit_inject_x, mut fut_bit_inject, mut bit_inject_args) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut cast_x, mut fut_cast, mut cast_args) = (Vec::new(), Vec::new(), Vec::new());
        let (mut cast_b2a_x, mut fut_cast_b2a, mut cast_b2a_args) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut fut_fulfilled, mut fulfilled_index) = (Vec::new(), Vec::new());

        self.into_iter()
            .zip_eq(fufilled.iter_mut())
            .for_each(|(f, fufilled)| match f {
                FutureRep3Ring::Pending(FutureOp::BitInject(x), args) => {
                    bit_inject_x.push(x);
                    fut_bit_inject.push(fufilled);
                    bit_inject_args.push(args);
                }
                FutureRep3Ring::Pending(FutureOp::CastToField(x), args) => {
                    cast_x.push(x);
                    fut_cast.push(fufilled);
                    cast_args.push(args);
                }
                FutureRep3Ring::Pending(FutureOp::CastToFieldB2A(x), args) => {
                    cast_b2a_x.push(x);
                    fut_cast_b2a.push(fufilled);
                    cast_b2a_args.push(args);
                }
                FutureRep3Ring::Pending(FutureOp::Fulfilled, args) => {
                    fut_fulfilled.push(fufilled);
                    fulfilled_index.push(args);
                }
                FutureRep3Ring::Ready(x) => {
                    *fufilled = x;
                }
                _ => unimplemented!(),
            });

        // Bit Inject
        {
            let c = if !bit_inject_x.is_empty() {
                io_ctx.par_chunks(bit_inject_x, None, |bit_inject_x, io_ctx| {
                    rep3_ring::conversion::bit_inject_from_bits_to_field_many(&bit_inject_x, io_ctx)
                })?
            } else {
                vec![]
            };

            fut_bit_inject
                .into_par_iter()
                .zip_eq(c.into_par_iter())
                .zip_eq(bit_inject_args)
                .for_each(|((f, c), args)| {
                    *f = map(c, args);
                });
        }

        // Cast
        {
            let c = if !cast_x.is_empty() {
                io_ctx.par_chunks(cast_x, None, |cast_x, io_ctx| {
                    rep3_ring::casts::ring_to_field_many_selector(&cast_x, io_ctx)
                })?
            } else {
                vec![]
            };

            fut_cast
                .into_par_iter()
                .zip_eq(c.into_par_iter())
                .zip_eq(cast_args)
                .for_each(|((f, c), args)| {
                    *f = map(c, args);
                });
        }

        // Cast B2A
        {
            let shares = if !cast_b2a_x.is_empty() {
                io_ctx.par_chunks(cast_b2a_x, None, |cast_b2a_x, io_ctx| {
                    rep3_ring::casts::binary_ring_to_field_many(&cast_b2a_x, io_ctx)
                })?
            } else {
                vec![]
            };

            fut_cast_b2a
                .into_par_iter()
                .zip_eq(shares.into_par_iter())
                .zip_eq(cast_b2a_args)
                .for_each(|((f, c), args)| {
                    *f = map(c, args);
                });
        }

        // Fulfilled
        {
            if !fulfilled_index.is_empty() {
                fut_fulfilled
                    .into_par_iter()
                    .zip_eq(fulfilled_index)
                    .for_each(|(f, index)| *f = map(Default::default(), index));
            }
        }

        Ok(fufilled)
    }
}

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
        io_ctx: &mut WorkerIoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(Rep3RingShare<R>, Args) -> T + Send + Sync,
    {
        let mut fufilled = vec![T::default(); self.len()];
        let (mut mul_a2b_x, mut mul_a2b_y, mut fut_mul_a2b, mut args_mul_a2b) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut a2b_x, mut fut_a2b, mut a2b_args) = (Vec::new(), Vec::new(), Vec::new());

        self.into_iter()
            .zip_eq(fufilled.iter_mut())
            .for_each(|(f, fufilled)| match f {
                FutureRep3Ring::Pending(FutureOp::RingMulA2B(a, b), args) => {
                    mul_a2b_x.push(a);
                    mul_a2b_y.push(b);
                    fut_mul_a2b.push(fufilled);
                    args_mul_a2b.push(args);
                }
                FutureRep3Ring::Pending(FutureOp::RingA2B(x), args) => {
                    a2b_x.push(x);
                    fut_a2b.push(fufilled);
                    a2b_args.push(args);
                }
                FutureRep3Ring::Ready(x) => {
                    *fufilled = x;
                }
                _ => unimplemented!(),
            });

        // Mul + A2B
        {
            let c = if !mul_a2b_x.is_empty() && !mul_a2b_y.is_empty() {
                let t = rep3_ring::arithmetic::mul_vec(&mul_a2b_x, &mul_a2b_y, io_ctx.main())?;
                rep3_ring::conversion::a2b_many(&t, io_ctx.main())?
            } else {
                vec![]
            };

            fut_mul_a2b
                .into_par_iter()
                .zip_eq(c.into_par_iter())
                .zip_eq(args_mul_a2b)
                .for_each(|((f, c), args)| {
                    *f = map(c, args);
                });
        }

        // A2B
        {
            let c = if !a2b_x.is_empty() {
                rep3_ring::conversion::a2b_many(&a2b_x, io_ctx.main())?
            } else {
                vec![]
            };

            fut_a2b
                .into_par_iter()
                .zip_eq(c.into_par_iter())
                .zip_eq(a2b_args)
                .for_each(|((f, c), args)| {
                    *f = map(c, args);
                });
        }

        Ok(fufilled)
    }
}
