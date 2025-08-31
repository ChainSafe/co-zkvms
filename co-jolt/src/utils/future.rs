use crate::field::JoltField;
use itertools::Itertools;
use mpc_core::protocols::{
    rep3::{
        self,
        network::{IoContext, Rep3Network},
        Rep3BigUintShare, Rep3PrimeFieldShare,
    },
    rep3_ring::{
        self,
        ring::{bit::Bit, int_ring::IntRing2k},
        Rep3RingShare,
    },
};

use rayon::prelude::*;
use tokio::io;

#[derive(Debug, Clone)]
pub enum FutureVal<F: JoltField, T, Args = ()> {
    Ready(T),
    Pending(FutureOp<F>, Args),
}

impl<F: JoltField, T, Extra> FutureVal<F, T, Extra> {
    pub fn as_ready(&self) -> &T {
        match self {
            FutureVal::Ready(t) => t,
            _ => panic!("FutureVal is not ready"),
        }
    }

    pub fn mul_args(a: Rep3PrimeFieldShare<F>, b: Rep3PrimeFieldShare<F>, args: Extra) -> Self {
        FutureVal::Pending(FutureOp::Mul(a, b), args)
    }
}

impl<F: JoltField, T> FutureVal<F, T> {
    pub fn b2a(a: Rep3BigUintShare<F>) -> Self {
        FutureVal::Pending(FutureOp::B2A(a), ())
    }

    pub fn a2b(a: Rep3PrimeFieldShare<F>) -> Self {
        FutureVal::Pending(FutureOp::A2B(a), ())
    }

    pub fn ring_a2b(a: Rep3RingShare<u32>) -> Self {
        FutureVal::Pending(FutureOp::RingA2B(a), ())
    }

    pub fn cast_to_field(a: Rep3RingShare<u32>) -> Self {
        FutureVal::Pending(FutureOp::CastToField(a), ())
    }

    pub fn cast_to_field_b2a(a: Rep3RingShare<u32>) -> Self {
        FutureVal::Pending(FutureOp::CastToFieldB2A(a), ())
    }

    pub fn bit_inject_to_field(a: Rep3RingShare<Bit>) -> Self {
        FutureVal::Pending(FutureOp::BitInject(a), ())
    }
}

pub trait FutureExt<F: JoltField, U, T, Args> {
    fn fufill_batched<N: Rep3Network, MapFn: Fn(U, Args) -> T + Send>(
        self,
        io_ctx: &mut IoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(U, Args) -> T + Send + Sync;
}

impl<F: JoltField, T, Args> FutureExt<F, Rep3PrimeFieldShare<F>, T, Args>
    for Vec<FutureVal<F, T, Args>>
where
    T: Clone + Default + Send,
    Args: Send + Copy,
{
    #[tracing::instrument(skip_all, name = "FutureVals::fufill_batched", level = "trace")]
    fn fufill_batched<N: Rep3Network, MapFn>(
        mut self,
        io_ctx: &mut IoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(Rep3PrimeFieldShare<F>, Args) -> T + Send + Sync,
    {
        let mut fufilled = vec![T::default(); self.len()];
        let (mut mul_x, mut mul_y, mut fut_muls, mut args_mul) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut b2a_x, mut fut_b2a, mut b2a_args) = (Vec::new(), Vec::new(), Vec::new());
        let (mut bit_inject_x, mut fut_bit_inject, mut bit_inject_args) =
            (Vec::new(), Vec::new(), Vec::new());
        let (mut cast_x, mut fut_cast, mut cast_args) = (Vec::new(), Vec::new(), Vec::new());
        let (mut cast_b2a_x, mut fut_cast_b2a, mut cast_b2a_args) =
            (Vec::new(), Vec::new(), Vec::new());

        self.into_iter()
            .zip_eq(fufilled.iter_mut())
            .for_each(|(f, fufilled)| match f {
                FutureVal::Pending(FutureOp::Mul(a, b), args) => {
                    mul_x.push(a);
                    mul_y.push(b);
                    fut_muls.push(fufilled);
                    args_mul.push(args);
                }
                FutureVal::Pending(FutureOp::B2A(x), args) => {
                    b2a_x.push(x);
                    fut_b2a.push(fufilled);
                    b2a_args.push(args);
                }
                FutureVal::Pending(FutureOp::BitInject(x), args) => {
                    bit_inject_x.push(x);
                    fut_bit_inject.push(fufilled);
                    bit_inject_args.push(args);
                }
                FutureVal::Pending(FutureOp::CastToField(x), args) => {
                    cast_x.push(x);
                    fut_cast.push(fufilled);
                    cast_args.push(args);
                }
                FutureVal::Pending(FutureOp::CastToFieldB2A(x), args) => {
                    cast_b2a_x.push(x);
                    fut_cast_b2a.push(fufilled);
                    cast_b2a_args.push(args);
                }
                FutureVal::Ready(x) => {
                    *fufilled = x;
                }
                _ => unimplemented!(),
            });
        // Multiply
        {
            let c = if !mul_x.is_empty() && !mul_y.is_empty() {
                rep3::arithmetic::mul_vec(&mul_x, &mul_y, io_ctx)?
            } else {
                vec![]
            };

            fut_muls
                .into_par_iter()
                .zip_eq(c.into_par_iter())
                .zip_eq(args_mul)
                .for_each(|((f, c), args)| {
                    *f = map(c, args);
                });
        }

        // B2A
        {
            let c = if !b2a_x.is_empty() {
                rep3::conversion::b2a_many(&b2a_x, io_ctx)?
            } else {
                vec![]
            };

            fut_b2a
                .into_par_iter()
                .zip_eq(c.into_par_iter())
                .zip_eq(b2a_args)
                .for_each(|((f, c), args)| {
                    *f = map(c, args);
                });
        }

        // Bit Inject
        {
            let c = if !bit_inject_x.is_empty() {
                rep3_ring::conversion::bit_inject_from_bits_to_field_many(&bit_inject_x, io_ctx)?
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
                rep3_ring::casts::ring_to_field_many_selector(&cast_x, io_ctx)?
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
                rep3_ring::casts::binary_ring_to_field_many(&cast_b2a_x, io_ctx)?
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

        Ok(self
            .into_par_iter()
            .map(|f| match f {
                FutureVal::Ready(t) => t,
                _ => unreachable!(),
            })
            .collect())
    }
}

impl<F: JoltField, T, Args> FutureExt<F, Rep3BigUintShare<F>, T, Args>
    for Vec<FutureVal<F, T, Args>>
where
    T: Send,
    Args: Send + Copy,
{
    #[tracing::instrument(skip_all, name = "FutureVals::fufill_batched", level = "trace")]
    fn fufill_batched<N: Rep3Network, MapFn>(
        mut self,
        io_ctx: &mut IoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(Rep3BigUintShare<F>, Args) -> T + Send + Sync,
    {
        // A2B
        {
            let (arithmetic, futures): (Vec<_>, Vec<&mut FutureVal<F, T, Args>>) = self
                .iter_mut()
                .filter_map(|f| match f {
                    FutureVal::Pending(FutureOp::A2B(a), _) => Some((std::mem::take(a), f)),
                    _ => None,
                })
                .unzip();

            let shares = if !arithmetic.is_empty() {
                rep3::conversion::a2b_many(&arithmetic, io_ctx)?
            } else {
                vec![]
            };

            futures
                .into_par_iter()
                .zip(shares.into_par_iter())
                .for_each(|(f, c)| match f {
                    FutureVal::Pending(FutureOp::A2B(..), args) => {
                        *f = FutureVal::Ready(map(c, *args));
                    }
                    _ => unreachable!(),
                });
        }

        Ok(self
            .into_par_iter()
            .map(|f| match f {
                FutureVal::Ready(t) => t,
                _ => unreachable!(),
            })
            .collect())
    }
}

#[derive(Debug, Clone)]
pub enum FutureOp<F: JoltField> {
    Mul(Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>),
    B2A(Rep3BigUintShare<F>),
    A2B(Rep3PrimeFieldShare<F>),
    RingA2B(Rep3PrimeFieldShare<F>),
    BitInject(Rep3RingShare<Bit>),
    CastToField(Rep3RingShare<u32>),
    CastToFieldB2A(Rep3RingShare<u32>),
}
