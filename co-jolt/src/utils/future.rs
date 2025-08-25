use crate::field::JoltField;
use itertools::Itertools;
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare, Rep3PrimeFieldShare,
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
        MapFn: Fn(Rep3PrimeFieldShare<F>, Args) -> T + Send + Sync,
    {
        // Multiply
        {
            let (a, b, futures): (Vec<_>, Vec<_>, Vec<&mut FutureVal<F, T, Args>>) = self
                .iter_mut()
                .filter_map(|f| match f {
                    FutureVal::Pending(FutureOp::Mul(a, b), _) => Some((*a, *b, f)),
                    _ => None,
                })
                .multiunzip();

            let c = if !a.is_empty() && !b.is_empty() {
                rep3::arithmetic::mul_vec(&a, &b, io_ctx)?
            } else {
                vec![]
            };

            futures
                .into_par_iter()
                .zip(c.into_par_iter())
                .for_each(|(f, c)| match f {
                    FutureVal::Pending(FutureOp::Mul(..), args) => {
                        *f = FutureVal::Ready(map(c, *args));
                    }
                    _ => unreachable!(),
                });
        }

        // B2A
        {
            let (binary, futures): (Vec<_>, Vec<&mut FutureVal<F, T, Args>>) = self
                .iter_mut()
                .filter_map(|f| match f {
                    FutureVal::Pending(FutureOp::B2A(a), _) => Some((std::mem::take(a), f)),
                    _ => None,
                })
                .unzip();

            let shares = if !binary.is_empty() {
                rep3::conversion::b2a_many(&binary, io_ctx)?
            } else {
                vec![]
            };

            futures
                .into_par_iter()
                .zip(shares.into_par_iter())
                .for_each(|(f, c)| match f {
                    FutureVal::Pending(FutureOp::B2A(..), args) => {
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
}
