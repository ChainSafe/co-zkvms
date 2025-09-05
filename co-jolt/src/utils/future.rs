use crate::field::JoltField;
use itertools::Itertools;
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare, Rep3PrimeFieldShare,
};

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub enum FutureRep3<F: JoltField, T, Args = ()> {
    Ready(T),
    Pending(FutureOp<F>, Args),
}

#[derive(Debug, Clone)]
pub enum FutureOp<F: JoltField> {
    // Out: Rep3PrimeFieldShare<F>
    Mul(Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>),
    Cmux(
        Rep3PrimeFieldShare<F>,
        Rep3PrimeFieldShare<F>,
        Rep3PrimeFieldShare<F>,
    ),
    B2A(Rep3BigUintShare<F>),

    // Out: Rep3BigUintShare<F>
    A2B(Rep3PrimeFieldShare<F>),
}

impl<F: JoltField, T, Extra> FutureRep3<F, T, Extra> {
    pub fn as_ready(&self) -> &T {
        match self {
            FutureRep3::Ready(t) => t,
            _ => panic!("FutureVal is not ready"),
        }
    }

    pub fn mul_args(a: Rep3PrimeFieldShare<F>, b: Rep3PrimeFieldShare<F>, args: Extra) -> Self {
        FutureRep3::Pending(FutureOp::Mul(a, b), args)
    }
}

impl<F: JoltField, T> FutureRep3<F, T> {
    // ===== Field Ops =====

    pub fn b2a(a: Rep3BigUintShare<F>) -> Self {
        FutureRep3::Pending(FutureOp::B2A(a), ())
    }

    pub fn mul(a: Rep3PrimeFieldShare<F>, b: Rep3PrimeFieldShare<F>) -> Self {
        FutureRep3::Pending(FutureOp::Mul(a, b), ())
    }

    pub fn cmux(
        cond: Rep3PrimeFieldShare<F>,
        truthy: Rep3PrimeFieldShare<F>,
        falsy: Rep3PrimeFieldShare<F>,
    ) -> Self {
        FutureRep3::Pending(FutureOp::Cmux(cond, truthy, falsy), ())
    }

    // ===== BigUint Ops =====

    pub fn a2b(a: Rep3PrimeFieldShare<F>) -> Self {
        FutureRep3::Pending(FutureOp::A2B(a), ())
    }
}

pub trait FutureExt<F: JoltField, U, T, Args> {
    fn fulfill_batched<N: Rep3Network, MapFn: Fn(U, Args) -> T + Send>(
        self,
        io_ctx: &mut IoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(U, Args) -> T + Send + Sync;
}

impl<F: JoltField, T, Args> FutureExt<F, Rep3PrimeFieldShare<F>, T, Args>
    for Vec<FutureRep3<F, T, Args>>
where
    T: Clone + Default + Send,
    Args: Send + Copy,
{
    #[tracing::instrument(skip_all, name = "FutureVals::fulfill_batched", level = "trace")]
    fn fulfill_batched<N: Rep3Network, MapFn>(
        self,
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
        let (mut conds, mut truthy, mut falsy, mut fut_cmux, mut cmux_args) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

        self.into_iter()
            .zip_eq(fufilled.iter_mut())
            .for_each(|(f, fufilled)| match f {
                FutureRep3::Pending(FutureOp::Mul(a, b), args) => {
                    mul_x.push(a);
                    mul_y.push(b);
                    fut_muls.push(fufilled);
                    args_mul.push(args);
                }
                FutureRep3::Pending(FutureOp::B2A(x), args) => {
                    b2a_x.push(x);
                    fut_b2a.push(fufilled);
                    b2a_args.push(args);
                }
                FutureRep3::Pending(FutureOp::Cmux(c, t, f), args) => {
                    conds.push(c);
                    truthy.push(t);
                    falsy.push(f);
                    fut_cmux.push(fufilled);
                    cmux_args.push(args);
                }
                FutureRep3::Ready(x) => {
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

        // Cmux
        {
            let c = if !conds.is_empty() {
                rep3::arithmetic::cmux_many(&conds, &truthy, &falsy, io_ctx)?
            } else {
                vec![]
            };

            fut_cmux
                .into_par_iter()
                .zip_eq(c.into_par_iter())
                .zip_eq(cmux_args)
                .for_each(|((f, c), args)| {
                    *f = map(c, args);
                });
        }

        Ok(fufilled)
    }
}

impl<F: JoltField, T, Args> FutureExt<F, Rep3BigUintShare<F>, T, Args>
    for Vec<FutureRep3<F, T, Args>>
where
    T: Send,
    Args: Send + Copy,
{
    #[tracing::instrument(skip_all, name = "FutureVals::fulfill_batched", level = "trace")]
    fn fulfill_batched<N: Rep3Network, MapFn>(
        mut self,
        io_ctx: &mut IoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(Rep3BigUintShare<F>, Args) -> T + Send + Sync,
    {
        // A2B
        {
            let (arithmetic, futures): (Vec<_>, Vec<&mut FutureRep3<F, T, Args>>) = self
                .iter_mut()
                .filter_map(|f| match f {
                    FutureRep3::Pending(FutureOp::A2B(a), _) => Some((std::mem::take(a), f)),
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
                    FutureRep3::Pending(FutureOp::A2B(..), args) => {
                        *f = FutureRep3::Ready(map(c, *args));
                    }
                    _ => unreachable!(),
                });
        }

        Ok(self
            .into_par_iter()
            .map(|f| match f {
                FutureRep3::Ready(t) => t,
                _ => unreachable!(),
            })
            .collect())
    }
}
