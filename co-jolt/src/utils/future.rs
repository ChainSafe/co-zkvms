use itertools::Itertools;
use crate::field::JoltField;
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
    Rep3PrimeFieldShare, Rep3BigUintShare,
};

use rayon::prelude::*;

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

    pub fn mul_args(
        a: Rep3PrimeFieldShare<F>,
        b: Rep3PrimeFieldShare<F>,
        args: Extra,
    ) -> Self {
        FutureVal::Pending(FutureOp::Mul(a, b), args)
    }
}

impl<F: JoltField> FutureVal<F, Rep3PrimeFieldShare<F>> {
    pub fn b2a(a: Rep3BigUintShare<F>) -> Self {
        FutureVal::Pending(FutureOp::B2A(a), ())
    }
}

pub trait FutureExt<F: JoltField, T, Args> {
    fn fufill_batched<N: Rep3Network, MapFn: Fn(Rep3PrimeFieldShare<F>, Args) -> T + Send>(
        self,
        io_ctx: &mut IoContext<N>,
        map: MapFn,
    ) -> eyre::Result<Vec<T>>
    where
        MapFn: Fn(Rep3PrimeFieldShare<F>, Args) -> T + Send + Sync;
}

impl<F: JoltField, T, Args> FutureExt<F, T, Args> for Vec<FutureVal<F, T, Args>>
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

        Ok(self
            .into_par_iter()
            .map(|f| match f {
                FutureVal::Ready(t) => t,
                _ => unreachable!(),
            })
            .collect())
    }
}

pub enum FutureOp<F: JoltField> {
    Mul(Rep3PrimeFieldShare<F>, Rep3PrimeFieldShare<F>),
    B2A(Rep3BigUintShare<F>),
}
