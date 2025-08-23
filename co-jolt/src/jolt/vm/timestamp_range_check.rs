use crate::field::JoltField;

use jolt_core::{
    jolt::vm::timestamp_range_check::{TimestampRangeCheckPolynomials, TimestampRangeCheckStuff},
    lasso::memory_checking::NoPreprocessing,
};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker},
};

use crate::{jolt::vm::witness::Rep3Polynomials, poly::Rep3MultilinearPolynomial};

pub type Rep3TimestampRangeCheckPolynomials<F> =
    TimestampRangeCheckStuff<Rep3MultilinearPolynomial<F>>;

impl<F: JoltField> Rep3Polynomials<F, NoPreprocessing> for Rep3TimestampRangeCheckPolynomials<F> {
    type PublicPolynomials = TimestampRangeCheckPolynomials<F>;

    #[tracing::instrument(
        skip_all,
        name = "Rep3TimestampRangeCheckPolynomials::stream_secret_shares",
        level = "trace"
    )]
    fn stream_secret_shares<R: rand::Rng, Network: Rep3NetworkCoordinator>(
        _: &NoPreprocessing,
        polynomials: Self::PublicPolynomials,
        _: &mut R,
        network: &mut Network,
    ) -> eyre::Result<()> {
        let polys = (0..3)
            .map(|_| Self {
                read_cts_read_timestamp: Rep3MultilinearPolynomial::public_vec(
                    polynomials.read_cts_read_timestamp.to_vec(),
                )
                .try_into()
                .unwrap(),
                read_cts_global_minus_read: Rep3MultilinearPolynomial::public_vec(
                    polynomials.read_cts_global_minus_read.to_vec(),
                )
                .try_into()
                .unwrap(),
                final_cts_read_timestamp: Rep3MultilinearPolynomial::public_vec(
                    polynomials.final_cts_read_timestamp.to_vec(),
                )
                .try_into()
                .unwrap(),
                final_cts_global_minus_read: Rep3MultilinearPolynomial::public_vec(
                    polynomials.final_cts_global_minus_read.to_vec(),
                )
                .try_into()
                .unwrap(),
                identity: polynomials
                    .identity
                    .as_ref()
                    .map(|poly| Rep3MultilinearPolynomial::public(poly.clone())),
            })
            .collect();

        network.send_requests(polys)?;

        Ok(())
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3TimestampRangeCheckPolynomials::receive_witness_share",
        level = "trace"
    )]
    fn receive_witness_share<Network: Rep3NetworkWorker>(
        _: &NoPreprocessing,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Self> {
        let polys = io_ctx.network().receive_request()?;
        Ok(polys)
    }

    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &NoPreprocessing,
        ops: &mut [crate::jolt::vm::JoltTraceStep<F, Instructions>],
        M: usize,
        network: rep3::network::IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: crate::jolt::instruction::JoltInstructionSet<F>
            + crate::jolt::instruction::Rep3JoltInstructionSet<F>,
        Network: rep3::network::Rep3Network,
    {
        unimplemented!()
    }

    fn combine_polynomials(
        preprocessing: &NoPreprocessing,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials> {
        unimplemented!()
    }
}
