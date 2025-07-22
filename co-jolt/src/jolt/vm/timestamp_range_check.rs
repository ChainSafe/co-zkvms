use jolt_core::{
    field::JoltField,
    jolt::vm::timestamp_range_check::{TimestampRangeCheckPolynomials, TimestampRangeCheckStuff},
    lasso::memory_checking::NoPreprocessing,
};
use mpc_core::protocols::rep3;

use crate::{jolt::vm::witness::Rep3Polynomials, poly::Rep3MultilinearPolynomial};

pub type Rep3TimestampRangeCheckPolynomials<F: JoltField> =
    TimestampRangeCheckStuff<Rep3MultilinearPolynomial<F>>;

impl<F: JoltField> Rep3Polynomials<F, NoPreprocessing> for Rep3TimestampRangeCheckPolynomials<F> {
    type PublicPolynomials = TimestampRangeCheckPolynomials<F>;

    fn generate_secret_shares<R: rand::Rng>(
        _: &NoPreprocessing,
        polynomials: Self::PublicPolynomials,
        _: &mut R,
    ) -> Vec<Self> {
        (0..3)
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
            .collect()
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
