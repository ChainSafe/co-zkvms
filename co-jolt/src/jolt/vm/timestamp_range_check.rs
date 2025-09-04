use crate::{
    field::JoltField,
    jolt::vm::read_write_memory::witness::{Rep3ProgramIO, Rep3ReadWriteMemoryPolynomials},
};

use jolt_core::{
    jolt::vm::timestamp_range_check::{TimestampRangeCheckPolynomials, TimestampRangeCheckStuff},
    lasso::memory_checking::NoPreprocessing,
};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker, WorkerIoContext},
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
        ops: &mut [crate::jolt::vm::JoltTraceStep<Instructions>],
        _: &Rep3ProgramIO<F>,
        M: usize,
        network: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: crate::jolt::instruction::JoltInstructionSet
            + crate::jolt::instruction::Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker,
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

pub fn get_timestamp_range_check_polynomials<F: JoltField, PCS, ProofTranscript>(
    rw_polys: &mut Rep3ReadWriteMemoryPolynomials<F>,
) -> TimestampRangeCheckPolynomials<F>
where
    PCS: crate::poly::commitment::Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: jolt_core::utils::transcript::Transcript,
{
    // Take timestamp polys to use for witness generation
    let mut polys = jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryPolynomials {
        t_read_rd: std::mem::take(rw_polys.t_read_rd.as_public_mut()),
        t_read_rs1: std::mem::take(rw_polys.t_read_rs1.as_public_mut()),
        t_read_rs2: std::mem::take(rw_polys.t_read_rs2.as_public_mut()),
        t_read_ram: std::mem::take(rw_polys.t_read_ram.as_public_mut()),
        ..Default::default()
    };

    let res = jolt_core::jolt::vm::timestamp_range_check::TimestampValidityProof::<
        F,
        PCS,
        ProofTranscript,
    >::generate_witness(&polys);

    // Put back
    std::mem::swap(&mut polys.t_read_ram, rw_polys.t_read_rd.as_public_mut());
    std::mem::swap(&mut polys.t_read_ram, rw_polys.t_read_rs1.as_public_mut());
    std::mem::swap(&mut polys.t_read_ram, rw_polys.t_read_rs2.as_public_mut());
    std::mem::swap(&mut polys.t_read_ram, rw_polys.t_read_ram.as_public_mut());
    res
}
