use crate::{
    field::JoltField,
    jolt::vm::witness::Rep3Polynomials,
    poly::{generate_poly_shares_rep3, Rep3MultilinearPolynomial},
};
use jolt_core::{
    jolt::vm::bytecode::{BytecodePolynomials, BytecodePreprocessing, BytecodeStuff},
};
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use rand::Rng;

pub type Rep3BytecodePolynomials<F> = BytecodeStuff<Rep3MultilinearPolynomial<F>>;

impl<F: JoltField> Rep3Polynomials<F, BytecodePreprocessing<F>> for Rep3BytecodePolynomials<F> {
    type PublicPolynomials = BytecodePolynomials<F>;

    #[tracing::instrument(
        skip_all,
        name = "Rep3BytecodePolynomials::stream_secret_shares",
        level = "trace"
    )]
    fn stream_secret_shares<R: Rng, Network: Rep3NetworkCoordinator>(
        _preprocessing: &BytecodePreprocessing<F>,
        polynomials: Self::PublicPolynomials,
        rng: &mut R,
        network: &mut Network,
    ) -> eyre::Result<()> {
        let v_imm = polynomials.v_read_write.last().unwrap();
        let mut v_imm_shares = generate_poly_shares_rep3(v_imm, rng);
        let polys = (0..3)
            .map(|i| {
                let v_read_write = [
                    Rep3MultilinearPolynomial::public_vec(polynomials.v_read_write[..5].to_vec()),
                    vec![std::mem::take(&mut v_imm_shares[i])],
                ]
                .concat()
                .try_into()
                .unwrap();
                BytecodeStuff {
                    a_read_write: Rep3MultilinearPolynomial::public(
                        polynomials.a_read_write.clone(),
                    ),
                    v_read_write: v_read_write,
                    t_read: Rep3MultilinearPolynomial::public(polynomials.t_read.clone()),
                    t_final: Rep3MultilinearPolynomial::public(polynomials.t_final.clone()),
                    ..Default::default()
                }
            })
            .collect();

        network.send_requests(polys)?;

        Ok(())
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3BytecodePolynomials::receive_witness_share",
        level = "trace"
    )]
    fn receive_witness_share<Network: Rep3NetworkWorker>(
        _: &BytecodePreprocessing<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Self> {
        let polys = io_ctx.network().receive_request()?;
        Ok(polys)
    }

    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &BytecodePreprocessing<F>,
        trace: &mut [crate::jolt::vm::JoltTraceStep<F, Instructions>],
        M: usize,
        network: mpc_core::protocols::rep3::network::IoContext<Network>,
    ) -> eyre::Result<Self>
    where
        Instructions: crate::jolt::instruction::JoltInstructionSet<F>
            + crate::jolt::instruction::Rep3JoltInstructionSet<F>,
        Network: mpc_core::protocols::rep3::network::Rep3Network,
    {
        todo!()
    }

    fn combine_polynomials(
        preprocessing: &BytecodePreprocessing<F>,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<Self::PublicPolynomials> {
        todo!()
    }
}
