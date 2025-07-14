use ark_std::cfg_iter;
use color_eyre::eyre::Result;
use eyre::Context;
use jolt_core::poly::{field::JoltField, structured_poly::StructuredCommitment};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use mpc_net::mpc_star::MpcStarNetWorker;

use crate::{
    poly::{Rep3DensePolynomial, Rep3StructuredOpeningProof},
    subprotocols::{
        commitment::DistributedCommitmentScheme,
        grand_product::{
            BatchedGrandProductProver, BatchedRep3GrandProductCircuit, Rep3GrandProductCircuit,
        },
    },
    utils, Rep3Polynomials as _,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub trait MemoryCheckingProverRep3Worker<F, CS, Network>
where
    F: JoltField,
    Network: Rep3Network + MpcStarNetWorker,
    CS: DistributedCommitmentScheme<F>,
    Self::Polynomials: StructuredCommitment<CS>,
{
    type Polynomials;
    type Rep3Polynomials: crate::Rep3Polynomials + ?Sized;
    type Preprocessing;
    type ReadWriteOpenings: Rep3StructuredOpeningProof<
        F,
        CS,
        Self::Polynomials,
        Rep3Polynomials = Self::Rep3Polynomials,
    >;
    type InitFinalOpenings: Rep3StructuredOpeningProof<
        F,
        CS,
        Self::Polynomials,
        Rep3Polynomials = Self::Rep3Polynomials,
    >;

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_memory_checking")]
    fn prove_memory_checking(
        preprocessing: &Self::Preprocessing,
        setup: &CS::Setup,
        polynomials: &Self::Rep3Polynomials,
        io_ctx: &mut IoContext<Network>,
    ) -> Result<()> {
        let (r_read_write, r_init_final) =
            Self::prove_grand_products(preprocessing, polynomials, io_ctx)
                .context("while proving grand products")?;

        Self::ReadWriteOpenings::open_rep3_worker(polynomials, &r_read_write, &mut io_ctx.network)
            .context("while opening read-write polynomials")?;
        Self::ReadWriteOpenings::prove_openings_rep3_worker(
            polynomials,
            &r_read_write,
            setup,
            &mut io_ctx.network,
        )
        .context("while proving read-write openings")?;
        Self::InitFinalOpenings::open_rep3_worker(polynomials, &r_init_final, &mut io_ctx.network)
            .context("while opening init-final polynomials")?;
        Self::InitFinalOpenings::prove_openings_rep3_worker(
            polynomials,
            &r_init_final,
            setup,
            &mut io_ctx.network,
        )
        .context("while proving init-final openings")?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_grand_products")]
    fn prove_grand_products(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        io_ctx: &mut IoContext<Network>,
    ) -> Result<(Vec<F>, Vec<F>)> {
        let (gamma, tau) = io_ctx.network.receive_request()?;
        io_ctx.network.send_response(polynomials.num_lookups())?;

        let (read_write_leaves, init_final_leaves) =
            Self::compute_leaves(preprocessing, polynomials, &gamma, &tau, io_ctx);

        let (read_write_circuit, read_write_hashes) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves, io_ctx)
                .context("while computing read-write grand product")?;
        let (init_final_circuit, init_final_hashes) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves, io_ctx)
                .context("while computing init-final grand product")?;

        io_ctx
            .network
            .send_response((read_write_hashes.clone(), init_final_hashes.clone()))?;

        let r_read_write =
            BatchedGrandProductProver::prove_worker(read_write_circuit, &mut io_ctx.network)
                .context("while proving read-write grand product")?;
        let r_init_final =
            BatchedGrandProductProver::prove_worker(init_final_circuit, &mut io_ctx.network)
                .context("while proving init-final grand product")?;

        Ok((r_read_write, r_init_final))
    }

    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        gamma: &F,
        tau: &F,
        io_ctx: &mut IoContext<Network>,
    ) -> (Vec<Rep3DensePolynomial<F>>, Vec<Rep3DensePolynomial<F>>);

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::read_write_grand_product")]
    fn read_write_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Rep3Polynomials,
        read_write_leaves: Vec<Rep3DensePolynomial<F>>,
        io_ctx: &mut IoContext<Network>,
    ) -> Result<(BatchedRep3GrandProductCircuit<F>, Vec<F>)> {
        let read_write_circuits: Vec<Rep3GrandProductCircuit<F>> =
            utils::fork_map(read_write_leaves, io_ctx, |leaves, io_ctx| {
                Rep3GrandProductCircuit::new(&leaves, io_ctx).unwrap()
            })?;

        let read_write_hashes: Vec<F> = cfg_iter!(read_write_circuits)
            .map(|circuit| circuit.evaluate())
            .collect();

        Ok((
            BatchedRep3GrandProductCircuit::new_batch(read_write_circuits),
            read_write_hashes,
        ))
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::init_final_grand_product")]
    fn init_final_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Rep3Polynomials,
        init_final_leaves: Vec<Rep3DensePolynomial<F>>,
        io_ctx: &mut IoContext<Network>,
    ) -> Result<(BatchedRep3GrandProductCircuit<F>, Vec<F>)> {
        let init_final_circuits: Vec<Rep3GrandProductCircuit<F>> =
            utils::fork_map(init_final_leaves, io_ctx, |leaves, io_ctx| {
                Rep3GrandProductCircuit::new(&leaves, io_ctx).unwrap()
            })?;

        let init_final_hashes: Vec<F> = cfg_iter!(init_final_circuits)
            .map(|circuit| circuit.evaluate())
            .collect();

        Ok((
            BatchedRep3GrandProductCircuit::new_batch(init_final_circuits),
            init_final_hashes,
        ))
    }
}
