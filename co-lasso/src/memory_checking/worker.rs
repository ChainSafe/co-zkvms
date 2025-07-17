use ark_std::cfg_iter;
use color_eyre::eyre::Result;
use eyre::Context;
use jolt_core::{
    jolt::vm::JoltPolynomials,
    lasso::memory_checking::{
        ExogenousOpenings, Initializable, MemoryCheckingProver, MultisetHashes,
        StructuredPolynomialData,
    },
    poly::dense_mlpoly::DensePolynomial,
    utils::{math::Math, transcript::Transcript},
};
use mpc_core::protocols::rep3::network::{IoContext, Rep3NetworkWorker};
use mpc_net::mpc_star::MpcStarNetWorker;

use crate::{
    field::JoltField,
    poly::{opening_proof::Rep3ProverOpeningAccumulator, Rep3DensePolynomial},
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

pub trait MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    Network: Rep3NetworkWorker,
{
    // type ReadWriteGrandProduct: BatchedGrandProduct<F, PCS, ProofTranscript> + Send + 'static;
    // type InitFinalGrandProduct: BatchedGrandProduct<F, PCS, ProofTranscript> + Send + 'static;

    type Rep3Polynomials: StructuredPolynomialData<Rep3DensePolynomial<F>>
        + crate::Rep3Polynomials
        + ?Sized;
    type Openings: StructuredPolynomialData<F> + Sync + Initializable<F, Self::Preprocessing>;
    type Commitments: StructuredPolynomialData<PCS::Commitment>;
    type ExogenousOpenings: ExogenousOpenings<F> + Sync;

    type Preprocessing;

    type MemoryCheckingProof: MemoryCheckingProver<
        F,
        PCS,
        ProofTranscript,
        Preprocessing = Self::Preprocessing,
    >;
    // type ReadWriteOpenings: Rep3StructuredOpeningProof<
    //     F,
    //     PCS,
    //     Self::Polynomials,
    //     Rep3Polynomials = Self::Rep3Polynomials,
    // >;
    // type InitFinalOpenings: Rep3StructuredOpeningProof<
    //     F,
    //     PCS,
    //     Self::Polynomials,
    //     Rep3Polynomials = Self::Rep3Polynomials,
    // >;

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_memory_checking")]
    fn prove_memory_checking(
        _pcs_setup: &PCS::Setup,
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let (r_read_write, r_init_final, multiset_hashes) =
            Self::prove_grand_products(preprocessing, polynomials, io_ctx)
                .context("while proving grand products")?;

        let read_write_batch_size =
            multiset_hashes.read_hashes.len() + multiset_hashes.write_hashes.len();
        let init_final_batch_size =
            multiset_hashes.init_hashes.len() + multiset_hashes.final_hashes.len();

        let (_, r_read_write_opening) =
            r_read_write.split_at(read_write_batch_size.next_power_of_two().log_2());
        let (_, r_init_final_opening) =
            r_init_final.split_at(init_final_batch_size.next_power_of_two().log_2());

        Self::compute_openings(
            opening_accumulator,
            polynomials,
            jolt_polynomials,
            r_read_write_opening,
            r_init_final_opening,
            io_ctx,
        )?;

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_grand_products")]
    fn prove_grand_products(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        io_ctx: &mut IoContext<Network>,
    ) -> Result<(Vec<F>, Vec<F>, MultisetHashes<F>)> {
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

        let multiset_hashes = Self::MemoryCheckingProof::uninterleave_hashes(
            preprocessing,
            read_write_hashes.clone(),
            init_final_hashes.clone(),
        );

        io_ctx
            .network
            .send_response((read_write_hashes.clone(), init_final_hashes.clone()))?;

        let r_read_write =
            BatchedGrandProductProver::prove_worker(read_write_circuit, &mut io_ctx.network)
                .context("while proving read-write grand product")?;
        let r_init_final =
            BatchedGrandProductProver::prove_worker(init_final_circuit, &mut io_ctx.network)
                .context("while proving init-final grand product")?;

        Ok((r_read_write, r_init_final, multiset_hashes))
    }

    fn compute_openings(
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &JoltPolynomials<F>,
        r_read_write: &[F],
        r_init_final: &[F],
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let read_write_polys: Vec<_> = [
            polynomials.read_write_values(),
            // Self::ExogenousOpenings::exogenous_data(jolt_polynomials),
        ]
        .concat();
        let (read_write_evals, eq_read_write) =
            Rep3DensePolynomial::batch_evaluate(&read_write_polys, r_read_write);

        opening_accumulator.append(
            &read_write_polys,
            DensePolynomial::new(eq_read_write),
            r_read_write.to_vec(),
            &read_write_evals,
            io_ctx,
        );

        let init_final_polys = polynomials.init_final_values();
        let (init_final_evals, eq_init_final) =
            Rep3DensePolynomial::batch_evaluate(&init_final_polys, r_init_final);

        opening_accumulator.append(
            &polynomials.init_final_values(),
            DensePolynomial::new(eq_init_final),
            r_init_final.to_vec(),
            &init_final_evals,
            io_ctx,
        );

        Ok(())
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
