use ark_std::cfg_iter;
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    jolt::vm::{JoltPolynomials, JoltStuff},
    lasso::memory_checking::{
        ExogenousOpenings, Initializable, MemoryCheckingProver, MultisetHashes,
        StructuredPolynomialData,
    },
    poly::{dense_mlpoly::DensePolynomial, multilinear_polynomial::PolynomialEvaluation},
    utils::{math::Math, transcript::Transcript},
};
use mpc_core::protocols::rep3::{
    network::{IoContext, IoContextPool, Rep3NetworkWorker},
    PartyID,
};
use mpc_net::mpc_star::MpcStarNetWorker;

use crate::{
    poly::{
        commitment::Rep3CommitmentScheme, opening_proof::Rep3ProverOpeningAccumulator,
        Rep3DensePolynomial, Rep3MultilinearPolynomial, Rep3PolysConversion,
    },
    subprotocols::grand_product::Rep3BatchedGrandProductWorker,
    utils,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub trait MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    Network: Rep3NetworkWorker,
{
    type ReadWriteGrandProduct: Rep3BatchedGrandProductWorker<F, PCS, ProofTranscript, Network>
        + 'static;
    type InitFinalGrandProduct: Rep3BatchedGrandProductWorker<F, PCS, ProofTranscript, Network>
        + 'static;

    type Rep3Polynomials: StructuredPolynomialData<Rep3MultilinearPolynomial<F>> + ?Sized;
    type Openings: StructuredPolynomialData<F> + Sync + Initializable<F, Self::Preprocessing>;
    type ExogenousOpenings: ExogenousOpenings<F> + Sync;

    type Preprocessing;

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_memory_checking")]
    fn prove_memory_checking(
        pcs_setup: &PCS::Setup,
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &JoltStuff<Rep3MultilinearPolynomial<F>>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let (r_read_write, r_init_final, (read_write_batch_size, init_final_batch_size)) =
            Self::prove_grand_products(
                preprocessing,
                polynomials,
                jolt_polynomials,
                opening_accumulator,
                io_ctx,
                pcs_setup,
            )
            .context("while proving grand products")?;

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
        jolt_polynomials: &JoltStuff<Rep3MultilinearPolynomial<F>>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        io_ctx: &mut IoContextPool<Network>,
        pcs_setup: &PCS::Setup,
    ) -> Result<(Vec<F>, Vec<F>, (usize, usize))> {
        let (gamma, tau) = tracing::trace_span!("receive_gamma_tau")
            .in_scope(|| io_ctx.network().receive_request())?;

        let (read_write_leaves, init_final_leaves) = Self::compute_leaves(
            preprocessing,
            polynomials,
            jolt_polynomials,
            &gamma,
            &tau,
            io_ctx,
        )?;

        let (mut read_write_circuit, read_write_hashes) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves, io_ctx)
                .context("while computing read-write grand product")?;
        let (mut init_final_circuit, init_final_hashes) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves, io_ctx)
                .context("while computing init-final grand product")?;

        io_ctx
            .network()
            .send_response((read_write_hashes.clone(), init_final_hashes.clone()))?;

        let r_read_write = read_write_circuit.prove_grand_product_worker(
            Some(opening_accumulator),
            Some(pcs_setup),
            io_ctx,
        )?;
        let r_init_final = init_final_circuit.prove_grand_product_worker(
            Some(opening_accumulator),
            Some(pcs_setup),
            io_ctx,
        )?;

        let read_write_batch_size = read_write_hashes.len();
        let init_final_batch_size = init_final_hashes.len();

        Ok((
            r_read_write,
            r_init_final,
            (read_write_batch_size, init_final_batch_size),
        ))
    }

    fn compute_openings(
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &JoltStuff<Rep3MultilinearPolynomial<F>>,
        r_read_write: &[F],
        r_init_final: &[F],
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let read_write_polys: Vec<&_> = polynomials
            .read_write_values()
            .into_iter()
            .chain(Self::ExogenousOpenings::exogenous_data(jolt_polynomials))
            .collect::<Vec<_>>();

        let (read_write_evals, eq_read_write) =
            Rep3MultilinearPolynomial::batch_evaluate(&read_write_polys, r_read_write);

        opening_accumulator.append(
            &read_write_polys,
            DensePolynomial::new(eq_read_write),
            r_read_write.to_vec(),
            &read_write_evals
                .iter()
                .map(|x| x.into_additive(io_ctx.id))
                .collect::<Vec<_>>(),
            io_ctx.main(),
        )?;
        tracing::info!("read_write_evals appended");

        let init_final_polys = polynomials.init_final_values();
        let (init_final_evals, eq_init_final) =
            Rep3MultilinearPolynomial::batch_evaluate(&init_final_polys, r_init_final);

        tracing::info!("init_final_evals appending");
        opening_accumulator.append(
            &polynomials.init_final_values(),
            DensePolynomial::new(eq_init_final),
            r_init_final.to_vec(),
            &init_final_evals
                .iter()
                .map(|x| x.into_additive(io_ctx.id))
                .collect::<Vec<_>>(),
            io_ctx.main(),
        )?;
        tracing::info!("init_final_evals appended");

        Ok(())
    }

    /// Computes the MLE of the leaves of the read, write, init, and final grand product circuits,
    /// one of each type per memory.
    /// Returns: (interleaved read/write leaves, interleaved init/final leaves)
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &JoltStuff<Rep3MultilinearPolynomial<F>>,
        gamma: &F,
        tau: &F,
        io_ctx: &mut IoContextPool<Network>,
    ) -> Result<(
        <Self::ReadWriteGrandProduct as Rep3BatchedGrandProductWorker<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::Leaves,
        <Self::InitFinalGrandProduct as Rep3BatchedGrandProductWorker<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::Leaves,
    )>;

    /// Constructs a batched grand product circuit for the read and write multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Rep3Polynomials,
        read_write_leaves: <Self::ReadWriteGrandProduct as Rep3BatchedGrandProductWorker<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::Leaves,
        io_ctx: &mut IoContextPool<Network>,
    ) -> Result<(Self::ReadWriteGrandProduct, Vec<F>)> {
        let batched_circuit = Self::ReadWriteGrandProduct::construct(read_write_leaves, io_ctx)?;
        let claims = batched_circuit.claimed_outputs();
        Ok((batched_circuit, claims))
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::init_final_grand_product")]
    fn init_final_grand_product(
        _preprocessing: &Self::Preprocessing,
        _polynomials: &Self::Rep3Polynomials,
        init_final_leaves: <Self::InitFinalGrandProduct as Rep3BatchedGrandProductWorker<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::Leaves,
        io_ctx: &mut IoContextPool<Network>,
    ) -> Result<(Self::InitFinalGrandProduct, Vec<F>)> {
        let batched_circuit = Self::InitFinalGrandProduct::construct(init_final_leaves, io_ctx)?;
        let claims = batched_circuit.claimed_outputs();
        Ok((batched_circuit, claims))
    }
}
