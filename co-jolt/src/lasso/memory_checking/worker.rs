use color_eyre::eyre::Result;
use eyre::Context;
use jolt_core::{
    jolt::vm::JoltStuff,
    lasso::memory_checking::{ExogenousOpenings, StructuredPolynomialData},
    poly::dense_mlpoly::DensePolynomial,
    utils::{math::Math, transcript::Transcript},
};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::{field::JoltField, jolt::vm::witness::WorkerInitializable};
use crate::{
    poly::{
        commitment::Rep3CommitmentScheme, opening_proof::Rep3OpeningAccumulatorWorker,
        Rep3MultilinearPolynomial,
    },
    subprotocols::grand_product::Rep3BatchedGrandProductWorker,
};

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
    type Openings: StructuredPolynomialData<F> + Sync + WorkerInitializable<F, Self::Preprocessing>;
    type ExogenousOpenings: ExogenousOpenings<F> + Sync;

    type Preprocessing;

    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_memory_checking")]
    fn prove_memory_checking(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &JoltStuff<Rep3MultilinearPolynomial<F>>,
        opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        tracing::info!("worker: prove_memory_checking - start");

        let (r_read_write, r_init_final, (read_write_batch_size, init_final_batch_size)) =
            Self::prove_grand_products(preprocessing, polynomials, jolt_polynomials, io_ctx)
                .context("while proving grand products")?;

        let r_read_write_opening =
            &r_read_write[read_write_batch_size.next_power_of_two().log_2()..];
        let r_init_final_opening =
            &r_init_final[init_final_batch_size.next_power_of_two().log_2()..];

        Self::compute_openings(
            preprocessing,
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
        io_ctx: &mut IoContextPool<Network>,
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

        let (mut read_write_circuit, read_write_hashes, read_write_batch_size) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves, io_ctx)
                .context("while computing read-write grand product")?;

        let (mut init_final_circuit, init_final_hashes, init_final_batch_size) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves, io_ctx)
                .context("while computing init-final grand product")?;

        if let Some(read_write_hashes) = read_write_hashes {
            io_ctx.network().send_response(read_write_hashes)?
        }
        if let Some(init_final_hashes) = init_final_hashes {
            io_ctx.network().send_response(init_final_hashes)?
        }

        let r_read_write = read_write_circuit.prove_grand_product_worker(io_ctx)?;

        let r_init_final = init_final_circuit.prove_grand_product_worker(io_ctx)?;

        Ok((
            r_read_write,
            r_init_final,
            (read_write_batch_size, init_final_batch_size),
        ))
    }

    fn compute_openings(
        _: &Self::Preprocessing,
        opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &JoltStuff<Rep3MultilinearPolynomial<F>>,
        r_read_write: &[F],
        r_init_final: &[F],
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        compute_openings::<F, Self::ExogenousOpenings, _, _>(
            opening_accumulator,
            polynomials,
            jolt_polynomials,
            r_read_write,
            r_init_final,
            io_ctx,
        )
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
    ) -> Result<(
        Self::ReadWriteGrandProduct,
        Option<Vec<AdditiveShare<F>>>,
        usize,
    )> {
        let (batched_circuit, full_batch_size) =
            Self::ReadWriteGrandProduct::construct(read_write_leaves, io_ctx)?;
        let claims = batched_circuit.claimed_outputs();
        Ok((batched_circuit, claims, full_batch_size))
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
    ) -> Result<(
        Self::InitFinalGrandProduct,
        Option<Vec<AdditiveShare<F>>>,
        usize,
    )> {
        let (batched_circuit, full_batch_size) =
            Self::InitFinalGrandProduct::construct(init_final_leaves, io_ctx)?;
        let claims = batched_circuit.claimed_outputs();
        Ok((batched_circuit, claims, full_batch_size))
    }
}

pub(crate) fn compute_openings<F: JoltField, ExoOpenings, Polynomials, Network: Rep3NetworkWorker>(
    opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
    polynomials: &Polynomials,
    jolt_polynomials: &JoltStuff<Rep3MultilinearPolynomial<F>>,
    r_read_write: &[F],
    r_init_final: &[F],
    io_ctx: &mut IoContextPool<Network>,
) -> eyre::Result<()>
where
    Polynomials: StructuredPolynomialData<Rep3MultilinearPolynomial<F>> + ?Sized,
    ExoOpenings: ExogenousOpenings<F> + Sync,
{
    let log_num_workers = io_ctx.network().log_num_workers();
    let worker_idx = io_ctx.worker_idx();

    let read_write_polys: Vec<&_> = polynomials
        .read_write_values_grand_product()
        .into_iter()
        .chain(ExoOpenings::exogenous_data(jolt_polynomials))
        .collect::<Vec<_>>();

    let (read_write_evals, eq_read_write) = Rep3MultilinearPolynomial::batch_evaluate_worker(
        &read_write_polys,
        r_read_write,
        log_num_workers,
        worker_idx,
    );

    opening_accumulator.append_send_claims(
        &read_write_polys,
        DensePolynomial::new(eq_read_write),
        r_read_write.to_vec(),
        &read_write_evals,
        io_ctx.main(),
    )?;

    let init_final_polys = polynomials.init_final_values();

    let (init_final_evals, eq_init_final) = Rep3MultilinearPolynomial::batch_evaluate_worker(
        &init_final_polys,
        r_init_final,
        log_num_workers,
        worker_idx,
    );

    opening_accumulator.append_send_claims(
        &polynomials.init_final_values(),
        DensePolynomial::new(eq_init_final),
        r_init_final.to_vec(),
        &init_final_evals,
        io_ctx.main(),
    )?;

    Ok(())
}
