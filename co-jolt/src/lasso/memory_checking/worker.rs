use color_eyre::eyre::Result;
use eyre::Context;
use jolt_core::{
    jolt::vm::JoltStuff,
    lasso::memory_checking::{ExogenousOpenings, Initializable, StructuredPolynomialData},
    poly::{dense_mlpoly::DensePolynomial, multilinear_polynomial::PolynomialEvaluation},
    utils::{math::Math, transcript::Transcript},
};
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};

use crate::{field::JoltField, jolt::vm::read_write_memory};
use crate::{
    poly::{
        commitment::Rep3CommitmentScheme, opening_proof::Rep3ProverOpeningAccumulator,
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
        tracing::info!("worker: prove_memory_checking - start");

        let (read_write, init_final) = Self::prove_grand_products(
            preprocessing,
            polynomials,
            jolt_polynomials,
            opening_accumulator,
            io_ctx,
            pcs_setup,
        )
        .context("while proving grand products")?;

        let r_read_write_point = read_write.map(|(r_read_write, read_write_batch_size)| {
            let (_, r_read_write_point) =
                r_read_write.split_at(read_write_batch_size.next_power_of_two().log_2());
            r_read_write_point.to_vec()
        });
        let r_init_final_point = init_final.map(|(r_init_final, init_final_batch_size)| {
            let (_, r_init_final_point) =
                r_init_final.split_at(init_final_batch_size.next_power_of_two().log_2());
            r_init_final_point.to_vec()
        });

        Self::compute_openings(
            opening_accumulator,
            polynomials,
            jolt_polynomials,
            r_read_write_point,
            r_init_final_point,
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
    ) -> Result<(Option<(Vec<F>, usize)>, Option<(Vec<F>, usize)>)> {
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

        let (read_write_circuit, read_write_hashes) = read_write_leaves
            .map(|leaves| {
                Self::read_write_grand_product(preprocessing, polynomials, leaves, io_ctx)
                    .context("while computing read-write grand product")
            })
            .transpose()?
            .unzip();

        let read_write_batch_size = read_write_hashes
            .map(|hashes| {
                let batch_size = hashes.len();
                io_ctx.network().send_response(hashes)?;
                eyre::Ok(batch_size)
            })
            .transpose()?;

        if io_ctx.party_idx() == 0 {
            println!("read_write_batch_size: {:?}", read_write_batch_size);
        }

        let (init_final_circuit, init_final_hashes) = init_final_leaves
            .map(|leaves| {
                Self::init_final_grand_product(preprocessing, polynomials, leaves, io_ctx)
                    .context("while computing init-final grand product")
            })
            .transpose()?
            .unzip();

        let init_final_batch_size = init_final_hashes
            .map(|hashes| {
                let batch_size = hashes.len();
                io_ctx.network().send_response(hashes)?;
                eyre::Ok(batch_size)
            })
            .transpose()?;

        if io_ctx.party_idx() == 0 {
            println!("init_final_batch_size: {:?}", init_final_batch_size);
        }

        let read_write = if let Some(mut circuit) = read_write_circuit {
            let batch_size = read_write_batch_size.expect("batch size expected");
            let r_read_write = circuit.prove_grand_product_worker(
                Some(opening_accumulator),
                Some(pcs_setup),
                io_ctx,
            )?;

            Some((r_read_write, batch_size))
        } else {
            None
        };

        let init_final = if let Some(mut circuit) = init_final_circuit {
            let batch_size = init_final_batch_size.expect("batch size expected");
            let r_init_final = circuit.prove_grand_product_worker(
                Some(opening_accumulator),
                Some(pcs_setup),
                io_ctx,
            )?;

            Some((r_init_final, batch_size))
        } else {
            None
        };

        Ok((read_write, init_final))
    }

    fn compute_openings(
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &JoltStuff<Rep3MultilinearPolynomial<F>>,
        r_read_write: Option<Vec<F>>,
        r_init_final: Option<Vec<F>>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let party_id = io_ctx.party_id();
        let log_num_workers = io_ctx.log_num_workers();

        if let Some(r_read_write) = r_read_write {
            let read_write_polys: Vec<&_> = polynomials
                .read_write_values()
                .into_iter()
                .chain(Self::ExogenousOpenings::exogenous_data(jolt_polynomials))
                .collect::<Vec<_>>();

            let (read_write_evals, eq_read_write) = Rep3MultilinearPolynomial::batch_evaluate(
                &read_write_polys,
                &r_read_write[..r_read_write.len() - log_num_workers],
            );

            io_ctx.network().send_response(
                read_write_evals
                    .iter()
                    .map(|x| x.into_additive(party_id))
                    .collect::<Vec<_>>(),
            )?;

            opening_accumulator.append_with_known_claim(
                &read_write_polys,
                DensePolynomial::new(eq_read_write),
                r_read_write.to_vec(),
                io_ctx.main(),
            )?;
        }
        println!("worker append read_write_polys opennings");

        if let Some(r_init_final) = r_init_final {
            let init_final_polys = polynomials.init_final_values();
            let (init_final_evals, eq_init_final) = Rep3MultilinearPolynomial::batch_evaluate(
                &init_final_polys,
                &r_init_final[..r_init_final.len() - log_num_workers],
            );
            io_ctx.network().send_response(
                init_final_evals
                    .iter()
                    .map(|x| x.into_additive(party_id))
                    .collect::<Vec<_>>(),
            )?;

            opening_accumulator.append_with_known_claim(
                &polynomials.init_final_values(),
                DensePolynomial::new(eq_init_final),
                r_init_final.to_vec(),
                io_ctx.main(),
            )?;
        }

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
        Option<
            <Self::ReadWriteGrandProduct as Rep3BatchedGrandProductWorker<
                F,
                PCS,
                ProofTranscript,
                Network,
            >>::Leaves,
        >,
        Option<
            <Self::InitFinalGrandProduct as Rep3BatchedGrandProductWorker<
                F,
                PCS,
                ProofTranscript,
                Network,
            >>::Leaves,
        >,
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
    ) -> Result<(Self::ReadWriteGrandProduct, Vec<AdditiveShare<F>>)> {
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
    ) -> Result<(Self::InitFinalGrandProduct, Vec<AdditiveShare<F>>)> {
        let batched_circuit = Self::InitFinalGrandProduct::construct(init_final_leaves, io_ctx)?;
        let claims = batched_circuit.claimed_outputs();
        Ok((batched_circuit, claims))
    }
}
