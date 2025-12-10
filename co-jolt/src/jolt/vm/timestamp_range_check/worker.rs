use eyre::Context;
use jolt_common::constants::MEMORY_OPS_PER_INSTRUCTION;
use jolt_core::{
    jolt::vm::timestamp_range_check::{
        ReadTimestampOpenings, TimestampRangeCheckOpenings, TimestampRangeCheckStuff,
        TimestampValidityProof,
    },
    lasso::memory_checking::{
        ExogenousOpenings, MemoryCheckingProver, NoPreprocessing, StructuredPolynomialData,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::{CompactPolynomial, SmallScalar},
        dense_mlpoly::DensePolynomial,
    },
    subprotocols::grand_product::{BatchedDenseGrandProduct, BatchedGrandProductProof},
    utils::{
        thread::drop_in_background_thread,
        transcript::{KeccakTranscript, Transcript},
    },
};
use mpc_core::protocols::rep3::{
    network::{IoContextPool, Rep3NetworkWorker},
    PartyID,
};

use rayon::prelude::*;
use snarks_core::math::Math;
use tokio::io;

use crate::{
    field::JoltField,
    jolt::vm::{witness::Rep3JoltPolynomials, Jolt},
    lasso::memory_checking::worker::MemoryCheckingProverRep3Worker,
    poly::{opening_proof::Rep3OpeningAccumulatorWorker, Rep3MultilinearPolynomial},
    subprotocols::grand_product::{Rep3BatchedGrandProduct, Rep3BatchedGrandProductWorker},
};

pub trait TimestampValidityDistributredWorker<F: JoltField, PCS, Network: Rep3NetworkWorker> {
    fn prove_distributed_worker(
        polynomials: &TimestampRangeCheckStuff<Rep3MultilinearPolynomial<F>>,
        jolt_polynomials: &Rep3JoltPolynomials<F>,
        opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()>;

    fn prove_grand_products_distributed(
        polynomials: &TimestampRangeCheckStuff<Rep3MultilinearPolynomial<F>>,
        jolt_polynomials: &Rep3JoltPolynomials<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Vec<F>>;
}

impl<
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
        Network: Rep3NetworkWorker,
    > TimestampValidityDistributredWorker<F, PCS, Network>
    for TimestampValidityProof<F, PCS, ProofTranscript>
{
    #[tracing::instrument(skip_all, name = "TimestampValidityProof::prove_distributed")]
    fn prove_distributed_worker(
        polynomials: &TimestampRangeCheckStuff<Rep3MultilinearPolynomial<F>>,
        jolt_polynomials: &Rep3JoltPolynomials<F>,
        opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        // temporary hack: only one worker computes timestamp grand product
        io_ctx.network().set_log_num_workers(0);
        let _r_grand_product = if io_ctx.worker_idx() == 0 {
            Self::prove_grand_products_distributed(polynomials, jolt_polynomials, io_ctx)?
        } else {
            vec![]
        };
        io_ctx.network().reset_log_num_workers();

        let r_grand_product: Vec<F> = io_ctx.network().receive_request()?;

        const BATCH_SIZE: usize = MEMORY_OPS_PER_INSTRUCTION * 6 + 1;
        let r_opening = &r_grand_product[BATCH_SIZE.next_power_of_two().log_2()..];

        let read_write_polys = [
            polynomials.read_write_values(),
            ReadTimestampOpenings::<F>::exogenous_data(jolt_polynomials),
        ]
        .concat();

        let (read_write_evals, chis) = Rep3MultilinearPolynomial::batch_evaluate_worker(
            &read_write_polys,
            r_opening,
            io_ctx.log_num_workers(),
            io_ctx.worker_idx(),
        );

        opening_accumulator.append_send_claims(
            &read_write_polys,
            DensePolynomial::new(chis),
            r_opening.to_vec(),
            &read_write_evals,
            io_ctx.main(),
        )?;

        Ok(())
    }

    #[tracing::instrument(
        skip_all,
        name = "TimestampValidityProof::prove_grand_products_distributed"
    )]
    fn prove_grand_products_distributed(
        polynomials: &TimestampRangeCheckStuff<Rep3MultilinearPolynomial<F>>,
        jolt_polynomials: &Rep3JoltPolynomials<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Vec<F>> {
        if io_ctx.party_idx() != 0 {
            return io_ctx
                .network()
                .recv(PartyID::ID0)
                .context("while receiving r");
        }

        let (gamma, tau): (F, F) = io_ctx.network().receive_request()?;

        let leaves = compute_leaves(polynomials, jolt_polynomials, &gamma, &tau, io_ctx);

        let (mut batched_circuit, _batch_size) =
            <BatchedDenseGrandProduct<F> as Rep3BatchedGrandProductWorker<
                F,
                PCS,
                ProofTranscript,
                Network,
            >>::construct(leaves, io_ctx)?;

        let hashes = <BatchedDenseGrandProduct<F> as Rep3BatchedGrandProductWorker<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::claimed_outputs(&batched_circuit)
        .unwrap();

        io_ctx.network().send_response(hashes)?;

        let r_grand_product = <BatchedDenseGrandProduct<F> as Rep3BatchedGrandProductWorker<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::prove_grand_product_worker(&mut batched_circuit, io_ctx)?;

        drop_in_background_thread(batched_circuit);

        Ok(r_grand_product)
    }
}

fn compute_leaves<F: JoltField, Network: Rep3NetworkWorker>(
    polynomials: &TimestampRangeCheckStuff<Rep3MultilinearPolynomial<F>>,
    jolt_polynomials: &Rep3JoltPolynomials<F>,
    gamma: &F,
    tau: &F,
    io_ctx: &IoContextPool<Network>,
) -> (Vec<F>, usize, usize) {
    let read_timestamps: [&CompactPolynomial<u32, F>; 4] = [
        (&jolt_polynomials.read_write_memory.t_read_rd)
            .as_public()
            .try_into()
            .unwrap(),
        (&jolt_polynomials.read_write_memory.t_read_rs1)
            .as_public()
            .try_into()
            .unwrap(),
        (&jolt_polynomials.read_write_memory.t_read_rs2)
            .as_public()
            .try_into()
            .unwrap(),
        (&jolt_polynomials.read_write_memory.t_read_ram)
            .as_public()
            .try_into()
            .unwrap(),
    ];
    let read_cts_read_timestamp: [&CompactPolynomial<u32, F>; MEMORY_OPS_PER_INSTRUCTION] =
        polynomials
            .read_cts_read_timestamp
            .iter()
            .map(|poly| poly.as_public().try_into().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    let read_cts_global_minus_read: [&CompactPolynomial<u32, F>; MEMORY_OPS_PER_INSTRUCTION] =
        polynomials
            .read_cts_global_minus_read
            .iter()
            .map(|poly| poly.as_public().try_into().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

    let m = read_timestamps[0].full_len();

    let read_write_leaves: Vec<Vec<F>> = (0..MEMORY_OPS_PER_INSTRUCTION)
        .into_par_iter()
        .flat_map(|i| {
            let read_fingerprints_0: Vec<F> = (0..m)
                .into_par_iter()
                .map(|j| {
                    read_timestamps[i][j].field_mul(*gamma)
                        + F::from_u32(read_cts_read_timestamp[i][j])
                        - *tau
                })
                .collect();
            let write_fingeprints_0 = read_fingerprints_0
                .par_iter()
                .map(|read_fingerprint| *read_fingerprint + F::one())
                .collect();

            let read_fingerprints_1: Vec<F> = (0..m)
                .into_par_iter()
                .map(|j| {
                    let global_minus_read = j as u32 - read_timestamps[i][j];
                    global_minus_read.field_mul(*gamma)
                        + F::from_u32(read_cts_global_minus_read[i][j])
                        - *tau
                })
                .collect();
            let write_fingeprints_1 = read_fingerprints_1
                .par_iter()
                .map(|read_fingerprint| *read_fingerprint + F::one())
                .collect();

            [
                read_fingerprints_0,
                write_fingeprints_0,
                read_fingerprints_1,
                write_fingeprints_1,
            ]
        })
        .collect();

    let mut leaves = read_write_leaves;

    let init_leaves: Vec<F> = (0..m)
        .into_par_iter()
        .map(|i| {
            // t = 0
            (i as u64).field_mul(*gamma) - *tau
        })
        .collect();

    let final_cts_read_timestamp: [&CompactPolynomial<u32, F>; MEMORY_OPS_PER_INSTRUCTION] =
        polynomials
            .final_cts_read_timestamp
            .iter()
            .map(|poly| poly.try_into().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    let final_cts_global_minus_read: [&CompactPolynomial<u32, F>; MEMORY_OPS_PER_INSTRUCTION] =
        polynomials
            .final_cts_global_minus_read
            .iter()
            .map(|poly| poly.try_into().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
    leaves.par_extend(
        (0..MEMORY_OPS_PER_INSTRUCTION)
            .into_par_iter()
            .flat_map(|i| {
                let final_fingerprints_0 = (0..m)
                    .into_par_iter()
                    .map(|j| F::from_u32(final_cts_read_timestamp[i][j]) + init_leaves[j])
                    .collect();

                let final_fingerprints_1 = (0..m)
                    .into_par_iter()
                    .map(|j| F::from_u32(final_cts_global_minus_read[i][j]) + init_leaves[j])
                    .collect();

                [final_fingerprints_0, final_fingerprints_1]
            }),
    );
    leaves.push(init_leaves);

    let batch_size = leaves.len();

    (leaves.concat(), batch_size, batch_size)
}
