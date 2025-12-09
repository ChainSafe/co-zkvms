use std::marker::PhantomData;

use crate::field::JoltField;
use crate::jolt::vm::jolt::witness::Rep3JoltPolynomialsExt;
use crate::jolt::vm::read_write_memory::witness::Rep3ProgramIO;
use crate::jolt::vm::timestamp_range_check;
use crate::lasso::memory_checking;
use crate::lasso::memory_checking::worker::MemoryCheckingProverRep3Worker;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::poly::Rep3MultilinearPolynomial;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use crate::subprotocols::sumcheck;
use crate::utils::transcript::TranscriptExt;
use crate::utils::types::Rep3Value;
use itertools::Itertools;
use jolt_core::jolt::vm::read_write_memory::{
    memory_address_to_witness_index, ReadWriteMemoryOpenings, ReadWriteMemoryPreprocessing,
    RegisterAddressOpenings,
};
use jolt_core::lasso::memory_checking::{ExogenousOpenings, StructuredPolynomialData};
use jolt_core::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use mpc_core::protocols::additive::AdditiveShare;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, Rep3PrimeFieldShare};
use rayon::prelude::*;

use jolt_common::constants::{MEMORY_OPS_PER_INSTRUCTION, RAM_START_ADDRESS};
use jolt_core::jolt::vm::timestamp_range_check::TimestampValidityProof;
use jolt_core::utils::transcript::Transcript;
use jolt_core::{
    poly::{dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial},
    utils::math::Math,
};

use super::witness::Rep3ReadWriteMemoryPolynomials;
use crate::jolt::vm::witness::Rep3JoltPolynomials;

const RS1: usize = 0;
const RS2: usize = 1;
const RD: usize = 2;
const RAM: usize = 3;

pub struct Rep3ReadWriteMemoryProver<F: JoltField, PCS, ProofTranscript, Network> {
    pub _marker: PhantomData<(F, PCS, ProofTranscript, Network)>,
}

impl<F, PCS, ProofTranscript, Network> Rep3ReadWriteMemoryProver<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: TranscriptExt,
    Network: Rep3NetworkWorker,
{
    #[tracing::instrument(skip_all, name = "Rep3ReadWriteMemory::prove")]
    pub fn prove(
        pcs_setup: &PCS::Setup,
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials: &mut Rep3JoltPolynomials<F>,
        program_io: &mut Rep3ProgramIO<F>,
        opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        Self::prove_memory_checking(
            pcs_setup,
            preprocessing,
            &polynomials.read_write_memory,
            polynomials,
            opening_accumulator,
            io_ctx,
        )?;

        tracing::info!("prove_memory_checking done");

        Self::prove_outputs(
            &mut polynomials.read_write_memory,
            program_io,
            opening_accumulator,
            io_ctx,
        )?;

        // let state: Option<ProofTranscript::State> = io_ctx.network().receive_request()?;

        // if io_ctx.worker_idx() == 0 {
        //     if let Some(state) = state {
        //         let mut transcript = ProofTranscript::from_state(state);
        //         let mut opening_accumulator_public =
        //             ProverOpeningAccumulator::<F, ProofTranscript>::new();

        //         let timestamp_range_check_polynomials =
        //             timestamp_range_check::get_timestamp_range_check_polynomials::<
        //                 F,
        //                 PCS,
        //                 ProofTranscript,
        //             >(&mut polynomials.read_write_memory);
        //         let jolt_polynomials =
        //             polynomials.take_exogenous_polynomials_for_timestamp_range_check();

        //         let timestamp_validity_proof =
        //             TimestampValidityProof::<F, PCS, ProofTranscript>::prove(
        //                 pcs_setup,
        //                 &timestamp_range_check_polynomials,
        //                 &jolt_polynomials,
        //                 &mut opening_accumulator_public,
        //                 &mut transcript,
        //             );

        //         // opening_accumulator
        //         //     .append_public(&opening_accumulator_public.openings[0], io_ctx.main())?;
        //         io_ctx
        //             .network()
        //             .send_response((timestamp_validity_proof, transcript.state()))?;
        //     } else {
        //         // opening_accumulator.receive_public_opening(io_ctx.main())?;
        //     }
        // }
        // tracing::info!("worker timestamp done");

        Ok(())
    }

    #[tracing::instrument(skip_all, name = "Rep3ReadWriteMemory::prove_outputs", level = "info")]
    fn prove_outputs(
        polynomials: &mut Rep3ReadWriteMemoryPolynomials<F>,
        program_io: &mut Rep3ProgramIO<F>,
        opening_accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<()> {
        let worker_idx = io_ctx.worker_idx();
        let memory_size = polynomials.v_final.full_len();
        let memory_size_worker = polynomials.v_final.len();
        assert!(memory_size_worker == memory_size / io_ctx.num_workers());
        let num_rounds = memory_size_worker.log_2();

        let r_eq: Vec<F> = io_ctx.network().receive_request::<Vec<F>>()?;
        let eq = MultilinearPolynomial::from(EqPolynomial::evals_worker(
            &r_eq,
            io_ctx.log_num_workers(),
            worker_idx,
        ));

        let input_start_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as u64;
        let ram_start_index =
            memory_address_to_witness_index(RAM_START_ADDRESS, &program_io.memory_layout) as u64;

        let offset = memory_size_worker * worker_idx;
        let cutoff = memory_size_worker * (worker_idx + 1);
        let io_witness_range: Vec<u8> = (offset as u64..cutoff as u64)
            .map(|i| {
                if i >= input_start_index && i < ram_start_index {
                    1
                } else {
                    0
                }
            })
            .collect();

        let mut sumcheck_polys: Vec<Rep3MultilinearPolynomial<F>> = vec![
            eq.into(),
            MultilinearPolynomial::from(io_witness_range).into(),
            std::mem::take(&mut polynomials.v_final),
            std::mem::take(&mut program_io.v_io),
        ];

        // (v_final - v_io) * eq * io_witness_range
        let party_id = io_ctx.party_id();
        let output_check_fn = |vals: &[Rep3Value<F>]| -> AdditiveShare<F> {
            vals[2]
                .sub(&vals[3], party_id)
                .mul_public(vals[0].as_public() * vals[1].as_public())
                .into_additive(party_id)
        };

        let r_sumcheck = sumcheck::distributed_prove_arbitrary_worker(
            num_rounds,
            &mut sumcheck_polys,
            output_check_fn,
            3,
            io_ctx,
        )?;

        // distributed_prove_arbitrary_worker computes sumcheck evals and poly binding "LowToHigh", different from Jolt where it's "HighToLow"
        // accordingly prover and verifier must reverse `r_sumcheck`
        // TODO: consider skipping first rounds since they produce zero round evals
        let r_sumcheck = r_sumcheck.into_iter().rev().collect::<Vec<_>>();

        let v_final = std::mem::take(&mut sumcheck_polys[2]);

        opening_accumulator.append(
            &[&v_final],
            DensePolynomial::new(EqPolynomial::evals(&r_sumcheck)),
            r_sumcheck.to_vec(),
            io_ctx.main(),
        )?;

        Ok(())
    }
}

impl<F, PCS, ProofTranscript, Network>
    MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>
    for Rep3ReadWriteMemoryProver<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    type ReadWriteGrandProduct = Rep3BatchedDenseGrandProduct<F>;
    type InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    type Rep3Polynomials = Rep3ReadWriteMemoryPolynomials<F>;
    type Openings = ReadWriteMemoryOpenings<F>;
    type ExogenousOpenings = RegisterAddressOpenings<F>;

    type Preprocessing = ReadWriteMemoryPreprocessing;

    fn compute_leaves(
        _preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        jolt_polynomials: &Rep3JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<(
        (Vec<Rep3PrimeFieldShare<F>>, usize, usize),
        (Vec<Rep3PrimeFieldShare<F>>, usize, usize),
    )> {
        let gamma_squared = gamma.square();
        let gamma = *gamma;

        let num_ops = polynomials.a_ram.full_len();
        let memory_size = polynomials.v_final.full_len();

        let a_rd: &CompactPolynomial<u8, F> = (&jolt_polynomials.bytecode.v_read_write[2])
            .try_into()
            .unwrap();
        let a_rs1: &CompactPolynomial<u8, F> = (&jolt_polynomials.bytecode.v_read_write[3])
            .try_into()
            .unwrap();
        let a_rs2: &CompactPolynomial<u8, F> = (&jolt_polynomials.bytecode.v_read_write[4])
            .try_into()
            .unwrap();
        let a_ram: &CompactPolynomial<u32, F> = (&polynomials.a_ram).try_into().unwrap();
        let v_read_rs1 = &polynomials.v_read_rs1.as_shared();
        let v_read_rs2 = &polynomials.v_read_rs2.as_shared();
        let v_read_rd = &polynomials.v_read_rd.as_shared();
        let v_read_ram = &polynomials.v_read_ram.as_shared();
        let v_write_rd = &polynomials.v_write_rd.as_shared();
        let v_write_ram = &polynomials.v_write_ram.as_shared();
        let t_read_rs1: &CompactPolynomial<u32, F> = (&polynomials.t_read_rs1).try_into().unwrap();
        let t_read_rs2: &CompactPolynomial<u32, F> = (&polynomials.t_read_rs2).try_into().unwrap();
        let t_read_rd: &CompactPolynomial<u32, F> = (&polynomials.t_read_rd).try_into().unwrap();
        let t_read_ram: &CompactPolynomial<u32, F> = (&polynomials.t_read_ram).try_into().unwrap();

        let party_id = io_ctx.party_id();

        let worker_idx = io_ctx.worker_idx();
        let num_workers = io_ctx.num_workers();
        let rw_batch_size_full = 2 * MEMORY_OPS_PER_INSTRUCTION;
        assert!(io_ctx.num_workers() <= rw_batch_size_full);
        let rw_batch_size_worker = rw_batch_size_full / num_workers;
        let chunk_size = if num_workers <= MEMORY_OPS_PER_INSTRUCTION {
            2 * num_ops
        } else {
            num_ops * 2 * MEMORY_OPS_PER_INSTRUCTION / num_workers
        };

        assert!(
            chunk_size >= num_ops,
            "memory trace spliting is unimplemented"
        );

        // ------------- read_write ------------- //

        let num_ops_worker = num_ops; // TODO: different when num_workers <= rw_batch_size_full
        let mut read_write_leaves: Vec<Rep3PrimeFieldShare<F>> =
            vec![Rep3PrimeFieldShare::zero_share(); rw_batch_size_worker * num_ops_worker];

        let reg_offset = MEMORY_OPS_PER_INSTRUCTION * worker_idx / num_workers; // 2 => [0, 2] 4 => [0, 1, 2, 3] 8 => [0, 0, 1, 1, 2, 2, 3, 3]

        for (i, chunk) in read_write_leaves.chunks_mut(chunk_size).enumerate() {
            if num_workers <= rw_batch_size_full || worker_idx % 2 == 0 {
                tracing::info!(
                    "worker_idx {} read_leaves chunk {} ",
                    worker_idx,
                    reg_offset + i
                );
                chunk[..num_ops_worker].par_iter_mut().enumerate().for_each(
                    |(j, read_fingerprint)| {
                        match reg_offset + i {
                            RS1 => {
                                *read_fingerprint = rep3::arithmetic::add_public(
                                    rep3::arithmetic::mul_public(v_read_rs1[j], gamma),
                                    t_read_rs1[j].field_mul(gamma_squared) + F::from_u8(a_rs1[j])
                                        - *tau,
                                    party_id,
                                );
                            }
                            RS2 => {
                                *read_fingerprint = rep3::arithmetic::add_public(
                                    rep3::arithmetic::mul_public(v_read_rs2[j], gamma),
                                    t_read_rs2[j].field_mul(gamma_squared) + F::from_u8(a_rs2[j])
                                        - *tau,
                                    party_id,
                                );
                            }
                            RD => {
                                *read_fingerprint = rep3::arithmetic::add_public(
                                    rep3::arithmetic::mul_public(v_read_rd[j], gamma),
                                    t_read_rd[j].field_mul(gamma_squared) + F::from_u8(a_rd[j])
                                        - *tau,
                                    party_id,
                                );
                            }
                            RAM => {
                                *read_fingerprint = rep3::arithmetic::add_public(
                                    rep3::arithmetic::mul_public(v_read_ram[j], gamma),
                                    t_read_ram[j].field_mul(gamma_squared) + F::from_u32(a_ram[j])
                                        - *tau,
                                    party_id,
                                );
                            }
                            _ => unreachable!(),
                        };
                    },
                );
            }

            if num_workers <= rw_batch_size_full || worker_idx % 2 != 0 {
                tracing::info!(
                    "worker_idx {} write_leaves chunk {} ",
                    worker_idx,
                    reg_offset + i
                );

                chunk[num_ops_worker..].par_iter_mut().enumerate().for_each(
                    |(j, write_fingerprint)| match reg_offset + i {
                        RS1 => {
                            *write_fingerprint = rep3::arithmetic::add_public(
                                rep3::arithmetic::mul_public(v_read_rs1[j], gamma),
                                (j as u64).field_mul(gamma_squared) + F::from_u8(a_rs1[j]) - *tau,
                                party_id,
                            );
                        }
                        RS2 => {
                            *write_fingerprint = rep3::arithmetic::add_public(
                                rep3::arithmetic::mul_public(v_read_rs2[j], gamma),
                                (j as u64).field_mul(gamma_squared) + F::from_u8(a_rs2[j]) - *tau,
                                party_id,
                            );
                        }
                        RD => {
                            *write_fingerprint = rep3::arithmetic::add_public(
                                rep3::arithmetic::mul_public(v_write_rd[j], gamma),
                                (j as u64).field_mul(gamma_squared) + F::from_u8(a_rd[j]) - *tau,
                                party_id,
                            );
                        }
                        RAM => {
                            *write_fingerprint = rep3::arithmetic::add_public(
                                rep3::arithmetic::mul_public(v_write_ram[j], gamma),
                                (j as u64).field_mul(gamma_squared) + F::from_u32(a_ram[j]) - *tau,
                                party_id,
                            );
                        }
                        _ => unreachable!(),
                    },
                );
            }
        }

        // ------------- init_final ------------- //

        let memory_size_worker = polynomials.v_final.len();
        // when num_workers >= 4 worker has sharded v_final and t_final, so we must offset index accordingly
        let offset = memory_size_worker * worker_idx * (num_workers >= 4) as usize;
        let mut init_final_fingeprints = Vec::with_capacity(memory_size_worker);
        let init_final_batch_size_worker = 1 + (!io_ctx.network().is_distributed()) as usize; // is_distributed ? 1 : 2

        if !io_ctx.network().is_distributed() || worker_idx < num_workers / 2 {
            let v_init = polynomials.v_init.as_ref().unwrap().as_shared();

            init_final_fingeprints.par_extend((0..memory_size).into_par_iter().map(|i| {
                rep3::arithmetic::add_public(
                    rep3::arithmetic::mul_public(v_init[i], gamma),
                    F::from_u32((offset + i) as u32) - *tau,
                    party_id,
                )
            }));
        }

        if !io_ctx.network().is_distributed() || worker_idx >= num_workers / 2 {
            let v_final = &polynomials.v_final.as_shared();
            let t_final: &CompactPolynomial<u32, F> = (&polynomials.t_final).try_into().unwrap();
            init_final_fingeprints.par_extend((0..memory_size).into_par_iter().map(|i| {
                rep3::arithmetic::add_public(
                    rep3::arithmetic::mul_public(v_final[i], gamma),
                    t_final[i].field_mul(gamma_squared) + F::from_u32((offset + i) as u32) - *tau,
                    party_id,
                )
            }));
        }

        tracing::info!(
            "read_write_leaves {} rw_batch_size_worker {} rw_batch_size_full {}",
            read_write_leaves.len(),
            rw_batch_size_worker,
            rw_batch_size_full
        );

        tracing::info!(
            "init_final_fingeprints {} init_final_batch_size_worker {}",
            init_final_fingeprints.len(),
            init_final_batch_size_worker
        );

        Ok((
            (read_write_leaves, rw_batch_size_worker, rw_batch_size_full),
            (init_final_fingeprints, init_final_batch_size_worker, 2),
        ))
    }
}
