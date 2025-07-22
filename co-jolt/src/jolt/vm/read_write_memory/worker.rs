use std::marker::PhantomData;

use crate::jolt::vm::jolt::witness::Rep3JoltPolynomialsExt;
use crate::jolt::vm::read_write_memory::witness::Rep3ProgramIO;
use crate::lasso::memory_checking::worker::MemoryCheckingProverRep3Worker;
use crate::poly::opening_proof::Rep3ProverOpeningAccumulator;
use crate::poly::{Rep3MultilinearPolynomial, Rep3PolysConversion};
use crate::subprotocols::commitment::DistributedCommitmentScheme;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use crate::subprotocols::sumcheck;
use crate::utils::element::SharedOrPublic;
use crate::utils::transcript::TranscriptExt;
use itertools::Itertools;
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::read_write_memory::{
    memory_address_to_witness_index, ReadWriteMemoryOpenings, ReadWriteMemoryPreprocessing,
    ReadWriteMemoryStuff, RegisterAddressOpenings,
};
use jolt_core::jolt::vm::timestamp_range_check::TimestampRangeCheckPolynomials;
use jolt_core::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::subprotocols::grand_product::BatchedDenseGrandProduct;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use mpc_core::protocols::rep3::network::{IoContext, Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};
use rayon::prelude::*;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_common::constants::{
    BYTES_PER_INSTRUCTION, MEMORY_OPS_PER_INSTRUCTION, RAM_START_ADDRESS, REGISTER_COUNT,
};
use jolt_common::rv_trace::{JoltDevice, MemoryLayout, MemoryOp};
use jolt_core::jolt::vm::{
    read_write_memory::{ReadWriteMemoryPolynomials, ReadWriteMemoryProof},
    timestamp_range_check::TimestampValidityProof,
    JoltCommitments, JoltPolynomials, JoltStuff, JoltTraceStep,
};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::utils::transcript::Transcript;
use jolt_core::{
    poly::{
        dense_mlpoly::DensePolynomial, eq_poly::EqPolynomial, identity_poly::IdentityPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{errors::ProofVerifyError, math::Math},
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
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: TranscriptExt,
    Network: Rep3NetworkWorker,
{
    pub fn prove(
        pcs_setup: &PCS::Setup,
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials: &mut Rep3JoltPolynomials<F>,
        program_io: &Rep3ProgramIO<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        Self::prove_memory_checking(
            pcs_setup,
            preprocessing,
            &polynomials.read_write_memory,
            polynomials,
            opening_accumulator,
            io_ctx,
        )?;

        Self::prove_outputs(
            &polynomials.read_write_memory,
            program_io,
            opening_accumulator,
            io_ctx,
        )?;

        let state: Option<ProofTranscript::State> = io_ctx.network.receive_request()?;

        if let Some(state) = state {
            let mut transcript = ProofTranscript::from_state(state);
            let mut opening_accumulator_public =
                ProverOpeningAccumulator::<F, ProofTranscript>::new();

            let timestamp_range_check_polynomials =
                polynomials.get_timestamp_range_check_polynomials();
            let jolt_polynomials =
                polynomials.get_exogenous_polynomials_for_timestamp_range_check();

            let timestamp_validity_proof = TimestampValidityProof::<F, PCS, ProofTranscript>::prove(
                pcs_setup,
                &timestamp_range_check_polynomials,
                &jolt_polynomials,
                &mut opening_accumulator_public,
                &mut transcript,
            );

            opening_accumulator.append_public(&opening_accumulator_public.openings[0], io_ctx)?;
            io_ctx
                .network
                .send_response((timestamp_validity_proof, transcript.state()))?;
        } else {
            opening_accumulator.receive_public_opening(io_ctx)?;
        }

        Ok(())
    }

    fn prove_outputs(
        polynomials: &Rep3ReadWriteMemoryPolynomials<F>,
        program_io: &Rep3ProgramIO<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        let memory_size = polynomials.v_final.len();
        if io_ctx.id == PartyID::ID0 {
            io_ctx.network.send_response(memory_size)?;
        }
        let num_rounds = memory_size.log_2();
        let r_eq: Vec<F> = io_ctx.network.receive_request()?;
        let eq = MultilinearPolynomial::from(EqPolynomial::evals(&r_eq));

        let input_start_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as u64;
        let ram_start_index =
            memory_address_to_witness_index(RAM_START_ADDRESS, &program_io.memory_layout) as u64;

        let io_witness_range: Vec<u8> = (0..memory_size as u64)
            .map(|i| {
                if i >= input_start_index && i < ram_start_index {
                    1
                } else {
                    0
                }
            })
            .collect();

        let mut v_io = vec![Rep3PrimeFieldShare::zero_share(); memory_size];
        let mut input_index = memory_address_to_witness_index(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        );
        // Convert input bytes into words and populate `v_io`
        for (i, word) in program_io.input_words.iter().enumerate() {
            v_io[input_index] = *word;
            input_index += 1;
        }
        let mut output_index = memory_address_to_witness_index(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        );
        // Convert output bytes into words and populate `v_io`
        for (i, word) in program_io.output_words.iter().enumerate() {
            v_io[output_index] = *word;
            output_index += 1;
        }

        // Copy panic bit
        v_io[memory_address_to_witness_index(
            program_io.memory_layout.panic,
            &program_io.memory_layout,
        )] = program_io.panic;

        v_io[memory_address_to_witness_index(
            program_io.memory_layout.termination,
            &program_io.memory_layout,
        )] = program_io.panic;

        let mut sumcheck_polys = vec![
            eq.into(),
            MultilinearPolynomial::from(io_witness_range).into(),
            polynomials.v_final.clone().into(),
            Rep3MultilinearPolynomial::from(v_io).into(),
        ];

        // eq * io_witness_range * (v_final - v_io)
        let output_check_fn = |vals: &[SharedOrPublic<F>]| -> F {
            rep3::arithmetic::mul_public(
                *vals[2].as_shared() - *vals[3].as_shared(),
                *vals[0].as_public() * *vals[1].as_public(),
            )
            .into_additive()
        };

        let (r_sumcheck, sumcheck_openings) = sumcheck::prove_arbitrary_worker::<F, _, Network>(
            &F::zero(),
            num_rounds,
            &mut sumcheck_polys,
            output_check_fn,
            3,
            io_ctx,
        )?;

        opening_accumulator.append(
            &[polynomials.v_final.as_shared()],
            DensePolynomial::new(EqPolynomial::evals(&r_sumcheck)),
            r_sumcheck.to_vec(),
            &[sumcheck_openings[2]],
            io_ctx,
        )?;

        Ok(())
    }
}

impl<F, PCS, ProofTranscript, Network>
    MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>
    for Rep3ReadWriteMemoryProver<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
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
        io_ctx: &mut mpc_core::protocols::rep3::network::IoContext<Network>,
    ) -> eyre::Result<(
        (Vec<Rep3PrimeFieldShare<F>>, usize),
        (Vec<Rep3PrimeFieldShare<F>>, usize),
    )> {
        let gamma_squared = gamma.square();
        let gamma = *gamma;

        let num_ops = polynomials.a_ram.len();
        let memory_size = polynomials.v_final.len();

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
        let v_read_rs1 = &polynomials.v_read_rs1;
        let v_read_rs2 = &polynomials.v_read_rs2;
        let v_read_rd = &polynomials.v_read_rd;
        let v_read_ram = &polynomials.v_read_ram;
        let v_write_rd = &polynomials.v_write_rd;
        let v_write_ram = &polynomials.v_write_ram;
        let t_read_rs1: &CompactPolynomial<u32, F> = (&polynomials.t_read_rs1).try_into().unwrap();
        let t_read_rs2: &CompactPolynomial<u32, F> = (&polynomials.t_read_rs2).try_into().unwrap();
        let t_read_rd: &CompactPolynomial<u32, F> = (&polynomials.t_read_rd).try_into().unwrap();
        let t_read_ram: &CompactPolynomial<u32, F> = (&polynomials.t_read_ram).try_into().unwrap();

        let party_id = io_ctx.id;

        let mut read_write_leaves: Vec<Rep3PrimeFieldShare<F>> =
            vec![Rep3PrimeFieldShare::zero_share(); 2 * MEMORY_OPS_PER_INSTRUCTION * num_ops];
        for (i, chunk) in read_write_leaves.chunks_mut(2 * num_ops).enumerate() {
            chunk[..num_ops]
                .par_iter_mut()
                .enumerate()
                .for_each(|(j, read_fingerprint)| {
                    match i {
                        RS1 => {
                            *read_fingerprint = rep3::arithmetic::add_public(
                                rep3::arithmetic::mul_public(v_read_rs1.as_shared()[j], gamma),
                                t_read_rs1[j].field_mul(gamma_squared) + F::from_u8(a_rs1[j])
                                    - *tau,
                                party_id,
                            );
                        }
                        RS2 => {
                            *read_fingerprint = rep3::arithmetic::add_public(
                                rep3::arithmetic::mul_public(v_read_rs2.as_shared()[j], gamma),
                                t_read_rs2[j].field_mul(gamma_squared) + F::from_u8(a_rs2[j])
                                    - *tau,
                                party_id,
                            );
                        }
                        RD => {
                            *read_fingerprint = rep3::arithmetic::add_public(
                                rep3::arithmetic::mul_public(v_read_rd.as_shared()[j], gamma),
                                t_read_rd[j].field_mul(gamma_squared) + F::from_u8(a_rd[j]) - *tau,
                                party_id,
                            );
                        }
                        RAM => {
                            *read_fingerprint = rep3::arithmetic::add_public(
                                rep3::arithmetic::mul_public(v_read_ram.as_shared()[j], gamma),
                                t_read_ram[j].field_mul(gamma_squared) + F::from_u32(a_ram[j])
                                    - *tau,
                                party_id,
                            );
                        }
                        _ => unreachable!(),
                    };
                });

            chunk[num_ops..].par_iter_mut().enumerate().for_each(
                |(j, write_fingerprint)| match i {
                    RS1 => {
                        *write_fingerprint = rep3::arithmetic::add_public(
                            rep3::arithmetic::mul_public(v_read_rs1.as_shared()[j], gamma),
                            (j as u64).field_mul(gamma_squared) + F::from_u8(a_rs1[j]) - *tau,
                            party_id,
                        );
                    }
                    RS2 => {
                        *write_fingerprint = rep3::arithmetic::add_public(
                            rep3::arithmetic::mul_public(v_read_rs2.as_shared()[j], gamma),
                            (j as u64).field_mul(gamma_squared) + F::from_u8(a_rs2[j]) - *tau,
                            party_id,
                        );
                    }
                    RD => {
                        *write_fingerprint = rep3::arithmetic::add_public(
                            rep3::arithmetic::mul_public(v_write_rd.as_shared()[j], gamma),
                            (j as u64).field_mul(gamma_squared) + F::from_u8(a_rd[j]) - *tau,
                            party_id,
                        );
                    }
                    RAM => {
                        *write_fingerprint = rep3::arithmetic::add_public(
                            rep3::arithmetic::mul_public(v_write_ram.as_shared()[j], gamma),
                            (j as u64).field_mul(gamma_squared) + F::from_u32(a_ram[j]) - *tau,
                            party_id,
                        );
                    }
                    _ => unreachable!(),
                },
            );
        }

        let v_init = polynomials.v_init.as_ref().unwrap();
        let init_fingerprints: Vec<Rep3PrimeFieldShare<F>> = (0..memory_size)
            .into_par_iter()
            .map(|i| {
                rep3::arithmetic::add_public(
                    rep3::arithmetic::mul_public(v_init.as_shared()[i], gamma),
                    (i as u64).field_mul(gamma_squared) + F::from_u32(i as u32) - *tau,
                    party_id,
                )
            })
            .collect();

        let v_final = &polynomials.v_final;
        let t_final: &CompactPolynomial<u32, F> = (&polynomials.t_final).try_into().unwrap();
        let final_fingerprints = (0..memory_size)
            .into_par_iter()
            .map(|i| {
                rep3::arithmetic::add_public(
                    rep3::arithmetic::mul_public(v_final.as_shared()[i], gamma),
                    t_final[i].field_mul(gamma_squared) + F::from_u32(i as u32) - *tau,
                    party_id,
                )
            })
            .collect();

        Ok((
            (read_write_leaves, 2 * MEMORY_OPS_PER_INSTRUCTION),
            ([init_fingerprints, final_fingerprints].concat(), 2),
        ))
    }

    fn num_lookups(polynomials: &Self::Rep3Polynomials) -> usize {
        polynomials.v_final.len()
    }
}
