use std::marker::PhantomData;

use crate::lasso::memory_checking::worker::MemoryCheckingProverRep3Worker;
use crate::subprotocols::commitment::DistributedCommitmentScheme;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::read_write_memory::{
    ReadWriteMemoryOpenings, ReadWriteMemoryPreprocessing, ReadWriteMemoryStuff,
    RegisterAddressOpenings,
};
use jolt_core::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::subprotocols::grand_product::BatchedDenseGrandProduct;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use mpc_core::protocols::rep3::network::{Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, Rep3PrimeFieldShare};
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
        todo!()
    }
}
