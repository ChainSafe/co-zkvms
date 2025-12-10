use std::marker::PhantomData;

use crate::field::JoltField;
use crate::lasso::memory_checking;
use crate::lasso::memory_checking::worker::MemoryCheckingProverRep3Worker;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::poly::Rep3MultilinearPolynomial;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use crate::utils::types::Rep3Value;
use jolt_core::jolt::vm::bytecode::{BytecodeOpenings, BytecodePreprocessing};
use jolt_core::jolt::vm::JoltStuff;
use jolt_core::lasso::memory_checking::{NoExogenousOpenings, StructuredPolynomialData};
use jolt_core::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::subprotocols::grand_product::BatchedDenseGrandProduct;
use jolt_core::utils::transcript::Transcript;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, Rep3PrimeFieldShare};
use rayon::prelude::*;

use super::witness::Rep3BytecodePolynomials;
use crate::jolt::vm::witness::Rep3JoltPolynomials;

pub struct Rep3BytecodeProver<F: JoltField, PCS, ProofTranscript, Network> {
    pub _marker: PhantomData<(F, PCS, ProofTranscript, Network)>,
}

impl<F, PCS, ProofTranscript, Network>
    MemoryCheckingProverRep3Worker<F, PCS, ProofTranscript, Network>
    for Rep3BytecodeProver<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkWorker,
{
    type ReadWriteGrandProduct = Rep3BatchedDenseGrandProduct<F>;
    type InitFinalGrandProduct = BatchedDenseGrandProduct<F>;

    type Rep3Polynomials = Rep3BytecodePolynomials<F>;
    type Openings = BytecodeOpenings<F>;
    type ExogenousOpenings = NoExogenousOpenings;

    type Preprocessing = BytecodePreprocessing<F>;

    #[tracing::instrument(skip_all, name = "Rep3BytecodeProver::compute_leaves", level = "trace")]
    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &Self::Rep3Polynomials,
        _jolt_polynomials: &Rep3JoltPolynomials<F>,
        gamma: &F,
        tau: &F,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<(
        (Vec<Rep3PrimeFieldShare<F>>, usize, usize),
        (Vec<F>, usize, usize),
    )> {
        let worker_idx = io_ctx.worker_idx();
        let num_workers = io_ctx.num_workers();
        let batch_size_worker = 1 + (!io_ctx.network().is_distributed()) as usize; // is_distributed ? 1 : 2

        let mut gamma_terms = [F::zero(); 7];
        let mut gamma_term = F::one();
        for i in 0..7 {
            gamma_term *= *gamma;
            gamma_terms[i] = gamma_term;
        }

        // ------------- read_write ------------- //
        let num_ops_worker = polynomials.a_read_write.full_len() / (num_workers >> 1).max(1); // when num_workers >= 4 worker has sharded polys
        let a: &CompactPolynomial<u32, F> = (&polynomials.a_read_write).try_into().unwrap();
        let v_address: &CompactPolynomial<u64, F> =
            (&polynomials.v_read_write[0]).try_into().unwrap();
        let v_bitflags: &CompactPolynomial<u64, F> =
            (&polynomials.v_read_write[1]).try_into().unwrap();
        let v_rd: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[2]).try_into().unwrap();
        let v_rs1: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[3]).try_into().unwrap();
        let v_rs2: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[4]).try_into().unwrap();
        let v_imm = &polynomials.v_read_write[5].as_shared();
        let t: &CompactPolynomial<u32, F> = (&polynomials.t_read).try_into().unwrap();

        let party_id = io_ctx.party_id();
        let read_leaves: Vec<_> = (0..num_ops_worker)
            .into_par_iter()
            .map(|i| {
                let public_term = a[i].field_mul(gamma_terms[0])
                    + v_address[i].field_mul(gamma_terms[1])
                    + v_bitflags[i].field_mul(gamma_terms[2])
                    + v_rd[i].field_mul(gamma_terms[3])
                    + v_rs1[i].field_mul(gamma_terms[4])
                    + v_rs2[i].field_mul(gamma_terms[5])
                    + t[i].field_mul(gamma_terms[6])
                    - tau;
                Rep3Value::Shared(v_imm[i])
                    .add_public(public_term, party_id)
                    .as_shared()
            })
            .collect();

        let mut read_write_fingeprints = Vec::with_capacity(num_ops_worker);

        // write_fingeprints first to avoid cloning read_leaves
        if !io_ctx.network().is_distributed() || worker_idx >= num_workers >> 1 {
            read_write_fingeprints.par_extend(read_leaves.par_iter().map(|leaf| {
                Rep3Value::Shared(*leaf)
                    .add_public(gamma_terms[6], party_id)
                    .as_shared()
            }))
        }

        if !io_ctx.network().is_distributed() || worker_idx < num_workers >> 1 {
            read_write_fingeprints.splice(0..0, read_leaves); // extend from back
        }

        // ------------- init_final ------------- //
        let bytecode_size_worker =
            preprocessing.v_init_final[0].full_len() / (num_workers >> 1).max(1);
        let v_address: &CompactPolynomial<u64, F> =
            (&preprocessing.v_init_final[0]).try_into().unwrap();
        let v_bitflags: &CompactPolynomial<u64, F> =
            (&preprocessing.v_init_final[1]).try_into().unwrap();
        let v_rd: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[2]).try_into().unwrap();
        let v_rs1: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[3]).try_into().unwrap();
        let v_rs2: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[4]).try_into().unwrap();
        let v_imm: &CompactPolynomial<i64, F> =
            (&preprocessing.v_init_final[5]).try_into().unwrap();

        let init_leaves: Vec<F> = (0..bytecode_size_worker)
            .into_par_iter()
            .map(|i| {
                F::from_i64(v_imm[i])
                    + (i as u64).field_mul(gamma_terms[0])
                    + v_address[i].field_mul(gamma_terms[1])
                    + v_bitflags[i].field_mul(gamma_terms[2])
                    + v_rd[i].field_mul(gamma_terms[3])
                    + v_rs1[i].field_mul(gamma_terms[4])
                    + v_rs2[i].field_mul(gamma_terms[5])
                    - tau
            })
            .collect();

        let mut init_final_fingeprints = Vec::with_capacity(num_ops_worker);

        if !io_ctx.network().is_distributed() || worker_idx >= num_workers >> 1 {
            let t_final: &CompactPolynomial<u32, F> = (&polynomials.t_final).try_into().unwrap();
            init_final_fingeprints.par_extend(
                init_leaves
                    .par_iter()
                    .enumerate()
                    .map(|(i, leaf)| *leaf + t_final[i].field_mul(gamma_terms[6])),
            )
        }

        if !io_ctx.network().is_distributed() || worker_idx < num_workers >> 1 {
            init_final_fingeprints.splice(0..0, init_leaves); // extend from back
        }

        Ok((
            (read_write_fingeprints, batch_size_worker, 2),
            (init_final_fingeprints, batch_size_worker, 2),
        ))
    }
}
