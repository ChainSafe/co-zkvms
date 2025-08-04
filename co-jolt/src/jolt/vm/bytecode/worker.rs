use std::marker::PhantomData;

use crate::lasso::memory_checking::worker::MemoryCheckingProverRep3Worker;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use crate::utils::element::SharedOrPublic;
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::bytecode::{BytecodeOpenings, BytecodePreprocessing};
use jolt_core::lasso::memory_checking::NoExogenousOpenings;
use jolt_core::poly::compact_polynomial::{CompactPolynomial, SmallScalar};
use jolt_core::utils::transcript::Transcript;
use mpc_core::protocols::rep3::network::{IoContext, Rep3NetworkWorker};
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
    // TODO: InitFinalGrandProduct can be computed publically
    type InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;

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
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<(
        (Vec<Rep3PrimeFieldShare<F>>, usize),
        (Vec<Rep3PrimeFieldShare<F>>, usize),
    )> {
        let num_ops = polynomials.a_read_write.len();
        let bytecode_size = preprocessing.v_init_final[0].len();

        let mut gamma_terms = [F::zero(); 7];
        let mut gamma_term = F::one();
        for i in 0..7 {
            gamma_term *= *gamma;
            gamma_terms[i] = gamma_term;
        }

        let a: &CompactPolynomial<u32, F> = (&polynomials.a_read_write).try_into().unwrap();
        let v_address: &CompactPolynomial<u64, F> =
            (&polynomials.v_read_write[0]).try_into().unwrap();
        let v_bitflags: &CompactPolynomial<u64, F> =
            (&polynomials.v_read_write[1]).try_into().unwrap();
        let v_rd: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[2]).try_into().unwrap();
        let v_rs1: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[3]).try_into().unwrap();
        let v_rs2: &CompactPolynomial<u8, F> = (&polynomials.v_read_write[4]).try_into().unwrap();
        let v_imm = &polynomials.v_read_write[5];
        let t: &CompactPolynomial<u32, F> = (&polynomials.t_read).try_into().unwrap();

        let read_leaves: Vec<_> = (0..num_ops)
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
                v_imm
                    .get_coeff(i)
                    .add_public(public_term, io_ctx.id)
                    .as_shared()
            })
            .collect();

        let write_leaves: Vec<_> = read_leaves
            .par_iter()
            .map(|leaf| {
                SharedOrPublic::Shared(*leaf)
                    .add_public(gamma_terms[6], io_ctx.id)
                    .as_shared()
            })
            .collect();

        let v_address: &CompactPolynomial<u64, F> =
            (&preprocessing.v_init_final[0]).try_into().unwrap();
        let v_bitflags: &CompactPolynomial<u64, F> =
            (&preprocessing.v_init_final[1]).try_into().unwrap();
        let v_rd: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[2]).try_into().unwrap();
        let v_rs1: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[3]).try_into().unwrap();
        let v_rs2: &CompactPolynomial<u8, F> = (&preprocessing.v_init_final[4]).try_into().unwrap();
        let v_imm: &CompactPolynomial<i64, F> =
            (&preprocessing.v_init_final[5]).try_into().unwrap();

        let init_leaves: Vec<F> = (0..bytecode_size)
            .into_par_iter()
            .map(|i| {
                F::from_i64(v_imm[i])
                    + (i as u64).field_mul(gamma_terms[0])
                    + v_address[i].field_mul(gamma_terms[1])
                    + v_bitflags[i].field_mul(gamma_terms[2])
                    + v_rd[i].field_mul(gamma_terms[3])
                    + v_rs1[i].field_mul(gamma_terms[4])
                    + v_rs2[i].field_mul(gamma_terms[5])
                    // + gamma_terms[6] * 0
                    - tau
            })
            .collect();

        let t_final: &CompactPolynomial<u32, F> = (&polynomials.t_final).try_into().unwrap();
        let final_leaves: Vec<F> = init_leaves
            .par_iter()
            .enumerate()
            .map(|(i, leaf)| *leaf + t_final[i].field_mul(gamma_terms[6]))
            .collect();

        let init_leaves = rep3::arithmetic::promote_to_trivial_shares(init_leaves, io_ctx.id);
        let final_leaves = rep3::arithmetic::promote_to_trivial_shares(final_leaves, io_ctx.id);

        // TODO(moodlezoup): avoid concat
        Ok((
            ([read_leaves, write_leaves].concat(), 2),
            ([init_leaves, final_leaves].concat(), 2),
        ))
    }
}
