use crate::lasso::memory_checking::{MemoryCheckingProver, Rep3MemoryCheckingProver};
use crate::poly::opening_proof::Rep3ProverOpeningAccumulator;
use crate::subprotocols::commitment::DistributedCommitmentScheme;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::read_write_memory::ReadWriteMemoryPreprocessing;
use jolt_core::poly::compact_polynomial::CompactPolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::subprotocols::grand_product::BatchedDenseGrandProduct;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
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

use crate::jolt::vm::witness::Rep3JoltPolynomials;

pub trait Rep3ReadWriteMemoryCoordinator<F, PCS, ProofTranscript, Network>:
    MemoryCheckingProver<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn prove_rep3(
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials: &mut Rep3JoltPolynomials<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        pcs_setup: &PCS::Setup,
        io_ctx: &mut Network,
    ) -> Result<(), ProofVerifyError>;
}

impl<F, PCS, ProofTranscript, Network> Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
    for ReadWriteMemoryProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    type Rep3ReadWriteGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    type Rep3InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    fn init_final_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
    ) -> Self::Rep3InitFinalGrandProduct {
        todo!()
    }
}

impl<F, PCS, ProofTranscript, Network>
    Rep3ReadWriteMemoryCoordinator<F, PCS, ProofTranscript, Network>
    for ReadWriteMemoryProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn prove_rep3(
        preprocessing: &ReadWriteMemoryPreprocessing,
        polynomials: &mut Rep3JoltPolynomials<F>,
        opening_accumulator: &mut Rep3ProverOpeningAccumulator<F>,
        pcs_setup: &PCS::Setup,
        io_ctx: &mut Network,
    ) -> Result<(), ProofVerifyError> {
        todo!()
    }
}
