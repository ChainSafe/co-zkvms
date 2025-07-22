use std::marker::PhantomData;

use crate::lasso::memory_checking::{MemoryCheckingProver, Rep3MemoryCheckingProver};
use crate::poly::opening_proof::Rep3ProverOpeningAccumulator;
use crate::subprotocols::commitment::DistributedCommitmentScheme;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use crate::subprotocols::sumcheck;
use jolt_core::field::JoltField;
use jolt_core::jolt::vm::read_write_memory::{OutputSumcheckProof, ReadWriteMemoryPreprocessing};
use jolt_core::poly::compact_polynomial::CompactPolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::opening_proof::ProverOpeningAccumulator;
use jolt_core::subprotocols::grand_product::BatchedDenseGrandProduct;
use jolt_core::utils::thread::unsafe_allocate_zero_vec;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::protocols::rep3::PartyID;
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
    Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    fn prove_rep3(
        pcs_setup: &PCS::Setup,
        preprocessing: &ReadWriteMemoryPreprocessing,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<ReadWriteMemoryProof<F, PCS, ProofTranscript>>;

    fn coordinate_prove_outputs(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<OutputSumcheckProof<F, PCS, ProofTranscript>>;
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
        pcs_setup: &PCS::Setup,
        preprocessing: &ReadWriteMemoryPreprocessing,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<ReadWriteMemoryProof<F, PCS, ProofTranscript>> {
        let memory_checking_proof =
            Self::coordinate_memory_checking(preprocessing, transcript, network)?;

        Ok(ReadWriteMemoryProof {
            memory_checking_proof,
            output_proof: todo!(),
            timestamp_validity_proof: todo!(),
        })
    }

    fn coordinate_prove_outputs(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<OutputSumcheckProof<F, PCS, ProofTranscript>> {
        let memory_size = network.receive_response(PartyID::ID0, 0, 0usize)?;
        let num_rounds = memory_size.log_2();
        let r_eq: Vec<F> = transcript.challenge_vector(num_rounds);
        network.broadcast_request(r_eq)?;
        let (sumcheck_proof, r_sumcheck, sumcheck_openings) =
            sumcheck::coordinate_prove_arbitrary::<F, _, Network>(
                num_rounds, transcript, network,
            )?;

        let sumcheck_openings = Rep3ProverOpeningAccumulator::receive_claims(transcript, network)?;

        Ok(OutputSumcheckProof {
            num_rounds,
            sumcheck_proof,
            opening: sumcheck_openings[2],
            _pcs: PhantomData,
        })
    }
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
