use crate::lasso::memory_checking::Rep3MemoryCheckingProver;
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use crate::field::JoltField;
use jolt_core::jolt::vm::bytecode::BytecodeProof;
use jolt_core::utils::transcript::Transcript;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;

impl<F, PCS, ProofTranscript, Network> Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
    for BytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    type Rep3ReadWriteGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    type Rep3InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;
}
