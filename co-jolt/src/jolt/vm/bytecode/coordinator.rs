use crate::field::JoltField;
use crate::lasso::memory_checking::{self, Rep3MemoryCheckingProver};
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::opening_proof::Rep3OpeningAccumulatorCoordinator;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use jolt_core::jolt::vm::bytecode::BytecodeProof;
use jolt_core::lasso::memory_checking::{
    ExogenousOpenings, Initializable, NoExogenousOpenings, StructuredPolynomialData,
};
use jolt_core::subprotocols::grand_product::BatchedDenseGrandProduct;
use jolt_core::utils::transcript::Transcript;
use mpc_core::protocols::additive;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use snarks_core::math::Math;

use rayon::prelude::*;
impl<F, PCS, ProofTranscript, Network> Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
    for BytecodeProof<F, PCS, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
{
    type Rep3ReadWriteGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    type Rep3InitFinalGrandProduct = BatchedDenseGrandProduct<F>;
}
