use crate::field::JoltField;
use crate::lasso::memory_checking::{self, Rep3MemoryCheckingProver};
use crate::poly::commitment::Rep3CommitmentScheme;
use crate::poly::opening_proof::Rep3OpeningAccumulatorCoordinator;
use crate::subprotocols::grand_product::Rep3BatchedDenseGrandProduct;
use jolt_core::jolt::vm::bytecode::BytecodeProof;
use jolt_core::lasso::memory_checking::{
    ExogenousOpenings, Initializable, NoExogenousOpenings, StructuredPolynomialData,
};
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

    type Rep3InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    #[tracing::instrument(skip_all, name = "ReadWriteMemoryProof::compute_openings")]
    fn receive_openings(
        read_write_chunk_size: usize,
        init_final_chunk_size: usize,
        preprocessing: &Self::Preprocessing,
        opening_accumulator: &mut Rep3OpeningAccumulatorCoordinator<F>,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(Self::Openings, Self::ExogenousOpenings)> {
        if !network.is_distributed() {
            return memory_checking::receive_openings::<
                F,
                _,
                Self::Openings,
                NoExogenousOpenings,
                _,
                _,
            >(
                read_write_chunk_size,
                init_final_chunk_size,
                preprocessing,
                opening_accumulator,
                transcript,
                network,
            );
        }

        let mut exogenous_openings = Self::ExogenousOpenings::default();
        let mut openings = Self::Openings::initialize(preprocessing);

        let read_write_evals: Vec<F> =
            additive::combine_additive_vec(network.receive_responses_from_subnets()?.remove(0));

        tracing::info!("read_write_evals: {:?}", read_write_evals);

        opening_accumulator.append_with_claims(
            read_write_chunk_size.log_2(),
            &read_write_evals,
            transcript,
            network,
        )?;

        let read_write_openings: Vec<_> = openings
            .read_write_values_grand_product_mut()
            .into_iter()
            .chain(exogenous_openings.openings_mut())
            .collect();

        read_write_openings
            .into_par_iter()
            .zip(read_write_evals.par_iter())
            .for_each(|(opening, eval)| {
                *opening = *eval;
            });

        let init_final_evals: Vec<F> =
            additive::combine_additive_vec(network.receive_responses_from_subnets()?.remove(0));

        tracing::info!("init_final_evals: {:?}", init_final_evals);

        opening_accumulator.append_with_claims(
            init_final_chunk_size.log_2(),
            &init_final_evals,
            transcript,
            network,
        )?;

        openings
            .init_final_values_mut()
            .into_par_iter()
            .zip(init_final_evals.par_iter())
            .for_each(|(opening, eval)| {
                *opening = *eval;
            });

        Ok((openings, exogenous_openings))
    }
}
