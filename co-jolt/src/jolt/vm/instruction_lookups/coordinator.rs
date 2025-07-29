use crate::{
    lasso::memory_checking::Rep3MemoryCheckingProver,
    poly::{commitment::Rep3CommitmentScheme, opening_proof::Rep3ProverOpeningAccumulator},
    subprotocols::{
        grand_product::Rep3BatchedDenseGrandProduct,
        sparse_grand_product::Rep3ToggledBatchedGrandProduct,
    },
};
use color_eyre::eyre::Result;
use eyre::Context;
use jolt_core::utils::transcript::{AppendToTranscript, Transcript};
use jolt_core::{
    field::JoltField,
    jolt::subtable::JoltSubtableSet,
    poly::unipoly::{CompressedUniPoly, UniPoly},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::math::Math,
};
use mpc_core::protocols::additive;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use std::marker::PhantomData;

use super::{
    InstructionLookupsPreprocessing, InstructionLookupsProof, PrimarySumcheck,
    PrimarySumcheckOpenings,
};
use crate::jolt::instruction::JoltInstructionSet;

impl<F, const C: usize, const M: usize, PCS, ProofTranscript, Instructions, Subtables>
    InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    Instructions: JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove")]
    pub fn prove_rep3<Network: Rep3NetworkCoordinator>(
        num_ops: usize,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> Result<InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>>
    {
        transcript.append_message(Self::protocol_name());

        let r_eq = transcript.challenge_vector::<F>(num_ops.log_2());
        network.broadcast_request(r_eq)?;

        let num_rounds = num_ops.log_2();

        let primary_sumcheck_proof =
            Self::prove_primary_sumcheck_rep3(num_rounds, transcript, network)
                .context("while proving primary sumcheck")?;

        let mut flag_evals = vec![F::zero(); Self::NUM_INSTRUCTIONS];
        let mut E_evals = vec![F::zero(); preprocessing.num_memories];
        let mut outputs_eval = F::zero();

        let claims = Rep3ProverOpeningAccumulator::<F>::receive_claims(transcript, network)?;

        E_evals
            .iter_mut()
            .chain(flag_evals.iter_mut())
            .chain([&mut outputs_eval])
            .zip(claims.into_iter())
            .for_each(|(eval, claim)| {
                *eval = claim;
            });

        // Create a single opening proof for the flag_evals and memory_evals
        let sumcheck_openings = PrimarySumcheckOpenings {
            E_poly_openings: E_evals,
            flag_openings: flag_evals,
            lookup_outputs_opening: outputs_eval,
        };

        let primary_sumcheck = PrimarySumcheck::<F, ProofTranscript> {
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            openings: sumcheck_openings,
            _marker: PhantomData,
        };

        let memory_checking_proof =
            Self::coordinate_memory_checking(preprocessing, num_ops, M, transcript, network)
                .context("while proving memory checking")?;

        Ok(InstructionLookupsProof {
            primary_sumcheck,
            memory_checking: memory_checking_proof,
            _instructions: PhantomData,
            _subtables: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_primary_sumcheck")]
    fn prove_primary_sumcheck_rep3<Network: Rep3NetworkCoordinator>(
        num_rounds: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<SumcheckInstanceProof<F, ProofTranscript>> {
        // Check all polys are the same size

        let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            let round_poly = UniPoly::from_coeff(additive::combine_field_element_vec::<F>(
                network.receive_responses()?,
            ));
            let compressed_round_poly = round_poly.compress();
            compressed_round_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_round_poly);
            let r_j = transcript.challenge_scalar::<F>();
            let new_claim = round_poly.evaluate(&r_j);
            network.broadcast_request((r_j, new_claim))?;
            random_vars.push(r_j);
        } // End rounds

        Ok(SumcheckInstanceProof::new(compressed_polys))
    }
}

use crate::subprotocols::grand_product::Rep3BatchedGrandProduct;

impl<F, const C: usize, const M: usize, PCS, ProofTranscript, Instructions, Subtables, Network>
    Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
    for InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: Rep3CommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Instructions: JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    Network: Rep3NetworkCoordinator,
{
    type Rep3ReadWriteGrandProduct = Rep3ToggledBatchedGrandProduct<F>;
    type Rep3InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    fn read_write_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
        num_lookups: usize,
    ) -> Rep3ToggledBatchedGrandProduct<F> {
        <Rep3ToggledBatchedGrandProduct<F> as Rep3BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::construct(num_lookups.log_2() + 1)
    }
}
