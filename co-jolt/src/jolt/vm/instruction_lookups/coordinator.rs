use co_lasso::{
    memory_checking::Rep3MemoryCheckingProver,
    subprotocols::{
        commitment::DistributedCommitmentScheme, grand_product::Rep3BatchedDenseGrandProduct,
        sparse_grand_product::Rep3ToggledBatchedGrandProduct,
    },
    utils::transcript::{AppendToTranscript, Transcript},
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::chain;
use jolt_core::{
    field::JoltField,
    poly::unipoly::{CompressedUniPoly, UniPoly},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::math::Math,
};
use mpc_core::protocols::rep3::{self, network::Rep3NetworkCoordinator, PartyID};
use mpc_core::protocols::{additive, rep3::Rep3PrimeFieldShare};
use mpc_net::mpc_star::MpcStarNetCoordinator;
use std::marker::PhantomData;

use super::{
    witness::Rep3InstructionLookupPolynomials, InstructionLookupPolynomials,
    InstructionLookupsPreprocessing, InstructionLookupsProof, PrimarySumcheck,
    PrimarySumcheckOpenings,
};
use crate::{
    jolt::{instruction::JoltInstructionSet, subtable::JoltSubtableSet},
    poly::eq_poly::EqPolynomial,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl<F, const C: usize, const M: usize, PCS, ProofTranscript, Instructions, Subtables>
    InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    Instructions: JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove")]
    pub fn prove_rep3<Network: Rep3NetworkCoordinator>(
        trace_length: usize,
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> Result<InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>>
    {
        transcript.append_message(Self::protocol_name());

        let r_eq = transcript.challenge_vector::<F>(trace_length.log_2());
        network.broadcast_request(r_eq)?;

        let num_rounds = trace_length.log_2();

        let (primary_sumcheck_proof, _r_primary_sumcheck, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck_rep3(num_rounds, transcript, network)
                .context("while proving primary sumcheck")?;

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
            Self::coordinate_memory_checking(preprocessing, transcript, network)
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
    ) -> eyre::Result<(
        SumcheckInstanceProof<F, ProofTranscript>,
        Vec<F>,
        Vec<F>,
        Vec<F>,
        F,
    )> {
        // Check all polys are the same size

        let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            let round_poly = UniPoly::from_coeff(
                additive::combine_field_element_vec::<F>(network.receive_responses(vec![])?),
            );
            let compressed_round_poly = round_poly.compress();
            compressed_round_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_round_poly);
            let r_j = transcript.challenge_scalar::<F>();
            let new_claim = round_poly.evaluate(&r_j);
            network.broadcast_request((r_j, new_claim))?;
            random_vars.push(r_j);
        } // End rounds

        // Pass evaluations at point r back in proof:
        // - flags(r) * NUM_INSTRUCTIONS
        // - E(r) * NUM_SUBTABLES

        // Polys are fully defined so we can just take the first (and only) evaluation
        // let flag_evals = (0..flag_polys.len()).map(|i| flag_polys[i][0]).collect();

        let flag_evals = network.receive_response(PartyID::ID0, 0, vec![])?;
        let [(me1, oe1), (me2, oe2), (me3, oe3)] = network
            .receive_responses((vec![], Rep3PrimeFieldShare::zero_share()))?
            .try_into()
            .unwrap();
        let memory_evals = rep3::combine_field_elements(&me1, &me2, &me3);
        let outputs_eval = rep3::combine_field_element(oe1, oe2, oe3);

        Ok((
            SumcheckInstanceProof::new(compressed_polys),
            random_vars,
            flag_evals,
            memory_evals,
            outputs_eval,
        ))
    }
}

use co_lasso::subprotocols::grand_product::Rep3BatchedGrandProduct;

impl<F, const C: usize, const M: usize, PCS, ProofTranscript, Instructions, Subtables, Network>
    Rep3MemoryCheckingProver<F, PCS, ProofTranscript, Network>
    for InstructionLookupsProof<C, M, F, PCS, Instructions, Subtables, ProofTranscript>
where
    F: JoltField,
    PCS: DistributedCommitmentScheme<F, ProofTranscript>,
    ProofTranscript: Transcript,
    Instructions: JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    Network: Rep3NetworkCoordinator,
{
    type Rep3ReadWriteGrandProduct = Rep3ToggledBatchedGrandProduct<F>;
    type Rep3InitFinalGrandProduct = Rep3BatchedDenseGrandProduct<F>;

    fn init_final_grand_product_rep3(
        _preprocessing: &Self::Preprocessing,
    ) -> Self::Rep3InitFinalGrandProduct {
        <Self::Rep3InitFinalGrandProduct as Rep3BatchedGrandProduct<
            F,
            PCS,
            ProofTranscript,
            Network,
        >>::construct(M.log_2())
    }
}
