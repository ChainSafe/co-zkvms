use co_lasso::{
    memory_checking::Rep3MemoryCheckingProver,
    poly::{Rep3DensePolynomial, Rep3StructuredOpeningProof},
    subprotocols::{
        commitment::{DistributedCommitmentScheme, PST13},
        grand_product::BatchedGrandProductProver,
    },
};
use color_eyre::eyre::Result;
use eyre::Context;
use itertools::{chain, interleave, Itertools};
use jolt_core::{
    poly::{
        field::JoltField,
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::{grand_product::BatchedGrandProductArgument, sumcheck::SumcheckInstanceProof},
    utils::{
        math::Math,
        transcript::{AppendToTranscript, ProofTranscript},
    },
};
use mpc_core::protocols::rep3::{self, PartyID};
use mpc_core::protocols::{additive, rep3::Rep3PrimeFieldShare};
use mpc_net::mpc_star::MpcStarNetCoordinator;
use std::marker::PhantomData;

use super::{
    witness::Rep3InstructionPolynomials, InstructionFinalOpenings, InstructionLookupsPreprocessing,
    InstructionLookupsProof, InstructionPolynomials, InstructionReadWriteOpenings, PrimarySumcheck,
    PrimarySumcheckOpenings,
};
use crate::{
    jolt::{instruction::JoltInstructionSet, subtable::JoltSubtableSet},
    poly::eq_poly::EqPolynomial,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type Preprocessing<F> = InstructionLookupsPreprocessing<F>;

impl<F: JoltField, const C: usize, const M: usize, CS, Lookups, Subtables>
    InstructionLookupsProof<C, M, F, CS, Lookups, Subtables>
where
    CS: DistributedCommitmentScheme<F>,
    Lookups: JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
{
    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove")]
    pub fn prove_rep3<Network: MpcStarNetCoordinator>(
        trace_length: usize,
        preprocessing: &Preprocessing<F>,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> Result<InstructionLookupsProof<C, M, F, CS, Lookups, Subtables>> {
        transcript.append_protocol_name(Self::protocol_name());

        let r_eq =
            transcript.challenge_vector::<F>(b"Jolt instruction lookups", trace_length.log_2());
        network.broadcast_request(r_eq)?;

        let num_rounds = trace_length.log_2();

        let (primary_sumcheck_proof, _r_primary_sumcheck, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck_rep3(num_rounds, transcript, network)?;

        // Create a single opening proof for the flag_evals and memory_evals
        let sumcheck_openings = PrimarySumcheckOpenings {
            E_poly_openings: E_evals,
            flag_openings: flag_evals,
            lookup_outputs_opening: outputs_eval,
        };

        let opening_proof = CS::distributed_batch_open(transcript, network)?;

        let primary_sumcheck = PrimarySumcheck::<F, CS> {
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            openings: sumcheck_openings,
            opening_proof,
        };

        let memory_checking_proof =
            Self::prove_memory_checking(M, preprocessing, network, transcript)?;

        Ok(InstructionLookupsProof {
            _instructions: PhantomData,
            primary_sumcheck,
            memory_checking: memory_checking_proof,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_primary_sumcheck")]
    fn prove_primary_sumcheck_rep3<Network: MpcStarNetCoordinator>(
        num_rounds: usize,
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<(SumcheckInstanceProof<F>, Vec<F>, Vec<F>, Vec<F>, F)> {
        // Check all polys are the same size

        let mut random_vars: Vec<F> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        let round_uni_poly = {
            let [p1, p2, p3] = network
                .receive_responses(UniPoly::from_coeff(vec![]))?
                .try_into()
                .unwrap();
            UniPoly::from_coeff(additive::combine_field_elements::<F>(
                &p1.as_slice(),
                &p2.as_slice(),
                &p3.as_slice(),
            ))
        };
        compressed_polys.push(round_uni_poly.compress());
        let r_j = Self::update_primary_sumcheck_transcript(round_uni_poly, transcript);
        network.broadcast_request(r_j)?;
        random_vars.push(r_j);

        for _round in 1..num_rounds {
            let round_uni_poly = {
                let [p1, p2, p3] = network
                    .receive_responses(UniPoly::from_coeff(vec![]))?
                    .try_into()
                    .unwrap();
                UniPoly::from_coeff(additive::combine_field_elements::<F>(
                    &p1.as_slice(),
                    &p2.as_slice(),
                    &p3.as_slice(),
                ))
            };
            compressed_polys.push(round_uni_poly.compress());
            let r_j = Self::update_primary_sumcheck_transcript(round_uni_poly, transcript);
            network.broadcast_request(r_j)?;
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

impl<F: JoltField, const C: usize, const M: usize, CS, Lookups, Subtables, Network>
    Rep3MemoryCheckingProver<F, CS, InstructionPolynomials<F, CS>, Network>
    for InstructionLookupsProof<C, M, F, CS, Lookups, Subtables>
where
    CS: DistributedCommitmentScheme<F>,
    Lookups: JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    Network: MpcStarNetCoordinator,
{
}

use mpc_net::mpc_star::MpcStarNetWorker;

impl<F: JoltField, C: DistributedCommitmentScheme<F>>
    Rep3StructuredOpeningProof<F, C, InstructionPolynomials<F, C>>
    for InstructionReadWriteOpenings<F>
{
    type Rep3Polynomials = Rep3InstructionPolynomials<F>;

    fn open_rep3<Network: MpcStarNetCoordinator>(
        _opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<Self> {
        let (dim_openings, read_openings, E_poly_openings): (
            Vec<Vec<F>>,
            Vec<Vec<F>>,
            Vec<Vec<F>>,
        ) = itertools::multiunzip(network.receive_responses((
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))?);

        let dim_openings =
            additive::combine_field_elements(&dim_openings[0], &dim_openings[1], &dim_openings[2]);
        let read_openings = additive::combine_field_elements(
            &read_openings[0],
            &read_openings[1],
            &read_openings[2],
        );

        let E_poly_openings = additive::combine_field_elements(
            &E_poly_openings[0],
            &E_poly_openings[1],
            &E_poly_openings[2],
        );

        let flag_openings = network.receive_response::<Vec<F>>(PartyID::ID0, 0, Vec::new())?;
        Ok(Self {
            dim_openings,
            read_openings,
            E_poly_openings,
            flag_openings,
        })
    }

    fn open_rep3_worker<Network: MpcStarNetWorker>(
        polynomials: &Self::Rep3Polynomials,
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<()> {
        // All of these evaluations share the lagrange basis polynomials.
        let chis = EqPolynomial::new(opening_point.to_vec()).evals();

        let dim_openings: Vec<F> = polynomials
            .dims
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect();
        let read_openings: Vec<F> = polynomials
            .read_cts
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect();
        let E_poly_openings: Vec<F> = polynomials
            .e_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect();
        let flag_openings: Vec<F> = polynomials
            .lookup_flag_polys
            .par_iter()
            .map(|poly| poly.evaluate_at_chi(&chis))
            .collect();

        network.send_response((dim_openings, read_openings, E_poly_openings))?;
        if network.party_id() == PartyID::ID0 {
            network.send_response(flag_openings)?;
        }

        Ok(())
    }

    fn prove_openings_rep3_worker<Network: MpcStarNetWorker>(
        polynomials: &Self::Rep3Polynomials,
        opening_point: &[F],
        setup: &C::Setup,
        network: &mut Network,
    ) -> eyre::Result<()> {
        let lookup_flag_polys = polynomials
            .lookup_flag_polys
            .iter()
            .map(|p| {
                Rep3DensePolynomial::new(rep3::arithmetic::promote_to_trivial_shares(
                    p.evals(),
                    network.party_id(),
                ))
            })
            .collect::<Vec<_>>();
        let read_write_polys = chain![
            polynomials.dims.iter(),
            polynomials.read_cts.iter(),
            polynomials.e_polys.iter(),
            lookup_flag_polys.iter(),
        ]
        .collect::<Vec<_>>();

        C::distributed_batch_open_worker(&read_write_polys, setup, opening_point, network)
    }

    fn prove_openings_rep3<Network: MpcStarNetCoordinator>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Self::Proof> {
        C::distributed_batch_open(transcript, network)
    }
}

impl<F: JoltField, C: DistributedCommitmentScheme<F>, Subtables: JoltSubtableSet<F>>
    Rep3StructuredOpeningProof<F, C, InstructionPolynomials<F, C>>
    for InstructionFinalOpenings<F, Subtables>
{
    type Rep3Polynomials = Rep3InstructionPolynomials<F>;

    fn open_rep3<Network: MpcStarNetCoordinator>(
        _opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<Self> {
        let final_openings = network.receive_responses::<Vec<F>>(Vec::new())?;
        let final_openings = additive::combine_field_elements(
            &final_openings[0],
            &final_openings[1],
            &final_openings[2],
        );
        Ok(Self {
            _subtables: PhantomData,
            final_openings,
            a_init_final: None,
            v_init_final: None,
        })
    }

    fn open_rep3_worker<Network: MpcStarNetWorker>(
        polynomials: &Self::Rep3Polynomials,
        opening_point: &[F],
        network: &mut Network,
    ) -> eyre::Result<()> {
        let final_openings = polynomials
            .final_cts
            .par_iter()
            .map(|final_cts_i| final_cts_i.evaluate(&opening_point))
            .collect::<Vec<_>>();

        network.send_response(final_openings)
    }

    fn prove_openings_rep3_worker<Network: MpcStarNetWorker>(
        polynomials: &Self::Rep3Polynomials,
        opening_point: &[F],
        setup: &<C>::Setup,
        network: &mut Network,
    ) -> eyre::Result<()> {
        C::distributed_batch_open_worker(
            &polynomials.final_cts.iter().collect::<Vec<_>>(),
            setup,
            opening_point,
            network,
        )
    }

    fn prove_openings_rep3<Network: MpcStarNetCoordinator>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Self::Proof> {
        C::distributed_batch_open(transcript, network)
    }
}
