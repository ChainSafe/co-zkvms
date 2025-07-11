use std::marker::PhantomData;

use co_spartan::mpc::{additive, rep3::Rep3PrimeFieldShare};
use eyre::Context;
use itertools::{interleave, Itertools};
use jolt_core::{
    lasso::memory_checking::MultisetHashes,
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
use mpc_net::mpc_star::MpcStarNetCoordinator;
// use mpc_net::mpc_star::MpcStarNetCoordinator;
use color_eyre::eyre::Result;

use crate::{
    grand_product::BatchedGrandProductProver,
    lasso::{LassoPreprocessing, LassoProof, MemoryCheckingProof, PrimarySumcheck},
    subtables::SubtableSet,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Rep3MemoryCheckingProver<const C: usize, const M: usize, F, Subtables, Network> {
    pub _marker: PhantomData<(F, Subtables, Network)>,
}

type Preprocessing<F> = LassoPreprocessing<F>;

impl<F: JoltField, const C: usize, const M: usize, Subtables, Network>
    Rep3MemoryCheckingProver<C, M, F, Subtables, Network>
where
    Subtables: SubtableSet<F>,
    Network: MpcStarNetCoordinator,
{
    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove")]
    pub fn prove(
        trace_length: usize,
        preprocessing: &Preprocessing<F>,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> Result<LassoProof<C, M, F>> {
        let r_eq =
            transcript.challenge_vector::<F>(b"Jolt instruction lookups", trace_length.log_2());
        network.broadcast_request(r_eq)?;

        let num_rounds = trace_length.log_2();

        let (primary_sumcheck_proof, r_primary_sumcheck, flag_evals, E_evals, outputs_eval) =
            Self::prove_primary_sumcheck(num_rounds, transcript, network)?;

        let primary_sumcheck = PrimarySumcheck {
            sumcheck_proof: primary_sumcheck_proof,
            num_rounds,
            // openings: sumcheck_openings,
            // opening_proof: sumcheck_opening_proof,
        };

        let memory_checking_proof =
            Self::prove_memory_checking(preprocessing, network, transcript)?;

        Ok(LassoProof {
            primary_sumcheck,
            memory_checking: memory_checking_proof,
        })
    }

    /// Prove Jolt primary sumcheck including instruction collation.
    ///
    /// Computes \sum{ eq(r,x) * [ flags_0(x) * g_0(E(x)) + flags_1(x) * g_1(E(x)) + ... + flags_{NUM_INSTRUCTIONS}(E(x)) * g_{NUM_INSTRUCTIONS}(E(x)) ]}
    /// via the sumcheck protocol.
    /// Note: These E(x) terms differ from term to term depending on the memories used in the instruction.
    ///
    /// Returns: (SumcheckProof, Random evaluation point, claimed evaluations of polynomials)
    ///
    /// Params:
    /// - `claim`: Claimed sumcheck evaluation.
    /// - `num_rounds`: Number of rounds to run sumcheck. Corresponds to the number of free bits or free variables in the polynomials.
    /// - `memory_polys`: Each of the `E` polynomials or "dereferenced memory" polynomials.
    /// - `flag_polys`: Each of the flag selector polynomials describing which instruction is used at a given step of the CPU.
    /// - `degree`: Degree of the inner sumcheck polynomial. Corresponds to number of evaluation points per round.
    /// - `transcript`: Fiat-shamir transcript.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "Rep3LassoProver::prove_primary_sumcheck")]
    fn prove_primary_sumcheck(
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

    #[tracing::instrument(skip_all, name = "Rep3MemoryCheckingProver::prove_memory_checking")]
    pub fn prove_memory_checking(
        preprocessing: &Preprocessing<F>,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> Result<MemoryCheckingProof<F>> {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products(preprocessing, network, transcript)
            .context("while proving grand products")?;

        Ok(MemoryCheckingProof {
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
        })
    }

    fn prove_grand_products(
        preprocessing: &Preprocessing<F>,
        network: &mut Network,
        transcript: &mut ProofTranscript,
    ) -> Result<(
        BatchedGrandProductArgument<F>,
        BatchedGrandProductArgument<F>,
        MultisetHashes<F>,
        Vec<F>,
        Vec<F>,
    )> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar::<F>(b"Memory checking gamma");
        let tau: F = transcript.challenge_scalar::<F>(b"Memory checking tau");
        network.broadcast_request((gamma, tau))?;
        let num_lookups = network.receive_responses(0usize)?[0];

        let (read_write_hashes_shares, init_final_hashes_shares): (Vec<Vec<_>>, Vec<Vec<_>>) =
            network
                .receive_responses((Vec::default(), Vec::default()))
                .context("while receiving hashes")?
                .into_iter()
                .unzip();

        assert_eq!(read_write_hashes_shares.len(), 3);
        assert_eq!(init_final_hashes_shares.len(), 3);

        let read_write_hashes = additive::combine_field_elements(
            &read_write_hashes_shares[0],
            &read_write_hashes_shares[1],
            &read_write_hashes_shares[2],
        );
        let init_final_hashes = additive::combine_field_elements(
            &init_final_hashes_shares[0],
            &init_final_hashes_shares[1],
            &init_final_hashes_shares[2],
        );

        let multiset_hashes =
            Self::uninterleave_hashes(preprocessing, &read_write_hashes, &init_final_hashes);
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let num_layers_read_write = (num_lookups).log_2() + 1; // +1 for the flag layer
        let num_layers_init_final = M.log_2();

        let (read_write_grand_product, r_read_write) = BatchedGrandProductProver::prove(
            read_write_hashes,
            num_layers_read_write,
            network,
            transcript,
        )?;
        let (init_final_grand_product, r_init_final) = BatchedGrandProductProver::prove(
            init_final_hashes,
            num_layers_init_final,
            network,
            transcript,
        )?;

        Ok((
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ))
    }

    fn uninterleave_hashes(
        preprocessing: &Preprocessing<F>,
        read_write_hashes: &[F],
        init_final_hashes: &[F],
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len() % 2, 0);
        let num_memories = read_write_hashes.len() / 2;

        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(Subtables::COUNT);
        let mut final_hashes = Vec::with_capacity(preprocessing.num_memories);
        let mut init_final_hashes = init_final_hashes.iter();
        for subtable_index in 0..Subtables::COUNT {
            // I
            init_hashes.push(*init_final_hashes.next().unwrap());
            // F F F F
            let memory_indices = &preprocessing.subtable_to_memory_indices[subtable_index];
            for _ in memory_indices {
                final_hashes.push(*init_final_hashes.next().unwrap());
            }
        }
        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        preprocessing: &Preprocessing<F>,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        assert_eq!(
            multiset_hashes.final_hashes.len(),
            preprocessing.num_memories
        );
        assert_eq!(
            multiset_hashes.write_hashes.len(),
            preprocessing.num_memories
        );
        assert_eq!(multiset_hashes.init_hashes.len(), Subtables::COUNT);

        (0..preprocessing.num_memories)
            .into_par_iter()
            .for_each(|i| {
                let read_hash = multiset_hashes.read_hashes[i];
                let write_hash = multiset_hashes.write_hashes[i];
                let init_hash =
                    multiset_hashes.init_hashes[preprocessing.memory_to_subtable_index[i]];
                let final_hash = multiset_hashes.final_hashes[i];

                assert_eq!(
                    init_hash * write_hash,
                    final_hash * read_hash,
                    "Multiset hashes don't match"
                );
            });
    }

    fn interleave_hashes(
        _preprocessing: &Preprocessing<F>,
        multiset_hashes: &MultisetHashes<F>,
    ) -> (Vec<F>, Vec<F>) {
        let read_write_hashes = interleave(
            multiset_hashes.read_hashes.clone(),
            multiset_hashes.write_hashes.clone(),
        )
        .collect();
        let init_final_hashes = interleave(
            multiset_hashes.init_hashes.clone(),
            multiset_hashes.final_hashes.clone(),
        )
        .collect();

        (read_write_hashes, init_final_hashes)
    }

    fn update_primary_sumcheck_transcript(
        round_uni_poly: UniPoly<F>,
        transcript: &mut ProofTranscript,
    ) -> F {
        round_uni_poly.append_to_transcript(b"poly", transcript);

        transcript.challenge_scalar::<F>(b"challenge_nextround")
    }
}
