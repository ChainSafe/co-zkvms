use std::marker::PhantomData;

use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::interleave;
use jolt_core::{
    lasso::memory_checking::MultisetHashes,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, dense_mlpoly::DensePolynomial,
        field::JoltField, structured_poly::StructuredCommitment,
    },
    subprotocols::grand_product::{BatchedGrandProductCircuit, GrandProductCircuit},
    utils::{
        errors::ProofVerifyError, mul_0_1_optimized, split_poly_flagged,
        transcript::ProofTranscript,
    },
};
use mpc_net::mpc_star::MpcStarNetCoordinator;
// use mpc_net::mpc_star::MpcStarNetCoordinator;
use color_eyre::eyre::Result;
use spartan::transcript::Transcript;
use tracing::trace_span;

use crate::{grand_product::BatchedGrandProductArgument, lasso::{LassoPolynomials, MemoryCheckingProof}, LassoPreprocessing, Rep3LassoPolynomials};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Rep3MemoryCheckingProver<
    const C: usize,
    const M: usize,
    N: MpcStarNetCoordinator,
    F: JoltField,
    // C: CommitmentScheme<Field = F>,
    Polynomials,
> {
    pub _marker: PhantomData<(F, Polynomials, N)>,
}


type Preprocessing<F> = LassoPreprocessing<F>;

impl<F: JoltField, const C: usize, const M: usize, N: MpcStarNetCoordinator>
    Rep3MemoryCheckingProver<C, M, N, F, LassoPolynomials<F>>
{
    pub fn prove_rep3(
        preprocessing: &Preprocessing<F>,
        network: &mut N,
        transcript: &mut ProofTranscript,
    ) -> MemoryCheckingProof<F, LassoPolynomials<F>> {
        let (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        ) = Self::prove_grand_products(preprocessing, network, transcript);

        MemoryCheckingProof {
            _polys: PhantomData,
            multiset_hashes,
            read_write_grand_product,
            init_final_grand_product,
        }
    }

    fn prove_grand_products(
        preprocessing: &Preprocessing<F>,
        network: &mut N,
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

        // transcript.append_protocol_name(Self::protocol_name());

        let (read_write_leaves, init_final_leaves) =
            Self::compute_leaves(preprocessing, polynomials, &gamma, &tau);

        let (read_write_circuit, read_write_hashes) =
            Self::read_write_grand_product(preprocessing, polynomials, read_write_leaves);
        let (init_final_circuit, init_final_hashes) =
            Self::init_final_grand_product(preprocessing, polynomials, init_final_leaves);

        let multiset_hashes =
            Self::uninterleave_hashes(preprocessing, read_write_hashes, init_final_hashes);
        Self::check_multiset_equality(preprocessing, &multiset_hashes);
        multiset_hashes.append_to_transcript(transcript);

        let (read_write_grand_product, r_read_write) =
            BatchedGrandProductArgument::prove(read_write_circuit, transcript);
        let (init_final_grand_product, r_init_final) =
            BatchedGrandProductArgument::prove(init_final_circuit, transcript);

        (
            read_write_grand_product,
            init_final_grand_product,
            multiset_hashes,
            r_read_write,
            r_init_final,
        )
    }

    fn compute_leaves(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        gamma: &F,
        tau: &F,
    ) -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>) {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dims[0].len();

        let read_write_leaves = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];

                let read_fingerprints: Vec<F> = (0..num_lookups)
                    .map(|i| {
                        let a = &polynomials.dims[dim_index][i];
                        let v = &polynomials.e_polys[memory_index][i];
                        let t = &polynomials.read_cts[memory_index][i];
                        mul_0_1_optimized(t, &gamma_squared) + mul_0_1_optimized(v, gamma) + a - tau
                    })
                    .collect();
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| *read_fingerprint + gamma_squared)
                    .collect();
                [
                    DensePolynomial::new(read_fingerprints),
                    DensePolynomial::new(write_fingerprints),
                ]
            })
            .collect();

        let init_final_leaves: Vec<DensePolynomial<F>> = preprocessing
            .materialized_subtables
            .par_iter()
            .enumerate()
            .flat_map_iter(|(subtable_index, subtable)| {
                let init_fingerprints: Vec<F> = (0..M)
                    .map(|i| {
                        let a = &F::from_u64(i as u64).unwrap();
                        let v = &subtable[i];
                        // let t = F::zero();
                        // Compute h(a,v,t) where t == 0
                        mul_0_1_optimized(v, gamma) + a - tau
                    })
                    .collect();

                let final_leaves: Vec<DensePolynomial<F>> = preprocessing
                    .subtable_to_memory_indices[subtable_index]
                    .iter()
                    .map(|memory_index| {
                        let final_cts = &polynomials.final_cts[*memory_index];
                        let final_fingerprints = (0..M)
                            .map(|i| {
                                init_fingerprints[i]
                                    + mul_0_1_optimized(&final_cts[i], &gamma_squared)
                            })
                            .collect();
                        DensePolynomial::new(final_fingerprints)
                    })
                    .collect();

                let mut polys = Vec::with_capacity(C + 1);
                polys.push(DensePolynomial::new(init_fingerprints));
                polys.extend(final_leaves);
                polys
            })
            .collect();

        (read_write_leaves, init_final_leaves)
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        preprocessing: &Preprocessing<F>,
        read_write_leaves: Vec<DensePolynomial<F>>,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>) {
        assert_eq!(read_write_leaves.len(), 2 * preprocessing.num_memories);

        let _span = trace_span!("LassoLookups: construct circuits");
        let _enter = _span.enter();

        let memory_flag_polys =
            Self::memory_flag_polys(preprocessing, &polynomials.lookup_flag_bitvectors);

        let read_write_circuits = read_write_leaves
            .par_iter()
            .enumerate()
            .map(|(i, leaves_poly)| {
                // Split while cloning to save on future cloning in GrandProductCircuit
                let memory_index = i / 2;
                let flag: &DensePolynomial<F> = &memory_flag_polys[memory_index];
                println!("flag: {:?}", flag.len());
                println!("leaves_poly: {:?}", leaves_poly.len());
                let (toggled_leaves_l, toggled_leaves_r) = split_poly_flagged(leaves_poly, flag);
                GrandProductCircuit::new_split(
                    DensePolynomial::new(toggled_leaves_l),
                    DensePolynomial::new(toggled_leaves_r),
                )
            })
            .collect::<Vec<GrandProductCircuit<F>>>();

        drop(_enter);
        drop(_span);

        let _span = trace_span!("InstructionLookups: compute hashes");
        let _enter = _span.enter();

        let read_write_hashes: Vec<F> = read_write_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        drop(_enter);
        drop(_span);

        let _span = trace_span!("InstructionLookups: the rest");
        let _enter = _span.enter();

        // Prover has access to memory_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
        let batched_circuits = BatchedGrandProductCircuit::new_batch_flags(
            read_write_circuits,
            memory_flag_polys,
            read_write_leaves,
        );

        drop(_enter);
        drop(_span);

        (batched_circuits, read_write_hashes)
    }

    /// Converts instruction flag polynomials into memory flag polynomials. A memory flag polynomial
    /// can be computed by summing over the instructions that use that memory: if a given execution step
    /// accesses the memory, it must be executing exactly one of those instructions.
    #[tracing::instrument(skip_all)]
    fn memory_flag_polys(
        preprocessing: &Preprocessing<F>,
        instruction_flag_bitvectors: &[Vec<u64>],
    ) -> Vec<DensePolynomial<F>> {
        let m = instruction_flag_bitvectors[0].len();

        (0..preprocessing.num_memories)
            .into_par_iter()
            .map(|memory_index| {
                let mut memory_flag_bitvector = vec![0u64; m];
                for instruction_index in 0..preprocessing.lookups.len() {
                    if preprocessing.lookup_to_memory_indices[instruction_index]
                        .contains(&memory_index)
                    {
                        memory_flag_bitvector
                            .iter_mut()
                            .zip(&instruction_flag_bitvectors[instruction_index])
                            .for_each(|(memory_flag, instruction_flag)| {
                                *memory_flag += instruction_flag
                            });
                    }
                }
                DensePolynomial::from_u64(&memory_flag_bitvector)
            })
            .collect()
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::init_final_grand_product")]
    fn init_final_grand_product(
        _preprocessing: &Preprocessing<F>,
        _polynomials: &Polynomials<F>,
        init_final_leaves: Vec<DensePolynomial<F>>,
    ) -> (BatchedGrandProductCircuit<F>, Vec<F>) {
        let init_final_circuits: Vec<GrandProductCircuit<F>> = init_final_leaves
            .par_iter()
            .map(|leaves| GrandProductCircuit::new(leaves))
            .collect();
        let init_final_hashes: Vec<F> = init_final_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        (
            BatchedGrandProductCircuit::new_batch(init_final_circuits),
            init_final_hashes,
        )
    }

    fn uninterleave_hashes(
        _preprocessing: &Preprocessing<F>,
        read_write_hashes: Vec<F>,
        init_final_hashes: Vec<F>,
    ) -> MultisetHashes<F> {
        assert_eq!(read_write_hashes.len() % 2, 0);
        let num_memories = read_write_hashes.len() / 2;

        let mut read_hashes = Vec::with_capacity(num_memories);
        let mut write_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            read_hashes.push(read_write_hashes[2 * i]);
            write_hashes.push(read_write_hashes[2 * i + 1]);
        }

        let mut init_hashes = Vec::with_capacity(num_memories);
        let mut final_hashes = Vec::with_capacity(num_memories);
        for i in 0..num_memories {
            init_hashes.push(init_final_hashes[2 * i]);
            final_hashes.push(init_final_hashes[2 * i + 1]);
        }

        MultisetHashes {
            read_hashes,
            write_hashes,
            init_hashes,
            final_hashes,
        }
    }

    fn check_multiset_equality(
        _preprocessing: &Preprocessing<F>,
        multiset_hashes: &MultisetHashes<F>,
    ) {
        let num_memories = multiset_hashes.read_hashes.len();
        assert_eq!(multiset_hashes.final_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.write_hashes.len(), num_memories);
        assert_eq!(multiset_hashes.init_hashes.len(), num_memories);

        (0..num_memories).into_par_iter().for_each(|i| {
            let read_hash = multiset_hashes.read_hashes[i];
            let write_hash = multiset_hashes.write_hashes[i];
            let init_hash = multiset_hashes.init_hashes[i];
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

    pub fn verify_memory_checking(
        preprocessing: &Preprocessing<F>,
        mut proof: MemoryCheckingProof<F, Polynomials<F>>,
        // commitments: &Polynomials::Commitment,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        // Fiat-Shamir randomness for multiset hashes
        let gamma: F = transcript.challenge_scalar(b"Memory checking gamma");
        let tau: F = transcript.challenge_scalar(b"Memory checking tau");

        // transcript.append_protocol_name(Self::protocol_name());

        Self::check_multiset_equality(preprocessing, &proof.multiset_hashes);
        proof.multiset_hashes.append_to_transcript(transcript);

        let (read_write_hashes, init_final_hashes) =
            Self::interleave_hashes(preprocessing, &proof.multiset_hashes);

        let (claims_read_write, r_read_write) = proof
            .read_write_grand_product
            .verify(&read_write_hashes, transcript);
        let (claims_init_final, r_init_final) = proof
            .init_final_grand_product
            .verify(&init_final_hashes, transcript);

        // proof.read_write_openings.verify_openings(
        //     generators,
        //     &proof.read_write_opening_proof,
        //     commitments,
        //     &r_read_write,
        //     transcript,
        // )?;
        // proof.init_final_openings.verify_openings(
        //     generators,
        //     &proof.init_final_opening_proof,
        //     commitments,
        //     &r_init_final,
        //     transcript,
        // )?;

        // proof
        //     .read_write_openings
        //     .compute_verifier_openings(&NoPreprocessing, &r_read_write);
        // proof
        //     .init_final_openings
        //     .compute_verifier_openings(preprocessing, &r_init_final);

        // Self::check_fingerprints(
        //     preprocessing,
        //     claims_read_write,
        //     claims_init_final,
        //     &proof.read_write_openings,
        //     &proof.init_final_openings,
        //     &gamma,
        //     &tau,
        // );

        Ok(())
    }
}
