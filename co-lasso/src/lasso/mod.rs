mod prover;

use std::{iter, marker::PhantomData};

use ark_std::cfg_into_iter;
use itertools::{chain, Itertools};
use jolt_core::{
    jolt::vm::instruction_lookups::{InstructionCommitment, InstructionPolynomials},
    poly::{
        commitment::commitment_scheme::{BatchType, CommitmentScheme},
        dense_mlpoly::DensePolynomial,
        field::JoltField,
        structured_poly::StructuredCommitment,
    },
    utils::math::Math,
};

use crate::{
    instructions::LookupSet,
    subtables::{LassoSubtable, SubtableIndices, SubtableSet},
};
pub use prover::{LassoProof, PrimarySumcheck, InstructionReadWriteOpenings, InstructionFinalOpenings};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct InstructionLookupsPreprocessing<F: JoltField> {
    pub subtable_to_memory_indices: Vec<Vec<usize>>,
    pub instruction_to_memory_indices: Vec<Vec<usize>>,
    pub memory_to_subtable_index: Vec<usize>,
    pub memory_to_dimension_index: Vec<usize>,
    pub materialized_subtables: Vec<Vec<F>>,
    // pub lookup_id_to_index: HashMap<LookupId, usize>,
    pub num_memories: usize,
    // pub lookups: BTreeMap<LookupId, Box<dyn LookupType<F>>>,
}

impl<F: JoltField> InstructionLookupsPreprocessing<F> {
    #[tracing::instrument(skip_all, name = "Lasso::preprocess")]
    pub fn preprocess<const C: usize, const M: usize, Lookups, Subtables>() -> Self
    where
        Lookups: LookupSet<F>,
        Subtables: SubtableSet<F>,
    {
        let materialized_subtables = Self::materialize_subtables::<M, Subtables>();

        // Build a mapping from subtable type => chunk indices that access that subtable type
        let mut subtable_indices: Vec<SubtableIndices> =
            vec![SubtableIndices::with_capacity(C); Subtables::COUNT];
        // let mut subtable_id_to_index = HashMap::with_capacity(subtables.len());
        for lookup in Lookups::iter() {
            for (subtable, indices) in lookup.subtables(C, M) {
                subtable_indices[Subtables::enum_index(subtable)].union_with(&indices);
            }
        }

        let mut subtable_to_memory_indices = Vec::with_capacity(Subtables::COUNT);
        let mut memory_to_subtable_index = vec![];
        let mut memory_to_dimension_index = vec![];

        let mut memory_index = 0;
        for (subtable_index, dimension_indices) in subtable_indices.iter().enumerate() {
            subtable_to_memory_indices
                .push((memory_index..memory_index + dimension_indices.len()).collect_vec());
            memory_to_subtable_index.extend(vec![subtable_index; dimension_indices.len()]);
            memory_to_dimension_index.extend(dimension_indices.iter());
            memory_index += dimension_indices.len();
        }
        let num_memories = memory_index;

        let mut lookup_to_memory_indices = vec![vec![]; Lookups::COUNT];
        for lookup_type in Lookups::iter() {
            for (subtable, dimension_indices) in lookup_type.subtables(C, M) {
                let memory_indices: Vec<_> = subtable_to_memory_indices
                    [Subtables::enum_index(subtable)]
                .iter()
                .filter(|memory_index| {
                    dimension_indices.contains(memory_to_dimension_index[**memory_index])
                })
                .collect();
                lookup_to_memory_indices[Lookups::enum_index(&lookup_type)].extend(memory_indices);
            }
        }

        Self {
            num_memories,
            materialized_subtables,
            subtable_to_memory_indices,
            memory_to_subtable_index,
            memory_to_dimension_index,
            instruction_to_memory_indices: lookup_to_memory_indices,
        }
    }

    fn materialize_subtables<const M: usize, Subtables: SubtableSet<F>>() -> Vec<Vec<F>> {
        let mut subtables = Vec::with_capacity(Subtables::COUNT);
        for subtable in Subtables::iter() {
            subtables.push(subtable.materialize(M));
        }
        subtables
    }
}

pub type LassoPolynomials<F: JoltField, CS> = InstructionPolynomials<F, CS>;

impl<const C: usize, const M: usize, F: JoltField, CS, Lookups, Subtables>
    LassoProof<C, M, F, CS, Lookups, Subtables>
where
    CS: CommitmentScheme<Field = F>,
    Lookups: LookupSet<F>,
    Subtables: SubtableSet<F>,
{
    #[tracing::instrument(skip_all, name = "Lasso::polynomialize")]
    pub fn polynomialize(
        preprocessing: &InstructionLookupsPreprocessing<F>,
        ops: &[Lookups],
    ) -> LassoPolynomials<F, CS> {
        let num_reads = ops.len().next_power_of_two();

        let subtable_lookup_indices = Self::subtable_lookup_indices(ops);

        let polys: Vec<_> = cfg_into_iter!(0..preprocessing.num_memories)
            .map(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                let access_sequence = &subtable_lookup_indices[dim_index];

                let mut final_cts_i = vec![0usize; M];
                let mut read_cts_i = vec![0usize; num_reads];
                let mut subtable_lookups = vec![F::ZERO; num_reads];

                for (j, lookup) in ops.iter().enumerate() {
                    let memories_used =
                        &preprocessing.instruction_to_memory_indices[Lookups::enum_index(lookup)];
                    if memories_used.contains(&memory_index) {
                        let memory_address = access_sequence[j];
                        // debug_assert!(memory_address < M);

                        let counter = final_cts_i[memory_address];
                        read_cts_i[j] = counter;
                        final_cts_i[memory_address] = counter + 1;
                        subtable_lookups[j] =
                            preprocessing.materialized_subtables[subtable_index][memory_address];
                    }
                }

                (
                    DensePolynomial::from_usize(&read_cts_i),
                    DensePolynomial::from_usize(&final_cts_i),
                    DensePolynomial::new(subtable_lookups),
                )
            })
            .collect();

        // Vec<(DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>)> -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>)
        let (read_cts, final_cts, e_polys) = polys.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut read_acc, mut final_acc, mut e_acc), (read, f, e)| {
                read_acc.push(read);
                final_acc.push(f);
                e_acc.push(e);
                (read_acc, final_acc, e_acc)
            },
        );
        let dims: Vec<_> = subtable_lookup_indices
            .into_iter()
            .take(C)
            .map(|mut access_sequence| {
                access_sequence.resize(access_sequence.len().next_power_of_two(), 0);
                DensePolynomial::from_usize(&access_sequence)
            })
            .collect();

        let mut lookup_flag_bitvectors: Vec<Vec<u64>> = vec![vec![0u64; num_reads]; Lookups::COUNT];

        for (j, lookup) in ops.iter().enumerate() {
            lookup_flag_bitvectors[Lookups::enum_index(lookup)][j] = 1;
        }

        let lookup_flag_polys: Vec<_> = lookup_flag_bitvectors
            .iter()
            .map(|flag_bitvector| DensePolynomial::from_u64(flag_bitvector))
            .collect();

        let mut lookup_outputs = Self::compute_lookup_outputs(ops);
        lookup_outputs.resize(num_reads, F::zero());

        let lookup_outputs = DensePolynomial::new(lookup_outputs);
        LassoPolynomials {
            _marker: PhantomData,
            dim: dims,
            read_cts,
            final_cts,
            instruction_flag_polys: lookup_flag_polys,
            instruction_flag_bitvectors: lookup_flag_bitvectors,
            E_polys: e_polys,
            lookup_outputs,
        }
    }

    fn compute_lookup_outputs(inputs: &[Lookups]) -> Vec<F> {
        cfg_into_iter!(inputs)
            .map(|lookup| lookup.lookup_entry())
            .collect()
    }

    #[tracing::instrument(skip_all, name = "LassoNode::subtable_lookup_indices")]
    fn subtable_lookup_indices(inputs: &[Lookups]) -> Vec<Vec<usize>> {
        let num_chunks = C;
        let log_M = M.log_2();

        let indices: Vec<_> = cfg_into_iter!(inputs)
            .map(|lookup| lookup.to_indices(C, log_M))
            .collect();

        let lookup_indices = (0..num_chunks)
            .map(|i| {
                indices
                    .iter()
                    .map(|indices| indices[i].clone())
                    .collect_vec()
            })
            .collect_vec();
        lookup_indices
    }
}
