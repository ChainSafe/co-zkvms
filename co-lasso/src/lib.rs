pub mod grand_product;
pub mod memory_checking;
pub mod subtables;

use std::{
    collections::{BTreeMap, HashMap},
    iter,
    marker::PhantomData,
    sync::Arc,
};

use ark_ff::{BigInteger, PrimeField};
use itertools::Itertools;
use jolt_core::poly::{dense_mlpoly::DensePolynomial, field::JoltField};
use mpc_core::{
    lut::LookupTableProvider,
    protocols::{
        rep3::{
            arithmetic, network::{IoContext, Rep3Network}, Rep3BigUintShare, Rep3PrimeFieldShare
        },
        rep3_ring::lut::{PublicPrivateLut, Rep3LookupTable},
    },
};

use crate::subtables::{LassoSubtable, LookupId, LookupType, SubtableIndices};

pub struct LassoPreprocessing<F: JoltField> {
    subtable_to_memory_indices: Vec<Vec<usize>>,
    lookup_to_memory_indices: Vec<Vec<usize>>,
    memory_to_subtable_index: Vec<usize>,
    memory_to_dimension_index: Vec<usize>,
    materialized_subtables: Vec<Vec<F>>,
    subtables_by_idx: Option<Vec<Box<dyn LassoSubtable<F>>>>,
    lookup_id_to_index: HashMap<LookupId, usize>,
    num_memories: usize,
    lookups: BTreeMap<LookupId, Box<dyn LookupType<F>>>,
}

impl<F: JoltField> LassoPreprocessing<F> {
    #[tracing::instrument(skip_all, name = "LassoNode::preprocess")]
    pub fn preprocess<const C: usize, const M: usize>(
        lookups: impl IntoIterator<Item = Box<dyn LookupType<F>>>,
    ) -> Self {
        let lookups = BTreeMap::from_iter(
            lookups
                .into_iter()
                .map(|lookup| (lookup.lookup_id(), lookup)),
        );
        println!("lookups: {:?}", lookups);

        let lookup_id_to_index: HashMap<_, _> = HashMap::from_iter(
            lookups
                .keys()
                .enumerate()
                .map(|(i, lookup_id)| (lookup_id.clone(), i)),
        );

        let subtables = lookups
            .values()
            .flat_map(|lookup| {
                lookup
                    .subtables(C, M)
                    .into_iter()
                    .map(|(subtable, _)| subtable)
            })
            .unique_by(|subtable| subtable.subtable_id())
            .collect_vec();

        // Build a mapping from subtable type => chunk indices that access that subtable type
        let mut subtable_indices: Vec<SubtableIndices> =
            vec![SubtableIndices::with_capacity(C); subtables.len()];
        let mut subtables_by_idx = vec![None; subtables.len()];
        let mut subtable_id_to_index = HashMap::with_capacity(subtables.len());
        for (_, lookup) in &lookups {
            for (subtable, indices) in lookup.subtables(C, M).into_iter() {
                let subtable_idx = subtable_id_to_index
                    .entry(subtable.subtable_id())
                    .or_insert_with(|| {
                        subtables
                            .iter()
                            .position(|s| s.subtable_id() == subtable.subtable_id())
                            .expect("Subtable not found")
                    });
                subtables_by_idx[*subtable_idx].get_or_insert(subtable);
                subtable_indices[*subtable_idx].union_with(&indices);
            }
        }

        let mut subtable_to_memory_indices = Vec::with_capacity(subtables.len());
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

        // instruction is a type of lookup
        // assume all instreuctions are the same first
        let mut lookup_to_memory_indices = vec![vec![]; lookups.len()];
        for (lookup_index, lookup_type) in lookups.values().enumerate() {
            for (subtable, dimension_indices) in lookup_type.subtables(C, M) {
                let memory_indices: Vec<_> = subtable_to_memory_indices
                    [subtable_id_to_index[&subtable.subtable_id()]]
                    .iter()
                    .filter(|memory_index| {
                        dimension_indices.contains(memory_to_dimension_index[**memory_index])
                    })
                    .collect();
                lookup_to_memory_indices[lookup_index].extend(memory_indices);
            }
        }

        let materialized_subtables = Self::materialize_subtables::<M>(&subtables)
            .into_iter()
            // .map(box_dense_poly)
            .collect_vec();

        Self {
            num_memories,
            materialized_subtables,
            subtable_to_memory_indices,
            memory_to_subtable_index,
            memory_to_dimension_index,
            lookup_to_memory_indices,
            subtables_by_idx: Some(
                subtables_by_idx
                    .into_iter()
                    .map(|s| s.unwrap())
                    .collect_vec(),
            ),
            lookup_id_to_index,
            lookups,
        }
    }

    fn materialize_subtables<const M: usize>(
        subtables: &[Box<dyn LassoSubtable<F>>],
    ) -> Vec<Vec<F>> {
        let mut s = Vec::with_capacity(subtables.len());
        for subtable in subtables.iter() {
            s.push(subtable.materialize(M));
        }
        s
    }
}

pub struct LassoPolynomials<F: JoltField> {
    /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
    /// indices at which the memories will be evaluated. Each `DensePolynomial` has size
    /// `m` (# lookups).
    pub dims: Vec<DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// read access counts to the memory. Each `DensePolynomial` has size `m` (# lookups).
    pub read_cts: Vec<DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// final access counts to the memory. Each `DensePolynomial` has size M, AKA subtable size.
    pub final_cts: Vec<DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// the evaluation of memory accessed at each step of the CPU. Each `DensePolynomial` has
    /// size `m` (# lookups).
    pub e_polys: Vec<DensePolynomial<F>>,

    /// Polynomial encodings for flag polynomials for each instruction.
    /// If using a single instruction this will be empty.
    /// NUM_INSTRUCTIONS sized, each polynomial of length 'm' (# lookups).
    ///
    /// Stored independently for use in sumcheck, combined into single DensePolynomial for commitment.
    pub lookup_flag_polys: Vec<DensePolynomial<F>>,

    /// Instruction flag polynomials as bitvectors, kept in this struct for more efficient
    /// construction of the memory flag polynomials in `read_write_grand_product`.
    pub lookup_flag_bitvectors: Vec<Vec<u64>>,

    /// The lookup output for each instruction of the execution trace.
    pub lookup_outputs: DensePolynomial<F>,
}

pub struct Rep3LassoWitnessSolver<F: JoltField, N: Rep3Network> {
    lut_provider: Rep3LookupTable<N>,
    io_context0: IoContext<N>,
    io_context1: IoContext<N>,
    phantom_data: PhantomData<F>,
}

pub struct Rep3LassoPreprocessing<F: JoltField> {
    subtable_to_memory_indices: Vec<Vec<usize>>,
    lookup_to_memory_indices: Vec<Vec<usize>>,
    memory_to_subtable_index: Vec<usize>,
    memory_to_dimension_index: Vec<usize>,
    materialized_subtables: Vec<PublicPrivateLut<F>>,
    subtables_by_idx: Option<Vec<Box<dyn LassoSubtable<F>>>>,
    lookup_id_to_index: HashMap<LookupId, usize>,
    num_memories: usize,
    lookups: BTreeMap<LookupId, Box<dyn LookupType<F>>>,
}

pub struct Rep3LassoPolynomials<F: JoltField> {
    /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
    /// indices at which the memories will be evaluated. Each `DensePolynomial` has size
    /// `m` (# lookups).
    pub dims: Vec<Vec<Rep3BigUintShare<F>>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// read access counts to the memory. Each `DensePolynomial` has size `m` (# lookups).
    pub read_cts: Vec<Vec<Rep3PrimeFieldShare<F>>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// final access counts to the memory. Each `DensePolynomial` has size M, AKA subtable size.
    pub final_cts: Vec<Vec<Rep3PrimeFieldShare<F>>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// the evaluation of memory accessed at each step of the CPU. Each `DensePolynomial` has
    /// size `m` (# lookups).
    pub e_polys: Vec<Vec<Rep3PrimeFieldShare<F>>>,

    /// Polynomial encodings for flag polynomials for each instruction.
    /// If using a single instruction this will be empty.
    /// NUM_INSTRUCTIONS sized, each polynomial of length 'm' (# lookups).
    ///
    /// Stored independently for use in sumcheck, combined into single DensePolynomial for commitment.
    pub lookup_flag_polys:  Vec<DensePolynomial<F>>,

    /// Instruction flag polynomials as bitvectors, kept in this struct for more efficient
    /// construction of the memory flag polynomials in `read_write_grand_product`.
    pub lookup_flag_bitvectors: Vec<Vec<u64>>,

    /// The lookup output for each instruction of the execution trace.
    pub lookup_outputs: Vec<Rep3BigUintShare<F>>,
}

impl<F: JoltField, N: Rep3Network> Rep3LassoWitnessSolver<F, N> {
    pub fn polynomialize(
        &mut self,
        preprocessing: &Rep3LassoPreprocessing<F>,
        inputs: &[Rep3BigUintShare<F>],
        lookups: &[LookupId],
        M: usize,
        C: usize,
    ) -> Rep3LassoPolynomials<F> {
        let num_reads = inputs.len().next_power_of_two();

        println!("lookups: {:?}", lookups.len());
        let subtable_lookup_indices: Vec<Vec<Rep3BigUintShare<F>>> =
            Self::subtable_lookup_indices(preprocessing, inputs, &lookups, M, C);

        let lookup_inputs = (0..inputs.len())
            .into_iter()
            .zip(lookups.into_iter())
            .map(|(i, lookup_id)| (i, preprocessing.lookup_id_to_index[lookup_id]))
            .collect_vec();

        let polys: Vec<_> = (0..preprocessing.num_memories)
            .into_iter()
            .map(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                let access_sequence = &subtable_lookup_indices[dim_index];

                let mut final_cts_i = vec![Rep3PrimeFieldShare::zero_share(); M];
                let mut read_cts_i = vec![Rep3PrimeFieldShare::zero_share(); num_reads];
                let mut subtable_lookups = vec![Rep3PrimeFieldShare::zero_share(); num_reads];

                for (j, lookup) in &lookup_inputs {
                    let memories_used = &preprocessing.lookup_to_memory_indices[*lookup];
                    if memories_used.contains(&memory_index) {
                        let memory_address = &access_sequence[*j];
                        // debug_assert!(memory_address < M);

                        let ohv = Rep3LookupTable::ohv_from_index_no_a2b_conversion(
                            memory_address.clone(),
                            M,
                            &mut self.io_context0,
                            &mut self.io_context1,
                        ).unwrap();

                        let mut counter = self.lut_provider.get_from_shared_lut_from_ohv(
                            &ohv,
                            &final_cts_i,
                            &mut self.io_context0,
                            &mut self.io_context1,
                        ).unwrap();
                        read_cts_i[*j] = counter;
                        counter = counter + arithmetic::promote_to_trivial_share(self.io_context0.id, F::one());

                        self.lut_provider.write_to_shared_lut_from_ohv(
                            &ohv,
                            counter,
                            &mut final_cts_i,
                            &mut self.io_context0,
                            &mut self.io_context1,
                        ).unwrap();

                        let subtable_lookup_share = Rep3LookupTable::get_from_lut_no_a2b_conversion(
                            memory_address.clone(),
                            &preprocessing.materialized_subtables[subtable_index],
                            &mut self.io_context0,
                            &mut self.io_context1,
                        ).unwrap();
                        subtable_lookups[*j] = subtable_lookup_share;
                    }
                }

             

                (
                    read_cts_i,
                    final_cts_i,
                    subtable_lookups,
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
                access_sequence.resize(access_sequence.len().next_power_of_two(), Rep3BigUintShare::zero_share());
                access_sequence
            })
            .collect();

        let mut lookup_flag_bitvectors: Vec<Vec<u64>> =
            vec![vec![0u64; num_reads]; preprocessing.lookups.len()];

        for (j, lookup_idx) in lookup_inputs.into_iter() {
            lookup_flag_bitvectors[lookup_idx][j] = 1;
        }

        let lookup_flag_polys: Vec<_> = lookup_flag_bitvectors
            .iter()
            .map(|flag_bitvector| DensePolynomial::from_u64(flag_bitvector))
            .collect();

        let mut lookup_outputs = Self::compute_lookup_outputs(&preprocessing, inputs, &lookups);
        lookup_outputs.resize(num_reads, Rep3BigUintShare::zero_share());

        let lookup_outputs = lookup_outputs;
        Rep3LassoPolynomials {
            dims,
            read_cts,
            final_cts,
            lookup_flag_polys,
            lookup_flag_bitvectors,
            e_polys,
            lookup_outputs,
        }
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoWitnessSolver::subtable_lookup_indices")]
    fn subtable_lookup_indices(
        preprocessing: &Rep3LassoPreprocessing<F>,
        inputs: &[Rep3BigUintShare<F>],
        lookups: &[LookupId],
        M: usize,
        C: usize,
    ) -> Vec<Vec<Rep3BigUintShare<F>>> {
        let num_rows: usize = inputs.len();
        let num_chunks = C;

        let indices: Vec<_> = (0..num_rows)
            .into_iter()
            .zip(lookups.into_iter())
            .map(|(i, lookup_id)| {
                let lookup = &preprocessing.lookups[lookup_id];
                let mut index_bits = inputs[i].to_le_bits();
                index_bits.truncate(lookup.chunk_bits(M).iter().sum());
                // if cfg!(feature = "sanity-check") {
                //     assert_eq!(
                //         usize_from_bits_le(&fe_to_bits_le(inputs[i])),
                //         usize_from_bits_le(&index_bits),
                //         "index {i} out of range",
                //     );
                // }
                let mut chunked_index = iter::repeat(Rep3BigUintShare::zero_share())
                    .take(num_chunks)
                    .collect_vec();
                let chunked_index_bits = lookup.subtable_indices_rep3(index_bits, M.ilog2() as usize);
                chunked_index
                    .iter_mut()
                    .zip(chunked_index_bits)
                    .map(|(chunked_index, index_bits)| {
                        *chunked_index = Rep3BigUintShare::from_le_bits(&index_bits);
                    })
                    .collect_vec();
                chunked_index
            })
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

    fn compute_lookup_outputs(
        preprocessing: &Rep3LassoPreprocessing<F>,
        inputs: &[Rep3BigUintShare<F>],
        lookups: &[LookupId],
    ) -> Vec<Rep3BigUintShare<F>> {
        (0..inputs.len())
            .into_iter()
            .zip(lookups.into_iter())
            .map(|(i, lookup_id)| preprocessing.lookups[lookup_id].output_rep3(&inputs[i]))
            .collect()
    }
}

pub fn polynomialize<F: JoltField>(
    preprocessing: &LassoPreprocessing<F>,
    inputs: &[F],
    lookups: &[LookupId],
    M: usize,
    C: usize,
) -> LassoPolynomials<F> {
    let num_reads = inputs.len().next_power_of_two();

    println!("lookups: {:?}", lookups.len());
    let subtable_lookup_indices = subtable_lookup_indices(preprocessing, inputs, &lookups, M, C);

    let lookup_inputs = (0..inputs.len())
        .into_iter()
        .zip(lookups.into_iter())
        .map(|(i, lookup_id)| (i, preprocessing.lookup_id_to_index[lookup_id]))
        .collect_vec();

    let polys: Vec<_> = (0..preprocessing.num_memories)
        .into_iter()
        .map(|memory_index| {
            let dim_index = preprocessing.memory_to_dimension_index[memory_index];
            let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
            let access_sequence = &subtable_lookup_indices[dim_index];

            let mut final_cts_i = vec![0usize; M];
            let mut read_cts_i = vec![0usize; num_reads];
            let mut subtable_lookups = vec![F::ZERO; num_reads];

            for (j, lookup) in &lookup_inputs {
                let memories_used = &preprocessing.lookup_to_memory_indices[*lookup];
                if memories_used.contains(&memory_index) {
                    let memory_address = access_sequence[*j];
                    // debug_assert!(memory_address < M);
                    println!("j: {:?}, lookup: {:?}, memory_address: {:?}", *j, lookup, memory_address);

                    let counter = final_cts_i[memory_address];
                    read_cts_i[*j] = counter;
                    final_cts_i[memory_address] = counter + 1;
                    // println!("final_cts_i: {:?}", final_cts_i);
                    subtable_lookups[*j] =
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

    let mut lookup_flag_bitvectors: Vec<Vec<u64>> =
        vec![vec![0u64; num_reads]; preprocessing.lookups.len()];

    for (j, lookup_idx) in lookup_inputs.into_iter() {
        lookup_flag_bitvectors[lookup_idx][j] = 1;
    }

    let lookup_flag_polys: Vec<_> = lookup_flag_bitvectors
        .iter()
        .map(|flag_bitvector| DensePolynomial::from_u64(flag_bitvector))
        .collect();

    let mut lookup_outputs = compute_lookup_outputs(preprocessing, inputs, &lookups);
    lookup_outputs.resize(num_reads, F::ZERO);

    let lookup_outputs = DensePolynomial::new(lookup_outputs);
    LassoPolynomials {
        dims,
        read_cts,
        final_cts,
        lookup_flag_polys,
        lookup_flag_bitvectors,
        e_polys,
        lookup_outputs,
    }
}

fn compute_lookup_outputs<F: JoltField>(
    preprocessing: &LassoPreprocessing<F>,
    inputs: &[F],
    lookups: &[LookupId],
) -> Vec<F> {
    (0..inputs.len())
        .into_iter()
        .zip(lookups.into_iter())
        .map(|(i, lookup_id)| preprocessing.lookups[lookup_id].output(&inputs[i]))
        .collect()
}

#[tracing::instrument(skip_all, name = "LassoNode::subtable_lookup_indices")]
fn subtable_lookup_indices<F: JoltField>(
    preprocessing: &LassoPreprocessing<F>,
    inputs: &[F],
    lookups: &[LookupId],
    M: usize,
    C: usize,
) -> Vec<Vec<usize>> {
    let num_rows: usize = inputs.len();
    let num_chunks = C;

    let indices: Vec<_> = (0..num_rows)
        .into_iter()
        .zip(lookups.into_iter())
        .map(|(i, lookup_id)| {
            let lookup = &preprocessing.lookups[lookup_id];
            let mut index_bits = fe_to_bits_le(inputs[i]);
            index_bits.truncate(lookup.chunk_bits(M).iter().sum());
            // if cfg!(feature = "sanity-check") {
            //     assert_eq!(
            //         usize_from_bits_le(&fe_to_bits_le(inputs[i])),
            //         usize_from_bits_le(&index_bits),
            //         "index {i} out of range",
            //     );
            // }
            let mut chunked_index = iter::repeat(0).take(num_chunks).collect_vec();
            let chunked_index_bits = lookup.subtable_indices(index_bits, M.ilog2() as usize);
            chunked_index
                .iter_mut()
                .zip(chunked_index_bits)
                .map(|(chunked_index, index_bits)| {
                    *chunked_index = usize_from_bits_le(&index_bits);
                })
                .collect_vec();
            chunked_index
        })
        .collect();

    let lookup_indices = (0..num_chunks)
        .map(|i| indices.iter().map(|indices| indices[i].clone()).collect_vec())
        .collect_vec();
    lookup_indices
    // todo!()
}

pub fn fe_to_bits_le<F: PrimeField>(fe: F) -> Vec<bool> {
    fe.into_bigint().to_bits_le()
}

pub fn usize_from_bits_le(bits: &[bool]) -> usize {
    bits.iter()
        .rev()
        .fold(0, |int, bit| (int << 1) + (*bit as usize))
}
