pub mod grand_product;
pub mod lasso;
pub mod subtables;
mod utils;
pub mod memory_checking;

use ark_ff::{BigInteger, PrimeField};
use ark_std::{cfg_chunks, cfg_into_iter, cfg_iter};
use color_eyre::eyre::Context;
use itertools::Itertools;
use jolt_core::poly::{dense_mlpoly::DensePolynomial, field::JoltField};
use mpc_core::protocols::{
    rep3::{
        arithmetic,
        network::{IoContext, Rep3Network},
        Rep3BigUintShare, Rep3PrimeFieldShare,
    },
    rep3_ring::lut::{PublicPrivateLut, Rep3LookupTable},
};
use std::{iter, marker::PhantomData};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::lasso::LassoPreprocessing;
use crate::subtables::{LookupId, LookupType};

pub struct Rep3LassoWitnessSolver<F: JoltField, N: Rep3Network> {
    lut_provider: Rep3LookupTable<N>,
    pub io_context0: IoContext<N>,
    pub io_context1: IoContext<N>,
    phantom_data: PhantomData<F>,
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
    pub lookup_flag_polys: Vec<DensePolynomial<F>>,

    /// Instruction flag polynomials as bitvectors, kept in this struct for more efficient
    /// construction of the memory flag polynomials in `read_write_grand_product`.
    pub lookup_flag_bitvectors: Vec<Vec<u64>>,

    /// The lookup output for each instruction of the execution trace.
    pub lookup_outputs: Vec<Rep3BigUintShare<F>>,
}

impl<F: JoltField, N: Rep3Network> Rep3LassoWitnessSolver<F, N> {
    pub fn new(net: N) -> color_eyre::Result<Self> {
        let mut io_context0 = IoContext::init(net).context("failed to initialize io context")?;
        let io_context1 = io_context0.fork().context("failed to fork io context")?;

        Ok(Self {
            lut_provider: Rep3LookupTable::new(),
            io_context0,
            io_context1,
            phantom_data: PhantomData,
        })
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoWitnessSolver::polynomialize")]
    pub fn polynomialize(
        &mut self,
        preprocessing: &LassoPreprocessing<F>,
        inputs: &[Rep3BigUintShare<F>],
        lookups: &[LookupId],
        M: usize,
        C: usize,
    ) -> Rep3LassoPolynomials<F> {
        let num_reads = inputs.len().next_power_of_two();

        let subtable_lookup_indices: Vec<Vec<Rep3BigUintShare<F>>> =
            Self::subtable_lookup_indices(preprocessing, inputs, &lookups, M, C);

        let lookup_inputs = (0..inputs.len())
            .into_iter()
            .zip(lookups.into_iter())
            .map(|(i, lookup_id)| (i, preprocessing.lookup_id_to_index[lookup_id]))
            .collect_vec();

        let materialized_subtable_luts = preprocessing
            .materialized_subtables
            .clone()
            .into_iter()
            .map(|subtable| PublicPrivateLut::Public(subtable))
            .collect_vec();

        // let access_sequence = utils::fork_chunks_flat_map(
        //     access_sequence,
        //     &mut self.io_context0,
        //     &mut self.io_context1,
        //     1 << 11,
        //     |memory_address, mut io_context0, mut io_context1| {
        //         Rep3LookupTable::ohv_from_index_no_a2b_conversion(
        //             memory_address,
        //             M,
        //             &mut io_context0,
        //             &mut io_context1,
        //         )
        //         .unwrap()
        //     },
        // );

        let polys = tracing::info_span!("compute_polys").in_scope(|| {
            utils::fork_map(
                0..preprocessing.num_memories,
                &mut self.io_context0,
                &mut self.io_context1,
                |memory_index, mut io_context0, mut io_context1| {
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
                                &mut io_context0,
                                &mut io_context1,
                            )
                            .unwrap();

                            let mut counter = Rep3LookupTable::get_from_shared_lut_from_ohv(
                                &ohv,
                                &final_cts_i,
                                &mut io_context0,
                                &mut io_context1,
                            )
                            .unwrap();
                            read_cts_i[*j] = counter;
                            counter = counter
                                + arithmetic::promote_to_trivial_share(io_context0.id, F::one());

                            Rep3LookupTable::write_to_shared_lut_from_ohv(
                                &ohv,
                                counter,
                                &mut final_cts_i,
                                &mut io_context0,
                                &mut io_context1,
                            )
                            .unwrap();

                            let subtable_lookup_share =
                                Rep3LookupTable::get_from_lut_no_a2b_conversion(
                                    memory_address.clone(),
                                    &materialized_subtable_luts[subtable_index],
                                    &mut io_context0,
                                    &mut io_context1,
                                )
                                .unwrap();
                            subtable_lookups[*j] = subtable_lookup_share;
                        }
                    }

                    (read_cts_i, final_cts_i, subtable_lookups)
                },
            )
        });

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
        let dims: Vec<_> = cfg_into_iter!(subtable_lookup_indices)
            .take(C)
            .map(|mut access_sequence| {
                access_sequence.resize(
                    access_sequence.len().next_power_of_two(),
                    Rep3BigUintShare::zero_share(),
                );
                access_sequence
            })
            .collect();

        let mut lookup_flag_bitvectors: Vec<Vec<u64>> =
            vec![vec![0u64; num_reads]; preprocessing.lookups.len()];

        for (j, lookup_idx) in lookup_inputs.into_iter() {
            lookup_flag_bitvectors[lookup_idx][j] = 1;
        }

        let lookup_flag_polys: Vec<_> = cfg_iter!(lookup_flag_bitvectors)
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
        preprocessing: &LassoPreprocessing<F>,
        inputs: &[Rep3BigUintShare<F>],
        lookups: &[LookupId],
        M: usize,
        C: usize,
    ) -> Vec<Vec<Rep3BigUintShare<F>>> {
        let num_rows: usize = inputs.len();
        let num_chunks = C;

        let indices: Vec<_> = cfg_into_iter!(0..num_rows)
            .zip(cfg_into_iter!(lookups))
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
                let chunked_index_bits =
                    lookup.subtable_indices_rep3(index_bits, M.ilog2() as usize);
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
        preprocessing: &LassoPreprocessing<F>,
        inputs: &[Rep3BigUintShare<F>],
        lookups: &[LookupId],
    ) -> Vec<Rep3BigUintShare<F>> {
        cfg_into_iter!(0..inputs.len())
            .zip(cfg_into_iter!(lookups))
            .map(|(i, lookup_id)| preprocessing.lookups[lookup_id].output_rep3(&inputs[i]))
            .collect()
    }
}
