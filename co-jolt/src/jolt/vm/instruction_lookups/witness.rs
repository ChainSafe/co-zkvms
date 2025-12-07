use std::{cmp::min, marker::PhantomData, u32};

use crate::{
    field::JoltField,
    jolt::vm::{read_write_memory::witness::Rep3ProgramIO, witness::WorkerInitializable},
    poly::{Rep3DensePolynomial, Rep3MultilinearPolynomial},
    utils::future_ring::{FutureRep3Ring, Rep3RingFutureExt},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use itertools::{izip, Itertools};
#[cfg(feature = "debug")]
use jolt_core::jolt::vm::instruction_lookups::InstructionLookupPolynomials;
use jolt_core::{
    jolt::vm::instruction_lookups::InstructionLookupStuff, lasso::memory_checking::Initializable,
    poly::multilinear_polynomial::MultilinearPolynomial,
};
use jolt_core::{jolt::vm::JoltStuff, utils::math::Math};
use mpc_core::protocols::{
    rep3::{
        network::{IoContext, IoContextPool, Rep3NetworkWorker},
        Rep3PrimeFieldShare,
    },
    rep3_ring::{self, ring::ring_impl::RingElement, Rep3RingShare},
};

use mpc_net::topology::MpcRingNetWorkerExt;
use rayon::prelude::*;

use crate::jolt::{
    instruction::Rep3JoltInstructionSet,
    vm::{
        instruction_lookups::InstructionLookupsPreprocessing, witness::Rep3Polynomials,
        JoltTraceStep,
    },
};

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct InstructionLookupsPreprocessingExt<const C: usize, F: JoltField> {
    pub subtable_to_memory_indices: Vec<Vec<usize>>, // Vec<Range<usize>>?
    pub instruction_to_memory_indices: Vec<Vec<usize>>,
    pub memory_to_subtable_index: Vec<usize>,
    pub memory_to_dimension_index: Vec<usize>,
    pub materialized_subtables: Vec<Vec<u32>>,
    pub num_memories: usize,
    pub read_memories_worker: Vec<usize>, // TODO: Range
    pub init_final_subtables_worker: Vec<(usize, bool, Vec<usize>)>,
    pub final_memories_worker: Vec<usize>, // TODO: Range
    pub _field: PhantomData<F>,
}

impl<F: JoltField, const C: usize> InstructionLookupsPreprocessingExt<C, F> {
    pub fn for_worker(
        verifier_preprocessing: InstructionLookupsPreprocessing<C, F>,
        num_workers: usize,
        worker_idx: usize,
    ) -> Self {
        let InstructionLookupsPreprocessing {
            subtable_to_memory_indices,
            instruction_to_memory_indices,
            memory_to_subtable_index,
            memory_to_dimension_index,
            materialized_subtables,
            num_memories,
            _field,
        } = verifier_preprocessing;

        let read_memories_worker =
            read_write_memories_for_worker(num_memories, num_workers, worker_idx);

        let init_final_subtables_worker =
            init_final_subtables_for_worker(&subtable_to_memory_indices, num_workers, worker_idx);

        let final_memories_worker = init_final_subtables_worker
            .iter()
            .flat_map(|(_, _, memories)| memories)
            .copied()
            .collect_vec();

        Self {
            subtable_to_memory_indices,
            instruction_to_memory_indices,
            memory_to_subtable_index,
            memory_to_dimension_index,
            materialized_subtables,
            num_memories,
            read_memories_worker,
            init_final_subtables_worker,
            final_memories_worker,
            _field: PhantomData,
        }
    }
}

pub type Rep3InstructionLookupPolynomials<F> = InstructionLookupStuff<Rep3MultilinearPolynomial<F>>;

impl<const C: usize, F: JoltField, T: CanonicalSerialize + CanonicalDeserialize + Default>
    WorkerInitializable<T, InstructionLookupsPreprocessingExt<C, F>> for InstructionLookupStuff<T>
{
    fn worker_initialize(preprocessing: &InstructionLookupsPreprocessingExt<C, F>) -> Self {
        Self {
            dim: std::iter::repeat_with(|| T::default()).take(C).collect(),
            read_cts: std::iter::repeat_with(|| T::default())
                .take(preprocessing.read_memories_worker.len())
                .collect(),
            final_cts: std::iter::repeat_with(|| T::default())
                .take(preprocessing.final_memories_worker.len())
                .collect(),
            E_polys: std::iter::repeat_with(|| T::default())
                .take(preprocessing.read_memories_worker.len())
                .collect(),
            instruction_flags: std::iter::repeat_with(|| T::default())
                .take(preprocessing.instruction_to_memory_indices.len())
                .collect(),
            lookup_outputs: T::default(),
            a_init_final: None,
            v_init_final: None,
        }
    }

    type VerifierPreprocessing = InstructionLookupsPreprocessing<C, F>;

    fn initialize_for_worker(
        preprocessing: &Self::VerifierPreprocessing,
        num_workers: usize,
        worker_idx: usize,
    ) -> Self
    where
        Self: Initializable<T, Self::VerifierPreprocessing>,
    {
        let read_memories_worker =
            read_write_memories_for_worker(preprocessing.num_memories, num_workers, worker_idx)
                .len();

        let final_memories_worker = init_final_subtables_for_worker(
            &preprocessing.subtable_to_memory_indices,
            num_workers,
            worker_idx,
        )
        .iter()
        .flat_map(|(_, _, memories)| memories)
        .count();

        let mut init = Self::initialize(preprocessing);
        init.read_cts.truncate(read_memories_worker);
        init.E_polys.truncate(read_memories_worker);
        init.final_cts.truncate(final_memories_worker);
        init
    }
}

impl<F: JoltField, const C: usize> Rep3Polynomials<F, InstructionLookupsPreprocessingExt<C, F>>
    for Rep3InstructionLookupPolynomials<F>
{
    #[cfg(feature = "debug")]
    type PublicPolynomials = InstructionLookupPolynomials<F>;

    #[tracing::instrument(skip_all, name = "InstructionLookups::generate_witness_rep3")]
    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &InstructionLookupsPreprocessingExt<C, F>,
        trace: &mut [JoltTraceStep<Instructions>],
        _: &Rep3ProgramIO<F>,
        // M: usize,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Rep3InstructionLookupPolynomials<F>>
    where
        Instructions: Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker + MpcRingNetWorkerExt,
    {
        let m = trace.len().next_power_of_two();
        let M = preprocessing.materialized_subtables[0].len();
        let worker_idx = io_ctx.worker_idx();
        let log_num_workers = io_ctx.log_num_workers();
        let num_workers = 1usize << log_num_workers;

        let m_worker = m / num_workers;
        let trace_worker_range =
            (worker_idx * m_worker)..min((worker_idx + 1) * m_worker, trace.len());

        Instructions::promote_public_operands_to_shared(
            trace.par_iter_mut().map(|op| &mut op.instruction_lookup),
            io_ctx.party_id(),
        );

        Instructions::populate_operands_casts(
            trace.par_iter_mut().map(|op| &mut op.instruction_lookup),
            io_ctx.main(),
        )?;

        let lookup_outputs =
            compute_lookup_outputs_rep3(&trace[trace_worker_range.clone()], m, m_worker, io_ctx)?;
        let subtable_lookup_indices =
            subtable_lookup_indices_rep3::<C, F, Network, Instructions>(trace, io_ctx, M)?;

        let id = io_ctx.main().id;
        let materialized_subtable_luts: Vec<Vec<_>> = preprocessing
            .materialized_subtables
            .par_iter()
            .map(|subtable| {
                subtable
                    .iter()
                    .map(|v| rep3_ring::arithmetic::promote_to_trivial_share(id, (*v).into()))
                    .collect::<Vec<_>>()
            })
            .collect();

        let polys = tracing::info_span!("compute_polys").in_scope(|| {
            io_ctx.par_iter_cyclic(0..preprocessing.num_memories, |memory_index, io_ctx| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                let access_sequence = &subtable_lookup_indices[dim_index];
                let subtable = &materialized_subtable_luts[subtable_index];

                let is_read_mem = preprocessing.read_memories_worker.contains(&memory_index);
                let is_final_mem = preprocessing.final_memories_worker.contains(&memory_index);

                let (used_ops, memory_addresses): (Vec<_>, Vec<_>) = trace
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| is_read_mem || is_final_mem || trace_worker_range.contains(j))
                    .filter_map(|(j, op)| {
                        if let Some(op) = &op.instruction_lookup {
                            let memories_used = &preprocessing.instruction_to_memory_indices
                                [<Instructions as Rep3JoltInstructionSet>::enum_index(op)];
                            if memories_used.contains(&memory_index) {
                                Some((j, access_sequence[j].clone()))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .unzip();

                let mut read_cts_i =
                    is_read_mem.then_some(vec![Rep3PrimeFieldShare::zero_share(); m]);
                let mut subtable_lookups = vec![Rep3PrimeFieldShare::zero_share(); m];

                let num_reads = used_ops.len();
                if num_reads == 0 {
                    return eyre::Ok((
                        read_cts_i,
                        is_final_mem.then_some(vec![Rep3PrimeFieldShare::zero_share(); M]),
                        subtable_lookups,
                    ));
                }

                // TODO: don't reuse rand OHV, preprocess for ops bound
                let _guard = tracing::trace_span!("rand_ohv").entered();
                let (r, rand_ohv) = {
                    let (r, e) = rep3_ring::gadgets::ohv::rand_ohv::<u32, _>(16, io_ctx)?;
                    let rand_ohv =
                        rep3_ring::conversion::bit_inject_from_bits_many::<u32, _>(&e, io_ctx)?;
                    (r, rand_ohv)
                };
                drop(_guard);

                let _guard = tracing::trace_span!("open_c").entered();
                let mem_addresses_c = rep3_ring::binary::open_vec(
                    memory_addresses.iter().map(|addr| &r ^ addr).collect(),
                    io_ctx,
                )?;
                drop(_guard);

                let mut used_read_indices = Vec::with_capacity(num_reads);
                let mut read_cts_local = Vec::with_capacity(num_reads);
                let mut used_E_indices = Vec::with_capacity(num_reads);
                let mut E_local = Vec::with_capacity(num_reads);

                let _guard =
                    tracing::trace_span!("ops_per_memory", memory_index, subtable_index).entered();

                let final_cts_i = if is_read_mem {
                    let mut final_cts = vec![Rep3RingShare::zero_share(); M];
                    for (i, j) in used_ops.iter().enumerate() {
                        let mut r_ct = io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                        lookup_read_increment_final(
                            &mut final_cts,
                            &rand_ohv,
                            mem_addresses_c[i],
                            &mut r_ct,
                        );

                        let mut e = io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                        generic_lookup(&subtable, &rand_ohv, mem_addresses_c[i], &mut e);

                        read_cts_local.push(r_ct);
                        used_read_indices.push(*j);
                        E_local.push(e);
                        used_E_indices.push(*j);
                    }

                    is_final_mem.then_some(final_cts)
                } else if is_final_mem {
                    let mut final_cts_i = vec![Rep3RingShare::zero_share(); M];

                    for (i, j) in used_ops.iter().enumerate() {
                        increment_final(&mut final_cts_i, &rand_ohv, mem_addresses_c[i]);
                        let mut e = io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                        generic_lookup(&subtable, &rand_ohv, mem_addresses_c[i], &mut e);

                        E_local.push(e);
                        used_E_indices.push(*j);
                    }
                    Some(final_cts_i)
                } else {
                    for (i, j) in used_ops.iter().enumerate() {
                        let mut e = io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                        generic_lookup(&subtable, &rand_ohv, mem_addresses_c[i], &mut e);
                        E_local.push(e);
                        used_E_indices.push(*j);
                    }
                    None
                };

                drop(_guard);

                let final_cts_i = final_cts_i.map(|shares| {
                    rep3_ring::casts::ring_to_field_many_selector(&shares, io_ctx).unwrap()
                });

                if let Some(read_cts_i) = read_cts_i.as_mut() {
                    let read_cts_i_b = io_ctx.network.reshare_many(&read_cts_local)?;
                    let used_read_cts_i = rep3_ring::casts::ring_to_field_many_selector(
                        &izip!(read_cts_local, read_cts_i_b)
                            .map(|(a, b)| Rep3RingShare { a, b })
                            .collect::<Vec<_>>(),
                        io_ctx,
                    )?;

                    izip!(used_read_cts_i, used_read_indices).for_each(|(r, j)| {
                        read_cts_i[j] = r;
                    });
                }

                let lookup_subtables_b = io_ctx.network.reshare_many(&E_local)?;
                let used_subtable_lookups = rep3_ring::casts::ring_to_field_many_selector(
                    &izip!(E_local, lookup_subtables_b)
                        .map(|(a, b)| Rep3RingShare { a, b })
                        .collect::<Vec<_>>(),
                    io_ctx,
                )?;

                izip!(used_subtable_lookups, used_E_indices).for_each(|(e, j)| {
                    subtable_lookups[j] = e;
                });

                if !is_read_mem {
                    subtable_lookups = subtable_lookups
                        .drain((worker_idx * m_worker)..(worker_idx + 1) * m_worker)
                        .collect();
                }

                Ok((read_cts_i, final_cts_i, subtable_lookups))
            })
        })?;

        let (read_cts, final_cts, e_polys) = polys.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut read_acc, mut final_acc, mut e_acc), (read_evals, final_evals, e)| {
                if let Some(read_evals) = read_evals {
                    read_acc.push(Rep3MultilinearPolynomial::from(read_evals));
                }
                if let Some(final_evals) = final_evals {
                    final_acc.push(Rep3MultilinearPolynomial::from(final_evals));
                }
                e_acc.push(Rep3MultilinearPolynomial::new_shard_shared(
                    e,
                    m,
                    log_num_workers,
                    worker_idx,
                ));
                (read_acc, final_acc, e_acc)
            },
        );

        let span = tracing::info_span!("compute_dim");
        let _guard = span.enter();
        let mut dim = vec![vec![Rep3PrimeFieldShare::<F>::zero_share(); m]; C];
        let (dim_muts, used_subtable_lookup_indices): (Vec<_>, Vec<_>) = subtable_lookup_indices
            .into_par_iter()
            .zip(dim.par_iter_mut())
            .flat_map(|(indices, dim)| {
                indices
                    .into_par_iter()
                    .zip(trace.par_iter())
                    .zip(dim.par_iter_mut())
                    .filter_map(|((index, op), dim)| {
                        if op.instruction_lookup.is_some() {
                            Some((dim, index))
                        } else {
                            None
                        }
                    })
            })
            .unzip();

        io_ctx
            .par_chunks(used_subtable_lookup_indices, None, |chunk, io_ctx| {
                rep3_ring::casts::binary_ring_to_field_many::<_, F, _>(&chunk, io_ctx)
            })?
            .into_par_iter()
            .zip_eq(dim_muts)
            .for_each(|(dim, dim_mut): (_, &mut _)| {
                *dim_mut = dim;
            });

        let dim = dim
            .into_iter()
            .map(|dim| {
                Rep3MultilinearPolynomial::new_shard_shared(dim, m, log_num_workers, worker_idx)
            })
            .collect();

        drop(_guard);
        let mut instruction_flag_bitvectors = vec![vec![0u8; m]; Instructions::COUNT];

        for (j, op) in trace.iter().enumerate() {
            if let Some(op) = &op.instruction_lookup {
                instruction_flag_bitvectors
                    [<Instructions as Rep3JoltInstructionSet>::enum_index(op)][j] = 1;
            }
        }

        // let party_id = io_ctx.party_id();
        let instruction_flags: Vec<_> = instruction_flag_bitvectors
            .into_par_iter()
            .map(|flag_bitvector| {
                Rep3MultilinearPolynomial::new_shard_public_u8(
                    flag_bitvector,
                    m,
                    log_num_workers,
                    worker_idx,
                )
            })
            .collect();

        Ok(Rep3InstructionLookupPolynomials {
            dim,
            read_cts,
            final_cts,
            instruction_flags,
            E_polys: e_polys,
            lookup_outputs,
            a_init_final: None,
            v_init_final: None,
        })
    }

    #[cfg(feature = "debug")]
    fn combine_polynomials(
        _: &InstructionLookupsPreprocessingExt<C, F>,
        polynomials_shares: Vec<Self>,
    ) -> InstructionLookupPolynomials<F> {
        use itertools::multizip;

        use crate::poly::combine_poly_shares_rep3;

        let [share1, share2, share3] = polynomials_shares.try_into().unwrap();

        let dim = multizip((share1.dim, share2.dim, share3.dim))
            .map(|(dim1, dim2, dim3)| {
                Rep3MultilinearPolynomial::combine_shares(vec![dim1, dim2, dim3])
            })
            .collect_vec();

        let read_cts = multizip((share1.read_cts, share2.read_cts, share3.read_cts))
            .map(|(read1, read2, read3)| {
                Rep3MultilinearPolynomial::combine_shares(vec![read1, read2, read3])
            })
            .collect_vec();

        let final_cts = multizip((share1.final_cts, share2.final_cts, share3.final_cts))
            .map(|(final1, final2, final3)| {
                Rep3MultilinearPolynomial::combine_shares(vec![final1, final2, final3])
            })
            .collect_vec();

        let e_polys = multizip((share1.E_polys, share2.E_polys, share3.E_polys))
            .map(|(e1, e2, e3)| Rep3MultilinearPolynomial::combine_shares(vec![e1, e2, e3]))
            .collect_vec();

        let lookup_outputs = MultilinearPolynomial::from(
            combine_poly_shares_rep3(vec![
                share1.lookup_outputs.try_into().unwrap(),
                share2.lookup_outputs.try_into().unwrap(),
                share3.lookup_outputs.try_into().unwrap(),
            ])
            .evals()
            .into_iter()
            .map(|x| x.to_u64().unwrap() as u32)
            .collect_vec(),
        );

        let instruction_flags = share1
            .instruction_flags
            .into_iter()
            .map(|p| p.try_into().unwrap())
            .collect::<Vec<_>>();

        InstructionLookupPolynomials {
            dim,
            read_cts,
            final_cts,
            instruction_flags,
            E_polys: e_polys,
            lookup_outputs,
            a_init_final: None,
            v_init_final: None,
        }
    }
}

fn increment_final(
    final_cts: &mut Vec<Rep3RingShare<u32>>,
    rand_ohv: &[Rep3RingShare<u32>],
    c: usize,
) {
    for (i, l) in final_cts.iter_mut().enumerate() {
        let e = rand_ohv[i ^ c];
        *l += e; // ohv_bit (either 0 or 1)
    }
}

fn lookup_read_increment_final(
    final_cts: &mut Vec<Rep3RingShare<u32>>,
    rand_ohv: &[Rep3RingShare<u32>],
    c: usize,
    counter: &mut RingElement<u32>,
) {
    for (i, l) in final_cts.iter_mut().enumerate() {
        let e = rand_ohv[i ^ c];
        *counter += e * *l;
        *l += e; // ohv_bit (either 0 or 1)
    }
}

fn generic_lookup(
    v: &[Rep3RingShare<u32>],
    rand_ohv: &[Rep3RingShare<u32>],
    c: usize,
    value: &mut RingElement<u32>,
) {
    for (i, l) in v.iter().enumerate() {
        let e = rand_ohv[i ^ c];
        *value += e * *l;
    }
}

/// Compute per-worker delta in *chunks* for given batch_size (in chunks) and num_workers (power of 2).
///
/// Old element-based version was:
///   N_worker = floor(N / W)
///   t = floor(log2(N_worker))
///   M_elems = 2^(t + L - 1)
///   P_layer = N_worker * 2^L
///   delta_elems = P_layer mod M_elems, with sign rule:
///       if delta_elems == M_elems/2 -> +delta_elems
///       else                         -> -delta_elems
///
/// Dividing by 2^L (chunk_size), this simplifies in *chunks* to:
///   M_chunks = 2^(t-1)
///   delta_chunks_base = N_worker mod M_chunks
///   if delta_chunks_base == 0      -> 0
///   else if delta_chunks_base == M_chunks/2 -> +delta_chunks_base
///   else                           -> -delta_chunks_base
pub fn calculate_delta_per_worker(batch_size: usize, num_workers: usize) -> usize {
    assert!(num_workers > 0 && num_workers.is_power_of_two());

    // N_worker = floor(N / W)
    let n_worker = batch_size / num_workers;
    if n_worker == 0 {
        return 0;
    }

    // t = floor(log2(N_worker))
    let t = (usize::BITS - 1 - n_worker.leading_zeros()) as u32;

    // For t == 0 or 1, the original element-wise delta is always 0.
    if t <= 1 {
        return 0;
    }

    let m_chunks: usize = 1usize << (t - 1);
    let delta_base: usize = n_worker % m_chunks;

    if delta_base == 0 {
        return 0;
    }

    delta_base
}

/// Given:
/// - num_memories = M (each memory = 2 chunks),
/// - num_workers = W (power of 2),
/// split the big polynomial [0 .. 2*M chunks) among workers using the delta trick,
/// and return which memories this `worker_idx` touches.
///
/// The big poly is in *chunks*:
///   N = 2 * M
///   N_worker = floor(N / W)
///   delta_chunks = calculate_delta_per_worker(N, W)
/// Non-last workers get `base_len_chunks = N_worker + delta_chunks` chunks;
/// last worker gets the remainder.
/// A memory i occupies chunks [2*i, 2*i + 2).
pub fn read_write_memories_for_worker(
    num_memories: usize,
    num_workers: usize,
    worker_idx: usize,
) -> Vec<usize> {
    assert!(num_memories > 0);
    assert!(num_workers > 0 && num_workers.is_power_of_two());
    assert!(worker_idx < num_workers);

    // Total chunks and per-worker baseline
    let n_chunks = 2 * num_memories; // N = M * 2
    let n_worker = n_chunks / num_workers; // floor(N/W)
    assert!(n_worker > 0, "not enough chunks per worker");

    // Shared delta in *chunks*
    let delta_chunks = calculate_delta_per_worker(n_chunks, num_workers);

    // Length (in chunks) of a non-last worker's portion
    let base_len_chunks = n_worker - delta_chunks;
    assert!(
        base_len_chunks > 0,
        "non-last worker chunk_len must be positive"
    );

    let total_chunks = n_chunks;

    // Compute this worker's chunk range [start_chunk, end_chunk)
    let (start_chunk, end_chunk) = if worker_idx + 1 < num_workers {
        let start = base_len_chunks * worker_idx;
        let end = start + base_len_chunks;
        (start, end)
    } else {
        // last worker gets the remainder
        let start = base_len_chunks * (num_workers - 1);
        let end = total_chunks;
        (start, end)
    };

    // Each memory i occupies chunks [2*i, 2*i + 2)
    let mut memories = Vec::new();
    for mem_idx in 0..num_memories {
        let mem_start = 2 * mem_idx;
        let mem_end = mem_start + 2;
        // non-empty intersection with [start_chunk, end_chunk)
        if mem_start < end_chunk && mem_end > start_chunk {
            memories.push(mem_idx);
        }
    }

    memories
}

/// For a given worker, return a sequence of "segments" in the global memory layout:
/// - `Option<usize>` is `Some(subtable_idx)` if the worker owns the header block of that subtable,
///   or `None` if it only owns some memories from that subtable.
/// - `Vec<usize>` are the memory indices (from subtable_to_memory_indices) that fall into this
///   worker's polynomial slice.
///
/// Layout in *blocks/chunks*:
///   for each subtable i:
///       [header_block] + [mem_block_0] + [mem_block_1] + ...
///
/// Splitting:
///   B = total_blocks
///   N = B
///   N_worker = floor(N / num_workers)
///   delta_chunks = calculate_delta_per_worker(N, num_workers)
///   non-last workers:  len_chunks = N_worker + delta_chunks
///   last worker:        len_chunks = N - len_chunks * (num_workers - 1)
///
/// A header of subtable i is at block index `pref[i]`.
/// A memory j in subtable i is at block index `pref[i] + 1 + j`.
pub fn init_final_subtables_for_worker(
    subtable_to_memory_indices: &[Vec<usize>],
    num_workers: usize, // power of two
    worker_idx: usize,
) -> Vec<(usize, bool, Vec<usize>)> {
    assert!(
        num_workers > 0 && num_workers.is_power_of_two(),
        "num_workers must be power of two"
    );
    assert!(worker_idx < num_workers, "worker_idx out of bounds");

    // Prefix sums in *block* space: each subtable contributes 1 header + |st| memory blocks.
    let mut pref = Vec::with_capacity(subtable_to_memory_indices.len() + 1);
    pref.push(0usize);
    for st in subtable_to_memory_indices {
        pref.push(pref.last().copied().unwrap() + 1 + st.len());
    }
    let total_blocks = *pref.last().unwrap(); // B == N
    assert!(total_blocks > 0, "no blocks to allocate");

    let batch_size = total_blocks; // N
    let n_worker = batch_size / num_workers; // floor(N / W)
    assert!(n_worker > 0, "not enough blocks per worker");

    // delta in *chunks/blocks*
    let delta_chunks = calculate_delta_per_worker(batch_size, num_workers);

    // Length of a non-last worker's slice, in blocks.
    let base_len_chunks = n_worker - delta_chunks;
    assert!(
        base_len_chunks > 0,
        "non-last worker slice must be positive"
    );

    let total_chunks = batch_size; // one chunk per block

    // Chunk range [start_chunk, end_chunk) for this worker.
    let (start_chunk, end_chunk) = if worker_idx + 1 < num_workers {
        let start = base_len_chunks * worker_idx;
        let end = start + base_len_chunks;
        (start, end)
    } else {
        let start = base_len_chunks * (num_workers - 1);
        let end = total_chunks;
        (start, end)
    };

    // Map chunk interval [start_chunk, end_chunk) back to per-subtable segments.
    let mut out: Vec<(usize, bool, Vec<usize>)> = Vec::new();

    for (i, st) in subtable_to_memory_indices.iter().enumerate() {
        let st_beg_block = pref[i];
        let st_end_block = pref[i + 1];

        if st_beg_block >= end_chunk {
            break; // past this worker's range
        }
        if st_end_block <= start_chunk {
            continue; // entirely before this worker's range
        }

        // Header block index
        let header_block = st_beg_block;
        let header_in_range = header_block >= start_chunk && header_block < end_chunk;

        // Memory blocks
        let mems_beg_block = st_beg_block + 1;
        let mut mems_for_worker = Vec::new();

        for (j, &mem_id) in st.iter().enumerate() {
            let blk = mems_beg_block + j;
            if blk >= end_chunk {
                break; // remaining mems from this subtable are beyond this worker
            }
            if blk >= start_chunk {
                mems_for_worker.push(mem_id);
            }
        }

        // Include subtable if either header or at least one memory is in range.
        if header_in_range || !mems_for_worker.is_empty() {
            out.push((i, header_in_range, mems_for_worker));
        }
    }

    out
}

#[tracing::instrument(skip_all, name = "Rep3LassoWitnessSolver::subtable_lookup_indices")]
fn subtable_lookup_indices_rep3<const C: usize, F, Network, Instructions>(
    ops: &[JoltTraceStep<Instructions>],
    io_ctx0: &mut IoContextPool<Network>,
    M: usize,
) -> eyre::Result<Vec<Vec<Rep3RingShare<u32>>>>
where
    F: JoltField,
    Network: Rep3NetworkWorker,
    Instructions: Rep3JoltInstructionSet,
{
    let num_chunks = C;
    let log_M = M.log_2();

    let id = io_ctx0.party_id();
    let futures: Vec<_> = ops
        .par_iter()
        .map(|op| {
            if let Some(lookup) = &op.instruction_lookup {
                lookup.to_indices_intermediate::<F>(id)
            } else {
                FutureRep3Ring::Ready(None)
            }
        })
        .collect();

    let intermediate = futures.fulfill_batched(io_ctx0, |res, _| Some(res))?;

    let indices: Vec<_> = ops
        .into_par_iter()
        .zip(intermediate)
        .map(|(lookup, intermediate)| {
            if let Some(lookup) = &lookup.instruction_lookup {
                lookup.to_indices_rep3(intermediate, C, log_M)
            } else {
                vec![Rep3RingShare::zero_share(); C]
            }
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
    Ok(lookup_indices)
}

#[tracing::instrument(skip_all, name = "Rep3LassoWitnessSolver::compute_lookup_outputs")]
fn compute_lookup_outputs_rep3<
    F: JoltField,
    Network: Rep3NetworkWorker,
    Instructions: Rep3JoltInstructionSet,
>(
    ops: &[JoltTraceStep<Instructions>],
    m: usize,
    num_reads: usize,
    io_ctx: &mut IoContextPool<Network>,
) -> eyre::Result<Rep3MultilinearPolynomial<F>> {
    let mut outputs_futures =
        vec![FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share()); ops.len()];
    let _guard = tracing::info_span!("group_by_instruction").entered();
    let ops_by_instruction: (
        Vec<Vec<&Instructions>>,
        Vec<Vec<&mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>>,
    ) = izip!(ops, outputs_futures.iter_mut())
        .filter_map(|(op, out)| op.instruction_lookup.as_ref().map(|op| (op, out)))
        .group_by(|(lookup, _)| Rep3JoltInstructionSet::enum_index(*lookup))
        .into_iter()
        .map(|(_, g)| g.unzip())
        .unzip();
    drop(_guard);

    let _guard = tracing::info_span!("trace_outputs_batched").entered();
    let _ = io_ctx.par_chunks(
        ops_by_instruction, // TODO: sort to distribute work evenly
        None,
        |ops, io_ctx: &mut IoContext<Network>| {
            ops.into_iter()
                .map(|(steps, out)| steps[0].output_batched(&steps, io_ctx, out))
                .collect::<eyre::Result<Vec<_>>>()
        },
    )?;
    drop(_guard);

    let mut outputs = outputs_futures.fulfill_batched(io_ctx, |res, _| res)?;

    outputs.resize(num_reads, Rep3PrimeFieldShare::zero_share());
    Ok(Rep3MultilinearPolynomial::new_shard_shared(
        outputs,
        m,
        io_ctx.log_num_workers(),
        io_ctx.worker_idx(),
    ))
}
