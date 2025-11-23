use std::{cmp::min, u32};

use crate::{
    field::JoltField,
    jolt::vm::read_write_memory::witness::Rep3ProgramIO,
    poly::Rep3MultilinearPolynomial,
    utils::future_ring::{FutureRep3Ring, Rep3RingFutureExt},
};
use ark_ff::Zero;
use itertools::{izip, Itertools};
#[cfg(feature = "debug")]
use jolt_core::jolt::vm::instruction_lookups::InstructionLookupPolynomials;
use jolt_core::utils::math::Math;
use jolt_core::{
    jolt::vm::instruction_lookups::InstructionLookupStuff,
    poly::multilinear_polynomial::MultilinearPolynomial,
};
use mpc_core::protocols::{
    rep3::{
        network::{IoContext, IoContextPool, Rep3NetworkWorker},
        PartyID, Rep3PrimeFieldShare,
    },
    rep3_ring::{self, ring::ring_impl::RingElement, Rep3RingShare},
};

use mpc_net::topology::MpcRingNetWorkerExt;
use rayon::prelude::*;
use tokio::io;

use crate::jolt::{
    instruction::Rep3JoltInstructionSet,
    vm::{
        instruction_lookups::InstructionLookupsPreprocessing, witness::Rep3Polynomials,
        JoltTraceStep,
    },
};

pub type Rep3InstructionLookupPolynomials<F> = InstructionLookupStuff<Rep3MultilinearPolynomial<F>>;

impl<F: JoltField, const C: usize> Rep3Polynomials<F, InstructionLookupsPreprocessing<C, F>>
    for Rep3InstructionLookupPolynomials<F>
{
    #[cfg(feature = "debug")]
    type PublicPolynomials = InstructionLookupPolynomials<F>;

    #[tracing::instrument(skip_all, name = "InstructionLookups::generate_witness_rep3")]
    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        trace: &mut [JoltTraceStep<Instructions>],
        _: &Rep3ProgramIO<F>,
        M: usize,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Rep3InstructionLookupPolynomials<F>>
    where
        Instructions: Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker + MpcRingNetWorkerExt,
    {
        let m = trace.len().next_power_of_two();
        println!("m={}", m);
        let worker_idx = io_ctx.worker_idx();
        let num_workers = 1usize << io_ctx.log_num_workers();

        let m_worker = m / num_workers;
        let m_worker_nv = m_worker.log_2();
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
            compute_lookup_outputs_rep3(&trace[trace_worker_range.clone()], m_worker, io_ctx)?;
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

        let read_memories =
            read_write_memories_for_worker(preprocessing.num_memories, num_workers, worker_idx);

        let final_memories = init_final_subtables_for_worker(
            &preprocessing.subtable_to_memory_indices,
            num_workers,
            worker_idx,
        )
        .map(|subtables| {
            subtables
                .into_iter()
                .flat_map(|(_, memories)| memories)
                .collect_vec()
        });

        if io_ctx.party_idx() == 0 {
            // if io_ctx.worker_idx() == 0 {
            //     println!(
            //         "preprocessing.subtable_to_memory_indices {:?}",
            //         preprocessing.subtable_to_memory_indices
            //     );

            //     for w in [2, 4, 8, 16] {
            //         // println!(
            //         //     "w={} | read_memories: {:?}",
            //         //     w,
            //         //     (0..w)
            //         //         .map(|i| read_write_memories_for_worker(
            //         //             preprocessing.num_memories,
            //         //             w,
            //         //             i
            //         //         ))
            //         //         .collect::<Vec<_>>()
            //         // );
            //         // println!(
            //         //     "w={} | final_memories: {:?}",
            //         //     w,
            //         //     (0..w)
            //         //         .map(|i| init_final_memories_for_worker(
            //         //             &preprocessing.subtable_to_memory_indices,
            //         //             preprocessing.num_memories,
            //         //             w,
            //         //             i
            //         //         ))
            //         //         .collect::<Vec<_>>()
            //         // );

            //         println!(
            //             "w={} | final_subtables: {:?}",
            //             w,
            //             (0..w)
            //                 .map(|i| init_final_subtables_for_worker(
            //                     &preprocessing.subtable_to_memory_indices,
            //                     // preprocessing.num_memories,
            //                     w,
            //                     i
            //                 ))
            //                 .collect::<Vec<_>>()
            //         );
            //     }
            // }
            println!(
                "w={} | read_memories: {:?}",
                io_ctx.worker_idx(),
                read_memories
            );

            println!(
                "w={} | final_memories: {:?}",
                io_ctx.worker_idx(),
                final_memories
            );
        }

        let polys = tracing::info_span!("compute_polys").in_scope(|| {
            io_ctx.par_iter_cyclic(0..preprocessing.num_memories, |memory_index, io_ctx| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                let access_sequence = &subtable_lookup_indices[dim_index];

                let is_read_cts = read_memories
                    .as_ref()
                    .is_some_and(|m| m.contains(&memory_index));

                let is_final_cts = final_memories
                    .as_ref()
                    .is_some_and(|m| m.contains(&memory_index));

                let trace_range = if is_read_cts || is_final_cts {
                    0..m
                } else {
                    trace_worker_range.clone()
                };

                let (used_ops, memory_addresses): (Vec<_>, Vec<_>) = trace[trace_range]
                    .iter()
                    .enumerate()
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

                let mut final_cts_i =
                    (is_read_cts || is_final_cts).then_some(vec![Rep3RingShare::zero_share(); M]);
                let mut read_cts_i =
                    is_read_cts.then_some(vec![Rep3PrimeFieldShare::zero_share(); m]);
                let e_poly_len = if is_read_cts { m } else { m_worker };
                let mut subtable_lookups = vec![Rep3PrimeFieldShare::zero_share(); e_poly_len];

                let num_reads = used_ops.len();
                if num_reads == 0 {
                    return eyre::Ok((
                        read_cts_i,
                        is_final_cts.then_some(vec![Rep3PrimeFieldShare::zero_share(); M]),
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
                let memory_addresses_c = rep3_ring::binary::open_vec(
                    memory_addresses.iter().map(|addr| &r ^ addr).collect(),
                    io_ctx,
                )?;
                drop(_guard);

                let mut read_cts_i_used_indices = Vec::with_capacity(num_reads);
                let mut read_cts_i_local_a = Vec::with_capacity(num_reads);
                let mut subtable_lookups_used_indices = Vec::with_capacity(e_poly_len);
                let mut subtable_lookups_local_a = Vec::with_capacity(e_poly_len);

                let _guard =
                    tracing::trace_span!("ops_per_memory", memory_index, subtable_index).entered();
                if let Some(final_cts_i) = final_cts_i.as_mut() {
                    for (i, j) in used_ops.iter().enumerate() {
                        let c = memory_addresses_c[i];

                        let mut counter = is_read_cts
                            .then(|| io_ctx.rngs.rand.masking_element::<RingElement<u32>>()); // todo conditional per is_read_cts

                        for (i, l) in final_cts_i.iter_mut().enumerate() {
                            let e = rand_ohv[i ^ c];
                            if let Some(counter) = counter.as_mut() {
                                *counter += e * *l;
                            }
                            *l += e; // ohv_bit (either 0 or 1)
                        }

                        if is_read_cts {
                            read_cts_i_local_a.push(counter.unwrap());
                            read_cts_i_used_indices.push(*j);

                            let mut subtable_lookup =
                                io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                            for (i, l) in materialized_subtable_luts[subtable_index]
                                .iter()
                                .enumerate()
                            {
                                let e = rand_ohv[i ^ c];
                                subtable_lookup += e * *l;
                            }

                            subtable_lookups_local_a.push(subtable_lookup);
                            subtable_lookups_used_indices.push(*j);
                        }
                    }
                }

                let e_trace_range = if is_final_cts {
                    trace_worker_range.clone()
                } else {
                    0..m_worker
                };

                for (i, mut j) in used_ops
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, j)| e_trace_range.contains(j))
                {
                    let c = memory_addresses_c[i];

                    let mut subtable_lookup =
                        io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                    for (i, l) in materialized_subtable_luts[subtable_index]
                        .iter()
                        .enumerate()
                    {
                        let e = rand_ohv[i ^ c];
                        subtable_lookup += e * *l;
                    }

                    if is_final_cts {
                        j -= m_worker * worker_idx;
                    };

                    subtable_lookups_local_a.push(subtable_lookup);
                    subtable_lookups_used_indices.push(j);
                }
                drop(_guard);

                let final_cts_i = is_final_cts.then(|| {
                    rep3_ring::casts::ring_to_field_many_selector(&final_cts_i.unwrap(), io_ctx)
                        .unwrap()
                });

                if let Some(read_cts_i) = read_cts_i.as_mut() {
                    let read_cts_i_b = io_ctx.network.reshare_many(&read_cts_i_local_a)?;
                    let used_read_cts_i = rep3_ring::casts::ring_to_field_many_selector(
                        &izip!(read_cts_i_local_a, read_cts_i_b)
                            .map(|(a, b)| Rep3RingShare { a, b })
                            .collect::<Vec<_>>(),
                        io_ctx,
                    )?;

                    izip!(used_read_cts_i, read_cts_i_used_indices).for_each(|(r, j)| {
                        read_cts_i[j] = r;
                    });
                }

                let lookup_subtables_b = io_ctx.network.reshare_many(&subtable_lookups_local_a)?;
                let used_subtable_lookups = rep3_ring::casts::ring_to_field_many_selector(
                    &izip!(subtable_lookups_local_a, lookup_subtables_b)
                        .map(|(a, b)| Rep3RingShare { a, b })
                        .collect::<Vec<_>>(),
                    io_ctx,
                )?;

                izip!(used_subtable_lookups, subtable_lookups_used_indices).for_each(|(e, j)| {
                    subtable_lookups[j] = e;
                });

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
                e_acc.push(Rep3MultilinearPolynomial::shard_from_shared_coeffs(
                    e,
                    m_worker_nv,
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
                Rep3MultilinearPolynomial::shard_from_shared_coeffs(dim, m_worker_nv, worker_idx)
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
                Rep3MultilinearPolynomial::shard_from_public_bytes(
                    flag_bitvector,
                    m_worker_nv,
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
        _: &InstructionLookupsPreprocessing<C, F>,
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

/// Largest feasible number of workers ≤ `req_workers` that actually receive memories,
/// with per-full-worker chunk size = C (power of two memories) and last worker non-empty.
/// Closed-form (no search):
///   W_eff = ceil( M / C* ),  where  C* = next_pow2( ceil(M / req_workers) )
pub fn read_write_effective_workers(num_memories: usize, log_num_workers: usize) -> usize {
    assert_ne!(num_memories, 0);
    let num_workers = 1 << log_num_workers;
    let base = num_memories.div_ceil(num_workers);
    let c = base.next_power_of_two();
    num_memories.div_ceil(c)
}

pub fn read_write_memories_for_worker(
    num_memories: usize,
    num_workers: usize,
    worker_idx: usize,
) -> Option<Vec<usize>> {
    assert!(
        num_memories != 0
            && num_workers != 0
            && num_workers.is_power_of_two()
            && worker_idx < num_workers,
        "Invalid inputs"
    );

    if num_workers == 1 {
        return Some((0..num_memories).collect());
    }

    // C = next power of two of ceil(M/W)
    let base = (num_memories + num_workers - 1) / num_workers; // ceil
    let mut c = 1usize;
    while c < base {
        c <<= 1;
    }

    // worker's contiguous bin [start, start+C); trailing workers may be empty (None)
    let start = worker_idx.saturating_mul(c);
    if start >= num_memories {
        return None; // this worker has no memories
    }
    let end = (start + c).min(num_memories);
    Some((start..end).collect())
}

/// Given the subtable layout and a requested power-of-two worker count,
/// return how many workers actually receive (≥1) memory under equal chunk size.
/// Chunks are in **blocks** (header+memories), size C blocks where
/// C = next_pow2( ceil(B / W_req) ), B = headers + memories.
/// If C == 1, only chunks landing on memory blocks allocate; result = total memories.
pub fn init_final_effective_workers(
    subtable_to_memory_indices: &[Vec<usize>],
    log_num_workers: usize,
) -> usize {
    let num_workers = 1 << log_num_workers;

    let init_blocks = subtable_to_memory_indices.len();
    let final_blocks: usize = subtable_to_memory_indices.iter().map(|v| v.len()).sum();
    let blocks = init_blocks + final_blocks;
    if blocks == 0 {
        return 0;
    }

    let base = blocks.div_ceil(num_workers);
    let c = base.next_power_of_two();
    let w_chunks = blocks.div_ceil(c);

    if c == 1 {
        // each chunk is one block; only memory blocks allocate
        final_blocks.min(w_chunks)
    } else {
        // with non-empty subtables (≥1 mem each), every chunk (full or last) contains a memory
        w_chunks
    }
}

// /// Assigns memories to a worker so that:
// /// - All workers share the same chunk size in blocks (C * M), with C a power of two,
// /// - First W-1 workers are full (C blocks), the last is partial (rem blocks, 1..=C),
// /// - If `num_workers` is infeasible, falls back to the largest W' < num_workers that is feasible,
// ///   and returns `None` if `worker_idx >= W'`.
// pub fn init_final_memories_for_worker(
//     subtable_to_memory_indices: &[Vec<usize>],
//     num_memories: usize,
//     num_workers: usize,
//     worker_idx: usize,
// ) -> Option<Vec<usize>> {
//     assert!(
//         num_memories != 0
//             && num_workers != 0
//             && num_workers.is_power_of_two()
//             && worker_idx < num_workers,
//         "Invalid inputs"
//     );

//     if num_workers == 1 {
//         return Some((0..num_memories).collect());
//     }

//     // Total blocks = headers + memories
//     let total_blocks: usize = subtable_to_memory_indices
//         .iter()
//         .map(|st| 1 + st.len())
//         .sum();
//     if total_blocks == 0 {
//         return None;
//     }

//     // Helper: pick C (power-of-two blocks per full worker) and rem for a given W
//     fn pick_c_rem(total_blocks: usize, workers: usize) -> Option<(usize, usize)> {
//         if workers < 2 {
//             return None;
//         }
//         let lower = total_blocks.div_ceil(workers);
//         let upper = total_blocks / (workers - 1);
//         let mut c = 1usize;
//         while c < lower {
//             c <<= 1;
//         }
//         if c > upper {
//             return None;
//         }
//         let rem = total_blocks - c * (workers - 1);
//         if rem == 0 {
//             return None;
//         } // last must be non-empty
//         Some((c, rem))
//     }

//     // Fallback search: largest W' <= num_workers that is feasible
//     let mut chosen_w = None;
//     let mut chosen_c_rem = None;
//     for w in (2..=num_workers).rev() {
//         if let Some(cr) = pick_c_rem(total_blocks, w) {
//             chosen_w = Some(w);
//             chosen_c_rem = Some(cr);
//             break;
//         }
//     }
//     let w_eff = chosen_w?;
//     let (c, rem) = chosen_c_rem?;

//     if worker_idx >= w_eff {
//         return None;
//     }

//     // Compute this worker's block range [start, end)
//     let (start_block, take_blocks) = if worker_idx + 1 < w_eff {
//         (c * worker_idx, c)
//     } else {
//         (c * (w_eff - 1), rem)
//     };
//     let end_block = start_block + take_blocks;

//     // Stream through blocks; collect memory ids that land in [start, end)
//     let mut cur = 0usize;
//     let mut out = Vec::new();
//     'outer: for st in subtable_to_memory_indices {
//         // header block
//         if cur >= end_block {
//             break;
//         }
//         cur += 1; // header occupies one block (never yields a memory)

//         // memory blocks
//         for &mem in st {
//             if cur >= end_block {
//                 break 'outer;
//             }
//             if cur >= start_block {
//                 out.push(mem);
//             }
//             cur += 1;
//         }
//     }
//     Some(out)
// }

/// For a given worker, return the exact memories per subtable that land in its chunk.
/// Chunks are in block-space (1 header + k memories), with equal chunk size C for all
/// but the last (which is shorter and padded). We choose
///   C = next_pow2( ceil(B / req_workers) ),  B = total blocks,
/// and fall back to W_eff = ceil(B / C) workers. If `worker_idx >= W_eff` → None.
///
/// Output pairs (st_idx, Vec<memory_ids>) only for subtables that contribute ≥1 memory.
pub fn init_final_subtables_for_worker(
    subtable_to_memory_indices: &[Vec<usize>],
    num_workers: usize, // power of two
    worker_idx: usize,
) -> Option<Vec<(usize, Vec<usize>)>> {
    assert!(
        num_workers != 0 && num_workers.is_power_of_two() && worker_idx < num_workers,
        "Invalid inputs"
    );

    if num_workers == 1 {
        return Some(
            subtable_to_memory_indices
                .iter()
                .cloned()
                .enumerate()
                .collect(),
        );
    }

    // Prefix sums in block space.
    let mut pref = Vec::with_capacity(subtable_to_memory_indices.len() + 1);
    pref.push(0usize);
    for st in subtable_to_memory_indices {
        pref.push(pref.last().copied().unwrap() + 1 + st.len()); // 1 header + |st| memories
    }
    let b = *pref.last().unwrap();
    if b == 0 {
        return None;
    }

    // Chunk params (no search): C and effective workers.
    let base = (b + num_workers - 1) / num_workers; // ceil(B/W_req)
    let c = base.next_power_of_two(); // chunk size in blocks
    let w_eff = (b + c - 1) / c; // ceil(B/C)
    if worker_idx >= w_eff {
        return None;
    }

    let start = if worker_idx + 1 < w_eff {
        c * worker_idx
    } else {
        c * (w_eff - 1)
    };
    let end = if worker_idx + 1 < w_eff { start + c } else { b };

    let mut out: Vec<(usize, Vec<usize>)> = Vec::new();

    for (i, st) in subtable_to_memory_indices.iter().enumerate() {
        let st_beg = pref[i];
        let st_end = pref[i + 1];
        if st_beg >= end {
            break;
        } // past the window
        if st_end <= start {
            continue;
        } // before the window

        // Overlap with this subtable's memory blocks (skip its header).
        let mems_beg_block = st_beg + 1;
        let ov_beg = start.max(mems_beg_block);
        let ov_end = end.min(st_end);
        if ov_beg < ov_end {
            let mem_start = ov_beg - mems_beg_block; // index into st[]
            let mem_end = mem_start + (ov_end - ov_beg); // exclusive
            if mem_start < mem_end {
                out.push((i, st[mem_start..mem_end].to_vec()));
            }
        }
    }
    Some(out)
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
    Ok(Rep3MultilinearPolynomial::from(outputs))
}
