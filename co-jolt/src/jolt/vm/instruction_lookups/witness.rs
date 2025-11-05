use std::u32;

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
        let worker_idx = io_ctx.worker_idx();
        let num_workers = 1usize << io_ctx.log_num_workers_per_party();

        Instructions::promote_public_operands_to_shared(
            trace.par_iter_mut().map(|op| &mut op.instruction_lookup),
            io_ctx.party_id(),
        );

        Instructions::populate_operands_casts(
            trace.par_iter_mut().map(|op| &mut op.instruction_lookup),
            io_ctx.main(),
        )?;

        let lookup_outputs = compute_lookup_outputs_rep3(&trace, m, io_ctx)?;
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

        let (final_cts_, used_ops, rand_ohvs, memory_addresses_c): (
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = itertools::multiunzip(
            tracing::info_span!("compute_final_cts_prefix")
                .in_scope(|| {
                    io_ctx.par_iter_cyclic(0..preprocessing.num_memories, |memory_index, io_ctx| {
                        let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                        let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                        let access_sequence = &subtable_lookup_indices[dim_index];

                        let (used_ops, memory_addresses): (Vec<_>, Vec<_>) = trace
                            .iter()
                            .enumerate()
                            .filter_map(|(j, op)| {
                                if let Some(op) = &op.instruction_lookup {
                                    let memories_used = &preprocessing
                                        .instruction_to_memory_indices
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

                        let mut final_cts_i = vec![Rep3RingShare::zero_share(); M];

                        let num_reads = used_ops.len();
                        if num_reads == 0 {
                            return eyre::Ok((vec![], vec![], vec![], vec![]));
                        }

                        // TODO: don't reuse rand OHV, preprocess for ops bound
                        let _guard = tracing::trace_span!("rand_ohv").entered();
                        let (r, rand_ohv) = {
                            let (r, e) = rep3_ring::gadgets::ohv::rand_ohv::<u32, _>(16, io_ctx)?;
                            let rand_ohv = rep3_ring::conversion::bit_inject_from_bits_many::<
                                u32,
                                _,
                            >(&e, io_ctx)?;
                            (r, rand_ohv)
                        };
                        drop(_guard);

                        let _guard = tracing::trace_span!("open_c").entered();
                        let memory_addresses_c = rep3_ring::binary::open_vec(
                            memory_addresses.iter().map(|addr| &r ^ addr).collect(),
                            io_ctx,
                        )?;
                        drop(_guard);

                        let _guard =
                            tracing::trace_span!("ops_per_memory", memory_index, subtable_index)
                                .entered();
                        for i in 0..num_reads {
                            let c = memory_addresses_c[i];

                            for (i, l) in final_cts_i.iter_mut().enumerate() {
                                let e = rand_ohv[i ^ c];
                                *l += e; // ohv_bit (either 0 or 1)
                            }
                        }
                        drop(_guard);

                        Ok((final_cts_i, used_ops, rand_ohv, memory_addresses_c))
                    })
                })?
                .into_iter(),
        );

        let span = tracing::info_span!("sync_final_cts_prefix");
        let _guard = span.enter();
        if (worker_idx + 1) % num_workers != 0 {
            io_ctx.network().send_next_link(final_cts_.clone())?;
        }
        let final_cts: Vec<Vec<Rep3RingShare<u32>>> = if worker_idx != 0 {
            tracing::info!("using prefix");
            io_ctx.network().resv_prev_link()?
        } else {
            tracing::info!("using empty");
            vec![vec![Rep3RingShare::zero_share(); M]; preprocessing.num_memories]
        };
        drop(_guard);

        let (read_cts, final_cts, e_polys): (Vec<_>, Vec<_>, Vec<_>) =
            tracing::info_span!("compute_polys")
                .in_scope(|| {
                    io_ctx.par_iter_cyclic(
                        izip!(final_cts, used_ops, rand_ohvs, memory_addresses_c).enumerate(),
                        |(
                            memory_index,
                            (mut final_cts_i, used_ops, rand_ohv, memory_addresses_c),
                        ),
                         io_ctx| {
                            let subtable_index =
                                preprocessing.memory_to_subtable_index[memory_index];

                            if final_cts_i.is_empty() {
                                final_cts_i = vec![Rep3RingShare::zero_share(); M];
                            }

                            let mut read_cts_i = vec![Rep3PrimeFieldShare::zero_share(); m];
                            let mut subtable_lookups = vec![Rep3PrimeFieldShare::zero_share(); m];

                            let num_reads = used_ops.len();
                            if num_reads == 0 {
                                return eyre::Ok((
                                    Rep3MultilinearPolynomial::from(read_cts_i),
                                    final_cts_i,
                                    Rep3MultilinearPolynomial::from(subtable_lookups),
                                ));
                            }

                            let mut read_cts_i_local_a = vec![RingElement::zero(); num_reads];
                            let mut lookup_subtables_local_a = vec![RingElement::zero(); num_reads];

                            let _guard = tracing::trace_span!(
                                "ops_per_memory",
                                memory_index,
                                subtable_index
                            )
                            .entered();
                            // let memory_addresses_c = memory_addresses_c.into_iter();
                            for i in 0..num_reads {
                                let c = memory_addresses_c[i];

                                let mut counter =
                                    io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                                for (i, l) in final_cts_i.iter_mut().enumerate() {
                                    let e = rand_ohv[i ^ c];
                                    counter += e * *l;
                                    *l += e; // ohv_bit (either 0 or 1)
                                }
                                let mut subtable_lookup =
                                    io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                                for (i, l) in materialized_subtable_luts[subtable_index]
                                    .iter()
                                    .enumerate()
                                {
                                    let e = rand_ohv[i ^ c];
                                    subtable_lookup += e * *l;
                                }

                                read_cts_i_local_a[i] = counter;
                                lookup_subtables_local_a[i] = subtable_lookup;
                            }
                            drop(_guard);

                            let _guard = tracing::trace_span!("luts_reshare").entered();
                            let read_cts_i_b = io_ctx.network.reshare_many(&read_cts_i_local_a)?;
                            let lookup_subtables_b =
                                io_ctx.network.reshare_many(&lookup_subtables_local_a)?;
                            drop(_guard);

                            let used_read_cts_i = izip!(read_cts_i_local_a, read_cts_i_b)
                                .map(|(a, b)| Rep3RingShare { a, b })
                                .collect::<Vec<_>>();
                            let used_subtable_lookups =
                                izip!(lookup_subtables_local_a, lookup_subtables_b)
                                    .map(|(a, b)| Rep3RingShare { a, b })
                                    .collect::<Vec<_>>();

                            let used_read_cts_i = rep3_ring::casts::ring_to_field_many_selector(
                                &used_read_cts_i,
                                io_ctx,
                            )
                            .unwrap();
                            // let final_cts_i =
                            //     rep3_ring::casts::ring_to_field_many_selector(&final_cts_i, io_ctx)
                            //         .unwrap();
                            let used_subtable_lookups =
                                rep3_ring::casts::ring_to_field_many_selector(
                                    &used_subtable_lookups,
                                    io_ctx,
                                )
                                .unwrap();

                            izip!(used_ops, used_read_cts_i, used_subtable_lookups,).for_each(
                                |(j, cts, e)| {
                                    read_cts_i[j] = cts;
                                    subtable_lookups[j] = e;
                                },
                            );

                            Ok((read_cts_i.into(), final_cts_i, subtable_lookups.into()))
                        },
                    )
                })?
                .into_iter()
                .multiunzip();

        let span = tracing::info_span!("sync_final_cts");
        let _guard = span.enter();
        let final_cts: Vec<Vec<Rep3RingShare<u32>>> = if (worker_idx + 1) % num_workers != 0 {
            io_ctx.network().resv_next_link()?
        } else {
            final_cts
        };
        if worker_idx != 0 {
            io_ctx.network().send_prev_link(final_cts.clone())?;
        }
        drop(_guard);

        let M_chunk_size = M / num_workers;
        let final_cts = tracing::info_span!("final_cts_ring_to_field").in_scope(|| {
            io_ctx.par_iter_cyclic(final_cts, |final_cts_i, io_ctx| {
                let final_cts_i = final_cts_i
                    [M_chunk_size * worker_idx..M_chunk_size * (worker_idx + 1)]
                    .to_vec();
                if final_cts_i
                    .iter()
                    .all(|c| *c == Rep3RingShare::zero_share())
                {
                    return Ok(vec![Rep3PrimeFieldShare::<F>::zero_share(); M_chunk_size].into());
                }
                let final_cts_i =
                    rep3_ring::casts::ring_to_field_many_selector(&final_cts_i, io_ctx).unwrap();

                Ok(final_cts_i.into())
            })
        })?;

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
            .map(Rep3MultilinearPolynomial::from)
            .collect();

        drop(_guard);

        let mut instruction_flag_bitvectors = vec![vec![0u8; m]; Instructions::COUNT];

        for (j, op) in trace.iter().enumerate() {
            if let Some(op) = &op.instruction_lookup {
                instruction_flag_bitvectors
                    [<Instructions as Rep3JoltInstructionSet>::enum_index(op)][j] = 1;
            }
        }

        let party_id = io_ctx.party_id();
        let instruction_flags: Vec<_> = instruction_flag_bitvectors
            .into_par_iter()
            .map(|flag_bitvector| {
                Rep3MultilinearPolynomial::public_with_trivial_share(
                    MultilinearPolynomial::from(flag_bitvector),
                    party_id,
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
