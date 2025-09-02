use std::{iter, u32};

use crate::{
    field::JoltField,
    jolt::{instruction::sub, trace},
    poly::{
        combine_poly_shares_rep3, generate_poly_shares_rep3, generate_poly_shares_rep3_vec,
        Rep3MultilinearPolynomial,
    },
    utils::{
        self,
        future::{FutureExt, FutureRep3},
        future_ring::{FutureRep3Ring, Rep3RingFutureExt},
    },
};
use ark_ff::{One, Zero};
use ark_std::cfg_into_iter;
use eyre::Context;
use itertools::{izip, multizip, Either, Itertools};
use jolt_core::{jolt::vm::instruction_lookups::InstructionLookupPolynomials, utils::math::Math};
use jolt_core::{
    jolt::vm::instruction_lookups::InstructionLookupStuff,
    poly::multilinear_polynomial::MultilinearPolynomial,
};
use mpc_core::protocols::{
    rep3::{
        self, arithmetic,
        network::{
            IoContext, IoContextPool, Rep3Network, Rep3NetworkCoordinator, Rep3NetworkWorker,
            WorkerIoContext,
        },
        PartyID, Rep3BigUintShare, Rep3PrimeFieldShare,
    },
    rep3_ring::{
        self,
        gadgets::ohv::ohv,
        lut::{PublicPrivateLut, Rep3LookupTable},
        ring::ring_impl::RingElement,
        Rep3RingShare,
    },
};
use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tokio::io;

use crate::jolt::{
    instruction::{JoltInstructionSet, Rep3JoltInstructionSet},
    vm::{
        instruction_lookups::InstructionLookupsPreprocessing, witness::Rep3Polynomials,
        JoltTraceStep,
    },
};

const _M: usize = 1 << 16;

pub type Rep3InstructionLookupPolynomials<F> = InstructionLookupStuff<Rep3MultilinearPolynomial<F>>;

impl<F: JoltField, const C: usize> Rep3Polynomials<F, InstructionLookupsPreprocessing<C, F>>
    for Rep3InstructionLookupPolynomials<F>
{
    type PublicPolynomials = InstructionLookupPolynomials<F>;

    // type Commitments = InstructionLookupCommitments<PCS, ProofTranscript>;

    #[tracing::instrument(skip_all, name = "InstructionLookupsProof::generate_witness_rep3")]
    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        ops: &mut [JoltTraceStep<Instructions>],
        M: usize,
        io_ctx: &mut WorkerIoContext<Network>,
    ) -> eyre::Result<Rep3InstructionLookupPolynomials<F>>
    where
        Instructions: Rep3JoltInstructionSet,
        Network: Rep3NetworkWorker,
    {
        // let mut network = BiNetwork::new(io_ctx)?;
        let num_reads = ops.len().next_power_of_two();

        // let operands = ops
        //     .iter()
        //     .find_map(|op| match op.instruction_lookup.as_ref() {
        //         Some(op) => Some(op.operands_rep3()),
        //         None => None,
        //     })
        //     .unwrap();

        // let operands_open = vec![operands.0.as_binary_share(), operands.1.as_binary_share()]
        //     .into_iter()
        //     .map(|x| {
        //         rep3_ring::binary::open(&x, io_ctx.main())
        //             .unwrap()
        //             .convert()
        //     })
        //     .collect::<Vec<_>>();

        // let operands_b2a = rep3_ring::conversion::b2a_many(
        //     vec![operands.0.as_binary_share(), operands.1.as_binary_share()],
        //     io_ctx.main(),
        // )?;
        // // let operands_b2a = vec![operands.0.as_binary_share(), operands.1.as_binary_share()]
        // //     .into_iter()
        // //     .map(|x| rep3_ring::conversion::b2a(&x, io_ctx.main()).unwrap())
        // //     .collect::<Vec<_>>();

        // let operands_b2a_open = rep3_ring::arithmetic::open_vec(&operands_b2a, io_ctx.main())?
        //     .into_iter()
        //     .map(|x| x.convert())
        //     .collect::<Vec<_>>();
        // let operands_check = vec![operands.0.as_public() as u32, operands.1.as_public() as u32];
        // assert_eq!(operands_open, operands_check);
        // assert_eq!(operands_b2a_open, operands_check);
        // println!("Operands checks passed");

        Instructions::promote_public_operands_to_shared(
            ops.par_iter_mut().map(|op| &mut op.instruction_lookup),
            io_ctx.party_id(),
        );

        Instructions::populate_operands_casts(
            ops.par_iter_mut().map(|op| &mut op.instruction_lookup),
            io_ctx.main(),
        )?;

        let lookup_outputs = compute_lookup_outputs_rep3(&ops, num_reads, io_ctx)?;
        let subtable_lookup_indices = subtable_lookup_indices_rep3::<C, F, Network, Instructions>(
            ops,
            &mut io_ctx.main(),
            M,
        )?;

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

        let (instructions_used, access_sequences): (Vec<Vec<usize>>, Vec<Vec<Rep3RingShare<u32>>>) =
            (0..preprocessing.num_memories)
                .into_par_iter()
                .map(|memory_index| {
                    let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                    let access_sequence = &subtable_lookup_indices[dim_index];
                    ops.iter()
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
                        .unzip()
                })
                .unzip();

        let mut ohvs = Rep3LookupTable::ohv_bits_from_index_no_a2b_conversion_many(
            access_sequences.par_iter().cloned().flatten(),
            M,
            io_ctx.main(),
        )?;

        let mut ohvs_by_memory = Vec::with_capacity(preprocessing.num_memories);
        instructions_used
            .iter()
            .for_each(|js| ohvs_by_memory.push(ohvs.drain(..js.len()).collect_vec()));
        assert!(ohvs.is_empty());

        let polys = tracing::info_span!("compute_polys").in_scope(|| {
            io_ctx.par_iter(
                (0..preprocessing.num_memories)
                    .into_par_iter()
                    .zip_eq(ohvs_by_memory),
                None,
                |(memory_index, ohvs), io_ctx| {
                    let subtable_index = preprocessing.memory_to_subtable_index[memory_index];

                    let mut final_cts_i = vec![Rep3RingShare::zero_share(); M];
                    let mut read_cts_i = vec![Rep3PrimeFieldShare::zero_share(); num_reads];
                    let mut subtable_lookups = vec![Rep3PrimeFieldShare::zero_share(); num_reads];
                    if ohvs.is_empty() {
                        return Ok((
                            Rep3MultilinearPolynomial::from(vec![
                                Rep3PrimeFieldShare::zero_share();
                                M
                            ]),
                            Rep3MultilinearPolynomial::from(read_cts_i),
                            Rep3MultilinearPolynomial::from(subtable_lookups),
                        ));
                    }

                    // let mut ohvs = rep3_ring::conversion::bit_inject_from_bits_to_field_many(
                    //     &ohvs.into_par_iter().flatten().collect::<Vec<_>>(),
                    //     io_ctx,
                    // )?;

                    // let (r, e_r) = rep3_ring::gadgets::ohv::rand_ohv::<u32, _>(M, io_ctx)?;
                    // let rand_ohv = rep3_ring::conversion::bit_inject_from_bits_many(&e_r, io_ctx)?;

                    let mut used_ops = Vec::with_capacity(ohvs.len());
                    let mut read_cts_i_local_a = Vec::with_capacity(ohvs.len());
                    let mut lookup_subtables_local_a = Vec::with_capacity(ohvs.len());
                    let mut ohvs = ohvs.into_iter();

                    let _guard =
                        tracing::trace_span!("ops_per_memory", memory_index, subtable_index)
                            .entered();
                    for (j, op) in ops.iter().enumerate() {
                        if let Some(op) = &op.instruction_lookup {
                            let memories_used = &preprocessing.instruction_to_memory_indices
                                [<Instructions as Rep3JoltInstructionSet>::enum_index(op)];
                            if memories_used.contains(&memory_index) {
                                // let memory_address = access_sequence[j];
                                let ohv =
                                    rep3_ring::conversion::bit_inject_from_bits_many::<u32, _>(
                                        &ohvs.next().unwrap(),
                                        io_ctx,
                                    )?;
                                // let ohv = ohvs.drain(..M).collect::<Vec<_>>();
                                // let c = rep3_ring::binary::open(&(r ^ memory_address), io_ctx)?;
                                // let c: usize = (c.0 as usize) & ((1 << M) - 1); // Mask potential overflows from non-well-defined input

                                let _guard = tracing::trace_span!("luts_rw_local").entered();
                                let mut counter =
                                    io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                                for (i, l) in final_cts_i.iter_mut().enumerate() {
                                    // let e = rand_ohv[i ^ c];
                                    let e = ohv[i];
                                    counter += e * *l;
                                    *l += e; // ohv_bit (either 0 or 1)
                                }
                                // let mut subtable_lookup =
                                //     io_ctx.rngs.rand.masking_element::<RingElement<u32>>();
                                let mut subtable_lookup = RingElement::zero();
                                for (i, l) in materialized_subtable_luts[subtable_index]
                                    .iter()
                                    .enumerate()
                                {
                                    // let e = rand_ohv[i ^ c];
                                    let e = ohv[i];
                                    subtable_lookup += e * *l;
                                }

                                read_cts_i_local_a.push(counter);
                                lookup_subtables_local_a.push(subtable_lookup);
                                used_ops.push(j);
                            }
                        }
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
                    let used_subtable_lookups = izip!(lookup_subtables_local_a, lookup_subtables_b)
                        .map(|(a, b)| Rep3RingShare { a, b })
                        .collect::<Vec<_>>();

                    let used_read_cts_i =
                        rep3_ring::casts::ring_to_field_many_selector(&used_read_cts_i, io_ctx)
                            .unwrap();
                    let final_cts_i =
                        rep3_ring::casts::ring_to_field_many_selector(&final_cts_i, io_ctx)
                            .unwrap();
                    let used_subtable_lookups = rep3_ring::casts::ring_to_field_many_selector(
                        &used_subtable_lookups,
                        io_ctx,
                    )
                    .unwrap();

                    izip!(
                        used_ops,
                        // read_cts_i_local_a,
                        used_read_cts_i,
                        // lookup_subtables_local_a,
                        used_subtable_lookups,
                    )
                    .for_each(|(j, cts, e)| {
                        read_cts_i[j] = cts;
                        subtable_lookups[j] = e;
                    });

                    Ok((
                        Rep3MultilinearPolynomial::from(read_cts_i),
                        Rep3MultilinearPolynomial::from(final_cts_i),
                        Rep3MultilinearPolynomial::from(subtable_lookups),
                    ))
                },
            )
        })?;

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

        let span = tracing::info_span!("compute_dim");
        let _guard = span.enter();
        let dim: Vec<_> = rep3_ring::casts::binary_ring_to_field_many(
            &subtable_lookup_indices.into_iter().flatten().collect_vec(),
            io_ctx.main(),
        )?
        .chunks_exact(ops.len())
        .map(|c| Rep3MultilinearPolynomial::from_shared_coeffs(c.to_vec()))
        .collect();
        drop(_guard);

        let mut instruction_flag_bitvectors: Vec<Vec<u64>> =
            vec![vec![0u64; num_reads]; Instructions::COUNT];

        for (j, op) in ops.iter().enumerate() {
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

    fn combine_polynomials(
        _: &InstructionLookupsPreprocessing<C, F>,
        polynomials_shares: Vec<Self>,
    ) -> eyre::Result<InstructionLookupPolynomials<F>> {
        let [share1, share2, share3] = polynomials_shares.try_into().unwrap();

        let dim = multizip((share1.dim, share2.dim, share3.dim))
            .map(|(dim1, dim2, dim3)| {
                Rep3MultilinearPolynomial::try_combine_shares(vec![dim1, dim2, dim3])
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        let read_cts = multizip((share1.read_cts, share2.read_cts, share3.read_cts))
            .map(|(read1, read2, read3)| {
                Rep3MultilinearPolynomial::try_combine_shares(vec![read1, read2, read3])
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        let final_cts = multizip((share1.final_cts, share2.final_cts, share3.final_cts))
            .map(|(final1, final2, final3)| {
                Rep3MultilinearPolynomial::try_combine_shares(vec![final1, final2, final3])
            })
            .collect::<eyre::Result<Vec<_>>>()?;

        let e_polys = multizip((share1.E_polys, share2.E_polys, share3.E_polys))
            .map(|(e1, e2, e3)| Rep3MultilinearPolynomial::try_combine_shares(vec![e1, e2, e3]))
            .collect::<eyre::Result<Vec<_>>>()?;

        let lookup_outputs = MultilinearPolynomial::from(
            combine_poly_shares_rep3(vec![
                share1.lookup_outputs.try_into()?,
                share2.lookup_outputs.try_into()?,
                share3.lookup_outputs.try_into()?,
            ])
            .evals()
            .into_iter()
            .map(|x| x.to_u64().unwrap() as u32)
            .collect_vec(),
        );

        let instruction_flags = share1
            .instruction_flags
            .into_iter()
            .map(|p| p.try_into())
            .collect::<eyre::Result<Vec<_>>>()?;

        Ok(InstructionLookupPolynomials {
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

    #[tracing::instrument(
        skip_all,
        name = "Rep3InstructionLookupPolynomials::stream_secret_shares",
        level = "trace"
    )]
    fn stream_secret_shares<R: Rng, Network: Rep3NetworkCoordinator>(
        _: &InstructionLookupsPreprocessing<C, F>,
        polynomials: InstructionLookupPolynomials<F>,
        rng: &mut R,
        network: &mut Network,
    ) -> eyre::Result<()> {
        let InstructionLookupStuff {
            dim,
            read_cts,
            final_cts,
            E_polys,
            lookup_outputs,
            instruction_flags,
            ..
        } = polynomials;

        let dim_shares = generate_poly_shares_rep3_vec(&dim, rng);
        network.send_requests_blocking(dim_shares)?;

        let read_cts_shares = generate_poly_shares_rep3_vec(&read_cts, rng);
        network.send_requests_blocking(read_cts_shares)?;

        let final_cts_shares = generate_poly_shares_rep3_vec(&final_cts, rng);
        network.send_requests_blocking(final_cts_shares)?;

        let e_polys_shares = generate_poly_shares_rep3_vec(&E_polys, rng);
        network.send_requests_blocking(e_polys_shares)?;

        let lookup_outputs_shares = generate_poly_shares_rep3(&lookup_outputs, rng);
        network.send_requests_blocking(lookup_outputs_shares)?;

        let instruction_flags_shares = [PartyID::ID0, PartyID::ID1, PartyID::ID2].map(|id| {
            Rep3MultilinearPolynomial::public_with_trivial_share_vec(instruction_flags.clone(), id)
        });
        network.send_requests_blocking(instruction_flags_shares.to_vec())?;

        Ok(())
    }

    #[tracing::instrument(
        skip_all,
        name = "Rep3InstructionLookupPolynomials::receive_witness_share",
        level = "trace"
    )]
    fn receive_witness_share<Network: Rep3NetworkWorker>(
        _: &InstructionLookupsPreprocessing<C, F>,
        io_ctx: &mut IoContextPool<Network>,
    ) -> eyre::Result<Self> {
        let dim = io_ctx.network().receive_request()?;
        let read_cts = io_ctx.network().receive_request()?;
        let final_cts = io_ctx.network().receive_request()?;
        let E_polys = io_ctx.network().receive_request()?;
        let lookup_outputs = io_ctx.network().receive_request()?;
        let instruction_flags = io_ctx.network().receive_request()?;
        Ok(Self {
            dim,
            read_cts,
            final_cts,
            E_polys,
            lookup_outputs,
            instruction_flags,
            a_init_final: None,
            v_init_final: None,
        })
    }
}

#[tracing::instrument(skip_all, name = "Rep3LassoWitnessSolver::subtable_lookup_indices")]
fn subtable_lookup_indices_rep3<const C: usize, F, Network, Instructions>(
    ops: &[JoltTraceStep<Instructions>],
    io_ctx0: &mut IoContext<Network>,
    M: usize,
) -> eyre::Result<Vec<Vec<Rep3RingShare<u32>>>>
where
    F: JoltField,
    Network: Rep3Network,
    Instructions: Rep3JoltInstructionSet,
{
    let num_chunks = C;
    let log_M = M.log_2();

    let id = io_ctx0.id;
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

    let intermediate = futures.fufill_batched(io_ctx0, |res, _| Some(res))?;

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
    io_ctx: &mut WorkerIoContext<Network>,
) -> eyre::Result<Rep3MultilinearPolynomial<F>> {
    let mut outputs_futures =
        vec![FutureRep3Ring::Ready(Rep3PrimeFieldShare::zero_share()); ops.len()];
    let ops_by_instruction: (
        Vec<Vec<&Instructions>>,
        Vec<Vec<&mut FutureRep3Ring<u32, Rep3PrimeFieldShare<F>>>>,
    ) = izip!(ops, outputs_futures.iter_mut())
        .filter_map(|(op, out)| op.instruction_lookup.as_ref().map(|op| (op, out)))
        .group_by(|(lookup, _)| Rep3JoltInstructionSet::enum_index(*lookup))
        .into_iter()
        .map(|(_, g)| g.unzip())
        .unzip();

    let _ = io_ctx.par_chunks(
        ops_by_instruction, // TODO: sort to distribute work evenly
        None,
        |ops, io_ctx: &mut IoContext<Network>| {
            ops.into_iter()
                .map(|(steps, out)| steps[0].output_batched(&steps, io_ctx, out))
                .collect::<eyre::Result<Vec<_>>>()
        },
    )?;

    let mut outputs = outputs_futures.fufill_batched(io_ctx.main(), |res, _| res)?;

    outputs.resize(num_reads, Rep3PrimeFieldShare::zero_share());
    Ok(Rep3MultilinearPolynomial::from(outputs))
}
