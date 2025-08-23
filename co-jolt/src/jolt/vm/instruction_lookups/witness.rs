use crate::{
    field::JoltField,
    poly::{
        combine_poly_shares_rep3, generate_poly_shares_rep3, generate_poly_shares_rep3_vec,
        Rep3MultilinearPolynomial,
    },
    utils::{self, Forkable},
};
use ark_std::cfg_into_iter;
use itertools::{multizip, Itertools};
use jolt_core::{
    jolt::vm::instruction_lookups::InstructionLookupPolynomials,
    utils::math::Math,
};
use jolt_core::{
    jolt::vm::instruction_lookups::InstructionLookupStuff,
    poly::multilinear_polynomial::MultilinearPolynomial,
};
use mpc_core::protocols::{
    rep3::{
        self, arithmetic,
        network::{
            IoContext, IoContextPool, Rep3Network, Rep3NetworkCoordinator, Rep3NetworkWorker,
        },
        PartyID, Rep3BigUintShare, Rep3PrimeFieldShare,
    },
    rep3_ring::lut::{PublicPrivateLut, Rep3LookupTable},
};
use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::jolt::{
    instruction::{JoltInstructionSet, Rep3JoltInstructionSet},
    vm::{
        instruction_lookups::InstructionLookupsPreprocessing, witness::Rep3Polynomials,
        JoltTraceStep,
    },
};

pub type Rep3InstructionLookupPolynomials<F> = InstructionLookupStuff<Rep3MultilinearPolynomial<F>>;

impl<F: JoltField, const C: usize> Rep3Polynomials<F, InstructionLookupsPreprocessing<C, F>>
    for Rep3InstructionLookupPolynomials<F>
{
    type PublicPolynomials = InstructionLookupPolynomials<F>;

    // type Commitments = InstructionLookupCommitments<PCS, ProofTranscript>;

    #[tracing::instrument(skip_all, name = "InstructionLookupsProof::generate_witness_rep3")]
    fn generate_witness_rep3<Instructions, Network>(
        preprocessing: &InstructionLookupsPreprocessing<C, F>,
        ops: &mut [JoltTraceStep<F, Instructions>],
        M: usize,
        io_ctx: IoContext<Network>,
    ) -> eyre::Result<Rep3InstructionLookupPolynomials<F>>
    where
        Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
        Network: Rep3Network,
    {
        let mut network = BiNetwork::new(io_ctx)?;
        let num_reads = ops.len().next_power_of_two();

        let subtable_lookup_indices = subtable_lookup_indices_rep3::<C, F, Network, Instructions>(
            ops,
            &mut network.io_ctx,
            M,
        )?;

        let materialized_subtable_luts = preprocessing
            .materialized_subtables
            .clone()
            .into_iter()
            .map(|subtable| {
                PublicPrivateLut::Public(subtable.into_iter().map(F::from_u32).collect_vec())
            })
            .collect_vec();

        let polys = tracing::info_span!("compute_polys").in_scope(|| {
            utils::fork_map(
                0..preprocessing.num_memories,
                &mut network,
                |memory_index, solver| {
                    let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                    let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                    let access_sequence = &subtable_lookup_indices[dim_index];

                    let mut final_cts_i = vec![Rep3PrimeFieldShare::zero_share(); M];
                    let mut read_cts_i = vec![Rep3PrimeFieldShare::zero_share(); num_reads];
                    let mut subtable_lookups = vec![Rep3PrimeFieldShare::zero_share(); num_reads];

                    for (j, op) in ops.iter().enumerate() {
                        if let Some(op) = &op.instruction_lookup {
                            let memories_used = &preprocessing.instruction_to_memory_indices
                                [<Instructions as Rep3JoltInstructionSet<F>>::enum_index(op)];
                            if memories_used.contains(&memory_index) {
                                let memory_address = &access_sequence[j];

                                let ohv = Rep3LookupTable::ohv_from_index_no_a2b_conversion(
                                    memory_address.clone(),
                                    M,
                                    &mut solver.io_ctx,
                                    &mut solver.io_ctx1,
                                )
                                .unwrap();

                                let mut counter = Rep3LookupTable::get_from_shared_lut_from_ohv(
                                    &ohv,
                                    &final_cts_i,
                                    &mut solver.io_ctx,
                                    &mut solver.io_ctx1,
                                )
                                .unwrap();
                                read_cts_i[j] = counter;
                                counter = counter
                                    + arithmetic::promote_to_trivial_share(
                                        solver.io_ctx.id,
                                        F::one(),
                                    );

                                Rep3LookupTable::write_to_shared_lut_from_ohv(
                                    &ohv,
                                    counter,
                                    &mut final_cts_i,
                                    &mut solver.io_ctx,
                                    &mut solver.io_ctx1,
                                )
                                .unwrap();

                                let subtable_lookup_share =
                                    Rep3LookupTable::get_from_lut_no_a2b_conversion(
                                        memory_address.clone(),
                                        &materialized_subtable_luts[subtable_index],
                                        &mut solver.io_ctx,
                                        &mut solver.io_ctx1,
                                    )
                                    .unwrap();
                                subtable_lookups[j] = subtable_lookup_share;
                            }
                        }
                    }

                    (
                        Rep3MultilinearPolynomial::from(read_cts_i),
                        Rep3MultilinearPolynomial::from(final_cts_i),
                        Rep3MultilinearPolynomial::from(subtable_lookups),
                    )
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

        let dim = tracing::info_span!("b2a dims").in_scope(|| {
            utils::fork_map(
                subtable_lookup_indices,
                &mut network.io_ctx,
                |access_sequence, mut io_ctx0| {
                    let mut dim = vec![
                        Rep3PrimeFieldShare::zero_share();
                        access_sequence.len().next_power_of_two()
                    ];
                    for i in 0..access_sequence.len() {
                        // TODO: b2a_many ?
                        dim[i] = rep3::conversion::b2a_selector(&access_sequence[i], &mut io_ctx0)
                            .unwrap();
                    }
                    Rep3MultilinearPolynomial::from(dim)
                },
            )
        })?;

        let mut instruction_flag_bitvectors: Vec<Vec<u64>> =
            vec![vec![0u64; num_reads]; Instructions::COUNT];

        for (j, op) in ops.iter().enumerate() {
            if let Some(op) = &op.instruction_lookup {
                instruction_flag_bitvectors
                    [<Instructions as Rep3JoltInstructionSet<F>>::enum_index(op)][j] = 1;
            }
        }

        let party_id = network.io_ctx.id;
        let instruction_flags: Vec<_> = instruction_flag_bitvectors
            .into_par_iter()
            .map(|flag_bitvector| {
                Rep3MultilinearPolynomial::public_with_trivial_share(
                    MultilinearPolynomial::from(flag_bitvector),
                    party_id,
                )
            })
            .collect();

        let lookup_outputs = compute_lookup_outputs_rep3(&ops, num_reads, &mut network.io_ctx)?;

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
    lookups: &mut [JoltTraceStep<F, Instructions>],
    io_ctx0: &mut IoContext<Network>,
    M: usize,
) -> eyre::Result<Vec<Vec<Rep3BigUintShare<F>>>>
where
    F: JoltField,
    Network: Rep3Network,
    Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
{
    Instructions::promote_public_operands_to_binary(
        lookups.par_iter_mut().map(|op| &mut op.instruction_lookup),
        io_ctx0.id,
    );

    Instructions::operands_a2b_many(
        lookups.par_iter_mut().map(|op| &mut op.instruction_lookup),
        io_ctx0,
    )?;

    let num_chunks = C;
    let log_M = M.log_2();

    let indices: Vec<_> = cfg_into_iter!(lookups)
        .map(|lookup| {
            if let Some(lookup) = &lookup.instruction_lookup {
                lookup.to_indices_rep3(C, log_M)
            } else {
                vec![Rep3BigUintShare::zero_share(); C]
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
    Network: Rep3Network,
    Instructions: JoltInstructionSet<F> + Rep3JoltInstructionSet<F>,
>(
    ops: &[JoltTraceStep<F, Instructions>],
    num_reads: usize,
    io_ctx: &mut IoContext<Network>,
) -> eyre::Result<Rep3MultilinearPolynomial<F>> {
    // TODO: use jolt_core::utils::chunk_map
    let mut outputs = ops
        .iter()
        .map(|op| {
            if let Some(op) = &op.instruction_lookup {
                op.output(io_ctx)
            } else {
                Ok(Rep3PrimeFieldShare::zero_share())
            }
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    outputs.resize(num_reads, Rep3PrimeFieldShare::zero_share());
    Ok(Rep3MultilinearPolynomial::from(outputs))
}

struct BiNetwork<Network: Rep3Network> {
    pub io_ctx: IoContext<Network>,
    pub io_ctx1: IoContext<Network>,
}

impl<Network: Rep3Network> BiNetwork<Network> {
    pub fn new(mut io_ctx: IoContext<Network>) -> eyre::Result<Self> {
        let io_ctx1 = io_ctx.fork()?;
        Ok(Self { io_ctx, io_ctx1 })
    }
}

impl<Network: Rep3Network> Forkable for BiNetwork<Network> {
    fn fork(&mut self) -> eyre::Result<Self> {
        Ok(Self {
            io_ctx: self.io_ctx.fork()?,
            io_ctx1: self.io_ctx1.fork()?,
        })
    }
}
