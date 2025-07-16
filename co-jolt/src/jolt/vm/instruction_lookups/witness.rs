use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, cfg_iter};
use co_lasso::{
    poly::{combine_poly_shares_rep3, generate_poly_shares_rep3, Rep3DensePolynomial},
    subprotocols::commitment::DistributedCommitmentScheme,
    utils::{self, Forkable},
};
use color_eyre::eyre::Context;
use itertools::{chain, multizip, Itertools};
use jolt_core::{
    poly::{
        commitment::commitment_scheme::BatchType, dense_mlpoly::DensePolynomial, field::JoltField,
    },
    utils::math::Math,
};
use mpc_core::protocols::{
    rep3::{
        self, arithmetic,
        network::{IoContext, Rep3Network},
        PartyID, Rep3BigUintShare, Rep3PrimeFieldShare,
    },
    rep3_ring::lut::{PublicPrivateLut, Rep3LookupTable},
};
use mpc_net::mpc_star::{MpcStarNetCoordinator, MpcStarNetWorker};
use rand::Rng;
use std::{iter, marker::PhantomData};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    jolt::{
        instruction::Rep3JoltInstructionSet,
        subtable::JoltSubtableSet,
        vm::instruction_lookups::{
            InstructionCommitment, InstructionLookupsPreprocessing, InstructionPolynomials,
        },
    },
    poly::commitment::commitment_scheme::CommitmentScheme,
};

#[derive(Debug, Clone, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3InstructionPolynomials<F: JoltField> {
    /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
    /// indices at which the memories will be evaluated. Each `DensePolynomial` has size
    /// `m` (# lookups).
    pub dim: Vec<Rep3DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// read access counts to the memory. Each `DensePolynomial` has size `m` (# lookups).
    pub read_cts: Vec<Rep3DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// final access counts to the memory. Each `DensePolynomial` has size M, AKA subtable size.
    pub final_cts: Vec<Rep3DensePolynomial<F>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// the evaluation of memory accessed at each step of the CPU. Each `DensePolynomial` has
    /// size `m` (# lookups).
    pub E_polys: Vec<Rep3DensePolynomial<F>>,

    /// Polynomial encodings for flag polynomials for each instruction.
    /// If using a single instruction this will be empty.
    /// NUM_INSTRUCTIONS sized, each polynomial of length 'm' (# lookups).
    ///
    /// Stored independently for use in sumcheck, combined into single DensePolynomial for commitment.
    pub instruction_flag_polys: Vec<DensePolynomial<F>>,

    /// Instruction flag polynomials as bitvectors, kept in this struct for more efficient
    /// construction of the memory flag polynomials in `read_write_grand_product`.
    pub instruction_flag_bitvectors: Vec<Vec<u64>>,

    /// The lookup output for each instruction of the execution trace.
    pub lookup_outputs: Rep3DensePolynomial<F>,
}

impl<F: JoltField> Rep3InstructionPolynomials<F> {
    pub fn commit<CS: DistributedCommitmentScheme<F>>(
        &self,
        setup: &CS::Setup,
        network: &mut impl MpcStarNetWorker,
    ) -> eyre::Result<()> {
        let dims_share_a = self
            .dim
            .iter()
            .map(|poly| poly.copy_share_a())
            .collect::<Vec<_>>();
        let read_cts_share_a = self
            .read_cts
            .iter()
            .map(|poly| poly.copy_share_a())
            .collect::<Vec<_>>();
        let e_polys_share_a = self
            .E_polys
            .iter()
            .map(|poly| poly.copy_share_a())
            .collect::<Vec<_>>();
        let lookup_outputs_share_a = self.lookup_outputs.copy_share_a();
        let trace_polys: Vec<&DensePolynomial<F>> = chain![
            dims_share_a.iter(),
            read_cts_share_a.iter(),
            e_polys_share_a.iter(),
            iter::once(&lookup_outputs_share_a),
        ]
        .collect();

        let final_cts = self
            .final_cts
            .iter()
            .map(|poly| poly.copy_poly_shares().0)
            .collect::<Vec<_>>();
        let trace_commitment = CS::batch_commit_polys_ref(&trace_polys, setup, BatchType::Big);
        let lookup_flag_polys_commitment = CS::batch_commit_polys_ref(
            &self.instruction_flag_polys.iter().collect::<Vec<_>>(),
            setup,
            BatchType::Big,
        );
        let final_commitment = CS::batch_commit_polys(&final_cts, setup, BatchType::Big);

        if network.party_id() == PartyID::ID0 {
            network.send_response(lookup_flag_polys_commitment)?;
        }

        network.send_response(InstructionCommitment::<CS> {
            trace_commitment,
            final_commitment,
        })
    }

    pub fn receive_commitments<CS: DistributedCommitmentScheme<F>>(
        network: &mut impl MpcStarNetCoordinator,
    ) -> eyre::Result<InstructionCommitment<CS>> {
        // lookup flag polys commitment are not secret shared
        let lookup_flag_polys_commitment: Vec<CS::Commitment> =
            network.receive_response(PartyID::ID0, 0, Default::default())?;

        let [share1, share2, share3] = network
            .receive_responses(InstructionCommitment::<CS> {
                trace_commitment: Default::default(),
                final_commitment: Default::default(),
            })?
            .try_into()
            .map_err(|_| eyre::eyre!("failed to receive commitments"))?;

        let mut trace_commitment = multizip((
            share1.trace_commitment,
            share2.trace_commitment,
            share3.trace_commitment,
        ))
        .map(|(trace1, trace2, trace3)| CS::combine_commitments(&[trace1, trace2, trace3]))
        .collect_vec();

        // need to insert lookup flag polys commitment after e_polys commitments and before lookup outputs commitment
        let after_e_polys_idx = trace_commitment.len().saturating_sub(1);
        trace_commitment.splice(
            after_e_polys_idx..after_e_polys_idx,
            lookup_flag_polys_commitment,
        );

        let final_commitment = multizip((
            share1.final_commitment,
            share2.final_commitment,
            share3.final_commitment,
        ))
        .map(|(final1, final2, final3)| CS::combine_commitments(&[final1, final2, final3]))
        .collect_vec();

        Ok(InstructionCommitment::<CS> {
            trace_commitment,
            final_commitment,
        })
    }
}

impl<F: JoltField> co_lasso::Rep3Polynomials for Rep3InstructionPolynomials<F> {
    fn num_lookups(&self) -> usize {
        self.dim[0].len()
    }
}

pub struct Rep3InstructionWitnessSolver<
    const C: usize,
    const M: usize,
    F: JoltField,
    CS,
    Lookups: Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    N: Rep3Network,
> {
    pub io_ctx0: IoContext<N>,
    pub io_ctx1: IoContext<N>,
    _marker: PhantomData<(F, CS, Lookups, Subtables)>,
}

impl<const C: usize, const M: usize, F: JoltField, CS, Instructions, Subtables, Network>
    Rep3InstructionWitnessSolver<C, M, F, CS, Instructions, Subtables, Network>
where
    CS: DistributedCommitmentScheme<F>,
    Instructions: Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    Network: Rep3Network,
{
    pub fn new(net: Network) -> color_eyre::Result<Self> {
        let mut io_context0 = IoContext::init(net).context("failed to initialize io context")?;
        let io_context1 = io_context0.fork().context("failed to fork io context")?;

        Ok(Self {
            io_ctx0: io_context0,
            io_ctx1: io_context1,
            _marker: PhantomData,
        })
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoWitnessSolver::polynomialize")]
    pub fn polynomialize(
        &mut self,
        preprocessing: &InstructionLookupsPreprocessing<F>,
        mut ops: Vec<Option<Instructions>>,
    ) -> eyre::Result<Rep3InstructionPolynomials<F>> {
        let num_reads = ops.len().next_power_of_two();

        let subtable_lookup_indices = self.subtable_lookup_indices(&mut ops)?;

        let materialized_subtable_luts = preprocessing
            .materialized_subtables
            .clone()
            .into_iter()
            .map(|subtable| PublicPrivateLut::Public(subtable))
            .collect_vec();

        let polys = tracing::info_span!("compute_polys").in_scope(|| {
            utils::fork_map(
                0..preprocessing.num_memories,
                self,
                |memory_index, solver| {
                    let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                    let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                    let access_sequence = &subtable_lookup_indices[dim_index];

                    let mut final_cts_i = vec![Rep3PrimeFieldShare::zero_share(); M];
                    let mut read_cts_i = vec![Rep3PrimeFieldShare::zero_share(); num_reads];
                    let mut subtable_lookups = vec![Rep3PrimeFieldShare::zero_share(); num_reads];

                    for (j, op) in ops.iter().enumerate() {
                        if let Some(op) = op {
                            let memories_used = &preprocessing.instruction_to_memory_indices
                                [Instructions::enum_index(op)];
                            if memories_used.contains(&memory_index) {
                                let memory_address = &access_sequence[j];

                                let ohv = Rep3LookupTable::ohv_from_index_no_a2b_conversion(
                                    memory_address.clone(),
                                    M,
                                    &mut solver.io_ctx0,
                                    &mut solver.io_ctx1,
                                )
                                .unwrap();

                                let mut counter = Rep3LookupTable::get_from_shared_lut_from_ohv(
                                    &ohv,
                                    &final_cts_i,
                                    &mut solver.io_ctx0,
                                    &mut solver.io_ctx1,
                                )
                                .unwrap();
                                read_cts_i[j] = counter;
                                counter = counter
                                    + arithmetic::promote_to_trivial_share(
                                        solver.io_ctx0.id,
                                        F::one(),
                                    );

                                Rep3LookupTable::write_to_shared_lut_from_ohv(
                                    &ohv,
                                    counter,
                                    &mut final_cts_i,
                                    &mut solver.io_ctx0,
                                    &mut solver.io_ctx1,
                                )
                                .unwrap();

                                let subtable_lookup_share =
                                    Rep3LookupTable::get_from_lut_no_a2b_conversion(
                                        memory_address.clone(),
                                        &materialized_subtable_luts[subtable_index],
                                        &mut solver.io_ctx0,
                                        &mut solver.io_ctx1,
                                    )
                                    .unwrap();
                                subtable_lookups[j] = subtable_lookup_share;
                            }
                        }
                    }

                    (
                        Rep3DensePolynomial::new(read_cts_i),
                        Rep3DensePolynomial::new(final_cts_i),
                        Rep3DensePolynomial::new(subtable_lookups),
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

        let dims = tracing::info_span!("b2a dims").in_scope(|| {
            utils::fork_map(
                subtable_lookup_indices,
                &mut self.io_ctx0,
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
                    Rep3DensePolynomial::new(dim)
                },
            )
        })?;

        let mut instruction_flag_bitvectors: Vec<Vec<u64>> =
            vec![vec![0u64; num_reads]; Instructions::COUNT];

        for (j, op) in ops.iter().enumerate() {
            if let Some(op) = op {
                instruction_flag_bitvectors[Instructions::enum_index(op)][j] = 1;
            }
        }

        let instruction_flag_polys: Vec<_> = cfg_iter!(instruction_flag_bitvectors)
            .map(|flag_bitvector| DensePolynomial::from_u64(flag_bitvector))
            .collect();

        let lookup_outputs = Self::compute_lookup_outputs(&ops, num_reads, &mut self.io_ctx0);

        Ok(Rep3InstructionPolynomials {
            dim: dims,
            read_cts,
            final_cts,
            instruction_flag_polys,
            instruction_flag_bitvectors,
            E_polys: e_polys,
            lookup_outputs,
        })
    }

    #[tracing::instrument(skip_all, name = "Rep3LassoWitnessSolver::subtable_lookup_indices")]
    fn subtable_lookup_indices(
        &mut self,
        lookups: &mut [Option<Instructions>],
    ) -> eyre::Result<Vec<Vec<Rep3BigUintShare<F>>>> {
        Instructions::operands_to_binary(lookups, &mut self.io_ctx0)?;

        let num_chunks = C;
        let log_M = M.log_2();

        let indices: Vec<_> = cfg_into_iter!(lookups)
            .map(|lookup| {
                if let Some(lookup) = lookup {
                    lookup.to_indices(C, log_M)
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
    fn compute_lookup_outputs(
        ops: &[Option<Instructions>],
        num_reads: usize,
        io_ctx: &mut IoContext<Network>,
    ) -> Rep3DensePolynomial<F> {
        let mut outputs = ops
            .iter()
            .map(|op| {
                if let Some(op) = op {
                    op.output(io_ctx)
                } else {
                    Rep3PrimeFieldShare::zero_share()
                }
            })
            .collect_vec();
        outputs.resize(num_reads, Rep3PrimeFieldShare::zero_share());
        Rep3DensePolynomial::new(outputs)
    }
}

impl<F: JoltField> Rep3InstructionPolynomials<F> {
    pub fn combine_polynomials<CS: CommitmentScheme<Field = F>>(
        polynomials_shares: Vec<Self>,
    ) -> InstructionPolynomials<F, CS> {
        let [share1, share2, share3] = polynomials_shares.try_into().unwrap();

        let dims = multizip((share1.dim, share2.dim, share3.dim))
            .map(|(dim1, dim2, dim3)| combine_poly_shares_rep3(vec![dim1, dim2, dim3]))
            .collect_vec();

        let read_cts = multizip((share1.read_cts, share2.read_cts, share3.read_cts))
            .map(|(read1, read2, read3)| combine_poly_shares_rep3(vec![read1, read2, read3]))
            .collect_vec();

        let final_cts = multizip((share1.final_cts, share2.final_cts, share3.final_cts))
            .map(|(final1, final2, final3)| combine_poly_shares_rep3(vec![final1, final2, final3]))
            .collect_vec();

        let e_polys = multizip((share1.E_polys, share2.E_polys, share3.E_polys))
            .map(|(e1, e2, e3)| combine_poly_shares_rep3(vec![e1, e2, e3]))
            .collect_vec();

        let lookup_outputs = combine_poly_shares_rep3(vec![
            share1.lookup_outputs,
            share2.lookup_outputs,
            share3.lookup_outputs,
        ]);

        InstructionPolynomials {
            _marker: PhantomData,
            dim: dims,
            read_cts,
            final_cts,
            instruction_flag_polys: share1.instruction_flag_polys.clone(),
            instruction_flag_bitvectors: share1.instruction_flag_bitvectors.clone(),
            E_polys: e_polys,
            lookup_outputs,
        }
    }
}

impl<const C: usize, const M: usize, F: JoltField, CS, Lookups, Subtables, Network> Forkable
    for Rep3InstructionWitnessSolver<C, M, F, CS, Lookups, Subtables, Network>
where
    CS: DistributedCommitmentScheme<F>,
    Lookups: Rep3JoltInstructionSet<F>,
    Subtables: JoltSubtableSet<F>,
    Network: Rep3Network,
{
    fn fork(&mut self) -> eyre::Result<Self> {
        Ok(Self {
            io_ctx0: self.io_ctx0.fork()?,
            io_ctx1: self.io_ctx1.fork()?,
            _marker: PhantomData,
        })
    }
}

impl<F, C> InstructionPolynomials<F, C>
where
    F: JoltField,
    C: CommitmentScheme<Field = F>,
{
    pub fn into_secret_shares_rep3<R: Rng>(
        self,
        rng: &mut R,
    ) -> eyre::Result<[Rep3InstructionPolynomials<F>; 3]> {
        let (dim0, dim1, dim2) = itertools::multiunzip(
            self.dim
                .iter()
                .map(|poly| generate_poly_shares_rep3(poly, rng)),
        );

        let (read_cts0, read_cts1, read_cts2) = itertools::multiunzip(
            self.read_cts
                .iter()
                .map(|poly| generate_poly_shares_rep3(poly, rng)),
        );

        let (final_cts0, final_cts1, final_cts2) = itertools::multiunzip(
            self.final_cts
                .iter()
                .map(|poly| generate_poly_shares_rep3(poly, rng)),
        );

        let (e_polys0, e_polys1, e_polys2) = itertools::multiunzip(
            self.E_polys
                .iter()
                .map(|poly| generate_poly_shares_rep3(poly, rng)),
        );

        let (lookup_outputs0, lookup_outputs1, lookup_outputs2) =
            generate_poly_shares_rep3(&self.lookup_outputs, rng);

        let p0 = Rep3InstructionPolynomials {
            dim: dim0,
            read_cts: read_cts0,
            final_cts: final_cts0,
            E_polys: e_polys0,
            lookup_outputs: lookup_outputs0,
            instruction_flag_polys: self.instruction_flag_polys.clone(),
            instruction_flag_bitvectors: self.instruction_flag_bitvectors.clone(),
        };

        let p1 = Rep3InstructionPolynomials {
            dim: dim1,
            read_cts: read_cts1,
            final_cts: final_cts1,
            E_polys: e_polys1,
            lookup_outputs: lookup_outputs1,
            instruction_flag_polys: self.instruction_flag_polys.clone(),
            instruction_flag_bitvectors: self.instruction_flag_bitvectors.clone(),
        };

        let p2 = Rep3InstructionPolynomials {
            dim: dim2,
            read_cts: read_cts2,
            final_cts: final_cts2,
            E_polys: e_polys2,
            lookup_outputs: lookup_outputs2,
            instruction_flag_polys: self.instruction_flag_polys,
            instruction_flag_bitvectors: self.instruction_flag_bitvectors,
        };

        Ok([p0, p1, p2])
    }
}
