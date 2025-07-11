use color_eyre::eyre::Result;
use eyre::Context;
use jolt_core::{
    poly::{dense_mlpoly::DensePolynomial, field::JoltField},
    utils::mul_0_1_optimized,
};
use mpc_core::protocols::rep3::{
    self,
    network::{IoContext, Rep3Network},
};
use mpc_net::mpc_star::MpcStarNetWorker;
use std::marker::PhantomData;
use tracing::trace_span;

use crate::{
    grand_product::{
        BatchedGrandProductProver, BatchedRep3GrandProductCircuit, Rep3GrandProductCircuit,
    },
    lasso::LassoPreprocessing,
    poly::Rep3DensePolynomial,
    subtables::{LookupSet, SubtableSet},
    utils::{self, split_rep3_poly_flagged},
    witness_solver::Rep3LassoPolynomials,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Rep3MemoryCheckingProver<const C: usize, const M: usize, F, Lookups, Subtables, Network>
where
    Network: Rep3Network,
{
    pub _marker: PhantomData<(F, Lookups, Subtables)>,
    pub io_ctx: IoContext<Network>,
}

type Preprocessing<F> = LassoPreprocessing<F>;
type Polynomials<F> = Rep3LassoPolynomials<F>;

impl<const C: usize, const M: usize, F: JoltField, Lookups, Subtables, Network>
    Rep3MemoryCheckingProver<C, M, F, Lookups, Subtables, Network>
where
    Lookups: LookupSet<F>,
    Subtables: SubtableSet<F>,
    Network: Rep3Network + MpcStarNetWorker,
{
    pub fn new(net: Network) -> color_eyre::Result<Self> {
        let io_ctx = IoContext::init(net).context("failed to initialize io context")?;
        Ok(Self {
            _marker: PhantomData,
            io_ctx,
        })
    }
    pub fn prove(
        &mut self,
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
    ) -> Result<()> {
        let _ = self.prove_grand_products(preprocessing, polynomials)?;

        Ok(())
    }

    fn prove_grand_products(
        &mut self,
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
    ) -> Result<(Vec<F>, Vec<F>)> {
        let (gamma, tau) = self.io_ctx.network.receive_request()?;
        self.io_ctx
            .network
            .send_response(polynomials.dims[0].len())?;

        let (read_write_leaves, init_final_leaves) = Self::compute_leaves(
            preprocessing,
            polynomials,
            &gamma,
            &tau,
            &mut self.io_ctx.network,
        );

        let (read_write_circuit, read_write_hashes) =
            self.read_write_grand_product(preprocessing, polynomials, read_write_leaves)?;
        let (init_final_circuit, init_final_hashes) =
            self.init_final_grand_product(preprocessing, polynomials, init_final_leaves)?;

        self.io_ctx
            .network
            .send_response((read_write_hashes.clone(), init_final_hashes.clone()))?;

        let r_read_write =
            BatchedGrandProductProver::prove_worker(read_write_circuit, &mut self.io_ctx.network)?;
        let r_init_final =
            BatchedGrandProductProver::prove_worker(init_final_circuit, &mut self.io_ctx.network)?;

        Ok((r_read_write, r_init_final))
    }

    fn compute_leaves(
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        gamma: &F,
        tau: &F,
        network: &mut Network,
    ) -> (Vec<Rep3DensePolynomial<F>>, Vec<Rep3DensePolynomial<F>>) {
        let gamma_squared = gamma.square();
        let num_lookups = polynomials.dims[0].len();
        let party_id = network.party_id();

        let read_write_leaves = (0..preprocessing.num_memories)
            .into_par_iter()
            .flat_map_iter(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];

                let read_fingerprints: Vec<_> = (0..num_lookups)
                    .map(|i| {
                        let a = &polynomials.dims[dim_index][i];
                        let v = &polynomials.e_polys[memory_index][i];
                        let t = &polynomials.read_cts[memory_index][i];
                        rep3::arithmetic::sub_shared_by_public(
                            (t * gamma_squared) + (v * *gamma) + *a,
                            *tau,
                            party_id,
                        )
                    })
                    .collect();
                let write_fingerprints = read_fingerprints
                    .iter()
                    .map(|read_fingerprint| {
                        rep3::arithmetic::add_public(*read_fingerprint, gamma_squared, party_id)
                    })
                    .collect();
                [
                    Rep3DensePolynomial::new(read_fingerprints),
                    Rep3DensePolynomial::new(write_fingerprints),
                ]
            })
            .collect();

        let init_final_leaves = preprocessing
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
                        mul_0_1_optimized(gamma, v) + a - tau
                    })
                    .collect();
                let init_fingerprints =
                    rep3::arithmetic::promote_to_trivial_shares(init_fingerprints, party_id);

                let final_leaves: Vec<_> = preprocessing.subtable_to_memory_indices[subtable_index]
                    .iter()
                    .map(|memory_index| {
                        let final_cts = &polynomials.final_cts[*memory_index];
                        let final_fingerprints = (0..M)
                            .map(|i| init_fingerprints[i] + (final_cts[i] * gamma_squared))
                            .collect();
                        Rep3DensePolynomial::new(final_fingerprints)
                    })
                    .collect();

                let mut polys = Vec::with_capacity(C + 1);
                polys.push(Rep3DensePolynomial::new(init_fingerprints));
                polys.extend(final_leaves);
                polys
            })
            .collect();

        (read_write_leaves, init_final_leaves)
    }

    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::read_write_grand_product")]
    fn read_write_grand_product(
        &mut self,
        preprocessing: &Preprocessing<F>,
        polynomials: &Polynomials<F>,
        read_write_leaves: Vec<Rep3DensePolynomial<F>>,
    ) -> Result<(BatchedRep3GrandProductCircuit<F>, Vec<F>)> {
        assert_eq!(read_write_leaves.len(), 2 * preprocessing.num_memories);

        let _span = trace_span!("construct_circuits");
        _span.enter();

        let memory_flag_polys =
            Self::memory_flag_polys(preprocessing, &polynomials.lookup_flag_bitvectors);

        let read_write_circuits = read_write_leaves
            // .par_iter()
            .iter()
            .enumerate()
            .map(|(i, leaves_poly)| {
                // Split while cloning to save on future cloning in GrandProductCircuit
                let memory_index = i / 2;
                let flag: &DensePolynomial<F> = &memory_flag_polys[memory_index];
                let (toggled_leaves_l, toggled_leaves_r) =
                    split_rep3_poly_flagged(leaves_poly, flag, self.io_ctx.network.party_id());
                Rep3GrandProductCircuit::new_split(
                    toggled_leaves_l,
                    toggled_leaves_r,
                    &mut self.io_ctx,
                )
            })
            .collect::<Result<Vec<Rep3GrandProductCircuit<F>>>>()?;

        drop(_span);

        let len = read_write_circuits[0].left_vec.len();
        self.io_ctx.network.send_response((
            read_write_circuits[0].left_vec[len - 1][0].clone(),
            read_write_circuits[0].right_vec[len - 1][0],
        ))?;

        let read_write_hashes: Vec<F> = trace_span!("compute_hashes").in_scope(|| {
            read_write_circuits
                .par_iter()
                .map(|circuit| circuit.evaluate())
                .collect()
        });

        // Prover has access to memory_flag_polys, which are uncommitted, but verifier can derive from instruction_flag commitments.
        let batched_circuits = BatchedRep3GrandProductCircuit::new_batch_flags(
            read_write_circuits,
            memory_flag_polys,
            read_write_leaves,
        );

        Ok((batched_circuits, read_write_hashes))
    }

    /// Constructs a batched grand product circuit for the init and final multisets associated
    /// with the given leaves. Also returns the corresponding multiset hashes for each memory.
    #[tracing::instrument(skip_all, name = "MemoryCheckingProver::init_final_grand_product")]
    fn init_final_grand_product(
        &mut self,
        _preprocessing: &Preprocessing<F>,
        _polynomials: &Polynomials<F>,
        init_final_leaves: Vec<Rep3DensePolynomial<F>>,
    ) -> Result<(BatchedRep3GrandProductCircuit<F>, Vec<F>)> {
        let init_final_circuits: Vec<Rep3GrandProductCircuit<F>> =
            utils::fork_map(init_final_leaves, &mut self.io_ctx, |leaves, io_ctx| {
                Rep3GrandProductCircuit::new(&leaves, io_ctx).unwrap()
            })?;

        let init_final_hashes: Vec<F> = init_final_circuits
            .par_iter()
            .map(|circuit| circuit.evaluate())
            .collect();

        Ok((
            BatchedRep3GrandProductCircuit::new_batch(init_final_circuits),
            init_final_hashes,
        ))
    }

    /// Converts instruction flag polynomials into memory flag polynomials. A memory flag polynomial
    /// can be computed by summing over the instructions that use that memory: if a given execution step
    /// accesses the memory, it must be executing exactly one of those instructions.
    #[tracing::instrument(skip_all)]
    fn memory_flag_polys(
        preprocessing: &Preprocessing<F>,
        flag_bitvectors: &[Vec<u64>],
    ) -> Vec<DensePolynomial<F>> {
        let m = flag_bitvectors[0].len();

        (0..preprocessing.num_memories)
            .into_par_iter()
            .map(|memory_index| {
                let mut memory_flag_bitvector = vec![0u64; m];
                for instruction_index in 0..Lookups::COUNT {
                    if preprocessing.lookup_to_memory_indices[instruction_index]
                        .contains(&memory_index)
                    {
                        memory_flag_bitvector
                            .iter_mut()
                            .zip(&flag_bitvectors[instruction_index])
                            .for_each(|(memory_flag, instruction_flag)| {
                                *memory_flag += instruction_flag
                            });
                    }
                }
                DensePolynomial::from_u64(&memory_flag_bitvector)
            })
            .collect()
    }
}
