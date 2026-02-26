use std::marker::PhantomData;

use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::{D, LOG_K_CHUNK};
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::network::{
    IoContextPool, Rep3NetworkCoordinator, Rep3NetworkWorker,
};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3_ring::edabits::EdaBitsPool;

use crate::field::JoltField;
use crate::poly::one_hot_polynomial::{compute_g_from_masked_indices, Rep3OneHotPolynomial};
use crate::poly::opening_proof::{Rep3OpeningAccumulator, Rep3OpeningAccumulatorWorker};
use crate::zkvm::dag::stage::{
    BatchedSumcheckInstance, BatchedSumcheckWorkerInstance, SumcheckStagesCoordinator,
    SumcheckStagesWorker,
};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

use self::booleanity::{Rep3BooleanitySumcheck, Rep3BooleanitySumcheckWorker};
use self::hamming_weight::{Rep3HammingWeightSumcheck, Rep3HammingWeightSumcheckWorker};
use self::ra_virtual::{Rep3InstructionRaSumcheck, Rep3InstructionRaSumcheckWorker};
use self::read_raf_checking::{Rep3ReadRafSumcheck, Rep3ReadRafSumcheckWorker};

pub mod booleanity;
pub mod hamming_weight;
pub mod ra_virtual;
pub mod read_raf_checking;

// ---------------------------------------------------------------------------
// compute_ra_evals
// ---------------------------------------------------------------------------

/// MPC version of vanilla `compute_ra_evals`. Computes eq-weighted histogram
/// of lookup index chunks using the RandOHV representation from witness gen.
///
/// For each chunk `i`, computes `G[i][k] = Σ_j eq(r_cycle, j) * [chunk_i(index_j) == k]`
/// as a secret-shared vector using the existing `compute_g_from_masked_indices`.
///
/// No MPC communication — all operations are `public * shared`.
fn compute_ra_evals<F: JoltField>(
    one_hot_polys: &[Rep3OneHotPolynomial<F>; D],
    eq_r_cycle: &[F],
) -> [Vec<Rep3PrimeFieldShare<F>>; D] {
    std::array::from_fn(|i| compute_g_from_masked_indices(&one_hot_polys[i], eq_r_cycle))
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

/// Stage3 init data for the ReadRaf + HammingWeight sumchecks.
struct LookupStage3Init<F: JoltField> {
    /// HammingWeight gamma powers (drawn from transcript by coordinator).
    hamming_gamma: [F; D],
    /// ReadRaf gamma (drawn from transcript by coordinator).
    read_raf_gamma: F,
    /// rv_claim = LookupOutput claim from Spartan outer.
    rv_claim: F,
    /// raf_claim = left_operand_claim + read_raf_gamma * right_operand_claim.
    raf_claim: F,
}

pub struct Rep3LookupsDagWorker<F: JoltField> {
    stage2: Option<([F; D], Vec<F::Challenge>)>,
    stage3: Option<LookupStage3Init<F>>,
    /// Shared G arrays computed in stage2, consumed in stage3.
    G: Option<[Vec<Rep3PrimeFieldShare<F>>; D]>,
    /// Public eq(r_cycle) evaluations.
    eq_r_cycle: Option<Vec<F>>,
    /// D Rep3OneHotPolynomials from witness gen, used for compute_ra_evals.
    pub one_hot_polys: [Rep3OneHotPolynomial<F>; D],
}

impl<F: JoltField> Rep3LookupsDagWorker<F> {
    pub fn new(one_hot_polys: [Rep3OneHotPolynomial<F>; D]) -> Self {
        Self {
            stage2: None,
            stage3: None,
            G: None,
            eq_r_cycle: None,
            one_hot_polys,
        }
    }

    pub fn set_stage2_init(&mut self, gamma: [F; D], r_address: Vec<F::Challenge>) {
        self.stage2 = Some((gamma, r_address));
    }

    pub fn set_stage3_init(
        &mut self,
        hamming_gamma: [F; D],
        read_raf_gamma: F,
        rv_claim: F,
        raf_claim: F,
    ) {
        self.stage3 = Some(LookupStage3Init {
            hamming_gamma,
            read_raf_gamma,
            rv_claim,
            raf_claim,
        });
    }

    pub fn stage3_instances<PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
        edabits_pool: EdaBitsPool<F>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F, N>> {
        let G = self.G.take().unwrap();
        let init = self
            .stage3
            .take()
            .expect("Rep3LookupsDagWorker stage3 init not set");

        // ReadRaf (created before HammingWeight, matching vanilla ordering)
        let eq_r_cycle = self.eq_r_cycle.take().unwrap();
        let one_hot_polys = self.one_hot_polys.clone();
        let cw = &mut sm.prover_state.cycle_witness;
        let lookup_indices = cw.lookup_indices.clone();
        let lookup_tables = std::mem::take(&mut cw.lookup_tables);
        let is_interleaved_operands = std::mem::take(&mut cw.is_interleaved_operands);
        let read_raf = Rep3ReadRafSumcheckWorker::new(
            init.read_raf_gamma,
            init.rv_claim,
            init.raf_claim,
            one_hot_polys,
            &eq_r_cycle,
            lookup_tables,
            is_interleaved_operands,
            lookup_indices,
            io_ctx,
            sm.party_id,
            edabits_pool,
        )
        .expect("Rep3ReadRafSumcheckWorker::new failed");

        let hamming_weight = Rep3HammingWeightSumcheckWorker::new(G, init.hamming_gamma);

        vec![
            BatchedSumcheckWorkerInstance::Secret(Box::new(read_raf)),
            BatchedSumcheckWorkerInstance::Secret(Box::new(hamming_weight)),
        ]
    }

    /// Run the worker-side RA virtualization sumcheck (stage 4).
    ///
    /// Receives init data (claim, r_address, r_cycle) from the coordinator
    /// and runs the dedicated proving loop with IO context for resharing.
    pub fn stage4_prove_worker<N: Rep3NetworkWorker>(
        &self,
        accumulator: &mut Rep3OpeningAccumulatorWorker<F>,
        io_ctx: &mut IoContextPool<N>,
    ) -> eyre::Result<Vec<F::Challenge>> {
        let (ra_input_claim, ra_r_address, ra_r_cycle): (F, Vec<F::Challenge>, Vec<F::Challenge>) =
            io_ctx.network().receive_request()?;
        let mut ra_worker = Rep3InstructionRaSumcheckWorker::new(
            &self.one_hot_polys,
            &ra_r_address,
            ra_r_cycle,
            ra_input_claim,
        );
        ra_virtual::prove_worker(&mut ra_worker, accumulator, io_ctx)
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
    SumcheckStagesWorker<F, PCS, N> for Rep3LookupsDagWorker<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F, N>> {
        let (gamma, r_address) = self
            .stage2
            .take()
            .expect("Rep3LookupsDagWorker stage2 init not set");
        let r_cycle = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .clone();
        let eq_r_cycle = EqPolynomial::evals(&r_cycle);
        let G = compute_ra_evals(&self.one_hot_polys, &eq_r_cycle);

        self.eq_r_cycle = Some(eq_r_cycle);
        self.G = Some(G.clone());

        let booleanity = Rep3BooleanitySumcheckWorker::new(
            gamma,
            r_address,
            G,
            &self.one_hot_polys,
            &r_cycle,
            sm.prover_state.cycle_witness.len(),
            sm.party_id,
        );

        vec![BatchedSumcheckWorkerInstance::Secret(Box::new(booleanity))]
    }
}

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub struct Rep3LookupsDag<F: JoltField> {
    _phantom: PhantomData<F>,
}

impl<F: JoltField> Rep3LookupsDag<F> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: JoltField> Default for Rep3LookupsDag<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: JoltField> Rep3LookupsDag<F> {
    /// Run the coordinator-side RA virtualization sumcheck (stage 4).
    ///
    /// Reads the InstructionRa opening from the accumulator, broadcasts init
    /// data to workers, and runs the dedicated proving loop.
    pub fn stage4_prove_coordinator<ProofTranscript, PCS, N>(
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
        network: &mut N,
    ) -> eyre::Result<Option<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>)>>
    where
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
        N: Rep3NetworkCoordinator,
    {
        use jolt_core::poly::opening_proof::OpeningId;

        let ra_key = OpeningId::Virtual(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );
        let has_ra_opening = sm.accumulator.openings.contains_key(&ra_key);

        // Tell workers whether stage 4 is active.
        network.broadcast_request(has_ra_opening)?;

        if !has_ra_opening {
            return Ok(None);
        }

        let (ra_point, ra_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, r_cycle) = ra_point.r.split_at(D * LOG_K_CHUNK);
        let r_address_chunks: Vec<Vec<F::Challenge>> =
            r_address.chunks(LOG_K_CHUNK).map(|c| c.to_vec()).collect();

        network.broadcast_request((ra_claim, r_address.to_vec(), r_cycle.to_vec()))?;

        let ra_coord = Rep3InstructionRaSumcheck::new(ra_claim, r_cycle.to_vec(), r_address_chunks);
        let result = ra_virtual::prove_coordinator(
            &ra_coord,
            &mut sm.accumulator,
            &mut sm.transcript,
            network,
        )?;

        Ok(Some(result))
    }
}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStagesCoordinator<F, ProofTranscript, PCS> for Rep3LookupsDag<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        let log_T = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .len();

        let booleanity = Rep3BooleanitySumcheck::new(&mut sm.transcript, log_T);

        vec![BatchedSumcheckInstance::Secret(Box::new(booleanity))]
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<BatchedSumcheckInstance<F, ProofTranscript>> {
        // ReadRaf (created before HammingWeight, matching vanilla ordering).
        // Draws gamma from transcript internally.
        let (_, rv_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );
        let (_, left_operand_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LeftLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let (_, right_operand_claim) = sm.accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::RightLookupOperand,
            SumcheckId::SpartanOuter,
        );
        let log_T = sm
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            )
            .0
            .r
            .len();

        let read_raf = Rep3ReadRafSumcheck::new(
            &mut sm.transcript,
            rv_claim,
            left_operand_claim,
            right_operand_claim,
            log_T,
        );

        let hamming_weight = Rep3HammingWeightSumcheck::new(&mut sm.transcript);

        vec![
            BatchedSumcheckInstance::Secret(Box::new(read_raf)),
            BatchedSumcheckInstance::Secret(Box::new(hamming_weight)),
        ]
    }
}
