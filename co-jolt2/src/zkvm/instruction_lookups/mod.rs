use std::sync::Arc;

use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::zkvm::instruction_lookups::D;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::network::{IoContextPool, Rep3NetworkWorker};
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;
use mpc_core::protocols::rep3_ring::edabits::PreprocessingPool;

use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::opening_proof::Rep3OpeningAccumulatorWorker;
use crate::zkvm::dag::stage::{BatchedSumcheckWorkerInstance, SumcheckStagesWorker};
use crate::zkvm::dag::state_manager::StateManagerWorker;
use jolt_core::field::JoltField;

use self::booleanity::Rep3BooleanitySumcheckWorker;
use self::hamming_weight::Rep3HammingWeightSumcheckWorker;
use self::ra_virtual::Rep3InstructionRaSumcheckWorker;
use self::read_raf_checking::Rep3ReadRafSumcheckWorker;

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
/// as a secret-shared vector.
///
/// No MPC communication — all operations are `public * shared`.
fn compute_ra_evals<F: JoltField>(
    one_hot_polys: &[Rep3OneHotPolynomial<F>; D],
    eq_r_cycle: &[F],
) -> [Arc<Vec<Rep3PrimeFieldShare<F>>>; D] {
    crate::poly::one_hot_polynomial::compute_g_from_masked_indices_many(one_hot_polys, eq_r_cycle)
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
    G: Option<[Arc<Vec<Rep3PrimeFieldShare<F>>>; D]>,
    /// Public eq(r_cycle) evaluations.
    eq_r_cycle: Option<Vec<F>>,
    /// D Rep3OneHotPolynomials from witness gen, used for compute_ra_evals.
    pub one_hot_polys: Arc<[Rep3OneHotPolynomial<F>; D]>,
}

impl<F: JoltField> Rep3LookupsDagWorker<F> {
    pub fn new(one_hot_polys: [Rep3OneHotPolynomial<F>; D]) -> Self {
        Self { stage2: None, stage3: None, G: None, eq_r_cycle: None, one_hot_polys: Arc::new(one_hot_polys) }
    }

    pub fn set_stage2_init(&mut self, gamma: [F; D], r_address: Vec<F::Challenge>) {
        self.stage2 = Some((gamma, r_address));
    }

    pub fn set_stage3_init(&mut self, hamming_gamma: [F; D], read_raf_gamma: F, rv_claim: F, raf_claim: F) {
        self.stage3 = Some(LookupStage3Init { hamming_gamma, read_raf_gamma, rv_claim, raf_claim });
    }

    pub fn stage3_instances<PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        io_ctx: &mut IoContextPool<N>,
        preproc: &mut PreprocessingPool<F>,
    ) -> Vec<BatchedSumcheckWorkerInstance<F, N>> {
        let G = self.G.take().unwrap();
        let init = self.stage3.take().expect("Rep3LookupsDagWorker stage3 init not set");

        // ReadRaf (created before HammingWeight, matching vanilla ordering)
        let eq_r_cycle = self.eq_r_cycle.take().unwrap();
        let one_hot_polys = self.one_hot_polys.clone();
        let rr = sm.prover_state.cycle_witness.take_read_raf();
        let read_raf = Rep3ReadRafSumcheckWorker::new(
            init.read_raf_gamma,
            init.rv_claim,
            init.raf_claim,
            one_hot_polys,
            &eq_r_cycle,
            rr.lookup_tables,
            rr.is_interleaved_operands,
            rr.lookup_indices,
            rr.right_operand_public_mask,
            io_ctx,
            sm.party_id,
            preproc,
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
        let mut ra_worker =
            Rep3InstructionRaSumcheckWorker::new(self.one_hot_polys.clone(), &ra_r_address, ra_r_cycle, ra_input_claim);
        ra_virtual::prove_worker(&mut ra_worker, accumulator, io_ctx)
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker> SumcheckStagesWorker<F, PCS, N>
    for Rep3LookupsDagWorker<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS>,
        _io_ctx: &mut IoContextPool<N>,
    ) -> Result<Vec<BatchedSumcheckWorkerInstance<F, N>>, eyre::Report> {
        let (gamma, r_address) = self.stage2.take().expect("Rep3LookupsDagWorker stage2 init not set");
        let r_cycle = sm
            .accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::LookupOutput, SumcheckId::SpartanOuter)
            .0
            .r
            .clone();
        let eq_r_cycle = EqPolynomial::evals(&r_cycle);
        let G = compute_ra_evals(&*self.one_hot_polys, &eq_r_cycle);

        self.eq_r_cycle = Some(eq_r_cycle);
        self.G = Some(G.clone());

        let booleanity = Rep3BooleanitySumcheckWorker::new(
            gamma,
            r_address,
            G,
            &*self.one_hot_polys,
            &r_cycle,
            sm.prover_state.cycle_witness.len(),
            sm.party_id,
        );

        Ok(vec![BatchedSumcheckWorkerInstance::Secret(Box::new(booleanity))])
    }
}

// ---------------------------------------------------------------------------
