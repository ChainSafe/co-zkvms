use std::marker::PhantomData;

use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::eq_poly::EqPolynomial;
use jolt_core::poly::opening_proof::SumcheckId;
use jolt_core::transcripts::Transcript;
use jolt_core::zkvm::instruction_lookups::D;
use jolt_core::zkvm::witness::VirtualPolynomial;
use mpc_core::protocols::rep3::network::Rep3NetworkWorker;
use mpc_core::protocols::rep3::Rep3PrimeFieldShare;

use crate::field::JoltField;
use crate::poly::one_hot_polynomial::{compute_g_from_masked_indices, Rep3OneHotPolynomial};
use crate::zkvm::dag::stage::{
    Rep3SumcheckInstance, Rep3SumcheckInstanceWorker, SumcheckStagesCoordinator,
    SumcheckStagesWorker,
};
use crate::zkvm::dag::state_manager::{StateManagerCoordinator, StateManagerWorker};

use self::booleanity::{Rep3BooleanitySumcheck, Rep3BooleanitySumcheckWorker};
use self::hamming_weight::{Rep3HammingWeightSumcheck, Rep3HammingWeightSumcheckWorker};

pub mod booleanity;
pub mod hamming_weight;

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

pub struct Rep3LookupsDagWorker<F: JoltField> {
    /// Shared G arrays computed in stage2, consumed in stage3.
    G: Option<[Vec<Rep3PrimeFieldShare<F>>; D]>,
    /// Public eq(r_cycle) evaluations.
    eq_r_cycle: Option<Vec<F>>,
    /// D Rep3OneHotPolynomials from witness gen, used for compute_ra_evals.
    one_hot_polys: [Rep3OneHotPolynomial<F>; D],
}

impl<F: JoltField> Rep3LookupsDagWorker<F> {
    pub fn new(one_hot_polys: [Rep3OneHotPolynomial<F>; D]) -> Self {
        Self {
            G: None,
            eq_r_cycle: None,
            one_hot_polys,
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>, N: Rep3NetworkWorker>
    SumcheckStagesWorker<F, PCS, N> for Rep3LookupsDagWorker<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerWorker<'_, F, PCS, N>,
    ) -> Vec<Box<dyn Rep3SumcheckInstanceWorker<F>>> {
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
            G,
            &self.one_hot_polys,
            &r_cycle,
            sm.prover_state.trace.len(),
            sm.io_ctx.party_id(),
        );

        vec![Box::new(booleanity)]
    }

    fn stage3_instances(
        &mut self,
        _sm: &mut StateManagerWorker<'_, F, PCS, N>,
    ) -> Vec<Box<dyn Rep3SumcheckInstanceWorker<F>>> {
        let G = self.G.take().unwrap();
        let hamming_weight = Rep3HammingWeightSumcheckWorker::new(G);

        vec![Box::new(hamming_weight)]
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

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStagesCoordinator<F, ProofTranscript, PCS> for Rep3LookupsDag<F>
{
    fn stage2_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> {
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

        vec![Box::new(booleanity)]
    }

    fn stage3_instances(
        &mut self,
        sm: &mut StateManagerCoordinator<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn Rep3SumcheckInstance<F, ProofTranscript>>> {
        let hamming_weight = Rep3HammingWeightSumcheck::new(&mut sm.transcript);

        vec![Box::new(hamming_weight)]
    }
}
