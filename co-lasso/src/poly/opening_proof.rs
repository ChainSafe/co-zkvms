pub use jolt_core::poly::opening_proof::*;
use mpc_core::protocols::rep3::{self, network::{IoContext, Rep3NetworkCoordinator, Rep3NetworkWorker}};

use crate::{
    field::JoltField,
    poly::{MultilinearPolynomial, Rep3DensePolynomial},
    utils::transcript::Transcript,
};

/// An opening computed by the prover.
pub struct Rep3ProverOpening<F: JoltField> {
    /// The polynomial being opened. May be a random linear combination
    /// of multiple polynomials all being opened at the same point.
    pub polynomial: Rep3DensePolynomial<F>,
    /// The multilinear extension EQ(x, opening_point). This is typically
    /// an intermediate value used to compute `claim`, but is also used in
    /// the `ProverOpeningAccumulator::prove_batch_opening_reduction` sumcheck.
    pub eq_poly: MultilinearPolynomial<F>,
    /// The point at which the `polynomial` is being evaluated.
    pub opening_point: Vec<F>,
    /// The claimed opening.
    pub claim: F,
}

impl<F: JoltField> Rep3ProverOpening<F> {
    fn new(
        polynomial: Rep3DensePolynomial<F>,
        eq_poly: MultilinearPolynomial<F>,
        opening_point: Vec<F>,
        claim: F,
    ) -> Self {
        Rep3ProverOpening {
            polynomial,
            eq_poly,
            opening_point,
            claim,
        }
    }
}

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
pub struct Rep3ProverOpeningAccumulator<F: JoltField> {
    openings: Vec<Rep3ProverOpening<F>>,
}

impl<F: JoltField> Rep3ProverOpeningAccumulator<F> {
    pub fn new() -> Self {
        Self { openings: vec![] }
    }

    pub fn len(&self) -> usize {
        self.openings.len()
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append")]
    pub fn append<Network: Rep3NetworkWorker>(
        &mut self,
        polynomials: &[&Rep3DensePolynomial<F>],
        eq_poly: MultilinearPolynomial<F>,
        opening_point: Vec<F>,
        claims: &[F],
        io_ctx: &mut IoContext<Network>,
    ) -> eyre::Result<()> {
        assert_eq!(polynomials.len(), claims.len());

        io_ctx.network.send_response(claims.to_vec())?;
        let (rho, batched_claim): (F, F) = io_ctx.network.receive_request()?;

        // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
        let mut rho_powers = vec![F::one()];
        for i in 1..polynomials.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let batched_poly = Rep3DensePolynomial::linear_combination(polynomials, &rho_powers);

        let opening = Rep3ProverOpening::new(batched_poly, eq_poly, opening_point, batched_claim);
        self.openings.push(opening);

        Ok(())
    }

    pub fn receive_claims<ProofTranscript: Transcript, Network: Rep3NetworkCoordinator>(
        transcript: &mut ProofTranscript,
        network: &mut Network,
    ) -> eyre::Result<Vec<F>> {
        let [share1, share2, share3] = network.receive_responses(vec![])?.try_into().unwrap();
        let claims = rep3::combine_field_elements(&share1, &share2, &share3);
        let rho: F = transcript.challenge_scalar();
        let mut rho_powers = vec![F::one()];
        for i in 1..claims.len() {
            rho_powers.push(rho_powers[i - 1] * rho);
        }
        // Compute the random linear combination of the claims
        let batched_claim: F = rho_powers
            .iter()
            .zip(claims.iter())
            .map(|(scalar, eval)| *scalar * *eval)
            .sum();
        network.broadcast_request((rho, batched_claim))?;
        Ok(claims)
    }
}
