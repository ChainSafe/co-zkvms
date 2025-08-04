#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use itertools::multizip;
use jolt_core::poly::{
    dense_mlpoly::DensePolynomial,
    multilinear_polynomial::{BindingOrder, PolynomialBinding},
};
use jolt_core::subprotocols::sumcheck::Bindable;
use jolt_core::utils::thread::drop_in_background_thread;
use mpc_core::protocols::rep3::network::{IoContext, Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::{self, PartyID};
use mpc_core::protocols::{additive, rep3::Rep3PrimeFieldShare};
use mpc_net::mpc_star::MpcStarNetWorker;
use rayon::prelude::*;
use tracing::trace_span;

use crate::poly::{Rep3DensePolynomial, Rep3MultilinearPolynomial};
use jolt_core::poly::split_eq_poly::{GruenSplitEqPolynomial, SplitEqPolynomial};
use jolt_core::utils::transcript::{AppendToTranscript, Transcript};

pub use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;

#[inline]
pub fn coordinate_eq_sumcheck_round<
    F: JoltField,
    ProofTranscript: Transcript,
    Network: Rep3NetworkCoordinator,
>(
    polys: &mut Vec<CompressedUniPoly<F>>,
    r: &mut Vec<F>,
    claim: &mut F,
    transcript: &mut ProofTranscript,
    network: &mut Network,
) -> eyre::Result<()> {
    let cubic_poly = UniPoly::from_coeff(additive::combine_field_element_vec(
        network.receive_responses()?,
    ));

    // Compress and add to transcript
    let compressed_poly = cubic_poly.compress();
    compressed_poly.append_to_transcript(transcript);

    // Derive challenge
    let r_i: F = transcript.challenge_scalar();
    r.push(r_i);
    polys.push(compressed_poly);

    // Evaluate for next round's claim
    *claim = cubic_poly.evaluate(&r_i);

    // Send next claim and challenge to workers
    network.broadcast_request((*claim, r_i))
}

#[inline]
#[tracing::instrument(skip_all, name = "process_eq_sumcheck_round_worker", level = "trace")]
pub fn process_eq_sumcheck_round_worker<F: JoltField, Network: Rep3NetworkWorker>(
    quadratic_evals: (F, F), // (t_i(0), t_i(infty))
    eq_poly: &mut GruenSplitEqPolynomial<F>,
    r: &mut Vec<F>,
    claim: &mut F,
    io_ctx: &mut IoContext<Network>,
) -> eyre::Result<F> {
    let scalar_times_w_i = eq_poly.current_scalar * eq_poly.w[eq_poly.current_index - 1];

    let cubic_poly = UniPoly::from_linear_times_quadratic_with_hint(
        // The coefficients of `eq(w[(n - i)..], r[..i]) * eq(w[n - i - 1], X)`
        [
            eq_poly.current_scalar - scalar_times_w_i,
            scalar_times_w_i + scalar_times_w_i - eq_poly.current_scalar,
        ],
        quadratic_evals.0,
        quadratic_evals.1,
        *claim,
    );

    // Send cubic poly to coordinator
    io_ctx.network.send_response(cubic_poly.as_vec())?;

    // Receive challenge and next claim
    let (next_claim, r_i) = io_ctx.network.receive_request()?;
    r.push(r_i);
    *claim = additive::promote_to_trivial_share(next_claim, io_ctx.network.get_id());

    // Bind eq_poly for next round
    eq_poly.bind(r_i);

    Ok(r_i)
}
