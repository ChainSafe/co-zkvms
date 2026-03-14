//! Point share operations for Rep3.
//!
//! Contains dot product operations using daPoint preprocessing tuples.

use crate::preprocessing::daPoint::DaPointsBatch;
use crate::protocols::rep3::PartyID;
use crate::protocols::rep3::network::{IoContext, Rep3Network};
use crate::protocols::rep3_ring::Rep3RingShare;
use crate::protocols::rep3_ring::ring::bit::Bit;
use crate::protocols::rep3_ring::ring::ring_impl::RingElement;
use ark_ec::CurveGroup;
use itertools::izip;

/// Securely compute this party's additive share of `Σ_i bits[i] · qs[i]`
/// using gamma-based daPoint tuples (no scalar muls online).
///
/// **Communication:** 1 round — P0 broadcasts N bits to P1 and P2.
/// Returns `C` (additive share): P0 contributes 0, P1 contributes X1, P2 contributes X2.
pub fn dot_product_dapoints<C, N>(
    bits: &[Rep3RingShare<Bit>],
    qs: &[C],
    batch: &DaPointsBatch<C>,
    io: &mut IoContext<N>,
) -> eyre::Result<C>
where
    C: CurveGroup,
    N: Rep3Network,
{
    let n = bits.len();
    eyre::ensure!(qs.len() == n, "dot_product_dapoints: qs length mismatch");
    eyre::ensure!(
        batch.alphas.len() == n || (io.id == PartyID::ID0 && batch.alphas.is_empty()),
        "dot_product_dapoints: batch.alphas length mismatch"
    );
    eyre::ensure!(
        batch.gammas.len() == n || (io.id != PartyID::ID0 && batch.gammas.len() == n),
        "dot_product_dapoints: batch.gammas length mismatch"
    );
    if n == 0 {
        return Ok(C::zero());
    }

    // Round 1: P0 broadcasts masked bits m[i] = x.a ^ x.b ^ gamma[i].
    let ms: Vec<RingElement<Bit>> = if io.id == PartyID::ID0 {
        let ms: Vec<RingElement<Bit>> = izip!(bits, &batch.gammas)
            .map(|(x, gamma)| RingElement(Bit::new(x.a.0.convert() ^ x.b.0.convert() ^ gamma.convert())))
            .collect();
        io.network.send_many(PartyID::ID1, &ms)?;
        io.network.send_many(PartyID::ID2, &ms)?;
        ms
    } else {
        io.network.recv_many(PartyID::ID0)?
    };

    // P0 contributes nothing.
    if io.id == PartyID::ID0 {
        return Ok(C::zero());
    }

    // P1/P2 compute beta and accumulate.
    let mut acc = C::zero();
    for (i, (m, x)) in ms.iter().zip(bits).enumerate() {
        let missing = match io.id {
            PartyID::ID1 => x.a.0,
            PartyID::ID2 => x.b.0,
            _ => unreachable!(),
        };
        let beta = m.0.convert() ^ missing.convert();

        let alpha = &batch.alphas[i];
        if beta {
            // beta=1: x_i * Q_i = Q_i - Gamma_i
            if io.id == PartyID::ID1 {
                acc += qs[i] - *alpha;
            } else {
                acc -= *alpha;
            }
        } else {
            // beta=0: x_i * Q_i = Gamma_i
            acc += *alpha;
        }
    }

    Ok(acc)
}
