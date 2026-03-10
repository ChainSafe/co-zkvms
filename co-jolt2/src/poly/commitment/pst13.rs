use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use crate::utils::types::MaybeShared;
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::Zero;
use ark_poly_commit::multilinear_pc::data_structures::{CommitterKey, Proof};
use ark_std::test_rng;
use jolt_core::field::JoltField;
use jolt_core::msm::VariableBaseMSM;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::transcripts::Transcript;
use mpc_core::protocols::rep3::network::Rep3NetworkWorker;
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;
use std::borrow::Borrow;

// Re-export types from coordinator (canonical definitions live there).
pub use co_jolt_coordinator::poly::commitment::pst13::{
    PST13, PST13Commitment, PST13Setup, PST13VerifierSetup,
};

use super::Rep3CommitmentScheme;

// =============================================================================
// Helpers (used by worker prove_rep3)
// =============================================================================

fn open<E: Pairing>(
    ck: &CommitterKey<E>,
    polynomial: &DensePolynomial<E::ScalarField>,
    point: &[E::ScalarField],
) -> (Proof<E>, E::ScalarField)
where
    E::ScalarField: JoltField,
{
    use ark_ff::One;

    let nv = polynomial.get_num_vars();
    assert_eq!(nv, ck.nv, "Invalid size of polynomial");
    let mut r: Vec<Vec<E::ScalarField>> = (0..nv + 1).map(|_| Vec::new()).collect();
    let mut q: Vec<Vec<E::ScalarField>> = (0..nv + 1).map(|_| Vec::new()).collect();

    r[nv] = polynomial.evals();

    let mut proofs = Vec::new();
    for i in 0..nv {
        let k = nv - i;
        let point_at_k = point[i];
        q[k] = vec![E::ScalarField::zero(); 1 << (k - 1)];
        r[k - 1] = vec![E::ScalarField::zero(); 1 << (k - 1)];
        for b in 0..(1 << (k - 1)) {
            q[k][b] = r[k][(b << 1) + 1] - r[k][b << 1];
            r[k - 1][b] = r[k][b << 1] * (E::ScalarField::one() - point_at_k)
                + r[k][(b << 1) + 1] * point_at_k;
        }
        let scalars: Vec<_> = (0..(1 << k)).map(|x| q[k][x >> 1]).collect();

        let pi_g = <E::G1 as VariableBaseMSM>::msm_field_elements(&ck.powers_of_g[i], &scalars)
            .unwrap()
            .into_affine();
        proofs.push(pi_g);
    }

    (Proof { proofs }, r[0][0])
}

// =============================================================================
// Rep3CommitmentScheme implementation (worker-side)
// =============================================================================

impl<ProofTranscript> Rep3CommitmentScheme<ark_bn254::Fr, ProofTranscript>
    for PST13<ark_bn254::Bn254>
where
    ProofTranscript: Transcript,
{
    #[tracing::instrument(skip_all, name = "PST13::commit_rep3", level = "trace")]
    fn commit_rep3(
        poly: &Rep3MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
        commit_to_public: bool,
    ) -> (
        MaybeShared<Self::Commitment>,
        MaybeShared<Self::OpeningProofHint>,
    ) {
        match poly {
            Rep3MultilinearPolynomial::Public(poly) => {
                if !commit_to_public {
                    return (MaybeShared::Public(None), MaybeShared::Public(None));
                }
                match poly {
                    MultilinearPolynomial::OneHot(one_hot) => {
                        let nv = one_hot.get_num_vars();
                        let t = one_hot.nonzero_indices.len();
                        let ck = setup.ck();
                        let srs = &ck.powers_of_g[0];
                        let mut g_product = <ark_bn254::Bn254 as Pairing>::G1::zero();
                        for j in 0..t {
                            if let Some(k) = one_hot.nonzero_indices[j] {
                                g_product += srs[k as usize * t + j];
                            }
                        }
                        let commitment = PST13Commitment {
                            nv,
                            g_product: g_product.into_affine(),
                        };
                        (
                            MaybeShared::Public(Some(commitment)),
                            MaybeShared::Public(Some(())),
                        )
                    }
                    _ => {
                        let (c, hint) = <Self as CommitmentScheme>::commit(poly, setup);
                        (
                            MaybeShared::Public(Some(c)),
                            MaybeShared::Public(Some(hint)),
                        )
                    }
                }
            }
            Rep3MultilinearPolynomial::Shared(shared_poly) => match shared_poly {
                Rep3SharedPoly::Dense(poly) => {
                    let poly_a =
                        MultilinearPolynomial::LargeScalars(poly.into_distributed_commit_form());
                    let (commitment, hint) = <Self as CommitmentScheme>::commit(&poly_a, setup);
                    (MaybeShared::Shared(commitment), MaybeShared::Shared(hint))
                }
                Rep3SharedPoly::OneHot(one_hot) => {
                    let (commitment, hint) = commit_one_hot_shared(one_hot, setup);
                    (MaybeShared::Shared(commitment), MaybeShared::Shared(hint))
                }
                Rep3SharedPoly::U64Scalars(_) => {
                    panic!("PST13 commit_rep3: U64Scalars unsupported (Dory-only)")
                }
                Rep3SharedPoly::RLC(_) => {
                    unreachable!("RLC polynomials should not be committed directly")
                }
            },
        }
    }

    #[tracing::instrument(skip_all, name = "PST13::batch_commit_rep3", level = "trace")]
    fn batch_commit_rep3<U>(
        polys: &[U],
        setup: &Self::ProverSetup,
        commit_to_public: bool,
    ) -> Vec<(
        MaybeShared<Self::Commitment>,
        MaybeShared<Self::OpeningProofHint>,
    )>
    where
        U: Borrow<Rep3MultilinearPolynomial<ark_bn254::Fr>> + Sync,
    {
        polys
            .par_iter()
            .map(|p| {
                <Self as Rep3CommitmentScheme<ark_bn254::Fr, ProofTranscript>>::commit_rep3(
                    p.borrow(),
                    setup,
                    commit_to_public,
                )
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "PST13::prove_rep3", level = "trace")]
    fn prove_rep3<Network>(
        poly: &Rep3MultilinearPolynomial<ark_bn254::Fr>,
        setup: &Self::ProverSetup,
        opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        _opening_hint: Option<Self::OpeningProofHint>,
        network: &mut Network,
    ) -> eyre::Result<()>
    where
        Network: Rep3NetworkWorker,
    {
        let opening_point_rev: Vec<ark_bn254::Fr> =
            opening_point.iter().rev().map(|c| (*c).into()).collect();

        let dense_poly = match poly {
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(dense)) => dense.copy_share_a(),
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(one_hot)) => {
                materialize_one_hot_share_a(one_hot)
            }
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::U64Scalars(_)) => {
                return Err(eyre::eyre!(
                    "PST13 prove_rep3: U64Scalars unsupported (Dory-only)"
                ));
            }
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::RLC(rlc)) => {
                materialize_rlc_share_a(rlc)
            }
            Rep3MultilinearPolynomial::Public(_) => {
                return Err(eyre::eyre!("prove_rep3 does not handle public polynomials"));
            }
        };

        let ck = setup.ck();
        let (pf, _claim) = open::<ark_bn254::Bn254>(&ck, &dense_poly, &opening_point_rev);
        network.send_response(pf.proofs)
    }

    fn combine_hints_rep3(
        _hints: Vec<MaybeShared<Self::OpeningProofHint>>,
        _coeffs: &[ark_bn254::Fr],
        _party_id: PartyID,
    ) -> Self::OpeningProofHint {
    }

    fn concat_commitments(a: &Self::Commitment, b: &Self::Commitment) -> Self::Commitment {
        PST13Commitment {
            nv: a.nv,
            g_product: (a.g_product + b.g_product).into_affine(),
        }
    }
}

// =============================================================================
// Shared OneHot commit: S[k] bucket + MSM(K)
// =============================================================================

#[tracing::instrument(skip_all, name = "PST13::commit_one_hot_shared", level = "trace")]
fn commit_one_hot_shared(
    one_hot: &Rep3OneHotPolynomial<ark_bn254::Fr>,
    setup: &PST13Setup<ark_bn254::Bn254>,
) -> (PST13Commitment<ark_bn254::Bn254>, ()) {
    type G1 = <ark_bn254::Bn254 as Pairing>::G1;

    let k_size = one_hot.K;
    let t = one_hot.masked_indices_c.len();
    let nv = one_hot.get_num_vars();
    let ck = setup.ck();
    let srs = &ck.powers_of_g[0];

    let buckets: Vec<G1> = (0..k_size)
        .into_par_iter()
        .map(|k| {
            let mut acc = G1::zero();
            for j in 0..t {
                if let Some(c) = one_hot.masked_indices_c[j] {
                    let m = k ^ (c as usize);
                    acc += srs[m * t + j];
                }
            }
            acc
        })
        .collect();

    let scalars: Vec<ark_bn254::Fr> = one_hot.rand_ohv_e_field.iter().map(|s| s.a).collect();

    let buckets_affine: Vec<<ark_bn254::Bn254 as Pairing>::G1Affine> =
        G1::normalize_batch(&buckets);

    let g_product = <G1 as VariableBaseMSM>::msm_field_elements(&buckets_affine, &scalars)
        .unwrap()
        .into_affine();

    (PST13Commitment { nv, g_product }, ())
}

// =============================================================================
// Helpers for prove_rep3: materialize dense coefficient vectors
// =============================================================================

fn materialize_one_hot_share_a(
    one_hot: &Rep3OneHotPolynomial<ark_bn254::Fr>,
) -> DensePolynomial<ark_bn254::Fr> {
    let k_size = one_hot.K;
    let t = one_hot.masked_indices_c.len();
    let mut coeffs = vec![ark_bn254::Fr::zero(); k_size * t];
    for j in 0..t {
        if let Some(c) = one_hot.masked_indices_c[j] {
            for k in 0..k_size {
                coeffs[k * t + j] = one_hot.rand_ohv_e_field[k ^ (c as usize)].a;
            }
        }
    }
    DensePolynomial::new(coeffs)
}

fn materialize_rlc_share_a(
    rlc: &crate::poly::rlc_polynomial::Rep3RLCPolynomial<ark_bn254::Fr>,
) -> DensePolynomial<ark_bn254::Fr> {
    use ark_ff::Field;

    let dense_len = rlc.dense_rlc.len();
    let max_one_hot_len = rlc
        .one_hot_rlc
        .iter()
        .filter_map(|(_, poly)| match poly.as_ref() {
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(oh)) => {
                Some(oh.K * oh.masked_indices_c.len())
            }
            _ => None,
        })
        .max()
        .unwrap_or(0);

    let full_len = dense_len.max(max_one_hot_len);
    let mut coeffs = vec![ark_bn254::Fr::zero(); full_len];

    for (i, share) in rlc.dense_rlc.iter().enumerate() {
        coeffs[i] = share.a;
    }

    for (coeff, poly) in &rlc.one_hot_rlc {
        match poly.as_ref() {
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(one_hot)) => {
                let t = one_hot.masked_indices_c.len();
                let k_size = one_hot.K;
                for j in 0..t {
                    if let Some(c) = one_hot.masked_indices_c[j] {
                        for k in 0..k_size {
                            coeffs[k * t + j] +=
                                *coeff * one_hot.rand_ohv_e_field[k ^ (c as usize)].a;
                        }
                    }
                }
            }
            Rep3MultilinearPolynomial::Public(MultilinearPolynomial::OneHot(one_hot)) => {
                let t = one_hot.nonzero_indices.len();
                for j in 0..t {
                    if let Some(k_idx) = one_hot.nonzero_indices[j] {
                        coeffs[k_idx as usize * t + j] += *coeff;
                    }
                }
            }
            _ => {}
        }
    }

    DensePolynomial::new(coeffs)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use jolt_core::poly::multilinear_polynomial::PolynomialEvaluation;
    use jolt_core::transcripts::Blake2bTranscript;
    use std::iter;

    type E = ark_bn254::Bn254;
    type F = ark_bn254::Fr;
    type ProofTranscript = Blake2bTranscript;

    #[test]
    fn test_pst13_commit_prove_verify_roundtrip() {
        let mut rng = test_rng();
        let num_vars = 4;
        let setup = PST13::<E>::setup_prover(num_vars);
        let verifier_setup = PST13::<E>::setup_verifier(&setup);

        let coeffs: Vec<F> = iter::repeat_with(|| F::rand(&mut rng))
            .take(1 << num_vars)
            .collect();
        let poly = MultilinearPolynomial::<F>::from(coeffs);

        let (commitment, hint) = PST13::<E>::commit(&poly, &setup);

        let point: Vec<<F as JoltField>::Challenge> =
            iter::repeat_with(|| <F as JoltField>::Challenge::rand(&mut rng))
                .take(num_vars)
                .collect();

        let mut transcript = ProofTranscript::new(b"test");
        let proof = PST13::<E>::prove(&setup, &poly, &point, hint, &mut transcript);

        let opening = poly.evaluate(&point.iter().map(|c| (*c).into()).collect::<Vec<F>>());

        let mut transcript = ProofTranscript::new(b"test");
        PST13::<E>::verify(
            &proof,
            &verifier_setup,
            &mut transcript,
            &point,
            &opening,
            &commitment,
        )
        .unwrap();
    }

    #[test]
    fn test_combine_commitments() {
        use ark_ff::One;
        let mut rng = test_rng();
        let setup = PST13::<E>::setup_prover(3);

        let rho = F::rand(&mut rng);
        let mut rho_powers = vec![F::one()];
        for i in 1..3 {
            rho_powers.push(rho_powers[i - 1] * rho);
        }

        let polys: Vec<_> = iter::repeat_with(|| {
            MultilinearPolynomial::<F>::from(
                iter::repeat_with(|| F::rand(&mut rng))
                    .take(1 << 3)
                    .collect::<Vec<_>>(),
            )
        })
        .take(3)
        .collect();

        let commitments: Vec<_> = polys
            .iter()
            .map(|p| PST13::<E>::commit(p, &setup).0)
            .collect();

        let combined = PST13::<E>::combine_commitments(&commitments, &rho_powers);

        let n = 1 << 3;
        let mut agg_coeffs = vec![F::zero(); n];
        for (poly, &coeff) in polys.iter().zip(rho_powers.iter()) {
            if let MultilinearPolynomial::LargeScalars(dp) = poly {
                for (j, val) in dp.Z.iter().enumerate() {
                    agg_coeffs[j] += coeff * val;
                }
            }
        }
        let agg_poly = MultilinearPolynomial::<F>::from(agg_coeffs);
        let (agg_commitment, _) = PST13::<E>::commit(&agg_poly, &setup);

        assert_eq!(combined, agg_commitment);
    }

    #[test]
    fn test_concat_commitments() {
        let mut rng = test_rng();
        let setup = PST13::<E>::setup_prover(4);

        let v1 = vec![F::from(1u64); 8];
        let v2 = vec![F::from(2u64); 8];
        let mut v1_ = vec![F::from(0u64); 16];
        v1_.splice(0..8, v1.clone());
        let mut v2_ = vec![F::from(0u64); 16];
        v2_.splice(8..16, v2.clone());
        let p1 = MultilinearPolynomial::<F>::from(v1_);
        let p2 = MultilinearPolynomial::<F>::from(v2_);
        let p = MultilinearPolynomial::<F>::from([v1, v2].concat());

        let (c1, _) = PST13::<E>::commit(&p1, &setup);
        let (c2, _) = PST13::<E>::commit(&p2, &setup);
        let (c, _) = PST13::<E>::commit(&p, &setup);

        let c_check = PST13Commitment {
            nv: c1.nv,
            g_product: (c1.g_product + c2.g_product).into_affine(),
        };

        assert_eq!(c_check, c);
    }
}
