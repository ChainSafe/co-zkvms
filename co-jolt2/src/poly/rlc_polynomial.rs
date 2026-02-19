use std::sync::Arc;

use ark_ec::CurveGroup;
use jolt_core::msm::VariableBaseMSM;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::utils::small_scalar::SmallScalar;
use mpc_core::protocols::rep3::{self, PartyID, Rep3PrimeFieldShare};
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::{Rep3MultilinearPolynomial, Rep3SharedPoly};

#[derive(Default, Clone, Debug)]
pub struct Rep3RLCPolynomial<F: JoltField> {
    /// Random linear combination of dense (i.e. length T) polynomials.
    pub dense_rlc: Vec<Rep3PrimeFieldShare<F>>,
    /// Random linear combination of one-hot polynomials (length T x K for some K).
    /// We store a vector of (coefficient, polynomial) pairs and lazily handle the
    /// linear combination in `commit_rows`.
    pub one_hot_rlc: Vec<(F, Arc<Rep3MultilinearPolynomial<F>>)>,
}

impl<F: JoltField> Rep3RLCPolynomial<F> {
    pub fn new() -> Self {
        Self {
            dense_rlc: vec![Rep3PrimeFieldShare::zero_share(); DoryGlobals::get_T()],
            one_hot_rlc: vec![],
        }
    }

    #[tracing::instrument(skip_all, name = "Rep3RLCPoly::linear_combination")]
    pub fn linear_combination(
        polynomials: Vec<Arc<Rep3MultilinearPolynomial<F>>>,
        coefficients: &[F],
        party_id: PartyID,
    ) -> Self {
        debug_assert_eq!(polynomials.len(), coefficients.len());

        let mut result = Rep3RLCPolynomial::<F>::new();

        let dense_indices: Vec<usize> = polynomials
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                !matches!(
                    p.as_ref(),
                    Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(_))
                )
            })
            .map(|(i, _)| i)
            .collect();

        if polynomials.iter().any(|p| {
            matches!(
                p.as_ref(),
                Rep3MultilinearPolynomial::Public(MultilinearPolynomial::OneHot(_))
            )
        }) {
            panic!("Public OneHot polynomials are not supported in Rep3RLCPolynomial");
        }

        if !dense_indices.is_empty() {
            let dense_len = result.dense_rlc.len();

            result.dense_rlc = (0..dense_len)
                .into_par_iter()
                .map(|i| {
                    let mut acc = Rep3PrimeFieldShare::<F>::zero_share();
                    for &poly_idx in &dense_indices {
                        let poly = polynomials[poly_idx].as_ref();
                        let coeff = coefficients[poly_idx];

                        match poly {
                            Rep3MultilinearPolynomial::Public(poly) => {
                                if i < poly.original_len() {
                                    let term = match poly {
                                        MultilinearPolynomial::U8Scalars(p) => {
                                            p.coeffs[i].field_mul(coeff)
                                        }
                                        MultilinearPolynomial::U16Scalars(p) => {
                                            p.coeffs[i].field_mul(coeff)
                                        }
                                        MultilinearPolynomial::U32Scalars(p) => {
                                            p.coeffs[i].field_mul(coeff)
                                        }
                                        MultilinearPolynomial::U64Scalars(p) => {
                                            p.coeffs[i].field_mul(coeff)
                                        }
                                        MultilinearPolynomial::I64Scalars(p) => {
                                            p.coeffs[i].field_mul(coeff)
                                        }
                                        MultilinearPolynomial::U128Scalars(p) => {
                                            p.coeffs[i].field_mul(coeff)
                                        }
                                        MultilinearPolynomial::I128Scalars(p) => {
                                            p.coeffs[i].field_mul(coeff)
                                        }
                                        MultilinearPolynomial::S128Scalars(p) => {
                                            p.coeffs[i].field_mul(coeff)
                                        }
                                        MultilinearPolynomial::LargeScalars(p) => {
                                            p.evals_ref()[i] * coeff
                                        }
                                        _ => unreachable!("Unexpected public polynomial variant"),
                                    };

                                    acc +=
                                        Rep3PrimeFieldShare::promote_from_trivial(&term, party_id);
                                }
                            }
                            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(p)) => {
                                let in_range = match p.global_chunk_range {
                                    Some((start, end)) => (i >= start) && (i < end),
                                    None => i < p.full_len(),
                                };
                                if !in_range {
                                    continue;
                                }

                                let share = match p.global_chunk_range {
                                    Some((start, _end)) => p.coeffs_ref()[i - start],
                                    None => p.coeffs_ref()[i],
                                };
                                acc += rep3::arithmetic::mul_public(share, coeff);
                            }
                            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(_)) => {
                                unreachable!("OneHot polynomials excluded from dense_indices")
                            }
                        }
                    }
                    acc
                })
                .collect();
        }

        for (i, poly) in polynomials.into_iter().enumerate() {
            if matches!(
                poly.as_ref(),
                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(_))
            ) {
                result.one_hot_rlc.push((coefficients[i], poly));
            }
        }

        result
    }

    #[tracing::instrument(skip_all, name = "Rep3RLCPolynomial::commit_rows")]
    pub fn commit_rows<G>(&self, bases: &[G::Affine]) -> eyre::Result<Vec<G>>
    where
        G: CurveGroup<ScalarField = F> + VariableBaseMSM + Send + Sync,
    {
        let num_rows = DoryGlobals::get_max_num_rows();
        let row_len = DoryGlobals::get_num_columns();

        let mut row_commitments = vec![G::zero(); num_rows];

        // Dense part: MSM against this party's additive share `a`.
        row_commitments.par_iter_mut().enumerate().try_for_each(
            |(row_idx, commitment)| -> eyre::Result<()> {
                let start = row_idx * row_len;
                if start >= self.dense_rlc.len() {
                    return Ok(());
                }
                let end = (start + row_len).min(self.dense_rlc.len());
                let dense_row = &self.dense_rlc[start..end];

                let scalars: Vec<F> = dense_row.iter().map(|s| s.a).collect();
                let msm_result: G = G::msm_field_elements(&bases[..scalars.len()], &scalars)
                    .map_err(|e| eyre::eyre!("msm_field_elements failed: {e:?}"))?;
                *commitment += msm_result;
                Ok(())
            },
        )?;

        // One-hot part: compute one-hot row commitment shares and scale-add by coefficient.
        for (coeff, poly) in self.one_hot_rlc.iter() {
            let mut rows: Vec<G> = match poly.as_ref() {
                Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::OneHot(one_hot)) => {
                    one_hot.commit_rows::<G>(bases)?
                }
                _ => {
                    eyre::bail!("Expected Shared(OneHot) polynomial in one_hot_rlc");
                }
            };

            rows.resize(num_rows, G::zero());

            for (dst, src) in row_commitments.iter_mut().zip(rows.into_iter()) {
                *dst += src * (*coeff);
            }
        }

        Ok(row_commitments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::test_rng;
    use ark_std::UniformRand;
    use ark_std::Zero;
    use jolt_core::ark_bn254::{Fr, G1Affine, G1Projective};
    use jolt_core::poly::dense_mlpoly::DensePolynomial;

    fn share_poly_rep3<F: JoltField>(
        coeffs: &[F],
        rng: &mut impl rand::Rng,
    ) -> [Vec<Rep3PrimeFieldShare<F>>; 3] {
        let mut party_coeffs: [Vec<Rep3PrimeFieldShare<F>>; 3] =
            std::array::from_fn(|_| Vec::with_capacity(coeffs.len()));

        for &c in coeffs {
            let shares = mpc_core::protocols::rep3::arithmetic::generate_shares_rep3(c, rng);
            party_coeffs[0].push(shares[0]);
            party_coeffs[1].push(shares[1]);
            party_coeffs[2].push(shares[2]);
        }

        party_coeffs
    }

    #[test]
    fn linear_combination_dense_correct() {
        let mut rng = test_rng();
        crate::poly::commitment::dory::test_support::init_dory_globals(256, 32);
        let t = DoryGlobals::get_T();
        assert_eq!(t, 32);

        let a = Fr::rand(&mut rng);
        let b = Fr::rand(&mut rng);

        let public_coeffs = (0..t).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let shared_plain = (0..t).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        let public_poly = Arc::new(Rep3MultilinearPolynomial::public(
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(public_coeffs.clone())),
        ));

        let shared_party_coeffs = share_poly_rep3(&shared_plain, &mut rng);
        let shared_polys: [Arc<Rep3MultilinearPolynomial<Fr>>; 3] = std::array::from_fn(|pid| {
            Arc::new(Rep3MultilinearPolynomial::from_shared_coeffs(
                shared_party_coeffs[pid].clone(),
            ))
        });

        let rlc0 = Rep3RLCPolynomial::linear_combination(
            vec![public_poly.clone(), shared_polys[0].clone()],
            &[a, b],
            PartyID::ID0,
        );
        let rlc1 = Rep3RLCPolynomial::linear_combination(
            vec![public_poly.clone(), shared_polys[1].clone()],
            &[a, b],
            PartyID::ID1,
        );
        let rlc2 = Rep3RLCPolynomial::linear_combination(
            vec![public_poly.clone(), shared_polys[2].clone()],
            &[a, b],
            PartyID::ID2,
        );

        let reconstructed = mpc_core::protocols::rep3::combine_field_elements::<Fr>(
            &rlc0.dense_rlc,
            &rlc1.dense_rlc,
            &rlc2.dense_rlc,
        );

        let expected = public_coeffs
            .iter()
            .zip(shared_plain.iter())
            .map(|(&p, &s)| a * p + b * s)
            .collect::<Vec<_>>();

        assert_eq!(reconstructed, expected);
    }

    #[test]
    fn commit_rows_dense_only_correct() {
        let mut rng = test_rng();
        crate::poly::commitment::dory::test_support::init_dory_globals(256, 32);
        let t = DoryGlobals::get_T();
        let row_len = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();

        let dense_plain = (0..t).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let dense_party_coeffs = share_poly_rep3(&dense_plain, &mut rng);

        let rlc_party: [Rep3RLCPolynomial<Fr>; 3] = std::array::from_fn(|pid| Rep3RLCPolynomial {
            dense_rlc: dense_party_coeffs[pid].clone(),
            one_hot_rlc: vec![],
        });

        let bases_proj = (0..row_len)
            .map(|_| G1Projective::rand(&mut rng))
            .collect::<Vec<_>>();
        let bases: Vec<G1Affine> = bases_proj.iter().map(|p| p.into_affine()).collect();

        let rows0 = rlc_party[0].commit_rows::<G1Projective>(&bases).unwrap();
        let rows1 = rlc_party[1].commit_rows::<G1Projective>(&bases).unwrap();
        let rows2 = rlc_party[2].commit_rows::<G1Projective>(&bases).unwrap();

        let mut reconstructed = vec![G1Projective::zero(); num_rows];
        for i in 0..num_rows {
            reconstructed[i] = rows0[i] + rows1[i] + rows2[i];
        }

        let mut expected = vec![G1Projective::zero(); num_rows];
        for (row_idx, dense_row) in dense_plain.chunks(row_len).enumerate() {
            let msm = G1Projective::msm_field_elements(&bases[..dense_row.len()], dense_row)
                .expect("msm_field_elements");
            expected[row_idx] += msm;
        }

        assert_eq!(reconstructed, expected);
    }
}
