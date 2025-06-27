use std::collections::LinkedList;
use std::io::empty;
use std::marker::PhantomData;
use std::ops::Div;
use std::ops::Index;
use std::ops::Mul;

use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ec::bls12::Bls12;
use ark_ec::pairing::Pairing;
use ark_ec::scalar_mul::fixed_base::FixedBase;
use ark_ec::AffineRepr;
use ark_ec::CurveGroup;
use ark_ec::VariableBaseMSM;
use ark_ff::Field;
use ark_ff::{One, PrimeField, UniformRand, Zero};
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverState;
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ZKProverState;
use ark_linear_sumcheck::ml_sumcheck::protocol::verifier::SubClaim;
use ark_linear_sumcheck::ml_sumcheck::protocol::ListOfProductsOfPolynomials;
use ark_linear_sumcheck::ml_sumcheck::protocol::PolynomialInfo;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_poly::evaluations;
use ark_poly::multivariate::{SparsePolynomial, SparseTerm, Term};
use ark_poly::polynomial;
use ark_poly::univariate::DensePolynomial;
use ark_poly::MultilinearExtension;
use ark_poly::Polynomial;
use ark_poly::{DenseMVPolynomial, DenseMultilinearExtension};
use ark_poly_commit::challenge::ChallengeGenerator;
use ark_poly_commit::marlin_pc::Commitment as MaskCommitment;
use ark_poly_commit::marlin_pc::CommitterKey;
use ark_poly_commit::marlin_pst13_pc::CommitterKey as MaskCommitterKey;
use ark_poly_commit::marlin_pst13_pc::MarlinPST13;
use ark_poly_commit::marlin_pst13_pc::Proof as MaskProof;
use ark_poly_commit::marlin_pst13_pc::Randomness;
use ark_poly_commit::marlin_pst13_pc::UniversalParams as MaskParam;
use ark_poly_commit::marlin_pst13_pc::VerifierKey as MaskVerifierKey;
use ark_poly_commit::multilinear_pc;
use ark_poly_commit::multilinear_pc::data_structures::Commitment as MLCommitment;
use ark_poly_commit::multilinear_pc::data_structures::CommitterKey as MLCommitterKey;
use ark_poly_commit::multilinear_pc::data_structures::Proof as MLProof;
use ark_poly_commit::multilinear_pc::data_structures::UniversalParams as MLPCParam;
use ark_poly_commit::multilinear_pc::data_structures::VerifierKey as MLVerifierKey;
use ark_poly_commit::multilinear_pc::MultilinearPC;
use ark_poly_commit::Error;
use ark_poly_commit::LabeledCommitment;
use ark_poly_commit::LabeledPolynomial;
use ark_poly_commit::PCRandomness;
use ark_poly_commit::PolynomialLabel;
use ark_std::{cfg_into_iter, test_rng};
use rayon::iter::*;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
// use ark_poly_commit::kzg10::Randomness;
// use ark_poly_commit::kzg10::Commitment;
// use ark_poly_commit::kzg10::KZG10;
// use ark_poly_commit::kzg10::Powers;
// use ark_poly_commit::kzg10::UniversalParams;
// use ark_poly_commit::marlin_pc::CommitterKey;
// use ark_poly_commit::kzg10::VerifierKey;
use ark_poly_commit::{DenseUVPolynomial, PolynomialCommitment};
// use ark_poly_commit::marlin_pc::MarlinKZG10;
use ark_linear_sumcheck::ml_sumcheck::MLSumcheck;
use ark_std::{end_timer, start_timer};
use blake2::digest::generic_array::typenum::Sum;
use rand::Rng;
use rand::RngCore;

use crate::errors;
use crate::snark::SumcheckProof;

use super::prover::ProverMessage;

type GOpen<E>
where
    E: Pairing,
= ark_poly_commit::marlin_pst13_pc::Proof<E>;

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone)]
pub struct ZKSumcheckProof<E: Pairing> {
    pub g_commit: MaskCommitment<E>,
    pub sumcheck_proof: SumcheckProof<E::ScalarField>,
    pub poly_info: PolynomialInfo,
    pub g_proof: MaskProof<E>,
    pub g_value: E::ScalarField,
}

pub struct SpecMultiCommit<E: Pairing, P: DenseMVPolynomial<E::ScalarField>> {
    _engine: PhantomData<E>,
    _poly: PhantomData<P>,
}

/// Generate 'num_variables' univariate mask polynomials with degree 'deg'.
/// The mask multivariate polynomial formed by univariate polynomials sums 0 on hypercube.
pub(crate) fn generate_mask_polynomial<F: Field>(
    mask_rng: &mut impl RngCore,
    num_variables: usize,
    deg: usize,
    sum_to_zero: bool,
) -> SparsePolynomial<F, SparseTerm> {
    let mut mask_polynomials: Vec<Vec<F>> = Vec::new();
    let mut sum_g = F::zero();
    for _ in 0..num_variables {
        let mut mask_poly = Vec::<F>::with_capacity(deg + 1);
        mask_poly.push(F::rand(mask_rng));
        sum_g += mask_poly[0] + mask_poly[0];
        for i in 1..deg + 1 {
            mask_poly.push(F::rand(mask_rng));
            sum_g += mask_poly[i];
        }
        mask_polynomials.push(mask_poly);
    }
    if sum_to_zero {
        mask_polynomials[0][0] -= sum_g / F::from(2u8);
    }
    let mut terms: Vec<(F, SparseTerm)> = Vec::new();
    for (var, variables_coef) in mask_polynomials.iter().enumerate() {
        variables_coef
            .iter()
            .enumerate()
            .for_each(|(degree, coef)| {
                terms.push((coef.clone(), SparseTerm::new(vec![(var, degree)])))
            });
    }

    SparsePolynomial::from_coefficients_vec(num_variables, terms)
}
pub fn get_polynomial<F: Field>(
    deg: usize,
    num_variables: usize,
) -> SparsePolynomial<F, SparseTerm> {
    let mut rng = Blake2s512Rng::setup();
    get_polynomial_with_rng(deg, num_variables, &mut rng)
}

#[inline]
pub fn get_polynomial_with_rng<F: Field>(
    deg: usize,
    num_variables: usize,
    rng: &mut impl Rng,
) -> SparsePolynomial<F, SparseTerm> {
    generate_mask_polynomial(rng, num_variables, deg, true)
}

// ZK Commitment for SparsePolynomial generated in function 'generate_mask_polynomial'. Only setup need to be rewritten,
impl<E, P> SpecMultiCommit<E, P>
where
    E: Pairing,
    P: DenseMVPolynomial<E::ScalarField> + Sync,
    P::Point: Index<usize, Output = E::ScalarField>,
{
    fn divide_at_point(p: &P, point: &P::Point) -> Vec<P>
    where
        P::Point: Index<usize, Output = E::ScalarField>,
    {
        let num_vars = p.num_vars();
        if p.is_zero() {
            return vec![P::zero(); num_vars];
        }
        let mut quotients = Vec::with_capacity(num_vars);
        // `cur` represents the current dividend
        let mut cur = p.clone();
        // Divide `cur` by `X_i - z_i`
        for i in 0..num_vars {
            let mut quotient_terms = Vec::new();
            let mut remainder_terms = Vec::new();
            for (mut coeff, term) in cur.terms() {
                // Since the final remainder is guaranteed to be 0, all the constant terms
                // cancel out so we don't need to keep track of them
                if term.is_constant() {
                    continue;
                }
                // If the current term contains `X_i` then divide appropiately,
                // otherwise add it to the remainder
                let mut term_vec = (&*term).to_vec();
                match term_vec.binary_search_by(|(var, _)| var.cmp(&i)) {
                    Ok(idx) => {
                        // Repeatedly divide the term by `X_i - z_i` until the remainder
                        // doesn't contain any `X_i`s
                        while term_vec[idx].1 > 1 {
                            // First divide by `X_i` and add the term to the quotient
                            term_vec[idx] = (i, term_vec[idx].1 - 1);
                            quotient_terms.push((coeff, P::Term::new(term_vec.clone())));
                            // Then compute the remainder term in-place
                            coeff *= &point[i];
                        }
                        // Since `X_i` is power 1, we can remove it entirely
                        term_vec.remove(idx);
                        quotient_terms.push((coeff, P::Term::new(term_vec.clone())));
                        remainder_terms.push((point[i] * &coeff, P::Term::new(term_vec)));
                    }
                    Err(_) => remainder_terms.push((coeff, term.clone())),
                }
            }
            quotients.push(P::from_coefficients_vec(num_vars, quotient_terms));
            // Set the current dividend to be the remainder of this division
            cur = P::from_coefficients_vec(num_vars, remainder_terms);
        }
        quotients
    }

    fn convert_to_bigints(p: &P) -> Vec<<E::ScalarField as PrimeField>::BigInt> {
        let plain_coeffs = ark_std::cfg_into_iter!(p.terms())
            .map(|(coeff, _)| coeff.into_bigint())
            .collect();
        plain_coeffs
    }

    pub fn special_setup<R: RngCore>(
        max_degree: usize,
        num_vars: Option<usize>,
        rng: &mut R,
    ) -> Result<MaskParam<E, P>, Error> {
        let num_vars = num_vars.ok_or(Error::InvalidNumberOfVariables)?;
        if num_vars < 1 {
            return Err(Error::InvalidNumberOfVariables);
        }
        if max_degree < 1 {
            return Err(Error::DegreeIsZero);
        }
        let setup_time = start_timer!(|| format!(
            "MarlinPST13::Setup with {} variables and max degree {}",
            num_vars, max_degree
        ));
        // Trapdoor evaluation points
        let mut betas = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            betas.push(E::ScalarField::rand(rng));
        }
        let result = Self::special_setup_with_beta(max_degree, Some(num_vars), rng, &betas);
        end_timer!(setup_time);
        result
    }

    fn special_setup_with_beta<R: RngCore>(
        max_degree: usize,
        num_vars: Option<usize>,
        rng: &mut R,
        betas: &Vec<E::ScalarField>,
    ) -> Result<MaskParam<E, P>, Error> {
        let num_vars = num_vars.ok_or(Error::InvalidNumberOfVariables)?;
        if num_vars < 1 {
            return Err(Error::InvalidNumberOfVariables);
        }
        if max_degree < 1 {
            return Err(Error::DegreeIsZero);
        }
        let setup_time = start_timer!(|| format!(
            "MarlinPST13::Setup with {} variables and max degree {}",
            num_vars, max_degree
        ));
        //let num_vars = 10;
        // Trapdoor evaluation points
        let betas = betas.clone();
        //let betas = &betas[2..];
        //let betas = betas.to_vec();
        // Generators
        //let g = special_g.unwrap_or_else(||E::G1::rand(rng));
        let g = E::G1::rand(rng);
        let gamma_g = E::G1::rand(rng);
        let h = E::G2::rand(rng);

        // Generate all possible monomials with `1 <= degree <= max_degree`
        let (powers_of_beta, mut powers_of_beta_terms): (Vec<_>, Vec<_>) = (1..=max_degree)
            .flat_map(|degree| {
                // Sample all combinations of `degree` variables from `variable_set`
                let terms: Vec<Vec<usize>> = (0..num_vars).map(|var| vec![var; degree]).collect();
                // For each multiset in `terms` evaluate the corresponding monomial at the
                // trapdoor and generate a `P::Term` object to index it
                ark_std::cfg_into_iter!(terms)
                    .map(|term| {
                        let value: E::ScalarField = term.iter().map(|e| betas[*e]).product();
                        let term = (0..num_vars)
                            .map(|var| (var, term.iter().filter(|e| **e == var).count()))
                            .collect();
                        (value, P::Term::new(term))
                    })
                    .collect::<Vec<_>>()
            })
            .unzip();

        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;
        let g_time = start_timer!(|| "Generating powers of G");
        let window_size = FixedBase::get_mul_window_size(max_degree + 1);
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g);
        let mut powers_of_g =
            FixedBase::msm::<E::G1>(scalar_bits, window_size, &g_table, &powers_of_beta);
        powers_of_g.push(g);
        powers_of_beta_terms.push(P::Term::new(vec![]));
        end_timer!(g_time);

        let gamma_g_time = start_timer!(|| "Generating powers of gamma * G");
        let window_size = FixedBase::get_mul_window_size(max_degree + 2);
        let gamma_g_table = FixedBase::get_window_table(scalar_bits, window_size, gamma_g);
        // Each element `i` of `powers_of_gamma_g` is a vector of length `max_degree+1`
        // containing `betas[i]^j \gamma G` for `j` from 1 to `max_degree+1` to support
        // up to `max_degree` queries
        let mut powers_of_gamma_g = vec![Vec::new(); num_vars];
        ark_std::cfg_iter_mut!(powers_of_gamma_g)
            .enumerate()
            .for_each(|(i, v)| {
                let mut powers_of_beta = Vec::with_capacity(max_degree);
                let mut cur = E::ScalarField::one();
                for _ in 0..=max_degree {
                    cur *= &betas[i];
                    powers_of_beta.push(cur);
                }
                *v = FixedBase::msm::<E::G1>(
                    scalar_bits,
                    window_size,
                    &gamma_g_table,
                    &powers_of_beta,
                );
            });
        end_timer!(gamma_g_time);

        let powers_of_g = E::G1::normalize_batch(&powers_of_g);
        let gamma_g = gamma_g.into_affine();
        let powers_of_gamma_g = powers_of_gamma_g
            .into_iter()
            .map(|v| E::G1::normalize_batch(&v))
            .collect();
        let beta_h: Vec<_> = betas.iter().map(|b| h.mul(b).into_affine()).collect();
        let h = h.into_affine();
        let prepared_h = h.into();
        let prepared_beta_h = beta_h.iter().map(|bh| (*bh).into()).collect();

        // Convert `powers_of_g` to a BTreeMap indexed by `powers_of_beta_terms`
        let powers_of_g = powers_of_beta_terms
            .into_iter()
            .zip(powers_of_g.into_iter())
            .collect();

        let pp = MaskParam {
            num_vars,
            max_degree,
            powers_of_g,
            gamma_g,
            powers_of_gamma_g,
            h,
            beta_h,
            prepared_h,
            prepared_beta_h,
        };
        end_timer!(setup_time);
        Ok(pp)
    }

    pub fn special_open<'a>(
        ck: &MaskCommitterKey<E, P>,
        labeled_polynomial: &LabeledPolynomial<E::ScalarField, P>,
        _commitments: impl IntoIterator<Item = &'a LabeledCommitment<MaskCommitment<E>>>,
        point: &P::Point,
        //rand: &'a Randomness<E, P>,
        _rng: Option<&mut dyn RngCore>,
    ) -> MaskProof<E>
    where
        P: 'a,
        Randomness<E, P>: 'a,
        MaskCommitment<E>: 'a,
    {
        // Compute random linear combinations of committed polynomials and randomness
        let mut p: P = P::zero();
        p += (E::ScalarField::one(), labeled_polynomial.polynomial());
        //let mut r = Randomness::empty() + rand;

        let open_time = start_timer!(|| format!("Opening polynomial of degree {}", p.degree()));
        let witness_time = start_timer!(|| "Computing witness polynomials");
        let witnesses = Self::divide_at_point(&p, point);
        /*let hiding_witnesses = if r.is_hiding() {
            Some(Self::divide_at_point(&r.blinding_polynomial, point))
        } else {
            None
        };*/
        end_timer!(witness_time);

        let witness_comm_time = start_timer!(|| "Computing commitment to witness polynomials");
        let w = witnesses
            .iter()
            .map(|w| {
                // Get the powers of `G` corresponding to the witness poly
                let powers_of_g = ark_std::cfg_iter!(w.terms())
                    .map(|(_, term)| *ck.powers_of_g.get(term).unwrap())
                    .collect::<Vec<_>>();
                // Convert coefficients to BigInt
                let witness_ints = Self::convert_to_bigints(&w);
                // Compute MSM
                <E::G1 as VariableBaseMSM>::msm_bigint(&powers_of_g, witness_ints.as_slice())
            })
            .collect::<Vec<_>>();
        end_timer!(witness_comm_time);

        // If the evaluation should be hiding, compute the MSM for `hiding_witnesses` and add
        // to the `w`. Additionally, compute the evaluation of `r` at `point`.
        /*let random_v = if let Some(hiding_witnesses) = hiding_witnesses {
            let witness_comm_time =
                start_timer!(|| "Computing commitment to hiding witness polynomials");
            ark_std::cfg_iter_mut!(w)
                .enumerate()
                .for_each(|(i, witness)| {
                    let hiding_witness = &hiding_witnesses[i];
                    // Get the powers of `\gamma G` corresponding to the terms of `hiding_witness`
                    let powers_of_gamma_g = hiding_witness
                        .terms()
                        .iter()
                        .map(|(_, term)| {
                            // Implicit Assumption: Each monomial in `hiding_witness` is univariate
                            let vars = term.vars();
                            match term.is_constant() {
                                true => ck.gamma_g,
                                false => ck.powers_of_gamma_g[vars[0]][term.degree() - 1],
                            }
                        })
                        .collect::<Vec<_>>();
                    // Convert coefficients to BigInt
                    let hiding_witness_ints = Self::convert_to_bigints(hiding_witness);
                    // Compute MSM and add result to witness
                    *witness += &<E::G1 as VariableBaseMSM>::msm_bigint(
                        &powers_of_gamma_g,
                        &hiding_witness_ints,
                    );
                });
            end_timer!(witness_comm_time);
            Some(r.blinding_polynomial.evaluate(point))
        } else {
            None
        };*/
        end_timer!(open_time);
        MaskProof {
            w: w.into_iter().map(|w| w.into_affine()).collect(),
            random_v: None, //random_v,
        }
    }
    /* pub fn setup<R: RngCore>(max_deg: usize, num_variables: usize, rng: &mut R) -> Vec<UniversalParams<E>>{
        let mut params = Vec::with_capacity(num_variables);
        for _ in 0..num_variables{
            params.push(KZG10::setup(max_deg, false, rng).unwrap())
        }
        params
    }

    pub(crate) fn trim_for_one(supported_deg: usize, param: &UniversalParams<E>, hiding_bound: usize) -> (Powers<E>, VerifierKey<E>){
        let powers_of_g = param.powers_of_g[..=supported_deg].to_vec();
        let powers_of_gamma_g = (0..=hiding_bound + 1)
        .map(|i| param.powers_of_gamma_g[&i])
        .collect::<Vec<_>>();
        let ck = Powers{
            powers_of_g: powers_of_g.into(),
            powers_of_gamma_g: powers_of_gamma_g.into()
        };
        let vk = VerifierKey {
            g: param.powers_of_g[0],
            gamma_g: param.powers_of_gamma_g[&0],
            h: param.h,
            beta_h: param.beta_h,
            prepared_h: param.prepared_h.clone(),
            prepared_beta_h: param.prepared_beta_h.clone(),
        };
        (ck, vk)
    }

    pub fn trim(params: &Vec<UniversalParams<E>>, deg: usize, hiding_bound: usize) -> (Vec<Powers<E>>, Vec<VerifierKey<E>>){
        let mut committer_keys: Vec<Powers<E>> = Vec::with_capacity(params.len());
        let mut verifier_keys: Vec<VerifierKey<E>> = Vec::with_capacity(params.len());
        for param in params{
            let (ck, vk) = Self::trim_for_one(deg, param, hiding_bound);
            committer_keys.push(ck);
            verifier_keys.push(vk);
        }
        (committer_keys, verifier_keys)
    }

    pub(crate) fn commit_for_one(ck: &Powers<E>, polynomial: &P, hiding_bound: usize, rng: Option<&mut dyn RngCore>)
            -> (Commitment<E>, Randomness<E::ScalarField, P>){
        KZG10::commit(ck, &polynomial, Some(hiding_bound), rng).unwrap()
    }

    pub fn commit(cks: &Vec<Powers<E>>, polynomials: &Vec<P>, hiding_bound: usize, rng: Option<&mut dyn RngCore>)
            -> (Commitment<E>, Vec<Randomness<E::ScalarField, P>>){
           let mut random_vec: Vec<Randomness<E::ScalarField, P>> = Vec::with_capacity(cks.len());
           let (mut sum, random_0) = Self::commit_for_one(&cks[0], &polynomials[0], hiding_bound, rng);
           random_vec.push(random_0);
           for i in 1..cks.len(){
               let (com, random) = Self::commit_for_one(&cks[i], &polynomials[i], hiding_bound, rng);
               sum += (E::ScalarField::from(1u8), &com);
               random_vec.push(random);
           }
           (sum, random_vec)
    }

    pub fn open(){

    } */
}

pub type ZKMLUniversalParam<E, P> = (MLPCParam<E>, MaskParam<E, P>);
pub type ZKMLCommitterKey<E, P> = (MLCommitterKey<E>, MaskCommitterKey<E, P>);
pub type ZKMLVerifierKey<E> = (MLVerifierKey<E>, MaskVerifierKey<E>);
pub type ZKMLProof<E: Pairing> = (MLProof<E>, E::ScalarField);
pub type ZKMLSponge<E: Pairing> = PoseidonSponge<E::ScalarField>;
pub type ZKMLCommitment<E: Pairing, P> = (MLCommitment<E>, LabeledPolynomial<E::ScalarField, P>);

/// Yoinked from ark-poly-commit lib.rs
/// Generate default parameters for alpha = 17, state-size = 8
///
/// WARNING: This poseidon parameter is not secure. Please generate
/// your own parameters according the field you use.
pub(crate) fn poseidon_parameters_for_test<F: PrimeField>() -> PoseidonConfig<F> {
    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;

    let mds = vec![
        vec![F::one(), F::zero(), F::one()],
        vec![F::one(), F::one(), F::zero()],
        vec![F::zero(), F::one(), F::one()],
    ];

    let mut ark = Vec::new();
    let mut ark_rng = test_rng();

    for _ in 0..(full_rounds + partial_rounds) {
        let mut res = Vec::new();

        for _ in 0..3 {
            res.push(F::rand(&mut ark_rng));
        }
        ark.push(res);
    }
    PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, ark, 2, 1)
}

fn test_sponge<E>() -> ZKMLSponge<E>
where
    E: Pairing,
{
    PoseidonSponge::new(&poseidon_parameters_for_test())
}
//todo: turn MLPCParam<E> to ZKMLUniversaalParam<E, P>
pub struct SRS<E, P>
where
    E: Pairing,
    P: DenseMVPolynomial<E::ScalarField>,
    P::Point: Index<usize, Output = E::ScalarField>,
{
    pub poly_srs: ZKMLUniversalParam<E, P>,
    pub mask_srs: MaskParam<E, P>,
}
impl<E, P> SRS<E, P>
where
    E: Pairing,
    P: DenseMVPolynomial<E::ScalarField>,
    P::Point: Index<usize, Output = E::ScalarField>,
{
    pub fn generate_srs<R: RngCore>(
        num_vars: usize,
        hiding_bound: usize,
        rng: &mut R,
    ) -> SRS<E, P> {
        let poly_srs = ZKMLCommit::<E, P>::setup(num_vars, hiding_bound, rng);
        let mask_srs = SpecMultiCommit::special_setup(5, Some(num_vars), rng).unwrap();
        SRS { poly_srs, mask_srs }
    }
}

pub struct ZKMLCommit<E: Pairing, P: DenseMVPolynomial<E::ScalarField>> {
    _engine: PhantomData<E>,
    _poly: PhantomData<P>,
}
type Sponge<E: Pairing> = PoseidonSponge<E::ScalarField>;
// Integrate multilinear PC and the special commitment above.
impl<E, P> ZKMLCommit<E, P>
where
    E: Pairing,
    P: DenseMVPolynomial<E::ScalarField> + Sync,
    P::Point: Index<usize, Output = E::ScalarField>,
{
    pub fn setup<R: RngCore>(
        num_vars: usize,
        hiding_bound: usize,
        rng: &mut R,
    ) -> ZKMLUniversalParam<E, P> {
        assert!(num_vars > 0, "constant polynomial not supported");
        let g: E::G1 = E::G1::rand(rng);
        let h: E::G2 = E::G2::rand(rng);
        let g = g.into_affine();
        let h = h.into_affine();
        let mut powers_of_g = Vec::new();
        let mut powers_of_h = Vec::new();
        let t: Vec<_> = (0..num_vars).map(|_| E::ScalarField::rand(rng)).collect();
        let scalar_bits = E::ScalarField::MODULUS_BIT_SIZE as usize;

        let mut eq: LinkedList<DenseMultilinearExtension<E::ScalarField>> =
            LinkedList::from_iter(eq_extension(&t).into_iter());
        let mut eq_arr = LinkedList::new();
        let mut base = eq.pop_back().unwrap().evaluations;

        for i in (0..num_vars).rev() {
            eq_arr.push_front(remove_dummy_variable(&base, i));
            if i != 0 {
                let mul = eq.pop_back().unwrap().evaluations;
                base = base
                    .into_iter()
                    .zip(mul.into_iter())
                    .map(|(a, b)| a * &b)
                    .collect();
            }
        }

        let mut pp_powers = Vec::new();
        let mut total_scalars = 0;
        for i in 0..num_vars {
            let eq = eq_arr.pop_front().unwrap();
            let pp_k_powers = (0..(1 << (num_vars - i))).map(|x| eq[x]);
            pp_powers.extend(pp_k_powers);
            total_scalars += 1 << (num_vars - i);
        }
        let window_size = FixedBase::get_mul_window_size(total_scalars);
        let g_table = FixedBase::get_window_table(scalar_bits, window_size, g.into_group());
        let h_table = FixedBase::get_window_table(scalar_bits, window_size, h.into_group());

        let pp_g = E::G1::normalize_batch(&FixedBase::msm(
            scalar_bits,
            window_size,
            &g_table,
            &pp_powers,
        ));
        let pp_h = E::G2::normalize_batch(&FixedBase::msm(
            scalar_bits,
            window_size,
            &h_table,
            &pp_powers,
        ));
        let mut start = 0;
        for i in 0..num_vars {
            let size = 1 << (num_vars - i);
            let pp_k_g = (&pp_g[start..(start + size)]).to_vec();
            let pp_k_h = (&pp_h[start..(start + size)]).to_vec();
            powers_of_g.push(pp_k_g);
            powers_of_h.push(pp_k_h);
            start += size;
        }

        // uncomment to measure the time for calculating vp
        // let vp_generation_timer = start_timer!(|| "VP generation");
        let h_mask = {
            let window_size = FixedBase::get_mul_window_size(num_vars);
            let h_table = FixedBase::get_window_table(scalar_bits, window_size, h.into_group());
            E::G2::normalize_batch(&FixedBase::msm(scalar_bits, window_size, &h_table, &t))
        };
        // end_timer!(vp_generation_timer);

        // MLPC is little endian, but pst13 is big endian
        //t = t.iter().copied().rev().collect();
        (
            MLPCParam {
                num_vars,
                g,
                h_mask,
                h,
                powers_of_g,
                powers_of_h,
            },
            SpecMultiCommit::special_setup_with_beta(hiding_bound, Some(num_vars), rng, &t)
                .unwrap(),
        )
    }

    pub fn trim(
        param: &ZKMLUniversalParam<E, P>,
        num_variables: usize,
        deg_for_mask: usize,
    ) -> (ZKMLCommitterKey<E, P>, ZKMLVerifierKey<E>) {
        let to_reduce = param.0.num_vars - num_variables;
        let ck1 = MLCommitterKey {
            powers_of_h: (&param.0.powers_of_h[to_reduce..]).to_vec(),
            powers_of_g: (&param.0.powers_of_g[to_reduce..]).to_vec(),
            g: param.0.g,
            h: param.0.h,
            nv: num_variables,
        };
        let vk1 = MLVerifierKey {
            nv: num_variables,
            g: param.0.g,
            h: param.0.h,
            h_mask_random: (&param.0.h_mask[to_reduce..]).to_vec(),
        };
        let (mut ck2, vk2) =
            MarlinPST13::<E, P, Sponge<E>>::trim(&param.1, deg_for_mask, 0, None).unwrap();
        ck2.powers_of_g = ck2
            .powers_of_g
            .iter()
            .filter(|(k, _)| k.is_constant() || k.vars()[0] >= to_reduce)
            .map(|(k, v)| {
                if (k.is_constant()) {
                    (k.clone(), v.clone())
                } else {
                    (
                        P::Term::new(vec![(k.vars()[0] - to_reduce, k.powers()[0])]),
                        v.clone(),
                    )
                }
            })
            .collect();
        ((ck1, ck2), (vk1, vk2))
    }

    pub fn commit_mask(
        ck: &MaskCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
        polynomial: &LabeledPolynomial<
            E::ScalarField,
            SparsePolynomial<E::ScalarField, SparseTerm>,
        >,
        rng: &mut impl Rng,
    ) -> <E as Pairing>::G1Affine {
        let polynomial_iter = std::iter::once(polynomial);
        let (hiding_commitment, _) = MarlinPST13::<
            E,
            SparsePolynomial<E::ScalarField, SparseTerm>,
            Sponge<E>,
        >::commit(&ck, polynomial_iter, Some(rng))
        .unwrap();
        hiding_commitment[0].commitment().comm.0
    }
    pub fn commit(
        ck: &ZKMLCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
        polynomial: &impl MultilinearExtension<E::ScalarField>,
        hiding_bound: usize,
        mask_num_var: Option<usize>,
        rng: &mut impl Rng,
    ) -> (ZKMLCommitment<E, SparsePolynomial<E::ScalarField, SparseTerm>>) {
        let mut p_hat = SparsePolynomial::<E::ScalarField, SparseTerm>::zero();
        if let Some(mask_num_vars) = mask_num_var {
            p_hat = generate_mask_polynomial(rng, mask_num_vars, hiding_bound, false);
        } else {
            p_hat = generate_mask_polynomial(rng, polynomial.num_vars(), hiding_bound, false);
        }
        let labeled_p_hat =
            LabeledPolynomial::new("p_hat".to_owned(), p_hat, Some(hiding_bound), None);
        //let labeled_p_hat_iter = std::iter::once(&labeled_p_hat);
        //let (hiding_commitment, randomness) = MarlinPST13::<E, SparsePolynomial<E::ScalarField, SparseTerm>, Sponge<E>>::commit(&ck.1, labeled_p_hat_iter, Some(rng)).unwrap();
        let hiding_commitment = Self::commit_mask(&ck.1, &labeled_p_hat, rng);
        let base_commitment = MultilinearPC::commit(&ck.0, polynomial).g_product;
        let hidden_commitment: E::G1Affine = (base_commitment + hiding_commitment).into();
        let commitment = MLCommitment {
            g_product: hidden_commitment,
            nv: polynomial.num_vars(),
        };
        (commitment, labeled_p_hat)
    }
    pub fn open_mask(
        ck: &MaskCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
        polynomial: &LabeledPolynomial<
            E::ScalarField,
            SparsePolynomial<E::ScalarField, SparseTerm>,
        >,
        point: &Vec<E::ScalarField>,
    ) -> (MaskProof<E>, E::ScalarField) {
        let proof =
            SpecMultiCommit::<E, SparsePolynomial<E::ScalarField, SparseTerm>>::special_open(
                &ck,
                polynomial,
                vec![],
                &point,
                None,
            );
        (proof, polynomial.evaluate(&point))
    }
    pub fn open(
        ck: &ZKMLCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
        polynomial: &impl MultilinearExtension<E::ScalarField>,
        p_hat: &LabeledPolynomial<E::ScalarField, SparsePolynomial<E::ScalarField, SparseTerm>>, // TODO: Represent this as a prg seed? how many bits?
        point: &[E::ScalarField],
        //rng: &mut impl Rng,
    ) -> ZKMLProof<E> {
        let base_proof = MultilinearPC::open(&ck.0, polynomial, point);
        let point = point.to_vec(); // todo add lifetime restriction
        let (hiding_proof, evaluation) = Self::open_mask(&ck.1, p_hat, &point);
        let hidden_proof_evals = base_proof
            .proofs
            .iter()
            .zip(hiding_proof.w.iter())
            .map(|(base_eval, hiding_eval)| (*base_eval + hiding_eval).into())
            .collect::<Vec<E::G1Affine>>();
        //modify MLProof
        (
            MLProof {
                proofs: hidden_proof_evals,
            },
            evaluation,
        )
    }
    pub fn check(
        vk: &ZKMLVerifierKey<E>,
        commitment: &MLCommitment<E>,
        point: &[E::ScalarField],
        value: E::ScalarField,
        proof: &ZKMLProof<E>,
    ) -> bool {
        let vk_ml = &vk.0;
        let vk_pst = &vk.1;
        let left = E::pairing(
            commitment.g_product.into_group() - vk_ml.g.mul(value) - vk_pst.g.mul(proof.1),
            vk_ml.h,
        );

        let scalar_size = E::ScalarField::MODULUS_BIT_SIZE as usize;
        let window_size = FixedBase::get_mul_window_size(vk_ml.nv);

        let h_table = FixedBase::get_window_table(scalar_size, window_size, vk_ml.h.into_group());
        let h_mul: Vec<E::G2> = FixedBase::msm(scalar_size, window_size, &h_table, point);

        let pairing_rights: Vec<_> = (0..vk_ml.nv)
            .map(|i| vk_ml.h_mask_random[i].into_group() - &h_mul[i])
            .collect();
        let pairing_rights: Vec<E::G2Affine> = E::G2::normalize_batch(&pairing_rights);
        let pairing_rights: Vec<E::G2Prepared> = pairing_rights
            .into_iter()
            .map(|x| E::G2Prepared::from(x))
            .collect();

        let pairing_lefts: Vec<E::G1Prepared> = proof
            .0
            .proofs
            .iter()
            .map(|x| E::G1Prepared::from(*x))
            .collect();

        let right = E::multi_pairing(pairing_lefts, pairing_rights);
        left == right
    }
}

/// fix first `pad` variables of `poly` represented in evaluation form to zero
fn remove_dummy_variable<F: Field>(poly: &[F], pad: usize) -> Vec<F> {
    if pad == 0 {
        return poly.to_vec();
    }
    if !poly.len().is_power_of_two() {
        panic!("Size of polynomial should be power of two. ")
    }
    let nv = ark_std::log2(poly.len()) as usize - pad;
    let table: Vec<_> = (0..(1 << nv)).map(|x| poly[x << pad]).collect();
    table
}

/// generate eq(t,x), a product of multilinear polynomials with fixed t.
/// eq(a,b) is takes extensions of a,b in {0,1}^num_vars such that if a and b in {0,1}^num_vars are equal
/// then this polynomial evaluates to 1.
fn eq_extension<F: Field>(t: &[F]) -> Vec<DenseMultilinearExtension<F>> {
    let dim = t.len();
    let mut result = Vec::new();

    for i in 0..dim {
        let ti = t[i];
        let poly = cfg_into_iter!(0..(1 << dim)).map(|x| {
            let xi = if ((x >> i) & 1) == 1 { F::one() } else { F::zero() };
            let ti_xi = if ((x >> i) & 1) == 1 { ti } else { F::zero() };
            ti_xi + ti_xi - xi - ti + F::one()
        }).collect();
        result.push(DenseMultilinearExtension::from_evaluations_vec(dim, poly));
    }

    result
}
/*pub fn wrap_poly_to_label_poly<F: PrimeField, P: Polynomial<F>>(label: String, polynomial: P, degree_bound: usize, hiding_bound: usize) -> LabeledPolynomial<F, P>{
    LabeledPolynomial::new(
        label,
        polynomial,
        Some(degree_bound),
        Some(hiding_bound)
    )


}*/

pub fn zk_sumcheck_prover_wrapper<
    E: Pairing,
    R: RngCore,
    R1: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
    S: CryptographicSponge,
>(
    list_poly: &ListOfProductsOfPolynomials<E::ScalarField>,
    fs_rng: &mut R1,
    mask_rng: &mut R,
    mask_key: &MaskCommitterKey<E, SparsePolynomial<E::ScalarField, SparseTerm>>,
    opening_challenge: &mut ChallengeGenerator<E::ScalarField, S>,
) -> (ZKSumcheckProof<E>, ZKProverState<E::ScalarField>) {
    let mask_poly = generate_mask_polynomial(
        mask_rng,
        list_poly.num_variables,
        list_poly.max_multiplicands,
        true,
    );
    let vec_mask_poly = vec![LabeledPolynomial::new(
        String::from("mask_poly_for_sumcheck"),
        mask_poly.clone(),
        Some(list_poly.max_multiplicands),
        None,
    )];
    let (mask_commit, mask_randomness) =
        MarlinPST13::<_, _, PoseidonSponge<E::ScalarField>>::commit(
            mask_key,
            &vec_mask_poly,
            Some(mask_rng),
        )
        .unwrap();
    let g_commit = mask_commit[0].commitment();
    let _ = fs_rng.feed(g_commit);
    let challenge = E::ScalarField::rand(fs_rng);
    let (pf, randomness) =
        MLSumcheck::prove_as_subprotocol_zk(fs_rng, list_poly, &mask_poly, challenge).unwrap();

    let opening = MarlinPST13::<_, SparsePolynomial<E::ScalarField, SparseTerm>, _>::open(
        &mask_key,
        &vec_mask_poly,
        &mask_commit,
        &randomness.prover_state.randomness,
        opening_challenge,
        &mask_randomness,
        None,
    );
    (
        ZKSumcheckProof {
            g_commit: *g_commit,
            sumcheck_proof: pf,
            poly_info: list_poly.info(),
            g_proof: opening.unwrap(),
            g_value: mask_poly.evaluate(&randomness.prover_state.randomness),
        },
        randomness,
    )
}

pub fn zk_sumcheck_verifier_wrapper<
    E: Pairing,
    R: RngCore + FeedableRNG<Error = ark_linear_sumcheck::Error>,
    S: CryptographicSponge,
>(
    mask_vk: &MaskVerifierKey<E>,
    proof: &ZKSumcheckProof<E>,
    fs_rng: &mut R,
    opening_challenge: &mut ChallengeGenerator<E::ScalarField, S>,
    claimed_sum: E::ScalarField,
) -> (bool, SubClaim<E::ScalarField>) {
    let _ = fs_rng.feed(&proof.g_commit);
    let challenge = E::ScalarField::rand(fs_rng);

    let subclaim = MLSumcheck::verify_as_subprotocol_zk(
        fs_rng,
        &proof.poly_info,
        claimed_sum,
        &proof.sumcheck_proof,
        challenge,
        proof.g_value,
    )
    .unwrap();

    let label_com = vec![LabeledCommitment::new(
        String::from("mask_poly_for_sumcheck"),
        proof.g_commit.clone(),
        None,
    )];
    let flag = MarlinPST13::<_, SparsePolynomial<E::ScalarField, SparseTerm>, _>::check(
        mask_vk,
        &label_com,
        &subclaim.point,
        vec![proof.g_value],
        &proof.g_proof,
        opening_challenge,
        None,
    )
    .unwrap();

    (flag, subclaim)
}

#[test]
pub(crate) fn test_zk() {
    let num_vars = 12;
    let mut rng = test_rng();
    use ark_bn254::{Bn254, Fr};
    type SCALAR = <Bn254 as Pairing>::ScalarField;
    type ZKML = ZKMLCommit<Bn254, SparsePolynomial<SCALAR, SparseTerm>>;
    let srs = ZKML::setup(num_vars, 5, &mut rng);
    let (ck, vk) = ZKML::trim(&srs, 10, 5);
    let rand_poly = DenseMultilinearExtension::<Fr>::rand(10, &mut rng);
    let (commitment, mask) = ZKML::commit(&ck, &rand_poly, 5, None, &mut rng);

    let point: Vec<_> = (0..10).map(|_| Fr::one()).collect();
    let commitment_ml = MultilinearPC::<Bn254>::commit(&ck.0, &rand_poly);
    /*let proof_ml = MultilinearPC::<Bn254>::open(&ck.0, &rand_poly, &point);
    let check_ml = MultilinearPC::<Bn254>::check(&vk.0, &commitment_ml, &point, rand_poly.evaluate(&point).unwrap() ,&proof_ml);
    let point = &point[..];
    assert!(check_ml);*/
    let proof = ZKML::open(&ck, &rand_poly, &mask, &point);
    let check = ZKML::check(
        &vk,
        &commitment,
        &point,
        rand_poly.evaluate(&point).unwrap(),
        &proof,
    );

    assert!(check);
}
