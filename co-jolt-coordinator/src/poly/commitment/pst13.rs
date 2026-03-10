use ark_bn254::{Bn254, Fr};
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{One, Zero};
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, CommitterKey, Proof, UniversalParams, VerifierKey},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use jolt_core::field::JoltField;
use jolt_core::msm::VariableBaseMSM;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::transcripts::{AppendToTranscript, Transcript};
use jolt_core::utils::errors::ProofVerifyError;
use mpc_core::protocols::rep3::network::Rep3NetworkCoordinator;
use mpc_core::MaybeShared;
use rayon::prelude::*;
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::Add;

use crate::poly::commitment::Rep3CommitmentScheme;

// =============================================================================
// Types
// =============================================================================

#[derive(Clone)]
pub struct PST13<E: Pairing> {
    _marker: PhantomData<E>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PST13Setup<E: Pairing> {
    pub uni_params: UniversalParams<E>,
}

impl<E: Pairing> Default for PST13Setup<E> {
    fn default() -> Self {
        Self {
            uni_params: UniversalParams {
                num_vars: 0,
                powers_of_g: vec![],
                powers_of_h: vec![],
                g: E::G1Affine::zero(),
                h: E::G2Affine::zero(),
                h_mask: vec![],
            },
        }
    }
}

impl<E: Pairing> PST13Setup<E> {
    pub fn ck(&self) -> CommitterKey<E> {
        MultilinearPC::trim(&self.uni_params, self.uni_params.num_vars).0
    }

    pub fn vk(&self) -> VerifierKey<E> {
        MultilinearPC::trim(&self.uni_params, self.uni_params.num_vars).1
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct PST13VerifierSetup<E: Pairing> {
    pub vk: VerifierKey<E>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize, Eq, PartialEq)]
pub struct PST13Commitment<E: Pairing> {
    pub nv: usize,
    pub g_product: E::G1Affine,
}

impl<E: Pairing> Default for PST13Commitment<E> {
    fn default() -> Self {
        Self {
            nv: 0,
            g_product: E::G1Affine::zero(),
        }
    }
}

impl<E: Pairing> Add for PST13Commitment<E> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            nv: self.nv + other.nv,
            g_product: (self.g_product + other.g_product).into_affine(),
        }
    }
}

impl<E: Pairing> AppendToTranscript for PST13Commitment<E> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"g_product");
        transcript.append_point(&self.g_product.into_group());
    }
}

impl<E: Pairing> From<&PST13Commitment<E>> for Commitment<E> {
    fn from(c: &PST13Commitment<E>) -> Self {
        Commitment {
            nv: c.nv,
            g_product: c.g_product,
        }
    }
}

// =============================================================================
// CommitmentScheme implementation
// =============================================================================

impl<E: Pairing> CommitmentScheme for PST13<E>
where
    E::ScalarField: JoltField,
{
    type Field = E::ScalarField;
    type ProverSetup = PST13Setup<E>;
    type VerifierSetup = PST13VerifierSetup<E>;
    type Commitment = PST13Commitment<E>;
    type Proof = Proof<E>;
    type BatchedProof = Proof<E>;
    type OpeningProofHint = ();

    #[tracing::instrument(skip_all, name = "PST13::setup_prover", level = "trace")]
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let mut rng = test_rng();
        let uni_params = MultilinearPC::setup(max_num_vars, &mut rng);
        PST13Setup { uni_params }
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        PST13VerifierSetup { vk: setup.vk() }
    }

    #[tracing::instrument(skip_all, name = "PST13::commit", level = "trace")]
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        let nv = poly.get_num_vars();
        let ck = setup.ck();
        let srs = &ck.powers_of_g[0];
        let g_product = match poly {
            MultilinearPolynomial::OneHot(oh) => {
                let t = oh.nonzero_indices.len();
                let mut acc = E::G1::zero();
                for (j, idx) in oh.nonzero_indices.iter().enumerate() {
                    if let Some(k) = idx {
                        acc += srs[*k as usize * t + j];
                    }
                }
                acc.into_affine()
            }
            MultilinearPolynomial::RLC(rlc) => {
                let t = rlc.dense_rlc.len();
                let mut acc =
                    <E::G1 as VariableBaseMSM>::msm_field_elements(&srs[..t], &rlc.dense_rlc)
                        .unwrap();
                for (coeff, oh_poly) in &rlc.one_hot_rlc {
                    if let MultilinearPolynomial::OneHot(oh) = oh_poly.as_ref() {
                        let oh_t = oh.nonzero_indices.len();
                        for (j, idx) in oh.nonzero_indices.iter().enumerate() {
                            if let Some(k) = idx {
                                acc += E::G1::from(srs[*k as usize * oh_t + j]) * *coeff;
                            }
                        }
                    }
                }
                acc.into_affine()
            }
            _ => {
                let size = 1 << nv;
                <E::G1 as VariableBaseMSM>::msm(&srs[..size], poly)
                    .unwrap()
                    .into_affine()
            }
        };
        (PST13Commitment { nv, g_product }, ())
    }

    #[tracing::instrument(skip_all, name = "PST13::batch_commit", level = "trace")]
    fn batch_commit<U>(
        polys: &[U],
        setup: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        let has_special = polys.iter().any(|p| {
            matches!(
                p.borrow(),
                MultilinearPolynomial::OneHot(_) | MultilinearPolynomial::RLC(_)
            )
        });
        if has_special {
            return polys
                .par_iter()
                .map(|p| Self::commit(p.borrow(), setup))
                .collect();
        }

        let ck = setup.ck();
        let powers_of_g = &ck.powers_of_g[0];
        let nv = polys[0].borrow().get_num_vars();
        let msm_size = 1 << nv;

        assert!(msm_size <= powers_of_g.len(), "Key length error");

        let commitments = <E::G1 as VariableBaseMSM>::batch_msm(&powers_of_g[..msm_size], polys);
        commitments
            .into_iter()
            .map(|c| {
                (
                    PST13Commitment {
                        nv,
                        g_product: c.into_affine(),
                    },
                    (),
                )
            })
            .collect()
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        let g_product: E::G1Affine = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, coeff)| commitment.borrow().g_product * coeff)
            .sum::<E::G1>()
            .into_affine();
        PST13Commitment {
            nv: commitments[0].borrow().nv,
            g_product,
        }
    }

    #[tracing::instrument(skip_all, name = "PST13::prove", level = "trace")]
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _hint: Self::OpeningProofHint,
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        assert_eq!(poly.get_num_vars(), opening_point.len());
        let opening_point_rev: Vec<Self::Field> =
            opening_point.iter().rev().map(|c| (*c).into()).collect();
        let dense = multilinear_to_dense(poly);
        open(&setup.ck(), &dense, &opening_point_rev).0
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let opening_point_rev: Vec<Self::Field> =
            opening_point.iter().rev().map(|c| (*c).into()).collect();
        if MultilinearPC::check(
            &setup.vk,
            &commitment.into(),
            &opening_point_rev,
            *opening,
            proof,
        ) {
            Ok(())
        } else {
            Err(ProofVerifyError::InternalError)
        }
    }

    fn protocol_name() -> &'static [u8] {
        b"PST13"
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn multilinear_to_dense<F: JoltField>(poly: &MultilinearPolynomial<F>) -> DensePolynomial<F> {
    use jolt_core::utils::small_scalar::SmallScalar;
    match poly {
        MultilinearPolynomial::LargeScalars(p) => DensePolynomial::new(p.evals()),
        MultilinearPolynomial::U8Scalars(p) => {
            DensePolynomial::new(p.coeffs.iter().map(|c| c.to_field()).collect())
        }
        MultilinearPolynomial::U16Scalars(p) => {
            DensePolynomial::new(p.coeffs.iter().map(|c| c.to_field()).collect())
        }
        MultilinearPolynomial::U32Scalars(p) => {
            DensePolynomial::new(p.coeffs.iter().map(|c| c.to_field()).collect())
        }
        MultilinearPolynomial::U64Scalars(p) => {
            DensePolynomial::new(p.coeffs.iter().map(|c| c.to_field()).collect())
        }
        MultilinearPolynomial::I64Scalars(p) => {
            DensePolynomial::new(p.coeffs.iter().map(|c| c.to_field()).collect())
        }
        MultilinearPolynomial::I128Scalars(p) => {
            DensePolynomial::new(p.coeffs.iter().map(|c| c.to_field()).collect())
        }
        MultilinearPolynomial::U128Scalars(p) => {
            DensePolynomial::new(p.coeffs.iter().map(|c| c.to_field()).collect())
        }
        MultilinearPolynomial::S128Scalars(p) => {
            DensePolynomial::new(p.coeffs.iter().map(|c| c.to_field()).collect())
        }
        MultilinearPolynomial::OneHot(oh) => {
            let k = oh.K;
            let t = oh.nonzero_indices.len();
            let mut coeffs = vec![F::zero(); k * t];
            for (j, idx) in oh.nonzero_indices.iter().enumerate() {
                if let Some(ki) = idx {
                    coeffs[*ki as usize * t + j] = F::one();
                }
            }
            DensePolynomial::new(coeffs)
        }
        MultilinearPolynomial::RLC(rlc) => {
            let t = rlc.dense_rlc.len();
            let total_size = if rlc.one_hot_rlc.is_empty() {
                t
            } else {
                let nv = poly.get_num_vars();
                1 << nv
            };
            let mut coeffs = vec![F::zero(); total_size];
            for (i, &val) in rlc.dense_rlc.iter().enumerate() {
                coeffs[i] = val;
            }
            for (coeff, oh_poly) in &rlc.one_hot_rlc {
                if let MultilinearPolynomial::OneHot(oh) = oh_poly.as_ref() {
                    let oh_t = oh.nonzero_indices.len();
                    for (j, idx) in oh.nonzero_indices.iter().enumerate() {
                        if let Some(ki) = idx {
                            coeffs[*ki as usize * oh_t + j] += *coeff;
                        }
                    }
                }
            }
            DensePolynomial::new(coeffs)
        }
    }
}

fn open<E: Pairing>(
    ck: &CommitterKey<E>,
    polynomial: &DensePolynomial<E::ScalarField>,
    point: &[E::ScalarField],
) -> (Proof<E>, E::ScalarField)
where
    E::ScalarField: JoltField,
{
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
// Coordinator Rep3CommitmentScheme implementation
// =============================================================================

impl<ProofTranscript> Rep3CommitmentScheme<Fr, ProofTranscript> for PST13<Bn254>
where
    ProofTranscript: Transcript,
{
    fn coordinate_prove<Network>(
        _setup: &Self::ProverSetup,
        _transcript: &mut ProofTranscript,
        network: &mut Network,
        _opening_point: &[<Fr as JoltField>::Challenge],
        _claimed_opening: &Fr,
        _commitment: &Self::Commitment,
    ) -> eyre::Result<Self::Proof>
    where
        Network: Rep3NetworkCoordinator,
    {
        let proofs = if network.is_distributed() {
            let subnet_proofs = network
                .receive_responses_from_subnets::<Vec<<Bn254 as Pairing>::G1Affine>>()?
                .into_iter()
                .map(|shares| {
                    let [pf0, pf1, pf2]: [Vec<_>; 3] = shares.try_into().unwrap();
                    itertools::multizip((pf0, pf1, pf2))
                        .map(|(a, b, c)| (a + b + c).into_affine())
                        .collect::<Vec<<Bn254 as Pairing>::G1Affine>>()
                })
                .collect::<Vec<_>>();

            let mut proofs = subnet_proofs.into_iter();
            let mut combined = proofs.next().unwrap();
            for next in proofs {
                for (acc, share) in combined.iter_mut().zip(next) {
                    *acc = (*acc + share).into_affine();
                }
            }
            combined
        } else {
            let [pf0, pf1, pf2]: [Vec<<Bn254 as Pairing>::G1Affine>; 3] =
                network.receive_responses()?.try_into().unwrap();

            itertools::multizip((pf0, pf1, pf2))
                .map(|(a, b, c)| (a + b + c).into_affine())
                .collect::<Vec<_>>()
        };

        Ok(Proof { proofs })
    }

    fn combine_commitment_shares(
        commitments: &[&MaybeShared<Self::Commitment>],
    ) -> Self::Commitment {
        let public = commitments
            .iter()
            .find(|c| matches!(c, MaybeShared::Public(Some(_))));
        let (g_product, nv) = match public {
            Some(MaybeShared::Public(Some(commitment))) => (commitment.g_product, commitment.nv),
            None => {
                let mut g_product = <Bn254 as Pairing>::G1::zero();
                let mut nv = None;
                for commitment in commitments {
                    match commitment {
                        MaybeShared::Shared(commitment) => {
                            g_product += commitment.g_product;
                            match nv {
                                Some(nv) => assert_eq!(nv, commitment.nv),
                                None => nv = Some(commitment.nv),
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                (g_product.into_affine(), nv.unwrap())
            }
            _ => unreachable!(),
        };
        PST13Commitment { nv, g_product }
    }

    fn combine_hint_shares(
        _hints: &[&MaybeShared<Self::OpeningProofHint>],
    ) -> Self::OpeningProofHint {
    }
}
