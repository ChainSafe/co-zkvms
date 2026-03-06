use crate::poly::one_hot_polynomial::Rep3OneHotPolynomial;
use crate::poly::{Rep3MultilinearPolynomial, Rep3SharedPoly};
use crate::utils::types::MaybeShared;
use ark_ec::pairing::Pairing;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{One, Zero};
use ark_poly_commit::multilinear_pc::{
    data_structures::{Commitment, CommitterKey, Proof, UniversalParams, VerifierKey},
    MultilinearPC,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::test_rng;
use itertools::izip;
use jolt_core::field::JoltField;
use jolt_core::msm::VariableBaseMSM;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::transcripts::{AppendToTranscript, Transcript};
use jolt_core::utils::errors::ProofVerifyError;
use mpc_core::protocols::rep3::network::{Rep3NetworkCoordinator, Rep3NetworkWorker};
use mpc_core::protocols::rep3::PartyID;
use rayon::prelude::*;
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::Add;

use super::Rep3CommitmentScheme;

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
    pub(crate) nv: usize,
    pub(crate) g_product: E::G1Affine,
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
                // Sparse commit: sum SRS elements at non-zero positions.
                // OneHot coeff layout: coeff[k * T + j] = 1 iff nonzero_indices[j] == Some(k).
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
                // RLC = dense part (length T) + one-hot contributions.
                // Dense part: MSM over first T SRS elements.
                let t = rlc.dense_rlc.len();
                let mut acc =
                    <E::G1 as VariableBaseMSM>::msm_field_elements(&srs[..t], &rlc.dense_rlc)
                        .unwrap();
                // One-hot contributions: each (coeff, one_hot_poly)
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
        // Mixed batches (OneHot/RLC alongside dense): commit each individually.
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

        // All-dense fast path: batch MSM.
        let ck = setup.ck();
        let powers_of_g = &ck.powers_of_g[0];
        let nv = polys[0].borrow().get_num_vars();
        let msm_size = 1 << nv;

        assert!(msm_size <= powers_of_g.len(), "Key length error");

        let commitments =
            <E::G1 as VariableBaseMSM>::batch_msm(&powers_of_g[..msm_size], polys);
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

/// Convert a `MultilinearPolynomial` to a `DensePolynomial` (field elements).
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
            // Determine total size from one_hot contributions
            let total_size = if rlc.one_hot_rlc.is_empty() {
                t
            } else {
                let nv = poly.get_num_vars();
                1 << nv
            };
            let mut coeffs = vec![F::zero(); total_size];
            // Dense part (first T coefficients)
            for (i, &val) in rlc.dense_rlc.iter().enumerate() {
                coeffs[i] = val;
            }
            // One-hot contributions
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

// =============================================================================
// open() helper — layered multilinear opening proof
// =============================================================================

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

        let pi_g = <E::G1 as VariableBaseMSM>::msm_field_elements(
            &ck.powers_of_g[i],
            &scalars,
        )
        .unwrap()
        .into_affine();
        proofs.push(pi_g);
    }

    (Proof { proofs }, r[0][0])
}

// =============================================================================
// Rep3CommitmentScheme implementation
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
                        // Sparse public commit: C = Σ_{j:active} srs[k(j)*T + j]
                        // Only T_active group additions, zero scalar muls.
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
                    let poly_a = MultilinearPolynomial::LargeScalars(
                        poly.into_distributed_commit_form(),
                    );
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

    #[tracing::instrument(skip_all, name = "PST13::coordinate_prove", level = "trace")]
    fn coordinate_prove<Network>(
        _setup: &Self::ProverSetup,
        _transcript: &mut ProofTranscript,
        network: &mut Network,
        _opening_point: &[<ark_bn254::Fr as JoltField>::Challenge],
        _claimed_opening: &ark_bn254::Fr,
        _commitment: &Self::Commitment,
    ) -> eyre::Result<Self::Proof>
    where
        Network: Rep3NetworkCoordinator,
    {
        let proofs = if network.is_distributed() {
            network
                .receive_responses_from_subnets::<Vec<<ark_bn254::Bn254 as Pairing>::G1Affine>>()?
                .into_iter()
                .map(|shares| {
                    let [pf0, pf1, pf2]: [Vec<_>; 3] = shares.try_into().unwrap();
                    itertools::multizip((pf0, pf1, pf2))
                        .map(|(a, b, c)| (a + b + c).into_affine())
                        .collect::<Vec<_>>()
                })
                .reduce(|prev, next| {
                    izip!(prev, next)
                        .map(|(p, n)| (p + n).into_affine())
                        .collect()
                })
                .unwrap()
        } else {
            let [pf0, pf1, pf2]: [Vec<<ark_bn254::Bn254 as Pairing>::G1Affine>; 3] =
                network.receive_responses()?.try_into().unwrap();

            itertools::multizip((pf0, pf1, pf2))
                .map(|(a, b, c)| (a + b + c).into_affine())
                .collect::<Vec<_>>()
        };

        Ok(Proof { proofs })
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
            Rep3MultilinearPolynomial::Shared(Rep3SharedPoly::Dense(dense)) => {
                dense.copy_share_a()
            }
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

    fn combine_commitment_shares(
        commitments: &[&MaybeShared<Self::Commitment>],
    ) -> Self::Commitment {
        let public = commitments
            .iter()
            .find(|c| matches!(c, MaybeShared::Public(Some(_))));
        let (g_product, nv) = match public {
            Some(MaybeShared::Public(Some(commitment))) => (commitment.g_product, commitment.nv),
            None => {
                let mut g_product = <ark_bn254::Bn254 as Pairing>::G1::zero();
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

/// Compute a PST13 commitment share for a shared one-hot polynomial.
///
/// Uses the S[k]-bucket approach: precompute public buckets
///   `S[k] = Σ_{j:active} srs[(k XOR c(j))*T + j]`
/// then evaluate `C_share = MSM(S[0..K], e_field[0..K].a)`.
///
/// Cost: K × T_active group additions + MSM(K).
///
/// NOTE: FWHT (as used in Dory's `commit_rows`) does NOT apply here.
/// Dory's FWHT XOR-convolution works because all K rows share the same column
/// generators: `out[k] = Σ_c s[c] · g[c XOR k]` with `s[c]` independent of `k`.
/// PST13's flat SRS has position-dependent generators: the bucket sum
/// `S[k] = Σ_c base_sum[c][k XOR c]` where `base_sum[c][m] = Σ_{j:c(j)=c} srs[m*T+j]`
/// depends on *both* `c` and `m = k XOR c`, preventing factorization into a
/// standard XOR-convolution.
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

    // Step 1: precompute S[k] = Σ_{j:active} srs[(k XOR c(j)) * T + j]
    // Parallelized over k.
    let buckets: Vec<G1> = (0..k_size)
        .into_par_iter()
        .map(|k| {
            let mut acc = G1::zero();
            for j in 0..t {
                if let Some(c) = one_hot.masked_indices_c[j] {
                    let m = k ^ (c as usize); // row index in SRS
                    acc += srs[m * t + j];
                }
            }
            acc
        })
        .collect();

    // Step 2: MSM(K) with shared scalars e_field[k].a
    let scalars: Vec<ark_bn254::Fr> = one_hot
        .rand_ohv_e_field
        .iter()
        .map(|s| s.a)
        .collect();

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

/// Materialize the `.a` share of a one-hot polynomial as a dense coefficient vector.
///
/// The open() algorithm requires dense sequential access to all coefficients.
/// For one-hot: `coeffs[k*T + j] = e_field[k XOR c(j)].a` for active j, else 0.
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

/// Materialize the `.a` share of an RLC polynomial as a dense coefficient vector.
///
/// Combines the dense_rlc part and one_hot_rlc contributions into a flat vector.
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

        let opening = poly.evaluate(
            &point
                .iter()
                .map(|c| (*c).into())
                .collect::<Vec<F>>(),
        );

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

        // Compute linear combination manually
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
