use std::marker::PhantomData;
use std::ops::Index;
use std::ops::Neg;
use std::ops::{AddAssign, SubAssign};
use std::rc::Rc;

use crate::mpc::rss::SSRandom;
use crate::mpc::utils::generate_ass_share_randomness;
use crate::mpc::SSOpen;
use ark_ec::bls12::Bls12;
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::Field;
use ark_ff::PrimeField;
use ark_ff::{One, UniformRand, Zero};
use ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::verifier::VerifierMsg;
use ark_linear_sumcheck::ml_sumcheck::protocol::{
    prover, IPForMLSumcheck, ListOfProductsOfPolynomials,
};
use ark_linear_sumcheck::ml_sumcheck::Proof;
use ark_linear_sumcheck::rng::{Blake2s512Rng, FeedableRNG};
use ark_linear_sumcheck::Error;
use ark_poly::DenseMultilinearExtension;
use ark_poly::MultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_iter_mut, test_rng};
use rand::RngCore;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::ops::{Add, Mul, Sub};

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct BeaverTriple<E: Pairing> {
    pub party: usize,
    pub worker_id: usize,
    pub peer: usize,
    pub a_vec: Vec<AssShare<E>>,
    pub b_vec: Vec<AssShare<E>>,
    pub ab_vec: Vec<AssShare<E>>,
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone, Copy)]
pub struct AssShare<E: Pairing> {
    pub party: usize,
    pub share_0: E::ScalarField,
}

impl<E: Pairing> Add<Self> for AssShare<E> {
    type Output = Self;
    fn add(self, rhs: AssShare<E>) -> <Self as Add<AssShare<E>>>::Output {
        assert_eq!(self.party, rhs.party);
        AssShare {
            party: self.party,
            share_0: self.share_0 + rhs.share_0,
        }
    }
}

impl<E: Pairing> Sub<Self> for AssShare<E> {
    type Output = Self;
    fn sub(self, rhs: AssShare<E>) -> Self::Output {
        assert_eq!(self.party, rhs.party);
        AssShare {
            party: self.party,
            share_0: self.share_0 - rhs.share_0,
        }
    }
}

impl<E: Pairing> Mul<E::ScalarField> for AssShare<E> {
    type Output = Self;
    fn mul(self, rhs: E::ScalarField) -> Self::Output {
        AssShare {
            party: self.party,
            share_0: self.share_0 * rhs,
        }
    }
}

impl<E: Pairing> AddAssign for AssShare<E> {
    fn add_assign(&mut self, rhs: AssShare<E>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
    }
}

impl<'a, E: Pairing> AddAssign<&'a AssShare<E>> for AssShare<E> {
    fn add_assign(&mut self, rhs: &AssShare<E>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 += rhs.share_0;
    }
}

impl<E: Pairing> SubAssign for AssShare<E> {
    fn sub_assign(&mut self, rhs: AssShare<E>) {
        assert_eq!(self.party, rhs.party);
        self.share_0 -= rhs.share_0;
    }
}

impl<E: Pairing> Zero for AssShare<E> {
    fn zero() -> Self {
        AssShare {
            party: 0,
            share_0: E::ScalarField::zero(),
        }
    }
    fn is_zero(&self) -> bool {
        self.share_0.is_zero()
    }
}

impl<E: Pairing> SSOpen<E::ScalarField> for AssShare<E> {
    fn open(shares: &[AssShare<E>]) -> <E as Pairing>::ScalarField {
        assert!(shares.len() == 2);
        let mut sum = E::ScalarField::zero();
        for ass in shares.iter() {
            sum += ass.share_0;
        }
        sum
    }
}

impl<E: Pairing> AssShare<E> {
    pub fn get_zero_ass(party: usize) -> Self {
        let mut ass = AssShare::<E>::zero();
        ass.party = party;
        ass
    }

    pub fn ass_mul(
        lhs: &AssShare<E>,
        rhs: &AssShare<E>,
        a: &AssShare<E>,
        b: &AssShare<E>,
        ab: &AssShare<E>,
        d: E::ScalarField,
        e: E::ScalarField,
    ) -> E::ScalarField {
        assert_eq!(lhs.party, rhs.party);
        if lhs.party == 0 {
            return d * e + d * b.share_0 + a.share_0 * e + ab.share_0;
        } else {
            return d * b.share_0 + a.share_0 * e + ab.share_0;
        }
    }

    pub fn get_zero_share<R: RngCore + FeedableRNG>(rng: &mut SSRandom<R>) -> E::ScalarField {
        let zero_share =
            E::ScalarField::rand(&mut rng.seed_1) - E::ScalarField::rand(&mut rng.seed_0);
        rng.update();
        zero_share
    }
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct AssPoly<E: Pairing> {
    pub party: usize,
    pub share_0: DenseMultilinearExtension<E::ScalarField>,
}

impl<E: Pairing> AssPoly<E> {
    pub fn new(party: usize, share_0: DenseMultilinearExtension<E::ScalarField>) -> Self {
        AssPoly { party, share_0 }
    }
    pub fn get_ass_by_idx(&self, i: usize) -> AssShare<E> {
        AssShare {
            party: self.party,
            share_0: self.share_0.index(i).clone(),
        }
    }
    pub fn fix_variables(&self, partial_point: &[E::ScalarField]) -> Self {
        AssPoly {
            party: self.party,
            share_0: self.share_0.fix_variables(partial_point),
        }
    }

    pub fn get_poly_by_rss(ass_vec: &Vec<AssShare<E>>, num_vars: usize) -> Self {
        let mut share_0 = Vec::with_capacity(1 << num_vars);

        let party = ass_vec[0].party;
        for share in ass_vec {
            share_0.push(share.share_0);
        }
        AssPoly {
            party,
            share_0: DenseMultilinearExtension::<E::ScalarField>::from_evaluations_vec(
                num_vars, share_0,
            ),
        }
    }
}

pub struct AssSumcheck<E: Pairing> {
    _pairing: PhantomData<E>,
}

// 1st round: pub * priv * priv
// 2nd round:
pub struct AssSumcheckProverState<E: Pairing> {
    pub secret_polys: Vec<AssPoly<E>>,
    pub pub_polys: Vec<DenseMultilinearExtension<E::ScalarField>>,
    pub randomness: Vec<E::ScalarField>,
    pub round: usize,
    pub num_vars: usize,
    pub party: usize,
    pub coef: Vec<E::ScalarField>,
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct AssSumcheckProverFirstMsg<E: Pairing> {
    pub evaluations: Vec<AssShare<E>>,
}

impl<E: Pairing> AssSumcheckProverFirstMsg<E> {
    pub fn open(msgs: &Vec<Self>) -> Vec<E::ScalarField> {
        assert!(msgs.len() == 2);
        let mut sum = vec![E::ScalarField::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].share_0;
            }
        }
        sum
    }

    pub fn open_to_msg(msgs: &Vec<Self>) -> ProverMsg<E::ScalarField> {
        assert!(msgs.len() == 2);
        let mut sum = vec![E::ScalarField::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].share_0;
            }
        }
        ProverMsg { evaluations: sum }
    }
}

#[derive(CanonicalDeserialize, CanonicalSerialize, Clone)]
pub struct AssSumcheckProverSecondMsg<E: Pairing> {
    pub evaluations: Vec<AssShare<E>>,
}

impl<E: Pairing> AssSumcheckProverSecondMsg<E> {
    pub fn open(msgs: &Vec<Self>) -> Vec<E::ScalarField> {
        assert!(msgs.len() == 2);
        let mut sum = vec![E::ScalarField::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].share_0;
            }
        }
        sum
    }

    pub fn open_to_msg(msgs: &Vec<Self>) -> ProverMsg<E::ScalarField> {
        assert!(msgs.len() == 2);
        let mut sum = vec![E::ScalarField::zero(); msgs[0].evaluations.len()];
        for msg in msgs {
            for i in 0..msg.evaluations.len() {
                sum[i] += msg.evaluations[i].share_0;
            }
        }
        ProverMsg { evaluations: sum }
    }
}

impl<E: Pairing> AssSumcheck<E> {
    pub fn first_sumcheck_init(
        v_a: &AssPoly<E>,
        v_b: &AssPoly<E>,
        v_c: &AssPoly<E>,
        pub1: &DenseMultilinearExtension<E::ScalarField>,
    ) -> AssSumcheckProverState<E> {
        let secret_polys = vec![v_a.clone(), v_b.clone(), v_c.clone()];
        let pub_polys = vec![pub1.clone()];
        AssSumcheckProverState {
            secret_polys,
            pub_polys,
            randomness: Vec::with_capacity(pub1.num_vars),
            round: 0,
            num_vars: pub1.num_vars,
            party: v_a.party,
            coef: vec![],
        }
    }

    pub fn second_sumcheck_init(
        v_a: &DenseMultilinearExtension<E::ScalarField>,
        v_b: &DenseMultilinearExtension<E::ScalarField>,
        v_c: &DenseMultilinearExtension<E::ScalarField>,
        z: &AssPoly<E>,
        v_msg: &Vec<E::ScalarField>,
    ) -> AssSumcheckProverState<E> {
        let pub_polys = vec![v_a.clone(), v_b.clone(), v_c.clone()];
        AssSumcheckProverState {
            secret_polys: vec![z.clone()],
            pub_polys,
            randomness: Vec::with_capacity(v_a.num_vars),
            round: 0,
            num_vars: v_a.num_vars,
            party: z.party,
            coef: v_msg.clone(),
        }
    }

    pub fn first_sumcheck_prove_round_stage_1(
        prover_state: &mut AssSumcheckProverState<E>,
        v_msg: &Option<VerifierMsg<E::ScalarField>>,
        beaver_vec_a: &Vec<Vec<AssShare<E>>>,
        beaver_vec_b: &Vec<Vec<AssShare<E>>>,
    ) -> (Vec<Vec<AssShare<E>>>, Vec<Vec<AssShare<E>>>) {
        if let Some(msg) = v_msg {
            if prover_state.round == 0 {
                panic!("first round should be prover first.");
            }
            prover_state.randomness.push(msg.randomness);

            // fix argument
            let i = prover_state.round;
            let r = prover_state.randomness[i - 1];
            cfg_iter_mut!(prover_state.secret_polys).for_each(|multiplicand| {
                *multiplicand = multiplicand.fix_variables(&[r]);
            });
            cfg_iter_mut!(prover_state.pub_polys).for_each(|multiplicand| {
                *multiplicand = multiplicand.fix_variables(&[r]);
            });

            if prover_state.round == prover_state.num_vars {
                prover_state.round += 1;
                return (Vec::new(), Vec::new());
            }
        } else if prover_state.round > 0 {
            panic!("verifier message is empty");
        }

        prover_state.round += 1;

        if prover_state.round > prover_state.num_vars {
            panic!("Prover is not active");
        }

        let i = prover_state.round;
        let nv = prover_state.num_vars;
        let degree = 3; // the degree of univariate polynomial sent by prover at this round
        let party = prover_state.party;

        let mut vec_d = Vec::new();
        let mut vec_e = Vec::new();

        for b in 0..1 << (nv - i) {
            let mut start_a = prover_state.secret_polys[0].get_ass_by_idx(b << 1);
            let step_a = prover_state.secret_polys[0].get_ass_by_idx((b << 1) + 1) - start_a;
            let mut start_b = prover_state.secret_polys[1].get_ass_by_idx(b << 1);
            let step_b = prover_state.secret_polys[1].get_ass_by_idx((b << 1) + 1) - start_b;
            let mut start_pub1 = prover_state.pub_polys[0][b << 1];
            let step_pub1 = prover_state.pub_polys[0][(b << 1) + 1] - start_pub1;

            let mut vec_d_pd = Vec::new();
            let mut vec_e_pd = Vec::new();
            for t in 0..degree + 1 {
                vec_d_pd.push(start_a - beaver_vec_a[b][t]);
                vec_e_pd.push(start_b - beaver_vec_b[b][t]);
                start_a += step_a;
                start_b += step_b;
            }

            vec_d.push(vec_d_pd);
            vec_e.push(vec_e_pd);
        }

        (vec_d, vec_e)
    }

    pub fn first_sumcheck_prove_round_stage_2<R: RngCore + FeedableRNG>(
        prover_state: &mut AssSumcheckProverState<E>,
        v_msg: &Option<VerifierMsg<E::ScalarField>>,
        beaver_vec_a: &Vec<Vec<AssShare<E>>>,
        beaver_vec_b: &Vec<Vec<AssShare<E>>>,
        beaver_vec_ab: &Vec<Vec<AssShare<E>>>,
        vec_d: &Vec<Vec<E::ScalarField>>,
        vec_e: &Vec<Vec<E::ScalarField>>,
        rng: &mut SSRandom<R>,
    ) -> AssSumcheckProverFirstMsg<E> {
        if prover_state.round > prover_state.num_vars {
            panic!("Prover is not active");
        }

        let i = prover_state.round;
        let nv = prover_state.num_vars;
        let degree = 3; // the degree of univariate polynomial sent by prover at this round
        let party = prover_state.party;

        #[cfg(not(feature = "parallel"))]
        let zeros = (
            vec![AssShare::<E>::get_zero_ass(party); degree + 1],
            vec![AssShare::<E>::get_zero_ass(party); degree + 1],
        );
        #[cfg(feature = "parallel")]
        let zeros = || {
            (
                vec![AssShare::<E>::get_zero_ass(party); degree + 1],
                vec![AssShare::<E>::get_zero_ass(party); degree + 1],
            )
        };

        // generate sum
        let fold_result = ark_std::cfg_into_iter!(0..1 << (nv - i), 1 << 10).fold(
            zeros,
            |(mut products_sum, mut product), b| {
                // In effect, this fold is essentially doing simply:
                // for b in 0..1 << (nv - i) {

                let mut start_a = prover_state.secret_polys[0].get_ass_by_idx(b << 1);
                let step_a = prover_state.secret_polys[0].get_ass_by_idx((b << 1) + 1) - start_a;
                let mut start_b = prover_state.secret_polys[1].get_ass_by_idx(b << 1);
                let step_b = prover_state.secret_polys[1].get_ass_by_idx((b << 1) + 1) - start_b;
                let mut start_pub1 = prover_state.pub_polys[0][b << 1];
                let step_pub1 = prover_state.pub_polys[0][(b << 1) + 1] - start_pub1;

                let mut t = 0;
                for p in product.iter_mut() {
                    *p = AssShare {
                        party: party,
                        share_0: AssShare::<E>::ass_mul(
                            &start_a,
                            &start_b,
                            &beaver_vec_a[b][t],
                            &beaver_vec_b[b][t],
                            &beaver_vec_ab[b][t],
                            vec_d[b][t],
                            vec_e[b][t],
                        ) * &start_pub1,
                    };
                    t += 1;
                    start_a += step_a;
                    start_b += step_b;
                    start_pub1 += step_pub1;
                }

                let mut start_c = prover_state.secret_polys[2].get_ass_by_idx(b << 1);
                let step_c = prover_state.secret_polys[2].get_ass_by_idx((b << 1) + 1) - start_c;
                let mut start_pub1 = prover_state.pub_polys[0][b << 1];
                let step_pub1 = prover_state.pub_polys[0][(b << 1) + 1] - start_pub1;

                for p in product.iter_mut() {
                    *p -= start_c * start_pub1;
                    start_c += step_c;
                    start_pub1 += step_pub1;
                }

                for t in 0..degree + 1 {
                    products_sum[t] += product[t];
                }

                (products_sum, product)
            },
        );

        #[cfg(not(feature = "parallel"))]
        let products_sum = fold_result.0;

        // When rayon is used, the `fold` operation results in a iterator of `Vec<F>` rather than a single `Vec<F>`. In this case, we simply need to sum them.
        #[cfg(feature = "parallel")]
        let mut products_sum = fold_result.map(|scratch| scratch.0).reduce(
            || vec![AssShare::<E>::get_zero_ass(party); degree + 1],
            |mut overall_products_sum, sublist_sum| {
                overall_products_sum
                    .iter_mut()
                    .zip(sublist_sum.iter())
                    .for_each(|(f, s)| *f += s);
                overall_products_sum
            },
        );

        for i in products_sum.iter_mut() {
            i.share_0 += AssShare::<E>::get_zero_share(rng);
        }

        AssSumcheckProverFirstMsg {
            evaluations: products_sum,
        }
    }

    pub fn second_sumcheck_prove_round<R: RngCore + FeedableRNG>(
        prover_state: &mut AssSumcheckProverState<E>,
        v_msg: &Option<VerifierMsg<E::ScalarField>>,
        rng: &mut SSRandom<R>,
    ) -> AssSumcheckProverSecondMsg<E> {
        if let Some(msg) = v_msg {
            if prover_state.round == 0 {
                panic!("first round should be prover first.");
            }
            prover_state.randomness.push(msg.randomness);

            // fix argument
            let i = prover_state.round;
            let r = prover_state.randomness[i - 1];
            cfg_iter_mut!(prover_state.secret_polys).for_each(|multiplicand| {
                *multiplicand = multiplicand.fix_variables(&[r]);
            });
            cfg_iter_mut!(prover_state.pub_polys).for_each(|multiplicand| {
                *multiplicand = multiplicand.fix_variables(&[r]);
            });

            if prover_state.round == prover_state.num_vars {
                prover_state.round += 1;
                return AssSumcheckProverSecondMsg {
                    evaluations: Vec::new(),
                };
            }
        } else if prover_state.round > 0 {
            panic!("verifier message is empty");
        }

        prover_state.round += 1;

        if prover_state.round > prover_state.num_vars {
            panic!("Prover is not active");
        }

        let i = prover_state.round;
        let nv = prover_state.num_vars;
        let degree = 2; // the degree of univariate polynomial sent by prover at this round
        let party = prover_state.party;

        #[cfg(not(feature = "parallel"))]
        let zeros = (
            vec![AssShare::<E>::get_zero_ass(party); degree + 1],
            vec![AssShare::<E>::get_zero_ass(party); degree + 1],
        );
        #[cfg(feature = "parallel")]
        let zeros = || {
            (
                vec![AssShare::<E>::get_zero_ass(party); degree + 1],
                vec![AssShare::<E>::get_zero_ass(party); degree + 1],
            )
        };

        // generate sum
        let fold_result = ark_std::cfg_into_iter!(0..1 << (nv - i), 1 << 10).fold(
            zeros,
            |(mut products_sum, mut product), b| {
                // In effect, this fold is essentially doing simply:
                // for b in 0..1 << (nv - i) {

                let mut start_a = prover_state.pub_polys[0][b << 1];
                let step_a = prover_state.pub_polys[0][(b << 1) + 1] - start_a;
                let mut start_b = prover_state.pub_polys[1][b << 1];
                let step_b = prover_state.pub_polys[1][(b << 1) + 1] - start_b;
                let mut start_c = prover_state.pub_polys[2][b << 1];
                let step_c = prover_state.pub_polys[2][(b << 1) + 1] - start_c;
                let mut start_z = prover_state.secret_polys[0].get_ass_by_idx(b << 1);
                let step_z = prover_state.secret_polys[0].get_ass_by_idx((b << 1) + 1) - start_z;

                for p in product.iter_mut() {
                    *p = start_z
                        * (start_a * prover_state.coef[0]
                            + start_b * prover_state.coef[1]
                            + start_c * prover_state.coef[2]);
                    start_a += step_a;
                    start_b += step_b;
                    start_c += step_c;
                    start_z += step_z;
                }

                for t in 0..degree + 1 {
                    products_sum[t] += &product[t];
                }

                (products_sum, product)
            },
        );

        #[cfg(not(feature = "parallel"))]
        let products_sum = fold_result.0;

        // When rayon is used, the `fold` operation results in a iterator of `Vec<F>` rather than a single `Vec<F>`. In this case, we simply need to sum them.
        #[cfg(feature = "parallel")]
        let mut products_sum = fold_result.map(|scratch| scratch.0).reduce(
            || vec![AssShare::<E>::get_zero_ass(party); degree + 1],
            |mut overall_products_sum, sublist_sum| {
                overall_products_sum
                    .iter_mut()
                    .zip(sublist_sum.iter())
                    .for_each(|(f, s)| *f += s);
                overall_products_sum
            },
        );
        for i in products_sum.iter_mut() {
            let mask_0 = AssShare::<E>::get_zero_share(rng);
            i.share_0 += mask_0;
        }
        AssSumcheckProverSecondMsg {
            evaluations: products_sum,
        }
    }
}

pub fn prove_as_subprotocol_first_round<E: Pairing, R: RngCore + FeedableRNG>(
    fs_rng: &mut impl FeedableRNG<Error = Error>,
    mut prover_state: Vec<AssSumcheckProverState<E>>,
    random_rng: &mut Vec<SSRandom<R>>,
    beaver_vec_a_0: &Vec<Vec<Vec<AssShare<E>>>>,
    beaver_vec_b_0: &Vec<Vec<Vec<AssShare<E>>>>,
    beaver_vec_ab_0: &Vec<Vec<Vec<AssShare<E>>>>,
    beaver_vec_a_1: &Vec<Vec<Vec<AssShare<E>>>>,
    beaver_vec_b_1: &Vec<Vec<Vec<AssShare<E>>>>,
    beaver_vec_ab_1: &Vec<Vec<Vec<AssShare<E>>>>,
) -> Result<(Proof<E::ScalarField>, Vec<AssSumcheckProverState<E>>), Error> {
    //fs_rng.feed(&polynomial.info())?;

    let num_vars = prover_state[0].num_vars;
    let mut verifier_msg = None;
    let mut prover_msgs = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let mut prover_msg = vec![];

        let (d_0, e_0) = AssSumcheck::<E>::first_sumcheck_prove_round_stage_1(
            &mut prover_state[0],
            &verifier_msg,
            &beaver_vec_a_0[round],
            &beaver_vec_b_0[round],
        );

        let (d_1, e_1) = AssSumcheck::<E>::first_sumcheck_prove_round_stage_1(
            &mut prover_state[1],
            &verifier_msg,
            &beaver_vec_a_1[round],
            &beaver_vec_b_1[round],
        );

        let mut e = Vec::new();
        let mut d = Vec::new();

        for i in 0..e_0.len() {
            let mut e_first = Vec::new();
            let mut d_first = Vec::new();
            for j in 0..e_0[i].len() {
                e_first.push(AssShare::open(&[e_0[i][j], e_1[i][j]]));
                d_first.push(AssShare::open(&[d_0[i][j], d_1[i][j]]));
            }
            e.push(e_first);
            d.push(d_first);
        }

        prover_msg.push(AssSumcheck::<E>::first_sumcheck_prove_round_stage_2(
            &mut prover_state[0],
            &verifier_msg,
            &beaver_vec_a_0[round],
            &beaver_vec_b_0[round],
            &beaver_vec_ab_0[round],
            &d,
            &e,
            &mut random_rng[0],
        ));

        prover_msg.push(AssSumcheck::<E>::first_sumcheck_prove_round_stage_2(
            &mut prover_state[1],
            &verifier_msg,
            &beaver_vec_a_1[round],
            &beaver_vec_b_1[round],
            &beaver_vec_ab_1[round],
            &d,
            &e,
            &mut random_rng[1],
        ));

        // let prover_msg = ProverFirstMsg::<E>::open(&prover_msg);
        // let prover_msg = ProverMsg {
        //     evaluations: prover_msg,
        // };
        let prover_msg = AssSumcheckProverFirstMsg::<E>::open_to_msg(&prover_msg);
        fs_rng.feed(&prover_msg)?;
        prover_msgs.push(prover_msg);
        verifier_msg = Some(IPForMLSumcheck::sample_round(fs_rng));
    }
    if let Some(msg) = verifier_msg {
        prover_state
            .iter_mut()
            .for_each(|x| x.randomness.push(msg.randomness));
    }
    Ok((prover_msgs, prover_state))
}

pub fn prove_as_subprotocol_second_round<E: Pairing, R: RngCore + FeedableRNG>(
    fs_rng: &mut impl FeedableRNG<Error = Error>,
    mut prover_state: Vec<AssSumcheckProverState<E>>,
    random_rng: &mut Vec<SSRandom<R>>,
) -> Result<(Proof<E::ScalarField>, Vec<AssSumcheckProverState<E>>), Error> {
    //fs_rng.feed(&polynomial.info())?;

    let num_vars = prover_state[0].num_vars;
    let mut verifier_msg = None;
    let mut prover_msgs = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let mut prover_msg = vec![];
        for party in 0..=1 {
            prover_msg.push(AssSumcheck::<E>::second_sumcheck_prove_round(
                &mut prover_state[party],
                &verifier_msg,
                &mut random_rng[party],
            ));
        }
        let prover_msg = AssSumcheckProverSecondMsg::<E>::open(&prover_msg);
        let prover_msg = ProverMsg {
            evaluations: prover_msg,
        };
        fs_rng.feed(&prover_msg)?;
        prover_msgs.push(prover_msg);
        verifier_msg = Some(IPForMLSumcheck::sample_round(fs_rng));
    }
    if let Some(msg) = verifier_msg {
        prover_state
            .iter_mut()
            .for_each(|x| x.randomness.push(msg.randomness));
    }
    Ok((prover_msgs, prover_state))
}

pub fn prove_as_subprotocol_test<F: Field>(
    fs_rng: &mut impl FeedableRNG<Error = Error>,
    polynomial: &ListOfProductsOfPolynomials<F>,
) -> Result<
    (
        Proof<F>,
        ark_linear_sumcheck::ml_sumcheck::protocol::prover::ProverState<F>,
    ),
    Error,
> {
    let mut prover_state = IPForMLSumcheck::prover_init(polynomial);
    let mut verifier_msg = None;
    let mut prover_msgs = Vec::with_capacity(polynomial.num_variables);
    for _ in 0..polynomial.num_variables {
        let prover_msg = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg);
        fs_rng.feed(&prover_msg)?;
        prover_msgs.push(prover_msg);
        verifier_msg = Some(IPForMLSumcheck::sample_round(fs_rng));
    }
    if let Some(msg) = verifier_msg {
        prover_state.randomness.push(msg.randomness);
    }
    Ok((prover_msgs, prover_state))
}
